"""Phase 2: agentic hint optimization (Proposer / Checker / Experimenter loop)."""
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

from .prompts import (
    PHASE2_CHECKER_SYSTEM,
    PHASE2_CHECKER_USER,
    PHASE2_EXPERIMENTER_SYSTEM,
    PHASE2_EXPERIMENTER_USER,
    PHASE2_PROPOSER_REINFORCE_SYSTEM,
    PHASE2_PROPOSER_REINFORCE_USER,
    PHASE2_PROPOSER_REPAIR_SYSTEM,
    PHASE2_PROPOSER_REPAIR_USER,
    _fmt,
)
from .providers import extract_json, get_client


PROPOSER_PARSE_RETRIES = 2
PROPOSER_RETRY_SLEEP = 0.25
DEFAULT_MAX_CONCURRENT = 4


def _kw(m: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in m.items() if k not in ("provider", "model", "tag", "max_concurrent")}


def _eq(a: Any, b: Any) -> bool:
    return str(a or "").strip().upper() == str(b or "").strip().upper()


def _row_get(row: Dict[str, Any], col: str, default: str = "") -> str:
    v = row.get(col, default)
    if v is None or (isinstance(v, float) and str(v) == "nan"):
        return default
    return str(v).strip()


def _hint_dumps(hint: List[str]) -> str:
    return json.dumps({"hint": hint}, ensure_ascii=False)


def _validate_hint(obj: Any) -> Optional[List[str]]:
    if not isinstance(obj, dict):
        return None
    h = obj.get("hint")
    if not isinstance(h, list) or not (1 <= len(h) <= 3):
        return None
    cleaned: List[str] = []
    for item in h:
        if not isinstance(item, str):
            return None
        s = item.strip()
        if not s:
            return None
        cleaned.append(s)
    return cleaned


def _validate_checker(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict", "")).strip().lower()
    if verdict not in ("pass", "revise"):
        return None
    feedback = obj.get("feedback", "")
    if not isinstance(feedback, str):
        return None
    hint = _validate_hint({"hint": obj.get("hint")})
    if hint is None:
        return None
    if verdict == "pass":
        feedback = ""
    return {"verdict": verdict, "feedback": feedback.strip(), "hint": hint}


def _validate_experimenter(obj: Any) -> Optional[Dict[str, str]]:
    if not isinstance(obj, dict):
        return None
    answer = obj.get("answer")
    reasoning = obj.get("reasoning")
    if not isinstance(answer, str) or not isinstance(reasoning, str):
        return None
    answer = answer.strip()
    if not answer:
        return None
    return {"answer": answer, "reasoning": reasoning.strip()}


@asynccontextmanager
async def _acq(sem: Optional[asyncio.Semaphore]):
    if sem is None:
        yield
    else:
        async with sem:
            yield


async def _call_role(
    client_for_provider: Callable[[str], Any],
    role_cfg: Dict[str, Any],
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    sem: Optional[asyncio.Semaphore],
) -> str:
    async with _acq(sem):
        client = client_for_provider(role_cfg["provider"])
        return await client.call(
            image_path=image_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=role_cfg["model"],
            **_kw(role_cfg),
        )


async def _propose(
    client_for_provider: Callable[[str], Any],
    proposer_cfg: Dict[str, Any],
    image_path: str,
    system_prompt: str,
    user_prompt: str,
    sem: Optional[asyncio.Semaphore],
) -> Optional[List[str]]:
    for _ in range(PROPOSER_PARSE_RETRIES + 1):
        try:
            text = await _call_role(
                client_for_provider, proposer_cfg, image_path,
                system_prompt, user_prompt, sem,
            )
        except Exception:
            text = ""
        hint = _validate_hint(extract_json(text))
        if hint is not None:
            return hint
        if PROPOSER_RETRY_SLEEP > 0:
            await asyncio.sleep(PROPOSER_RETRY_SLEEP)
    return None


async def _check(
    client_for_provider: Callable[[str], Any],
    checker_cfg: Dict[str, Any],
    image_path: str,
    user_prompt: str,
    sem: Optional[asyncio.Semaphore],
) -> Optional[Dict[str, Any]]:
    try:
        text = await _call_role(
            client_for_provider, checker_cfg, image_path,
            PHASE2_CHECKER_SYSTEM, user_prompt, sem,
        )
    except Exception:
        return None
    return _validate_checker(extract_json(text))


async def _experiment(
    client_for_provider: Callable[[str], Any],
    exp_cfg: Dict[str, Any],
    image_path: str,
    user_prompt: str,
    sem: Optional[asyncio.Semaphore],
) -> Optional[Dict[str, str]]:
    try:
        text = await _call_role(
            client_for_provider, exp_cfg, image_path,
            PHASE2_EXPERIMENTER_SYSTEM, user_prompt, sem,
        )
    except Exception:
        return None
    return _validate_experimenter(extract_json(text))


async def _run_row(
    cfg: Dict[str, Any],
    client_for_provider: Callable[[str], Any],
    row: Dict[str, Any],
    max_rounds: int,
    sems: Dict[str, Optional[asyncio.Semaphore]],
) -> Dict[str, Any]:
    image_path = _row_get(row, "image_path")
    question = _row_get(row, "question")
    caption = _row_get(row, "caption")
    gt_answer = _row_get(row, "gt_answer")
    base_answer = _row_get(row, "base_answer")
    base_reasoning = _row_get(row, "base_reasoning")
    mode = _row_get(row, "mode").lower()
    target_tag = _row_get(row, "target_tag")
    is_repair = mode == "repair"

    proposer_cfg = cfg["proposer"]
    checker_cfg = cfg["checker"]
    exp_cfg = cfg["experimenters"][target_tag]
    skip_checker = bool(cfg.get("skip_checker", False))

    if is_repair:
        proposer_system = PHASE2_PROPOSER_REPAIR_SYSTEM
        proposer_template = PHASE2_PROPOSER_REPAIR_USER
    else:
        proposer_system = PHASE2_PROPOSER_REINFORCE_SYSTEM
        proposer_template = PHASE2_PROPOSER_REINFORCE_USER

    feedback = ""
    last_exp_answer = ""
    last_exp_reasoning = ""
    candidates: List[Dict[str, Any]] = []
    first_pass_idx: Optional[int] = None
    trajectory: Dict[str, Any] = {"rounds": []}

    for rd in range(1, int(max_rounds) + 1):
        intro = f"Round {rd}\n\n" if rd > 1 else ""
        proposer_user = _fmt(
            proposer_template,
            INTRO=intro,
            QUESTION=question,
            CAPTION=caption,
            GROUND_TRUTH_ANSWER=gt_answer,
            BASE_ANSWER=base_answer,
            BASE_REASONING=base_reasoning,
            OPTIONAL_CHECKER_FEEDBACK=feedback,
            OPTIONAL_EXPERIMENTER_LAST_ANSWER=last_exp_answer,
            OPTIONAL_EXPERIMENTER_LAST_REASONING=last_exp_reasoning,
        )
        hint = await _propose(
            client_for_provider, proposer_cfg, image_path,
            proposer_system, proposer_user, sems.get("proposer"),
        )
        if hint is None:
            trajectory["rounds"].append({
                "round": rd, "stage": "proposer", "error": "parse_failed",
            })
            continue

        verdict: Optional[str] = None
        checker_feedback = ""
        used_hint = list(hint)

        if not skip_checker:
            checker_user = _fmt(
                PHASE2_CHECKER_USER,
                QUESTION=question,
                CAPTION=caption,
                GROUND_TRUTH_ANSWER=gt_answer,
                BASE_ANSWER=base_answer,
                BASE_REASONING=base_reasoning,
                HINT_JSON=_hint_dumps(hint),
            )
            checked = await _check(
                client_for_provider, checker_cfg, image_path,
                checker_user, sems.get("checker"),
            )
            if checked is None:
                verdict = "revise"
                checker_feedback = "checker_parse_failed"
            else:
                verdict = checked["verdict"]
                checker_feedback = checked["feedback"]
                used_hint = checked["hint"]
            if verdict == "pass" and first_pass_idx is None:
                first_pass_idx = len(candidates)

        feedback = checker_feedback
        hint_json = _hint_dumps(used_hint)
        exp_user = _fmt(PHASE2_EXPERIMENTER_USER, QUESTION=question, HINT_JSON=hint_json)
        experimented = await _experiment(
            client_for_provider, exp_cfg, image_path,
            exp_user, sems.get("experimenter"),
        )
        if experimented is None:
            exp_answer = ""
            exp_reasoning = ""
            exp_correct = False
        else:
            exp_answer = experimented["answer"]
            exp_reasoning = experimented["reasoning"]
            exp_correct = _eq(exp_answer, gt_answer)

        last_exp_answer = exp_answer
        last_exp_reasoning = exp_reasoning

        candidates.append({
            "round": rd,
            "hint": used_hint,
            "verdict": verdict,
            "feedback": checker_feedback,
            "exp_answer": exp_answer,
            "exp_reasoning": exp_reasoning,
            "exp_correct": exp_correct,
        })
        trajectory["rounds"].append({
            "round": rd,
            "hint_json": hint_json,
            "checker_verdict": verdict,
            "checker_feedback": checker_feedback,
            "exp_answer": exp_answer,
            "exp_correct": exp_correct,
        })

        if exp_correct:
            outcome = "repair_success" if is_repair else "reinforce_success"
            return {
                "hint_json": hint_json,
                "selected_round": rd,
                "outcome": outcome,
                "trajectory": json.dumps(trajectory, ensure_ascii=False),
            }

    if not candidates:
        return {
            "hint_json": "",
            "selected_round": 0,
            "outcome": "no_candidates",
            "trajectory": json.dumps(trajectory, ensure_ascii=False),
        }

    if is_repair:
        if not skip_checker and first_pass_idx is not None:
            chosen = candidates[first_pass_idx]
        else:
            chosen = candidates[-1]
        return {
            "hint_json": _hint_dumps(chosen["hint"]),
            "selected_round": chosen["round"],
            "outcome": "partial_repair",
            "trajectory": json.dumps(trajectory, ensure_ascii=False),
        }

    return {
        "hint_json": "",
        "selected_round": 0,
        "outcome": "discard",
        "trajectory": json.dumps(trajectory, ensure_ascii=False),
    }


def _make_sems(cfg: Dict[str, Any]) -> Dict[str, Optional[asyncio.Semaphore]]:
    out: Dict[str, Optional[asyncio.Semaphore]] = {}
    for role in ("proposer", "checker"):
        n = int((cfg.get(role) or {}).get("max_concurrent") or DEFAULT_MAX_CONCURRENT)
        out[role] = asyncio.Semaphore(max(1, n))
    n_exp = DEFAULT_MAX_CONCURRENT
    for em in (cfg.get("experimenters") or {}).values():
        n_exp = max(n_exp, int(em.get("max_concurrent") or DEFAULT_MAX_CONCURRENT))
    out["experimenter"] = asyncio.Semaphore(max(1, n_exp))
    return out


async def _amain(cfg: Dict[str, Any]) -> None:
    df = pd.read_csv(cfg["input_csv"])
    max_rounds = int(cfg["max_hint_rounds"])

    clients: Dict[str, Any] = {}

    def client_for_provider(prov: str) -> Any:
        if prov not in clients:
            clients[prov] = get_client(prov)
        return clients[prov]

    sems = _make_sems(cfg)
    rows = [df.iloc[i].to_dict() for i in range(len(df))]
    tasks = [_run_row(cfg, client_for_provider, r, max_rounds, sems) for r in rows]
    results = await asyncio.gather(*tasks)

    df["hint_json"] = [r["hint_json"] for r in results]
    df["selected_round"] = [r["selected_round"] for r in results]
    df["outcome"] = [r["outcome"] for r in results]
    df["trajectory"] = [r["trajectory"] for r in results]
    df.to_csv(cfg["input_csv"], index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
