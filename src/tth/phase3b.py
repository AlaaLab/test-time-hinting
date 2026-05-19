import argparse
import asyncio
import json

import torch
import yaml
from datasets import Dataset
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from .prompts import PHASE2_EXPERIMENTER_SYSTEM, PHASE2_EXPERIMENTER_USER, _fmt
from .providers import extract_json, get_client


def _kw(m):
    return {k: v for k, v in m.items() if k not in ("provider", "model", "tag")}


def _eq(a, b):
    return str(a or "").strip().upper() == str(b or "").strip().upper()


def _build_dataset(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ip = r["images"][0]
            img = Image.open(ip).convert("RGB")
            text = r["messages"][0]["content"]
            prompt = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }]
            rows.append({
                "prompt": prompt,
                "images": [img],
                "image_path": ip,
                "row_id": int(r["row_id"]),
                "question_text": r["question_text"],
                "answer_gt": r["answer_gt"],
                "base_correct_by_tag_json": json.dumps(r["base_correct_by_tag"], sort_keys=True),
            })
    return Dataset.from_list(rows)


def _extract_text(comp):
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp:
        last = comp[-1]
        if isinstance(last, dict):
            c = last.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                out = []
                for blk in c:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        out.append(str(blk.get("text", "")))
                return "".join(out)
    if isinstance(comp, dict):
        c = comp.get("content")
        if isinstance(c, str):
            return c
    return str(comp)


def _per_target(hinted_correct, base_correct):
    if hinted_correct and not base_correct:
        return 1.0
    if hinted_correct and base_correct:
        return 0.0
    if not hinted_correct and base_correct:
        return -1.0
    return -0.5


def _make_reward_fn(cfg):
    tags = list(cfg["reward_target_tags"])
    targets = cfg["target_models"]
    clients = {}

    def ci(prov):
        if prov not in clients:
            clients[prov] = get_client(prov)
        return clients[prov]

    async def _one_target(tag, image_path, question, hint_json, gt):
        m = targets[tag]
        u = _fmt(PHASE2_EXPERIMENTER_USER, QUESTION=question, HINT_JSON=hint_json)
        try:
            t = await ci(m["provider"]).call(
                image_path=image_path,
                system_prompt=PHASE2_EXPERIMENTER_SYSTEM,
                user_prompt=u,
                model=m["model"],
                **_kw(m),
            )
        except Exception:
            return False
        o = extract_json(t) or {}
        return _eq(o.get("answer", ""), gt)

    async def _one(comp, image_path, question, gt, base_corr):
        text = _extract_text(comp)
        obj = extract_json(text)
        if obj and isinstance(obj.get("hint"), list):
            hj = json.dumps({"hint": obj["hint"]}, ensure_ascii=False, sort_keys=True)
        else:
            hj = text or ""
        coros = [_one_target(t, image_path, question, hj, gt) for t in tags]
        hc = await asyncio.gather(*coros)
        rewards = [_per_target(bool(h), bool(base_corr.get(t, False))) for t, h in zip(tags, hc)]
        return sum(rewards) / len(rewards) if rewards else 0.0

    async def _all(comps, ips, qs, gts, bcs):
        return list(await asyncio.gather(*[
            _one(comps[i], ips[i], qs[i], gts[i], bcs[i]) for i in range(len(comps))
        ]))

    def reward_fn(prompts, completions, **kwargs):
        n = len(completions)
        ips = list(kwargs.get("image_path") or [""] * n)
        qs = list(kwargs.get("question_text") or [""] * n)
        gts = list(kwargs.get("answer_gt") or [""] * n)
        bcjs = list(kwargs.get("base_correct_by_tag_json") or ["{}"] * n)
        bcs = [json.loads(s) for s in bcjs]
        return asyncio.run(_all(completions, ips, qs, gts, bcs))

    return reward_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    base = AutoModelForImageTextToText.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, device_map="auto",
    )
    policy = PeftModel.from_pretrained(base, cfg["init_from"], is_trainable=True)
    proc = AutoProcessor.from_pretrained(cfg["model"])
    ds = _build_dataset(cfg["train_jsonl"])
    rfn = _make_reward_fn(cfg)

    grpo_cfg = GRPOConfig(output_dir=cfg["output_dir"], **cfg["grpo_args"])
    trainer = GRPOTrainer(
        model=policy,
        reward_funcs=rfn,
        args=grpo_cfg,
        train_dataset=ds,
        processing_class=proc,
    )
    trainer.train()
    trainer.model.save_pretrained(cfg["output_dir"] + "/final_adapter")


if __name__ == "__main__":
    main()
