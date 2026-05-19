import argparse
import asyncio

import pandas as pd
import yaml

from .prompts import PHASE1_VQA_SYSTEM, PHASE1_VQA_USER, _fmt
from .providers import extract_json, get_client


async def _one(client, m, image_path, question):
    user = _fmt(PHASE1_VQA_USER, QUESTION=question)
    kw = {k: v for k, v in m.items() if k not in ("provider", "model", "tag")}
    text = await client.call(
        image_path=image_path,
        system_prompt=PHASE1_VQA_SYSTEM,
        user_prompt=user,
        model=m["model"],
        **kw,
    )
    obj = extract_json(text) or {}
    return obj.get("answer", "") or "", obj.get("reasoning", "") or "", text or ""


async def _amain(cfg):
    df = pd.read_csv(cfg["input_csv"])
    n = len(df)
    clients = {}
    for m in cfg["models"]:
        prov = m["provider"]
        if prov not in clients:
            clients[prov] = get_client(prov)
        cli = clients[prov]
        tasks = [
            _one(cli, m, str(df.at[i, "image_path"]), str(df.at[i, "question"]))
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)
        tag = m["tag"]
        df[f"{tag}_answer"] = [r[0] for r in results]
        df[f"{tag}_reasoning"] = [r[1] for r in results]
        df[f"{tag}_raw_text"] = [r[2] for r in results]
    df.to_csv(cfg["input_csv"], index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    asyncio.run(_amain(cfg))


if __name__ == "__main__":
    main()
