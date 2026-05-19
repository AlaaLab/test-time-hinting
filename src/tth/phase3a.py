import argparse
import subprocess

import yaml


LLM_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MERGER_TARGETS = (
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.deepstack_merger_list.0.linear_fc1",
    "model.visual.deepstack_merger_list.0.linear_fc2",
    "model.visual.deepstack_merger_list.1.linear_fc1",
    "model.visual.deepstack_merger_list.1.linear_fc2",
    "model.visual.deepstack_merger_list.2.linear_fc1",
    "model.visual.deepstack_merger_list.2.linear_fc2",
)
VIT_LEAFS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def _vit_paths(n_blocks, last_n):
    if last_n is None:
        rng = range(n_blocks)
    else:
        rng = range(n_blocks - last_n, n_blocks)
    return [f"model.visual.blocks.{i}.{leaf}" for i in rng for leaf in VIT_LEAFS]


def _resolve_targets(scope, n_blocks):
    if scope == "full_FT":
        return [], "full"
    targets = list(LLM_TARGETS)
    if scope == "llm_only":
        return targets, "lora"
    targets += list(MERGER_TARGETS)
    if scope == "llm+merger":
        return targets, "lora"
    if scope == "llm+merger+vit_last2":
        return targets + _vit_paths(n_blocks, 2), "lora"
    if scope == "llm+merger+vit_full":
        return targets + _vit_paths(n_blocks, None), "lora"
    raise ValueError(f"unknown tuning_scope: {scope!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)

    args = ["swift", "sft", "--model", str(cfg["model"]), "--dataset", str(cfg["dataset"])]
    if cfg.get("val_dataset"):
        args += ["--val_dataset", str(cfg["val_dataset"])]
    args += ["--output_dir", str(cfg["output_dir"])]

    if "tuning_scope" in cfg:
        n_blocks = int(cfg.get("vit_n_blocks", 24))
        targets, tuner = _resolve_targets(cfg["tuning_scope"], n_blocks)
        args += ["--tuner_type", tuner]
        if targets:
            args += ["--target_modules", *targets]

    sft = dict(cfg["sft_args"])
    ebs = sft.pop("effective_batch_size", None)
    if ebs is not None and "gradient_accumulation_steps" not in sft:
        pdbs = int(sft["per_device_train_batch_size"])
        sft["gradient_accumulation_steps"] = max(1, int(ebs) // max(1, pdbs))
    for k, v in sft.items():
        args.extend([f"--{k}", str(v)])

    for x in cfg.get("extra_args", []):
        args.append(str(x))

    subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
