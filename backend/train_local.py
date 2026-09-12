"""
Local LoRA training for one persona.

1. Rebuilds train/valid.jsonl with the selected model's tokenizer, so
   example lengths are measured with the template that will actually be used.
2. Runs lora_train.py in a subprocess (frees all GPU memory when it exits)
   and relays its progress.
3. Trains into a staging directory and swaps it in only on success, so a
   failed retrain never leaves a half-written persona behind.

Usage:
    python train_local.py --name hudson --model qwen-3b
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import prepare_data
from generation import resolve_settings

BACKEND_DIR = Path(__file__).resolve().parent
EVENT_PREFIX = "@@personabot "
MLX_PROGRESS_RE = re.compile(r"^(Iter \d+: |Calculating loss)")

MODELS = {
    # tier drives batch size / gradient checkpointing (16 GB unified memory).
    # "unsloth" is the matching 4-bit CUDA build for the Colab notebook.
    "qwen-3b": {"repo": "mlx-community/Qwen2.5-3B-Instruct-4bit", "tier": "small",
                "unsloth": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"},
    "qwen-7b": {"repo": "mlx-community/Qwen2.5-7B-Instruct-4bit", "tier": "large",
                "unsloth": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"},
    "llama-3b": {"repo": "mlx-community/Llama-3.2-3B-Instruct-4bit", "tier": "small",
                 "unsloth": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"},
    "llama-8b": {"repo": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "tier": "large",
                 "unsloth": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"},
    # Base (non-instruct) checkpoints: no assistant-persona tuning to fight.
    # Qwen2.5 base ships the same chat template, so the pipeline is unchanged.
    "qwen-3b-base": {"repo": "mlx-community/Qwen2.5-3B-4bit", "tier": "small",
                     "unsloth": "unsloth/Qwen2.5-3B-bnb-4bit"},
    "qwen-7b-base": {"repo": "mlx-community/Qwen2.5-7B-4bit", "tier": "large",
                     "unsloth": "unsloth/Qwen2.5-7B-bnb-4bit"},
}


def pick_hparams(n_train, model_key, rank=16, num_layers=None, learning_rate=1e-5):
    """About one epoch with warmup + cosine decay. Validation runs ~10 times;
    early stopping and checkpoint selection handle overfitting, so the budget
    no longer has to be cut short up front."""
    tier = MODELS[model_key]["tier"]
    batch_size = 2 if tier == "small" else 1
    grad_accum = 1 if tier == "small" else 2
    iters = max(60, min(4000, math.ceil(n_train / batch_size)))
    updates = max(1, iters // grad_accum)
    warmup = max(3, updates // 20)
    return {
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accum,
        "iters": iters,
        "steps_per_eval": max(10, iters // 10),
        "steps_per_report": max(5, iters // 40),
        "num_layers": num_layers if num_layers is not None else (24 if tier == "small" else 16),
        "rank": rank,
        "grad_checkpoint": tier != "small",
        "learning_rate": learning_rate,
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [learning_rate, max(1, updates - warmup), learning_rate * 0.1],
            "warmup": warmup,
            "warmup_init": learning_rate * 0.05,
        },
    }


def staging_dir_for(adapter_dir):
    adapter_dir = Path(adapter_dir)
    return adapter_dir.with_name(adapter_dir.name + ".staging")


def swap_in(staging, adapter_dir):
    """Replace adapter_dir with staging, keeping the user's settings.json."""
    staging, adapter_dir = Path(staging), Path(adapter_dir)
    old_settings = adapter_dir / "settings.json"
    if old_settings.exists() and not (staging / "settings.json").exists():
        shutil.copy(old_settings, staging / "settings.json")
    backup = adapter_dir.with_name(adapter_dir.name + ".old")
    shutil.rmtree(backup, ignore_errors=True)
    if adapter_dir.exists():
        adapter_dir.rename(backup)
    staging.rename(adapter_dir)
    shutil.rmtree(backup, ignore_errors=True)


def persona_meta_from(data_meta, **extra):
    """Everything chat needs to reproduce the training-time prompt."""
    keys = ("persona", "display_name", "me_label", "prompt_style", "context_turns",
            "system_prompt", "reply_tokens_p95", "format_version")
    return {**{k: data_meta.get(k) for k in keys}, **extra}


def run_training(name, data_dir, adapter_dir, model_key="qwen-3b", learning_rate=1e-5,
                 rank=16, num_layers=None, style_select=True, log=print):
    data_dir, adapter_dir = Path(data_dir), Path(adapter_dir)
    repo = MODELS[model_key]["repo"]

    log(f"Sizing examples with the {repo} tokenizer…")
    data_meta = prepare_data.build_dataset(data_dir, tokenizer_model=repo)
    log(f"{data_meta['n_train']} train rows ({data_meta['n_train_replies']} replies) · {data_meta['n_val']} validation rows")
    if data_meta["n_train"] < 4 or data_meta["n_val"] < 2:
        raise RuntimeError("Not enough examples to train — upload a longer chat export.")

    hp = pick_hparams(data_meta["n_train"], model_key, rank=rank, num_layers=num_layers,
                      learning_rate=learning_rate)
    staging = staging_dir_for(adapter_dir)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if data_meta["prompt_style"] == "retrieval":
        shutil.copy(data_dir / "exemplars.jsonl", staging / "exemplars.jsonl")

    stored_settings = {}
    if (adapter_dir / "settings.json").exists():
        stored_settings = json.loads((adapter_dir / "settings.json").read_text())
    config = {
        "model": repo,
        "data_dir": str(data_dir),
        "adapter_dir": str(staging),
        "seed": 0,
        "batch_size": hp["batch_size"],
        "grad_accumulation_steps": hp["grad_accumulation_steps"],
        "iters": hp["iters"],
        "steps_per_eval": hp["steps_per_eval"],
        "steps_per_report": hp["steps_per_report"],
        "max_seq_length": data_meta["max_seq_length"],
        "grad_checkpoint": hp["grad_checkpoint"],
        "num_layers": hp["num_layers"],
        "lora_parameters": {"rank": hp["rank"], "scale": 20.0, "dropout": 0.05},
        "lr_schedule": hp["lr_schedule"],
        "patience": 3,
        "candidate_tolerance": 0.02,
        "max_candidates": 3,
        "style_select": style_select,
        "style_samples": 48,
        "generation": resolve_settings(stored_settings, data_meta["reply_tokens_p95"]),
    }
    config_path = staging / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    cmd = [sys.executable, str(BACKEND_DIR / "lora_train.py"), "--config", str(config_path)]
    log("Running: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            cwd=str(BACKEND_DIR), bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith(EVENT_PREFIX):
            log(_describe_event(json.loads(line[len(EVENT_PREFIX):])))
        elif line and not MLX_PROGRESS_RE.match(line):  # already reported via events
            log(line)
            tail = (tail + [line])[-15:]
    if proc.wait() != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("Training failed:\n" + "\n".join(tail))

    selection = json.loads((staging / "selection.json").read_text())
    persona_meta = persona_meta_from(
        data_meta,
        model=repo,
        model_key=model_key,
        trained_on="local",
        best_val=selection["chosen_val"],
        best_iter=selection["chosen_steps"],
        base_val=selection["base_val"],
        style=selection["style"],
        warnings=selection["warnings"],
        hparams=hp,
    )
    (staging / "persona_meta.json").write_text(json.dumps(persona_meta, indent=2, ensure_ascii=False))
    swap_in(staging, adapter_dir)
    log(f"Done — kept the checkpoint after {selection['chosen_steps']} steps.")
    return persona_meta


def _describe_event(evt):
    kind = evt["event"]
    if kind == "val":
        return f"step {evt['steps']}: validation loss {evt['loss']:.3f}"
    if kind == "train":
        return f"step {evt['iter']}: train loss {evt['loss']:.3f} · {evt['tokens_per_sec']} tok/s · {evt['peak_mem_gb']} GB"
    if kind == "early_stop":
        return f"Stopping early: no improvement for {evt['patience']} evals (best at step {evt['best_steps']})."
    if kind == "style":
        return f"Style check, step {evt['steps']}: distance {evt['style_distance']:.3f} (val {evt['val_loss']:.3f})"
    if kind == "dataset":
        skipped = evt["skipped_long"] + evt["skipped_template"]
        return f"Tokenized {evt['train']} train / {evt['val']} val rows" + (f" ({skipped} skipped)" if skipped else "")
    if kind in ("warning", "error"):
        return f"{kind.upper()}: {evt['message']}"
    if kind == "done":
        return f"Selected step {evt['chosen_steps']} (val {evt['chosen_val']:.3f})"
    return json.dumps(evt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Persona id (data/processed/<name>)")
    ap.add_argument("--model", dest="model_key", choices=list(MODELS), default="qwen-3b")
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=None,
                    help="Layers to adapt (-1 for all). Defaults to 24 (small models) / 16 (large).")
    ap.add_argument("--no_style_select", action="store_true",
                    help="Pick the checkpoint by validation loss only (skips generating samples).")
    args = ap.parse_args()

    root = BACKEND_DIR.parent
    meta = run_training(
        args.name, root / "data" / "processed" / args.name, root / "models" / "adapters" / args.name,
        model_key=args.model_key, learning_rate=args.learning_rate, rank=args.rank,
        num_layers=args.num_layers, style_select=not args.no_style_select,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
