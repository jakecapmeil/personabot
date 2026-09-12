"""
Wraps `mlx_lm.lora` to LoRA fine-tune a local base model on one persona's
prepared data, then keeps the best-val checkpoint (mlx_lm itself only saves
periodic + final checkpoints — it doesn't pick a winner, and the final one
is whatever iteration training happened to stop on, which past experience
on this project says is often already past the point where it started
overfitting).

Usage:
    python train_local.py --persona hudson --data_dir ../data/processed/hudson \
        --adapter_dir ../models/adapters/hudson
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

MODELS = {
    # key -> (repo, tier). tier drives batch/num_layers/grad_checkpoint
    # defaults below — "small" fits comfortably on 16GB with headroom,
    # "large" needs grad checkpointing to stay under budget.
    "qwen-3b": ("mlx-community/Qwen2.5-3B-Instruct-4bit", "small"),
    "qwen-7b": ("mlx-community/Qwen2.5-7B-Instruct-4bit", "large"),
    "llama-3b": ("mlx-community/Llama-3.2-3B-Instruct-4bit", "small"),
    "llama-8b": ("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "large"),
}

ITER_RE = re.compile(r"Iter (\d+): (Val|Train) loss ([\d.]+)")


def pick_hparams(n_train, model_key, rank=16, num_layers=None):
    """Small-corpus defaults.

    Measured directly on this project's ~3,200-example corpus: LoRA on a
    pretrained instruct model overfits *fast* — val loss bottomed well under
    a tenth of a single epoch in (iter ~100 of 150, batch 2) and was already
    climbing back by iter 150. A pretrained model already knows English and
    conversation structure; the adapter only has to steer style, so it
    doesn't need — and shouldn't get — many passes over a small corpus.
    Budget under 1 epoch, with frequent eval checkpoints early on so the
    best-val promotion in run_training() actually has a fine-grained best
    checkpoint to find rather than 20% jumps.

    rank/num_layers control adapter *capacity* (how much of the person's
    voice it can absorb), which is a separate axis from iters/epochs (how
    long training runs) — mlx_lm's own default is rank 8, quite low headroom
    for shifting content/personality rather than just tone; num_layers
    defaults to a larger fraction of the network than mlx_lm's own default
    (16, regardless of model size) since the base model's total layer count
    differs a lot between the "small" and "large" tiers."""
    tier = MODELS[model_key][1]
    batch_size = 2 if tier == "small" else 1
    epochs = 0.5
    iters = max(150, min(800, round(n_train * epochs / batch_size)))
    steps_per_eval = max(15, iters // 20)
    if num_layers is None:
        num_layers = 24 if tier == "small" else 16
    return {
        "batch_size": batch_size,
        "iters": iters,
        "steps_per_eval": steps_per_eval,
        "num_layers": num_layers,
        "rank": rank,
        "grad_checkpoint": tier != "small",
    }


def run_training(persona, data_dir, adapter_dir, model_key="qwen-7b", learning_rate=1e-5,
                  max_seq_length=512, rank=16, num_layers=None, extra_hparams=None):
    data_dir = Path(data_dir)
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((data_dir / "meta.json").read_text())
    hp = pick_hparams(meta["n_train"], model_key, rank=rank, num_layers=num_layers)
    if extra_hparams:
        hp.update(extra_hparams)

    # mlx_lm.lora only accepts lora_parameters (rank/scale/dropout) via a YAML
    # -c config, not a CLI flag — write one so the adapter isn't silently
    # stuck at mlx_lm's own default (rank 8). scale=20 is mlx_lm's default
    # scaling of the low-rank update; kept fixed since it's tuned for that
    # default learning-rate range, not something this project has evidence
    # to move.
    lora_config_path = adapter_dir / "lora_config.yaml"
    lora_config_path.write_text(yaml.safe_dump({
        "lora_parameters": {"rank": hp["rank"], "dropout": 0.05, "scale": 20.0},
    }))

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODELS[model_key][0],
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--fine-tune-type", "lora",
        "--batch-size", str(hp["batch_size"]),
        "--iters", str(hp["iters"]),
        "--num-layers", str(hp["num_layers"]),
        "--learning-rate", str(learning_rate),
        "--max-seq-length", str(max_seq_length),
        "--steps-per-report", str(max(10, hp["steps_per_eval"] // 2)),
        "--steps-per-eval", str(hp["steps_per_eval"]),
        "--save-every", str(hp["steps_per_eval"]),
        "--val-batches", "5",  # 20 spent ~28% of wall time on eval; 10 still worked fine, 5 trims further
        "--mask-prompt",
        "-c", str(lora_config_path),
    ]
    if hp["grad_checkpoint"]:
        cmd.append("--grad-checkpoint")

    print("Running:", " ".join(cmd))
    # Early stopping: measured on this project's own corpus, LoRA on a
    # pretrained model overfits within well under one epoch (best val at
    # ~10% of a 150-iter test run, already rising by the end) — so running
    # the full --iters budget routinely wastes most of the run's wall-clock
    # time after the best checkpoint is already found. PATIENCE consecutive
    # evals with no improvement stops the subprocess ourselves; this counts
    # as a normal, successful stop (we already have the best-val checkpoint
    # on disk), not an error.
    PATIENCE = 4
    best_iter, best_val, evals_since_best = None, float("inf"), 0
    early_stopped = False

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
        m = ITER_RE.search(line)
        if m:
            it, kind, val = int(m.group(1)), m.group(2), float(m.group(3))
            if val != val:  # NaN check (NaN != NaN); a bad batch has poisoned
                proc.terminate()  # every parameter downstream — no point continuing.
                raise RuntimeError(
                    f"Training diverged: {kind} loss went NaN at iter {it}. "
                    "Most likely a training example is too long for --max-seq-length "
                    "and got its target reply truncated off (see prepare_data.py's "
                    "length-safe trimming) — check for unusually long messages in "
                    "the uploaded chat export."
                )
            if kind == "Val":
                if val < best_val:
                    best_val, best_iter, evals_since_best = val, it, 0
                else:
                    evals_since_best += 1
                    if evals_since_best >= PATIENCE:
                        print(
                            f"\nEarly stopping: val loss hasn't improved on iter "
                            f"{best_iter} (val {best_val:.3f}) for {PATIENCE} evals.",
                            flush=True,
                        )
                        proc.terminate()
                        early_stopped = True
                        break
    if not early_stopped:
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"mlx_lm.lora exited with code {proc.returncode}")
    else:
        proc.wait(timeout=30)

    # Promote the best-val checkpoint to the adapter file mlx_lm.generate loads.
    if best_iter is not None:
        ckpt = adapter_dir / f"{best_iter:07d}_adapters.safetensors"
        if ckpt.exists():
            shutil.copy(ckpt, adapter_dir / "adapters.safetensors")
            print(f"Promoted iter {best_iter} (val {best_val:.3f}) to adapters.safetensors")

    # mlx_lm saves a full-size checkpoint (tens-hundreds of MB, scales with
    # rank/num_layers) at every eval — clean up everything except the one
    # already-copied adapters.safetensors, or these pile up per persona and
    # bloat the download/share zip with checkpoints nobody will ever load.
    for stray in adapter_dir.glob("*_adapters.safetensors"):
        stray.unlink()

    persona_meta = {
        "persona": persona,
        "model": MODELS[model_key][0],
        "model_key": model_key,
        "best_val": best_val,
        "best_iter": best_iter,
        "hparams": hp,
        # Carried over so chat_infer.py uses the exact same system prompt at
        # inference that the adapter was trained against (including any
        # mined signature-phrase priming) — a mismatch here wastes some of
        # what LoRA training actually learned.
        "system_prompt": meta.get("system_prompt"),
    }
    (adapter_dir / "persona_meta.json").write_text(json.dumps(persona_meta, indent=2))
    return persona_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--model", dest="model_key", choices=list(MODELS), default="qwen-7b")
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--rank", type=int, default=16,
                     help="LoRA rank — higher absorbs more of the person's voice, "
                          "at the cost of training time. mlx_lm's own default is 8.")
    ap.add_argument("--num_layers", type=int, default=None,
                     help="Layers to adapt (-1 for all). Defaults to 24 (small models) / 16 (large).")
    args = ap.parse_args()

    meta = run_training(
        args.persona, args.data_dir, args.adapter_dir,
        model_key=args.model_key, learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        rank=args.rank, num_layers=args.num_layers,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
