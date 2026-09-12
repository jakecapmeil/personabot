"""
Brings a Colab-trained persona back into personabot.

The notebook (colab_export.py) downloads `persona_adapter.zip`: the PEFT LoRA
weights (~100-300 MB) plus `personabot_adapter.json`. Instead of shipping a
~15 GB merged model and re-quantizing it (which blurs the small LoRA delta),
the adapter is converted to mlx-lm's format and applied at full precision on
top of the same 4-bit base model local training uses:

    PEFT:  y = W x + (alpha / r) * B (A x)       A: (r, in)    B: (out, r)
    mlx:   y = W x + scale * ((x @ a) @ b)       a: (in, r)    b: (r, out)
    =>     a = A.T,  b = B.T,  scale = alpha / r

Older notebooks produced `merged_model.zip` (a full HF model). That still
imports, converted with 8-bit quantization instead of 4-bit.

Usage:
    python import_colab.py --name hudson --zip ~/Downloads/persona_adapter.zip
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

import train_local

PEFT_KEY_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<module>.*?\.layers\.(?P<layer>\d+)\.(?P<sub>.+?))"
    r"\.lora_(?P<ab>[AB])(?:\.default)?\.weight$"
)


def convert_peft_state(state, rank, alpha):
    """PEFT LoRA tensors (numpy) -> (mlx adapter weights, lora keys, skipped keys)."""
    weights, subs, skipped = {}, set(), []
    for key, value in state.items():
        m = PEFT_KEY_RE.match(key)
        if not m:
            skipped.append(key)
            continue
        value = np.asarray(value, dtype=np.float32)
        if m.group("ab") == "A":
            if value.shape[0] != rank:
                raise ValueError(f"{key}: expected rank {rank}, got shape {value.shape}")
            weights[m.group("module") + ".lora_a"] = np.ascontiguousarray(value.T)
        else:
            if value.shape[1] != rank:
                raise ValueError(f"{key}: expected rank {rank}, got shape {value.shape}")
            weights[m.group("module") + ".lora_b"] = np.ascontiguousarray(value.T)
        subs.add(m.group("sub"))
    modules = {k.rsplit(".", 1)[0] for k in weights}
    for module in modules:
        if module + ".lora_a" not in weights or module + ".lora_b" not in weights:
            raise ValueError(f"Incomplete LoRA pair for {module}")
    if not weights:
        raise ValueError("No LoRA weights found in the adapter.")
    return weights, sorted(subs), skipped


def mlx_adapter_config(repo, rank, alpha, keys):
    return {
        "fine_tune_type": "lora",
        "model": repo,
        "num_layers": -1,  # all layers; modules the adapter doesn't cover keep a zero delta
        "lora_parameters": {"rank": rank, "scale": alpha / rank, "dropout": 0.0, "keys": keys},
    }


def _find_root(extracted, marker):
    extracted = Path(extracted)
    if (extracted / marker).exists():
        return extracted
    for child in extracted.iterdir():
        if child.is_dir() and (child / marker).exists():
            return child
    return None


def _import_adapter(root, staging, log):
    import mlx.core as mx

    info = json.loads((root / "personabot_adapter.json").read_text())
    model_key = info["model_key"]
    if model_key not in train_local.MODELS:
        raise ValueError(f"Unknown model '{model_key}' in personabot_adapter.json")
    repo = train_local.MODELS[model_key]["repo"]
    rank, alpha = int(info["rank"]), float(info["alpha"])

    log(f"Converting PEFT adapter (rank {rank}, alpha {alpha:g}) for {repo}…")
    state = {k: np.array(v.astype(mx.float32)) for k, v in mx.load(str(root / "adapter_model.safetensors")).items()}
    weights, keys, skipped = convert_peft_state(state, rank, alpha)
    if skipped:
        log(f"Ignored {len(skipped)} non-LoRA tensors (e.g. {skipped[0]})")
    mx.save_safetensors(str(staging / "adapters.safetensors"), {k: mx.array(v) for k, v in weights.items()})
    (staging / "adapter_config.json").write_text(json.dumps(mlx_adapter_config(repo, rank, alpha, keys), indent=2))
    return {"model": repo, "model_key": model_key, "trained_on": "colab"}


def _import_merged(root, staging, adapter_dir, log):
    mlx_model_dir = staging / "mlx_model"
    cmd = [sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(root),
           "--mlx-path", str(mlx_model_dir), "-q", "--q-bits", "8"]
    log("Legacy merged model — converting with 8-bit quantization: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Path after staging is swapped into place.
    return {"model": str(adapter_dir / "mlx_model"), "model_key": "colab-merged", "merged": True,
            "trained_on": "colab"}


def import_zip(name, zip_path, data_dir, adapter_dir, log=print):
    data_dir, adapter_dir = Path(data_dir), Path(adapter_dir)
    data_meta = json.loads((data_dir / "meta.json").read_text())
    staging = train_local.staging_dir_for(adapter_dir)
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            log(f"Extracting {Path(zip_path).name}…")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            if (root := _find_root(tmp, "personabot_adapter.json")) is not None:
                extra = _import_adapter(root, staging, log)
            elif (root := _find_root(tmp, "config.json")) is not None:
                extra = _import_merged(root, staging, adapter_dir, log)
            else:
                raise FileNotFoundError(
                    "This zip has neither personabot_adapter.json nor config.json — "
                    "is it the persona_adapter.zip from the Colab notebook?"
                )
        if data_meta.get("prompt_style") == "retrieval" and (data_dir / "exemplars.jsonl").exists():
            shutil.copy(data_dir / "exemplars.jsonl", staging / "exemplars.jsonl")
        persona_meta = train_local.persona_meta_from(data_meta, **extra)
        (staging / "persona_meta.json").write_text(json.dumps(persona_meta, indent=2, ensure_ascii=False))
        train_local.swap_in(staging, adapter_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    log(f"Imported — '{name}' is ready to chat.")
    return persona_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Persona id, matching data/processed/<name>/")
    ap.add_argument("--zip", required=True, help="persona_adapter.zip downloaded from the Colab notebook")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    import_zip(args.name, args.zip, root / "data" / "processed" / args.name,
               root / "models" / "adapters" / args.name)


if __name__ == "__main__":
    main()
