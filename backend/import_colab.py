"""
Brings a Colab-trained model back into personabot: converts the merged
HF-format model (from colab_export.py's notebook, step 5) to MLX format
locally, and registers it as a ready-to-chat persona.

This is the bridge between the Colab/Unsloth path (fast, HF/PEFT adapter
format) and the local MLX chat server (mlx_lm) — merging in Colab first
means there's no adapter-format mismatch to resolve, just a plain model
conversion, which only runs on Apple Silicon (mlx_lm.convert is MLX-only,
so this step cannot happen inside Colab itself).

Usage:
    python import_colab.py --name hudson --zip ~/Downloads/merged_model.zip
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def _find_model_root(extracted_dir):
    """The zip may contain the model files directly at its root, or nested
    one level under the directory name Colab saved to — handle both."""
    extracted_dir = Path(extracted_dir)
    if (extracted_dir / "config.json").exists():
        return extracted_dir
    for child in extracted_dir.iterdir():
        if child.is_dir() and (child / "config.json").exists():
            return child
    raise FileNotFoundError(
        f"No config.json found in {extracted_dir} or its immediate subdirectories — "
        "is this the merged_model.zip from the Colab notebook's step 5?"
    )


def import_merged_model(name, persona, zip_path, adapter_dir, system_prompt=None, quantize=True):
    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    mlx_model_dir = adapter_dir / "mlx_model"

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        hf_model_dir = _find_model_root(tmp)

        if mlx_model_dir.exists():
            shutil.rmtree(mlx_model_dir)

        cmd = [
            sys.executable, "-m", "mlx_lm", "convert",
            "--hf-path", str(hf_model_dir),
            "--mlx-path", str(mlx_model_dir),
        ]
        if quantize:
            cmd.append("-q")

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    persona_meta = {
        "persona": persona,
        "model": str(mlx_model_dir),
        "model_key": "colab",
        "merged": True,  # chat_infer.py: load the model directly, no adapter_path
        "system_prompt": system_prompt,
    }
    (adapter_dir / "persona_meta.json").write_text(json.dumps(persona_meta, indent=2))
    print(f"Imported. {mlx_model_dir} is ready to chat as '{name}'.")
    return persona_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Persona id, matching data/processed/<name>/")
    ap.add_argument("--zip", required=True, help="merged_model.zip downloaded from the Colab notebook")
    ap.add_argument("--no-quantize", action="store_true", help="Keep full precision instead of 4-bit")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed" / args.name
    adapter_dir = root / "models" / "adapters" / args.name

    meta = json.loads((data_dir / "meta.json").read_text())
    import_merged_model(
        args.name, meta["persona"], args.zip, adapter_dir,
        system_prompt=meta.get("system_prompt"), quantize=not args.no_quantize,
    )


if __name__ == "__main__":
    main()
