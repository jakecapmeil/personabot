"""
Generates a self-contained Colab notebook that fine-tunes Qwen2.5-7B-Instruct
with Unsloth on a persona's prepared data — the "cloud" backend. Faster than
local (a free Colab GPU beats the M-series' unified-memory GPU for this), at
the cost of running outside your machine.

The notebook embeds train.jsonl/valid.jsonl directly (base64-encoded in a
code cell) so there's exactly one download-then-drag-in step instead of
three — download this notebook, drop it into Colab, run it. The privacy
tradeoff of that: **the exported .ipynb file itself now contains the
persona's texts**. Nothing leaves your machine until you choose to upload
it to Colab (same as before), but don't casually share the .ipynb around —
it's not just code anymore.

I have not been able to execute this notebook myself (no GPU/Colab access
from here) — unlike the local MLX path, which I ran end-to-end. The Unsloth
API surface below is correct as of my training data, but if a cell errors
on a version mismatch, Unsloth's own docs/GitHub are the fastest fix.

Usage:
    python colab_export.py --data_dir ../data/processed/hudson --persona Hudson --out hudson_colab.ipynb
"""
import argparse
import base64
import json
from pathlib import Path

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
_CHUNK = 120  # wrap each base64 literal to this many chars/line for readability


def cell(cell_type, lines):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else []),
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def _embed_as_python(varname, text):
    """A `varname = ("...", "...", ...)` tuple of base64 chunks, joined and
    decoded at runtime — avoids one unreadably long source line."""
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    chunks = [b64[i:i + _CHUNK] for i in range(0, len(b64), _CHUNK)] or [""]
    lines = [f"{varname} = ("]
    lines += [f'    "{c}"' for c in chunks]
    lines += [")"]
    return lines


def build_notebook(persona, train_jsonl, valid_jsonl):
    n_train = sum(1 for l in train_jsonl.splitlines() if l.strip())
    n_valid = sum(1 for l in valid_jsonl.splitlines() if l.strip())

    data_cell_lines = (
        ["import base64, pathlib", ""]
        + _embed_as_python("_train_b64", train_jsonl)
        + [""]
        + _embed_as_python("_valid_b64", valid_jsonl)
        + [
            "",
            "pathlib.Path('train.jsonl').write_bytes(base64.b64decode(''.join(_train_b64)))",
            "pathlib.Path('valid.jsonl').write_bytes(base64.b64decode(''.join(_valid_b64)))",
            f"print(f'Wrote train.jsonl ({n_train} examples) and valid.jsonl ({n_valid} examples)')",
        ]
    )

    cells = [
        cell("markdown", [
            f"# {persona} chatbot — cloud fine-tune (Unsloth + Colab)",
            "",
            "Runtime: **Runtime > Change runtime type > T4 GPU** (or better) before running.",
            "",
            f"This notebook trains a LoRA adapter on `{persona}`'s texting style. "
            f"**This file contains {persona}'s prepared training data** "
            f"({n_train} train / {n_valid} val examples), embedded so there's nothing "
            "to separately upload — just run the cells in order. Don't share this "
            "`.ipynb` file around casually; treat it like the conversation export it "
            "was built from.",
        ]),
        cell("code", [
            "!pip install -q unsloth trl==0.9.6",
        ]),
        cell("markdown", ["## 1. Write out the embedded training data"]),
        cell("code", data_cell_lines),
        cell("markdown", ["## 2. Load the base model (4-bit) + attach LoRA"]),
        cell("code", [
            "from unsloth import FastLanguageModel",
            "",
            f"model, tokenizer = FastLanguageModel.from_pretrained(",
            f'    model_name="{MODEL_NAME}",',
            "    max_seq_length=1024,",
            "    load_in_4bit=True,",
            ")",
            "",
            "model = FastLanguageModel.get_peft_model(",
            "    model,",
            "    r=32,",
            "    lora_alpha=32,",
            "    lora_dropout=0.05,",
            "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",",
            "                    \"gate_proj\", \"up_proj\", \"down_proj\"],",
            "    use_gradient_checkpointing=\"unsloth\",",
            ")",
        ]),
        cell("markdown", [
            "## 3. Load the data and format with the model's chat template",
        ]),
        cell("code", [
            "import json",
            "from datasets import Dataset",
            "",
            "def load_jsonl(path):",
            "    return [json.loads(l) for l in open(path)]",
            "",
            "train_rows = load_jsonl('train.jsonl')",
            "valid_rows = load_jsonl('valid.jsonl')",
            "",
            "def to_text(example):",
            "    return {\"text\": tokenizer.apply_chat_template(",
            "        example[\"messages\"], tokenize=False, add_generation_prompt=False)}",
            "",
            "train_ds = Dataset.from_list(train_rows).map(to_text)",
            "valid_ds = Dataset.from_list(valid_rows).map(to_text)",
            "print(train_ds[0][\"text\"][:400])",
        ]),
        cell("markdown", [
            "## 4. Train (loss masked to the persona's reply tokens only)",
        ]),
        cell("code", [
            "from trl import SFTTrainer, SFTConfig",
            "from unsloth.chat_templates import train_on_responses_only",
            "",
            "trainer = SFTTrainer(",
            "    model=model,",
            "    tokenizer=tokenizer,",
            "    train_dataset=train_ds,",
            "    eval_dataset=valid_ds,",
            "    dataset_text_field=\"text\",",
            "    max_seq_length=1024,",
            "    args=SFTConfig(",
            "        per_device_train_batch_size=4,",
            "        gradient_accumulation_steps=4,",
            "        num_train_epochs=3,",
            "        learning_rate=2e-4,",
            "        warmup_ratio=0.05,",
            "        lr_scheduler_type=\"cosine\",",
            "        logging_steps=10,",
            "        eval_strategy=\"steps\",",
            "        eval_steps=50,",
            "        save_strategy=\"steps\",",
            "        save_steps=50,",
            "        load_best_model_at_end=True,",
            "        metric_for_best_model=\"eval_loss\",",
            "        output_dir=\"outputs\",",
            "        report_to=\"none\",",
            "    ),",
            ")",
            "",
            "# Only train on the assistant's final reply, not the prompt/history —",
            "# same reasoning as --mask-prompt in the local MLX path.",
            "trainer = train_on_responses_only(",
            "    trainer,",
            "    instruction_part=\"<|im_start|>user\\n\",",
            "    response_part=\"<|im_start|>assistant\\n\",",
            ")",
            "",
            "trainer.train()",
        ]),
        cell("markdown", [
            "## 5. Merge the adapter into the base model, then download",
            "",
            "The LoRA adapter alone is HF/PEFT format and **not** loadable by "
            "personabot's local MLX chat server. Merging it into the base weights "
            "produces a plain fine-tuned model with no adapter-format mismatch — "
            "download the zip below, then back on your Mac run:",
            "",
            "```bash",
            "cd personabot/backend",
            "python3 import_colab.py --name <persona-name> --zip ~/Downloads/merged_model.zip",
            "```",
            "",
            "That converts it to MLX format (`mlx_lm.convert`, quantized to match the "
            "local models) and registers it as a ready-to-chat persona in the app.",
        ]),
        cell("code", [
            "from google.colab import files",
            "",
            "merged_dir = \"merged_model\"",
            "model.save_pretrained_merged(merged_dir, tokenizer, save_method=\"merged_16bit\")",
            "",
            "import shutil",
            "shutil.make_archive(\"merged_model\", \"zip\", merged_dir)",
            "files.download(\"merged_model.zip\")",
        ]),
        cell("markdown", [
            "## 6. Chat with it right here (optional)",
            "",
            "Skippable if you're importing into personabot instead — see step 5.",
        ]),
        cell("code", [
            "FastLanguageModel.for_inference(model)",
            "",
            f'SYSTEM = ("You are {persona}, texting a close friend. Reply exactly the way "',
            f'          "{persona} actually texts: their real tone, slang, punctuation, "',
            '          "capitalization, and typical message length. Never mention being an AI.")',
            "",
            "history = [{\"role\": \"system\", \"content\": SYSTEM}]",
            "while True:",
            "    msg = input(\"Me: \")",
            "    if not msg:",
            "        break",
            "    history.append({\"role\": \"user\", \"content\": msg})",
            "    prompt = tokenizer.apply_chat_template(",
            "        history, tokenize=False, add_generation_prompt=True)",
            "    inputs = tokenizer(prompt, return_tensors=\"pt\").to(\"cuda\")",
            "    out = model.generate(**inputs, max_new_tokens=120, temperature=0.8,",
            "                          top_p=0.9, do_sample=True)",
            "    reply = tokenizer.decode(",
            "        out[0][inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)",
            f'    print(f"{persona}: {{reply}}")',
            "    history.append({\"role\": \"assistant\", \"content\": reply})",
        ]),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"{persona}_finetune.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="data/processed/<persona>/ directory")
    ap.add_argument("--persona", required=True, help="Display name, e.g. Hudson")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_jsonl = (data_dir / "train.jsonl").read_text()
    valid_jsonl = (data_dir / "valid.jsonl").read_text()

    nb = build_notebook(args.persona, train_jsonl, valid_jsonl)
    with open(args.out, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {args.out} — upload it to https://colab.research.google.com")


if __name__ == "__main__":
    main()
