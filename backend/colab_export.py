"""
Generates the Colab notebook for cloud training (Unsloth on a free T4).

It trains the same objective as local training: the loss covers exactly the
replies listed in each row's "train_turns" (so persona messages that appear
as context are never trained twice), for one epoch with early stopping, on
the model selected in personabot's settings.

It downloads only the LoRA adapter (`persona_adapter.zip`, ~100-300 MB),
which import_colab.py converts to mlx-lm format — no multi-GB merged model
and no lossy re-quantization.

The notebook contains no chat data; train.jsonl / valid.jsonl are uploaded
into it separately.

This has not been run end-to-end on Colab from here. If Unsloth changes its
API, its docs are the fastest fix; the training loop itself is plain
transformers.Trainer.

Usage:
    python colab_export.py --name hudson --model qwen-3b --out hudson_colab.ipynb
"""
import argparse
import json
from pathlib import Path

import train_local

INTRO = """# {display_name} — cloud fine-tune (Unsloth + Colab)

**Runtime → Change runtime type → T4 GPU** before running.

Trains a LoRA adapter on `{model_repo}` from `train.jsonl` / `valid.jsonl`
(downloaded alongside this notebook). The last cell downloads
`persona_adapter.zip`; import it on the persona's card in personabot."""

INSTALL = """!pip install -q unsloth"""

UPLOAD = """from google.colab import files
print('Select train.jsonl and valid.jsonl together:')
uploaded = files.upload()
assert 'train.jsonl' in uploaded and 'valid.jsonl' in uploaded, \\
    'Upload both train.jsonl and valid.jsonl — they were downloaded alongside this notebook.'"""

LOAD_MODEL = """from unsloth import FastLanguageModel

MODEL_KEY = {model_key!r}
MAX_SEQ_LENGTH = {max_seq_length}
RANK = {rank}
ALPHA = {rank}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={unsloth_repo!r},
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=0,
    target_modules=TARGET_MODULES,
    use_gradient_checkpointing="unsloth",
    random_state=0,
)"""

DATA = """import json
from datasets import Dataset

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def ids(messages, add_generation_prompt):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=False)

def encode(row):
    messages = row["messages"]
    input_ids = ids(messages, False)
    labels = [-100] * len(input_ids)
    for i in row["train_turns"]:
        start = len(ids(messages[:i], True))
        end = len(ids(messages[:i + 1], False))
        labels[start:end] = input_ids[start:end]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

def build(path):
    rows = [encode(r) for r in load_jsonl(path)]
    rows = [r for r in rows if len(r["input_ids"]) <= MAX_SEQ_LENGTH and any(l != -100 for l in r["labels"])]
    return Dataset.from_list(rows)

train_ds, valid_ds = build("train.jsonl"), build("valid.jsonl")
print(len(train_ds), "train rows ·", len(valid_ds), "validation rows")"""

TRAIN = """import math
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer, TrainingArguments
from unsloth import is_bfloat16_supported

BATCH, ACCUM = 4, 2
steps_per_epoch = max(1, math.ceil(len(train_ds) / (BATCH * ACCUM)))
eval_every = max(5, steps_per_epoch // 10)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    args=TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        num_train_epochs=1,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=max(1, eval_every // 2),
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="steps",
        save_steps=eval_every,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        report_to="none",
        seed=0,
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()"""

EXPORT = """import os, shutil
from peft import get_peft_model_state_dict
from safetensors.torch import save_file

os.makedirs("persona_adapter", exist_ok=True)
state = {k: v.detach().float().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items()}
save_file(state, "persona_adapter/adapter_model.safetensors")
with open("persona_adapter/personabot_adapter.json", "w") as f:
    json.dump({"format": "peft-lora", "model_key": MODEL_KEY, "rank": RANK, "alpha": ALPHA,
               "target_modules": TARGET_MODULES}, f, indent=2)
shutil.make_archive("persona_adapter", "zip", "persona_adapter")
files.download("persona_adapter.zip")"""

CHAT = """FastLanguageModel.for_inference(model)
SYSTEM = {system_prompt!r}

history = [{{"role": "system", "content": SYSTEM}}]
while True:
    msg = input("Me: ")
    if not msg:
        break
    history.append({{"role": "user", "content": msg}})
    inputs = tokenizer.apply_chat_template(
        history, add_generation_prompt=True, return_tensors="pt", return_dict=False).to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=96, temperature=0.8, min_p=0.05, do_sample=True)
    reply = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    print({display_name!r} + ":", reply)
    history.append({{"role": "assistant", "content": reply}})"""


def _cell(cell_type, source):
    lines = source.split("\n")
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]],
    }
    if cell_type == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def build_notebook(data_meta, model_key, rank=16):
    model = train_local.MODELS[model_key]
    fmt = {
        "display_name": data_meta["display_name"],
        "model_key": model_key,
        "model_repo": model["unsloth"],
        "unsloth_repo": model["unsloth"],
        "rank": int(rank),
        "max_seq_length": data_meta.get("max_seq_length", 1024),
        "system_prompt": data_meta["system_prompt"],
    }
    cells = [
        _cell("markdown", INTRO.format(**fmt)),
        _cell("code", INSTALL),
        _cell("markdown", "## 1. Upload the prepared data"),
        _cell("code", UPLOAD),
        _cell("markdown", "## 2. Load the base model (4-bit) and attach LoRA"),
        _cell("code", LOAD_MODEL.format(**fmt)),
        _cell("markdown", "## 3. Tokenize, masking the loss to the persona's replies"),
        _cell("code", DATA),
        _cell("markdown", "## 4. Train (one epoch, best checkpoint by validation loss, early stopping)"),
        _cell("code", TRAIN),
        _cell("markdown", "## 5. Download the adapter\n\nImport `persona_adapter.zip` on the persona's card "
                          "in personabot, or run `python3 backend/import_colab.py --name <persona> --zip "
                          "~/Downloads/persona_adapter.zip`."),
        _cell("code", EXPORT),
        _cell("markdown", "## 6. Chat with it here (optional)\n\nUses the base identity prompt; personas "
                          "with retrieved examples get those in personabot."),
        _cell("code", CHAT.format(**fmt)),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"{data_meta['display_name']}_finetune.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Persona id (data/processed/<name>)")
    ap.add_argument("--model", dest="model_key", choices=list(train_local.MODELS), default="qwen-3b")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import prepare_data

    data_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / args.name
    meta = prepare_data.build_dataset(data_dir, tokenizer_model=train_local.MODELS[args.model_key]["repo"])
    Path(args.out).write_text(json.dumps(build_notebook(meta, args.model_key, args.rank), indent=1))
    print(f"Wrote {args.out} — upload it to https://colab.research.google.com with "
          f"{data_dir / 'train.jsonl'} and {data_dir / 'valid.jsonl'}")


if __name__ == "__main__":
    main()
