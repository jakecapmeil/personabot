"""
LoRA training entry point, run as a subprocess by train_local.py:

    python lora_train.py --config <adapter_staging_dir>/run_config.json

Built on mlx_lm's trainer (pinned to mlx-lm 0.31.3), with three changes the
`mlx_lm lora` CLI can't express:

1. Loss masks cover *every* persona reply listed in a row's "train_turns",
   not only the last message, so a chunk of conversation trains all of its
   replies once instead of re-encoding overlapping windows.
2. Validation uses the whole (bounded) validation set every eval. Candidate
   checkpoints are saved at the exact weights that were evaluated, and
   training stops after `patience` evals without improvement.
3. Among checkpoints within `candidate_tolerance` of the best validation
   loss, the one whose generated replies are closest to the person's real
   texting style (eval_style.py) is kept.

Progress is reported as "@@personabot {json}" lines on stdout.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers, print_trainable_parameters

import eval_style
from generation import add_end_of_turn_eos

EVENT_PREFIX = "@@personabot "


def emit(event, **data):
    print(EVENT_PREFIX + json.dumps({"event": event, **data}), flush=True)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class MaskedChatDataset:
    """Tokenizes rows once. Items are (tokens, target_mask), where
    target_mask[j] = 1 means token j+1 is a trained reply token."""

    def __init__(self, rows, tokenizer, max_seq_length):
        self.items = []
        self.skipped_long = 0
        self.skipped_template = 0
        for row in rows:
            item = self._encode(row, tokenizer)
            if item is None:
                self.skipped_template += 1
            elif len(item[0]) > max_seq_length:
                self.skipped_long += 1
            else:
                self.items.append(item)

    @staticmethod
    def _encode(row, tokenizer):
        messages = row["messages"]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        mask = np.zeros(max(len(tokens) - 1, 0), dtype=np.float32)
        for i in row["train_turns"]:
            start = len(tokenizer.apply_chat_template(messages[:i], add_generation_prompt=True))
            prefix_end = tokenizer.apply_chat_template(messages[:i + 1], add_generation_prompt=False)
            end = len(prefix_end)
            if prefix_end != tokens[:end] or start >= end:
                return None  # template isn't prefix-stable; can't locate this reply safely
            mask[start - 1:end - 1] = 1.0
        if not mask.any():
            return None
        return tokens, mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def itemlen(self, idx):
        return len(self.items[idx][0])


def iterate_batches(dataset, batch_size, max_seq_length, loop=False, seed=None, comm_group=None):
    if len(dataset) < batch_size:
        raise ValueError(f"Need at least {batch_size} examples, got {len(dataset)}.")
    order = sorted(range(len(dataset)), key=dataset.itemlen)
    batches = [order[i:i + batch_size] for i in range(0, len(order) - batch_size + 1, batch_size)]
    while True:
        for b in np.random.permutation(len(batches)):
            items = [dataset[j] for j in batches[b]]
            longest = max(len(tokens) for tokens, _ in items)
            width = min(1 + 32 * ((longest + 31) // 32), max_seq_length)
            width = max(width, longest)
            tokens_arr = np.zeros((len(items), width), dtype=np.int32)
            mask_arr = np.zeros((len(items), width - 1), dtype=np.float32)
            for row, (tokens, mask) in enumerate(items):
                tokens_arr[row, :len(tokens)] = tokens
                mask_arr[row, :len(mask)] = mask
            yield mx.array(tokens_arr), mx.array(mask_arr)
        if not loop:
            break


def masked_loss(model, batch, mask):
    logits = model(batch[:, :-1])
    ce = nn.losses.cross_entropy(logits, batch[:, 1:]) * mask
    ntoks = mask.sum()
    return ce.astype(mx.float32).sum() / mx.maximum(ntoks, 1), ntoks


class StopTraining(Exception):
    pass


class TrainingDiverged(Exception):
    pass


class CheckpointSelector(TrainingCallback):
    def __init__(self, model, ckpt_dir, patience, tolerance):
        self.model = model
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.patience = patience
        self.tolerance = tolerance
        self.best_val, self.best_iter = math.inf, None
        self.base_val = None
        self.evals_since_best = 0
        self.saved = {}  # steps -> (val_loss, path)
        self.history = []

    def on_train_loss_report(self, info):
        loss = info["train_loss"]
        emit("train", iter=info["iteration"], loss=loss, lr=info["learning_rate"],
             tokens_per_sec=round(info["tokens_per_second"], 1), peak_mem_gb=round(info["peak_memory"], 2))
        if not math.isfinite(loss):
            raise TrainingDiverged(f"train loss became {loss} at iter {info['iteration']}")

    def on_val_loss_report(self, info):
        steps, val = info["iteration"], info["val_loss"]  # steps completed before this eval
        if not math.isfinite(val):
            raise TrainingDiverged(f"validation loss became {val} after {steps} steps")
        if self.base_val is None:
            self.base_val = val
        self.history.append({"steps": steps, "val_loss": val})
        emit("val", steps=steps, loss=val)

        if val < self.best_val:
            self.best_val, self.best_iter, self.evals_since_best = val, steps, 0
        else:
            self.evals_since_best += 1

        limit = self.best_val * (1 + self.tolerance)
        if val <= limit:
            path = self.ckpt_dir / f"{steps:07d}.safetensors"
            mx.save_safetensors(str(path), dict(tree_flatten(self.model.trainable_parameters())))
            self.saved[steps] = (val, path)
        for s, (v, p) in list(self.saved.items()):
            if v > limit:
                p.unlink(missing_ok=True)
                del self.saved[s]

        if self.evals_since_best >= self.patience:
            emit("early_stop", best_steps=self.best_iter, best_val=self.best_val, patience=self.patience)
            raise StopTraining()


def select_by_style(model, tokenizer, candidates, val_rows, cfg):
    pairs = eval_style.sample_contexts(val_rows, cfg["style_samples"])
    reals = [real for _, real in pairs]
    contexts = [ctx for ctx, _ in pairs]
    results = []
    model.eval()
    for steps, (val, path) in candidates:
        model.load_weights(str(path), strict=False)
        mx.eval(model.parameters())
        tic = time.perf_counter()
        generated = eval_style.generate_for_contexts(model, tokenizer, pairs, cfg["generation"])
        report = eval_style.compare(reals, generated, contexts)
        emit("style", steps=steps, val_loss=val, style_distance=report["style_distance"],
             seconds=round(time.perf_counter() - tic, 1))
        results.append({"steps": steps, "val_loss": val, "path": path, "report": report,
                        "samples": list(zip(reals, generated))[:6]})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.loads(Path(ap.parse_args().config).read_text())

    np.random.seed(cfg["seed"])
    mx.random.seed(cfg["seed"])
    data_dir, adapter_dir = Path(cfg["data_dir"]), Path(cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg['model']}", flush=True)
    model, tokenizer = load(cfg["model"])
    add_end_of_turn_eos(tokenizer)

    train_rows = load_jsonl(data_dir / "train.jsonl")
    val_rows = load_jsonl(data_dir / "valid.jsonl")
    train_set = MaskedChatDataset(train_rows, tokenizer, cfg["max_seq_length"])
    val_set = MaskedChatDataset(val_rows, tokenizer, cfg["max_seq_length"])
    emit("dataset", train=len(train_set), val=len(val_set),
         skipped_long=train_set.skipped_long + val_set.skipped_long,
         skipped_template=train_set.skipped_template + val_set.skipped_template)
    if len(val_set) < cfg["batch_size"] or len(train_set) < cfg["batch_size"]:
        raise SystemExit("Not enough usable examples to train — upload a longer chat export.")

    model.freeze()
    linear_to_lora_layers(model, cfg["num_layers"], cfg["lora_parameters"])
    print_trainable_parameters(model)
    (adapter_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "model": cfg["model"],
        "num_layers": cfg["num_layers"],
        "lora_parameters": cfg["lora_parameters"],
    }, indent=2))

    optimizer = optim.Adam(learning_rate=build_schedule(cfg["lr_schedule"]))
    args = TrainingArgs(
        batch_size=cfg["batch_size"],
        iters=cfg["iters"],
        val_batches=-1,
        steps_per_report=cfg["steps_per_report"],
        steps_per_eval=cfg["steps_per_eval"],
        steps_per_save=cfg["iters"] + 1,  # checkpoints are saved by the selector instead
        adapter_file=str(adapter_dir / "adapters.safetensors"),
        max_seq_length=cfg["max_seq_length"],
        grad_checkpoint=cfg["grad_checkpoint"],
        grad_accumulation_steps=cfg["grad_accumulation_steps"],
    )
    ckpt_dir = adapter_dir / "checkpoints"
    selector = CheckpointSelector(model, ckpt_dir, cfg["patience"], cfg["candidate_tolerance"])

    layer_type = type(model.layers[0])
    original_call = layer_type.__call__  # grad checkpointing patches the class; undo before generating
    early_stopped = False
    try:
        train(model=model, optimizer=optimizer, train_dataset=train_set, val_dataset=val_set,
              args=args, loss=masked_loss, iterate_batches=iterate_batches, training_callback=selector)
    except StopTraining:
        early_stopped = True
    except TrainingDiverged as e:
        emit("error", message=str(e))
        raise SystemExit(f"Training diverged: {e}")
    finally:
        layer_type.__call__ = original_call

    if not selector.saved:
        raise SystemExit("No checkpoint was saved — validation never ran.")
    candidates = sorted(selector.saved.items(), key=lambda kv: kv[1][0])[:cfg["max_candidates"]]

    warnings = []
    if selector.best_iter == 0:
        warnings.append(
            "Training never beat the untrained base model on validation loss. The chosen adapter "
            "may barely differ from the base model; try a lower learning rate or a longer export."
        )
        emit("warning", message=warnings[-1])

    style_results = []
    if cfg["style_select"] and len(val_rows) > 0:
        style_results = select_by_style(model, tokenizer, candidates, val_rows, cfg)
        chosen = min(style_results, key=lambda r: (r["report"]["style_distance"], r["val_loss"]))
        chosen_steps, chosen_val, chosen_path = chosen["steps"], chosen["val_loss"], chosen["path"]
        chosen_style = chosen["report"]
    else:
        chosen_steps, (chosen_val, chosen_path) = candidates[0]
        chosen_style = None

    shutil.copy(chosen_path, adapter_dir / "adapters.safetensors")
    selection = {
        "early_stopped": early_stopped,
        "base_val": selector.base_val,
        "best_val": selector.best_val,
        "best_steps": selector.best_iter,
        "chosen_steps": chosen_steps,
        "chosen_val": chosen_val,
        "style": {k: v for k, v in chosen_style.items() if k not in ("real", "generated")} if chosen_style else None,
        "candidates": [
            {"steps": r["steps"], "val_loss": r["val_loss"], "style_distance": r["report"]["style_distance"],
             "samples": [{"real": a, "generated": b} for a, b in r["samples"]]}
            for r in style_results
        ],
        "val_history": selector.history,
        "warnings": warnings,
    }
    (adapter_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    emit("done", chosen_steps=chosen_steps, chosen_val=chosen_val,
         style_distance=chosen_style["style_distance"] if chosen_style else None)


if __name__ == "__main__":
    main()
