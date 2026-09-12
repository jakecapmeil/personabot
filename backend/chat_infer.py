"""
Loads a persona's base model + LoRA adapter once and serves generations.
Keeps a small in-process cache so switching between personas already chatted
with in this session doesn't reload the (multi-GB) base model each time —
but only one persona's weights are resident at a time to stay within 16GB.
"""
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Fallback only — real adapters carry their own trained system_prompt
# (persona_meta.json, written by train_local.py from prepare_data.py's
# mined signature-phrase priming). Used only if that's ever missing.
FALLBACK_SYSTEM_TEMPLATE = (
    "You are {persona}, texting a close friend. Reply exactly the way "
    "{persona} actually texts: their real tone, slang, punctuation, "
    "capitalization, and typical message length (short replies are fine — "
    "don't pad them out). Never mention being an AI or a model."
)

DEFAULT_SETTINGS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 48,
    "repetition_penalty": 1.15,
    "repetition_context": 24,
}

_cache = {"persona": None, "model": None, "tokenizer": None, "meta": None}


def _load_persona(adapter_dir):
    meta = json.loads((Path(adapter_dir) / "persona_meta.json").read_text())
    if meta.get("merged"):
        # Colab-trained + locally merged/converted (import_colab.py) — a
        # plain fine-tuned MLX model, no separate adapter to attach.
        model, tokenizer = load(meta["model"])
    else:
        model, tokenizer = load(meta["model"], adapter_path=str(adapter_dir))
    return meta, model, tokenizer


def load_settings(adapter_dir):
    path = Path(adapter_dir) / "settings.json"
    settings = dict(DEFAULT_SETTINGS)
    if path.exists():
        settings.update(json.loads(path.read_text()))
    return settings


def save_settings(adapter_dir, settings):
    path = Path(adapter_dir) / "settings.json"
    merged = load_settings(adapter_dir)
    merged.update(settings)
    path.write_text(json.dumps(merged, indent=2))
    return merged


def _repetition_penalty_processor(penalty, context_size):
    """A simple recency-window repetition penalty (GPT-style): divide the
    logits of recently-used tokens so the model is discouraged from reusing
    them. Without this, a lightly-trained adapter tends to collapse onto a
    handful of stock replies — the same failure mode the from-scratch model
    in hudson-bot/ hit, addressed there the same way."""
    if not penalty or penalty <= 1.0:
        return None

    def processor(tokens, logits):
        if tokens.size == 0:
            return logits
        recent = tokens[-context_size:]
        idx = mx.array(sorted(set(recent.tolist())))
        vals = logits[..., idx]
        vals = mx.where(vals > 0, vals / penalty, vals * penalty)
        logits[..., idx] = vals
        return logits

    return processor


def reply(adapter_dir, persona, history, **overrides):
    """
    history: list of {"role": "user"|"assistant", "content": str}, ending
    with the newest user message. Returns the persona's reply text.

    Generation settings (temperature/top_p/max_tokens/repetition_penalty) are
    read from settings.json in adapter_dir when present, overridden by any
    keyword passed in here, falling back to DEFAULT_SETTINGS otherwise —
    letting the settings UI change behavior without every caller needing to
    know every knob.
    """
    adapter_dir = str(adapter_dir)
    if _cache["persona"] != adapter_dir:
        meta, model, tokenizer = _load_persona(adapter_dir)
        _cache.update(persona=adapter_dir, model=model, tokenizer=tokenizer, meta=meta)

    meta = _cache["meta"]
    tokenizer = _cache["tokenizer"]
    settings = load_settings(adapter_dir)
    settings.update({k: v for k, v in overrides.items() if v is not None})

    system_content = meta.get("system_prompt") or FALLBACK_SYSTEM_TEMPLATE.format(persona=persona)
    messages = [{"role": "system", "content": system_content}] + history
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    sampler = make_sampler(temp=settings["temperature"], top_p=settings["top_p"])
    logits_processors = None
    penalty_fn = _repetition_penalty_processor(
        settings.get("repetition_penalty"), settings.get("repetition_context", 24)
    )
    if penalty_fn is not None:
        logits_processors = [penalty_fn]

    text = generate(
        _cache["model"], tokenizer, prompt=prompt,
        max_tokens=settings["max_tokens"], sampler=sampler,
        logits_processors=logits_processors, verbose=False,
    )
    return text.strip()
