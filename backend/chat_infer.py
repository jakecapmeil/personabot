"""
Serves chat replies for one persona at a time (16 GB budget).

- The prompt is rebuilt exactly as in training: same system prompt (with
  retrieved exemplars for the "retrieval" style) and a history window of the
  length the adapter was trained on.
- The KV cache from the previous turn is reused for the shared prompt prefix,
  so each turn only prefills the new messages.
- A lock serializes loading and generation: concurrent share-link chats can't
  swap weights mid-reply or load two models at once.
"""
import json
import threading
from pathlib import Path

import prepare_data
from generation import (
    SETTINGS_VERSION, add_end_of_turn_eos, clean_reply, generate_reply, resolve_settings, sticky_window,
)
from retrieval import ExemplarIndex

LEGACY_CONTEXT_TURNS = 4

_lock = threading.RLock()
_state = {"key": None, "meta": None, "model": None, "tokenizer": None, "index": None,
          "prompt_cache": None, "cache_tokens": []}


def load_persona(adapter_dir):
    from mlx_lm import load

    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / "persona_meta.json").read_text())
    if meta.get("merged"):
        model, tokenizer = load(meta["model"])  # legacy merged Colab import
    else:
        model, tokenizer = load(meta["model"], adapter_path=str(adapter_dir))
    add_end_of_turn_eos(tokenizer)
    return meta, model, tokenizer


def read_settings_file(adapter_dir):
    path = Path(adapter_dir) / "settings.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _persona_meta(adapter_dir):
    path = Path(adapter_dir) / "persona_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_settings(adapter_dir):
    return resolve_settings(read_settings_file(adapter_dir), _persona_meta(adapter_dir).get("reply_tokens_p95"))


def save_settings(adapter_dir, updates):
    """Stores only values the user actually set, so default changes reach
    personas nobody has customized."""
    path = Path(adapter_dir) / "settings.json"
    stored = read_settings_file(adapter_dir)
    if stored.get("version") != SETTINGS_VERSION:
        stored = {}
    stored.update({k: v for k, v in updates.items() if v is not None})
    stored["version"] = SETTINGS_VERSION
    path.write_text(json.dumps(stored, indent=2))
    return load_settings(adapter_dir)


def unload():
    with _lock:
        _state.update(key=None, meta=None, model=None, tokenizer=None, index=None,
                      prompt_cache=None, cache_tokens=[])
        try:
            import gc

            import mlx.core as mx
            gc.collect()
            mx.clear_cache()
        except ImportError:
            pass


def loaded_adapter():
    return _state["key"]


def _ensure_loaded(adapter_dir):
    key = str(Path(adapter_dir).resolve())
    if _state["key"] == key:
        return
    unload()
    meta, model, tokenizer = load_persona(adapter_dir)
    index = None
    exemplars = Path(adapter_dir) / "exemplars.jsonl"
    if meta.get("prompt_style") == "retrieval" and exemplars.exists():
        index = ExemplarIndex.load(exemplars)
    _state.update(key=key, meta=meta, model=model, tokenizer=tokenizer, index=index)


def build_messages(meta, history, index=None):
    """System prompt + trained-length history window, matching training."""
    keep = meta.get("context_turns") or LEGACY_CONTEXT_TURNS
    window = sticky_window(history, keep)
    display_name = meta.get("display_name") or meta.get("persona") or "them"
    if meta.get("format_version", 1) < 2:
        system = meta.get("system_prompt") or prepare_data.base_system_prompt(display_name)
    else:
        exemplars = []
        if index is not None:
            query = next((m["content"] for m in reversed(window) if m["role"] == "user"), "")
            exemplars = index.search(query, k=prepare_data.EXEMPLARS_PER_PROMPT) if query else []
        system = prepare_data.render_system_prompt(display_name, exemplars)
    return [{"role": "system", "content": system}] + window


def _common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _reusable_prefix(prompt_tokens):
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    cache, cached = _state["prompt_cache"], _state["cache_tokens"]
    if cache is None or not can_trim_prompt_cache(cache):
        return 0
    if getattr(cache[0], "offset", None) != len(cached):
        return 0
    keep = min(_common_prefix(cached, prompt_tokens), len(prompt_tokens) - 1)
    drop = len(cached) - keep
    if drop and trim_prompt_cache(cache, drop) != drop:
        return 0
    return keep


def reply(adapter_dir, history, **overrides):
    """history: [{"role": "user"|"assistant", "content": str}, ...] ending
    with the newest user message. Returns the reply; separate bubbles are
    separated by "\n"."""
    from mlx_lm.models.cache import make_prompt_cache

    with _lock:
        _ensure_loaded(adapter_dir)
        meta, model, tokenizer = _state["meta"], _state["model"], _state["tokenizer"]
        settings = resolve_settings(read_settings_file(adapter_dir), meta.get("reply_tokens_p95"), overrides)
        messages = build_messages(meta, history, _state["index"])
        prompt_tokens = list(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        prefix = _reusable_prefix(prompt_tokens)
        if prefix == 0:
            _state["prompt_cache"] = make_prompt_cache(model)
        cache = _state["prompt_cache"]
        _state["cache_tokens"] = []  # invalid until generation finishes cleanly
        text, finish, generated = generate_reply(model, tokenizer, prompt_tokens, settings,
                                                 prompt_cache=cache, cached_prefix=prefix)
        offset = getattr(cache[0], "offset", None)
        all_tokens = prompt_tokens + generated
        _state["cache_tokens"] = all_tokens[:offset] if offset is not None and offset <= len(all_tokens) else []
        if not _state["cache_tokens"]:
            _state["prompt_cache"] = None
        return clean_reply(text, finish)
