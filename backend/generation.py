"""
Shared generation settings and helpers for chat (chat_infer.py) and for
checkpoint selection during training (lora_train.py).

mlx is imported lazily inside the functions that need it, so the pure
helpers here stay importable (and testable) without Apple Silicon.
"""
from parser import PLACEHOLDERS

SETTINGS_VERSION = 2
DEFAULT_SETTINGS = {
    "temperature": 0.8,
    "top_p": 1.0,           # disabled; min_p does the tail-trimming
    "min_p": 0.05,
    "max_tokens": None,     # None -> sized from the persona's own reply lengths
    "repetition_penalty": 1.05,
    "repetition_context": 32,
}
# Values the old settings UI wrote into settings.json wholesale; treated as
# "never chosen" so personas pick up the new defaults.
LEGACY_DEFAULTS = {
    "temperature": 0.7, "top_p": 0.9, "max_tokens": 48,
    "repetition_penalty": 1.15, "repetition_context": 24,
}
FALLBACK_MAX_TOKENS = 96

# Chat-template end-of-turn tokens. mlx_lm stops on the tokenizer's eos
# token, which for base (non-instruct) checkpoints is not the end-of-turn
# token the adapter learned to emit.
END_OF_TURN_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>")


def default_max_tokens(reply_tokens_p95):
    if not reply_tokens_p95:
        return FALLBACK_MAX_TOKENS
    return max(32, min(256, round(reply_tokens_p95 * 1.5) + 8))


def resolve_settings(stored, reply_tokens_p95=None, overrides=None):
    settings = dict(DEFAULT_SETTINGS)
    stored = dict(stored or {})
    if stored.get("version") != SETTINGS_VERSION:
        stored = {k: v for k, v in stored.items() if LEGACY_DEFAULTS.get(k) != v}
    settings.update({k: v for k, v in stored.items() if k in DEFAULT_SETTINGS and v is not None})
    settings.update({k: v for k, v in (overrides or {}).items() if k in DEFAULT_SETTINGS and v is not None})
    if not settings["max_tokens"]:
        settings["max_tokens"] = default_max_tokens(reply_tokens_p95)
    return settings


def sticky_window(history, keep, stride=None):
    """At least the last `keep` messages (fewer than keep + stride), with a
    window start that only moves every `stride` messages. The model always
    sees at least the history length it was trained with, and the prompt
    prefix stays stable for several turns so the KV cache is reused."""
    n = len(history)
    if keep <= 0 or n <= keep:
        return list(history)
    stride = stride or max(2, keep // 2)
    start = ((n - keep) // stride) * stride
    return list(history[start:])


def clean_reply(text, finish_reason=None):
    """Strip placeholders, blank lines and back-to-back duplicate bubbles; if
    generation hit max_tokens, drop the cut-off tail instead of showing half
    a word."""
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line and line not in PLACEHOLDERS and (not lines or line != lines[-1]):
            lines.append(line)
    if finish_reason == "length" and lines:
        if len(lines) > 1:
            lines = lines[:-1]
        else:
            s = lines[0]
            cut = max(s.rfind(". "), s.rfind("! "), s.rfind("? "))
            if cut >= len(s) // 2:
                s = s[:cut + 1]
            else:
                space = s.rfind(" ")
                if space > len(s) // 2:
                    s = s[:space]
            lines = [s]
    return "\n".join(lines)


def add_end_of_turn_eos(tokenizer):
    vocab = tokenizer.get_vocab()
    for token in END_OF_TURN_TOKENS:
        if token in vocab:
            tokenizer.add_eos_token(token)


def make_repetition_penalty(penalty, context_size):
    """Repetition penalty over *generated* tokens only.

    mlx_lm hands logits processors some of the prompt too — all of it in
    mlx-lm <=0.27 and on main, only the final token in 0.28-0.31. Penalizing
    the prompt suppresses the stop token and the "\n" before the reply (so
    "lol" can't just end, and multi-bubble replies are discouraged) and
    penalizes echoing the user's own words. The processor runs once per
    sampled token, so the k-th call has exactly k generated tokens at the end
    of `tokens`, whatever prompt prefix the installed version includes."""
    import mlx.core as mx

    calls = 0

    def processor(tokens, logits):
        nonlocal calls
        n_generated, calls = calls, calls + 1
        if n_generated == 0:
            return logits
        recent = tokens[-min(n_generated, context_size):]
        selected = logits[:, recent]
        selected = mx.where(selected < 0, selected * penalty, selected / penalty)
        logits[:, recent] = selected
        return logits

    return processor


def generate_reply(model, tokenizer, prompt_tokens, settings, prompt_cache=None, cached_prefix=0):
    """Generate one reply. prompt_tokens is the full prompt; the first
    `cached_prefix` tokens must already be in prompt_cache.
    Returns (raw_text, finish_reason, generated_token_ids)."""
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler

    feed = prompt_tokens[cached_prefix:]
    sampler = make_sampler(
        temp=settings["temperature"], top_p=settings["top_p"], min_p=settings["min_p"]
    )
    processors = None
    if settings["repetition_penalty"] and settings["repetition_penalty"] > 1.0:
        processors = [make_repetition_penalty(settings["repetition_penalty"], settings["repetition_context"])]

    parts, tokens, finish = [], [], None
    for response in stream_generate(
        model, tokenizer, mx.array(feed), max_tokens=settings["max_tokens"],
        sampler=sampler, logits_processors=processors, prompt_cache=prompt_cache,
    ):
        parts.append(response.text)
        tokens.append(response.token)
        finish = response.finish_reason
    return "".join(parts), finish, tokens
