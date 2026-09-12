"""
Turn (sender, text) turns into windowed chat-format training examples for
mlx_lm's LoRA trainer (ChatDataset: {"messages": [...]}), with a
random-shard train/val split.

Each example ends with one assistant turn (the persona's real reply) and
carries up to `context_turns` preceding turns as history, so the model
learns to track conversation flow rather than just isolated reply style.
Low-content replies ("lol", "ok") are kept, not filtered — the goal is to
reproduce the persona's actual ratio of short/long replies.

Random-shard split (not a plain shuffle of examples, and not a chronological
tail split): examples are windows that overlap their neighbors, so a plain
per-example shuffle would leak near-duplicate windows into both train and
val. Splitting by contiguous shards of the turn list — and only building
windows within a shard — avoids that leakage while still interleaving
different eras/topics into val, which is what actually measures fit
(see hudson-bot's train.py for the same lesson learned the hard way).
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from parser import parse_file

SYSTEM_TEMPLATE = (
    "You are {persona}, texting a close friend. Reply exactly the way "
    "{persona} actually texts: their real tone, slang, punctuation, "
    "capitalization, and typical message length (short replies are fine — "
    "don't pad them out). Never mention being an AI or a model."
)


_JUNK_MARKERS = ("http://", "https://", "sticker", "attachment", "stickercache",
                 "loved “", "laughed at “", "emphasized “")
_MAX_PHRASE_CHARS = 70  # hard cap on any single mined phrase actually inserted


def _is_spammy(text):
    """Reject repetition spam (e.g. a message that's just "haha" repeated
    hundreds of times) that word-count filtering alone doesn't catch: such a
    line can have very few whitespace-separated "words" while still being
    thousands of characters long. Compare against a hard char cap first
    (cheapest check), then the unique-character ratio."""
    if len(text) > _MAX_PHRASE_CHARS:
        return True
    if len(text) >= 20 and len(set(text.lower())) / len(text) < 0.25:
        return True
    return False


def mine_signature_phrases(turns, persona, max_short=6, max_long=3):
    """Pull a few of the person's actual phrases straight from their corpus:
    their most frequent short interjections/catchphrases, plus a few longer,
    distinctive lines. Priming the system prompt with real examples of their
    voice is a free lever on top of LoRA training itself — it costs no extra
    training time and directly nudges generation toward their real phrasing.

    Two failure modes learned from running this on a real corpus: (1) a
    single freakishly long message (repeated "haha..." spam, hundreds of
    chars) can bloat the *entire* system prompt past the training token
    budget, silently zeroing out every training example; (2) attachment/
    sticker placeholder text and full addresses/URLs are real corpus lines
    but not "voice" worth priming with, and can leak location/PII into
    every generation. Filtered for both below.
    """
    texts = [t for s, t in turns if s == persona]
    texts = [t for t in texts if not _is_spammy(t)
             and not any(m in t.lower() for m in _JUNK_MARKERS)]
    counts = Counter(texts)

    short = [t for t, c in counts.most_common() if c >= 2 and len(t.split()) <= 4]
    short = short[:max_short]

    long_candidates = sorted(
        {t for t in texts if 6 <= len(t.split()) <= 18},
        key=len, reverse=True,
    )
    long_lines = long_candidates[:max_long]

    return short, long_lines


def _clip(text, n=_MAX_PHRASE_CHARS):
    return text if len(text) <= n else text[:n].rstrip() + "…"


def build_system_prompt(persona, short_phrases=None, long_lines=None):
    base = SYSTEM_TEMPLATE.format(persona=persona)
    if not short_phrases and not long_lines:
        return base
    bits = []
    if short_phrases:
        bits.append("common short replies: " + ", ".join(f"\"{_clip(p)}\"" for p in short_phrases))
    if long_lines:
        bits.append("example longer messages: " + " | ".join(f"\"{_clip(p)}\"" for p in long_lines))
    return base + " For reference, here's how they actually write — " + "; ".join(bits) + "."

# mlx_lm.lora truncates sequences longer than --max-seq-length from the END.
# Our examples put the target reply LAST (system -> history -> reply), so an
# untrimmed long example gets its reply chopped off entirely, leaving zero
# loss-eligible tokens for --mask-prompt -> a 0/0 NaN loss that permanently
# corrupts training from that step on. Trim history (oldest first) at
# data-prep time so the full reply always survives; this must match the
# --max-seq-length actually used at train time, with margin for the
# generation-prompt tokens mlx_lm's mask-prompt offset adds.
MAX_SEQ_LENGTH = 512
LENGTH_MARGIN = 24  # headroom for chat-template/generation-prompt overhead

_tokenizer_cache = {}


def _get_tokenizer(model_repo):
    if model_repo not in _tokenizer_cache:
        from transformers import AutoTokenizer
        _tokenizer_cache[model_repo] = AutoTokenizer.from_pretrained(model_repo)
    return _tokenizer_cache[model_repo]


def _token_len(tokenizer, messages):
    # return_dict=False is required here: with it defaulted True, tokenize=True
    # returns a BatchEncoding (keys: input_ids, attention_mask) and len() on
    # that silently gives 2 every time, not the token count — which made an
    # earlier version of this function never actually trim anything.
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=False
    )
    return len(ids)


def build_examples(turns, persona, me_label="Me", context_turns=4, tokenizer=None,
                    max_tokens=MAX_SEQ_LENGTH - LENGTH_MARGIN, system_content=None):
    """One example per persona turn, with up to context_turns of history.

    If tokenizer is given, history is trimmed (oldest turns dropped first) so
    the full example — including the target reply — fits within max_tokens.
    An example is dropped only if system+reply alone still doesn't fit
    (a single reply longer than the whole budget), which should be rare.

    system_content, when given, should be the exact same string used at
    inference time (chat_infer.py reads it back from persona_meta.json) —
    training the adapter against one system prompt and then chatting against
    a differently-worded one wastes some of what the adapter learned.
    """
    if system_content is None:
        system_content = SYSTEM_TEMPLATE.format(persona=persona)

    examples = []
    for i, (sender, text) in enumerate(turns):
        if sender != persona:
            continue
        start = max(0, i - context_turns)
        history = list(turns[start:i])
        if not history:
            continue  # need at least one turn of context to reply to

        system_msg = {"role": "system", "content": system_content}
        target_msg = {"role": "assistant", "content": text}

        if tokenizer is not None:
            while history:
                messages = [system_msg] + [
                    {"role": "user" if s == me_label else "assistant", "content": t}
                    for s, t in history
                ] + [target_msg]
                if _token_len(tokenizer, messages) <= max_tokens:
                    break
                history = history[1:]  # drop the oldest turn of context first
            else:
                messages = [system_msg, target_msg]
            if _token_len(tokenizer, messages) > max_tokens:
                continue  # even system + reply alone doesn't fit; skip it
        else:
            messages = [system_msg] + [
                {"role": "user" if s == me_label else "assistant", "content": t}
                for s, t in history
            ] + [target_msg]

        examples.append({"messages": messages})
    return examples


def shard_split(turns, persona, me_label, context_turns, val_frac, seed, tokenizer=None,
                 system_content=None):
    n = len(turns)
    n_shards = min(40, max(8, n // 40))
    shard = n // n_shards
    rng = random.Random(seed)
    order = list(range(n_shards))
    rng.shuffle(order)
    n_val = max(1, int(n_shards * val_frac))
    val_shards = set(order[:n_val])

    train_examples, val_examples = [], []
    for s in range(n_shards):
        lo = s * shard
        hi = n if s == n_shards - 1 else (s + 1) * shard
        chunk = turns[lo:hi]
        ex = build_examples(chunk, persona, me_label, context_turns, tokenizer=tokenizer,
                             system_content=system_content)
        (val_examples if s in val_shards else train_examples).extend(ex)
    return train_examples, val_examples


DEFAULT_TOKENIZER_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
# Both the 7B and 3B Qwen2.5-Instruct MLX builds share the same tokenizer and
# chat template, so counting with one is valid for either training target.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Uploaded chat export (.txt)")
    ap.add_argument("--out_dir", required=True, help="Where to write train.jsonl / valid.jsonl")
    ap.add_argument("--me_label", default="Me")
    ap.add_argument("--context_turns", type=int, default=4)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer_model", default=DEFAULT_TOKENIZER_MODEL,
                     help="Used only to count tokens for length-safe trimming; "
                          "Qwen2.5 3B/7B share the same tokenizer.")
    args = ap.parse_args()
    tokenizer = _get_tokenizer(args.tokenizer_model)

    turns, persona = parse_file(args.input, me_label=args.me_label)
    short_phrases, long_lines = mine_signature_phrases(turns, persona)
    system_content = build_system_prompt(persona, short_phrases, long_lines)

    train_ex, val_ex = shard_split(
        turns, persona, args.me_label, args.context_turns, args.val_frac, args.seed,
        tokenizer=tokenizer, system_content=system_content,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ex in [("train.jsonl", train_ex), ("valid.jsonl", val_ex)]:
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for row in ex:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "persona": persona,
        "me_label": args.me_label,
        "n_turns": len(turns),
        "n_train": len(train_ex),
        "n_val": len(val_ex),
        "context_turns": args.context_turns,
        "system_prompt": system_content,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
