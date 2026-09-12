"""
Style evaluation: does the persona *text like them*?

Validation loss measures how well the model predicts exact words, which
bottoms out early and says little about voice. This compares generated
replies to the person's real replies for the same validation contexts:

- reply length distribution (word-count quantiles, log scale)
- lowercase-start rate, trailing-punctuation rate, emoji rate, multi-bubble rate
- distinct-2 (a deficit means the model repeats itself more than they do)
- copy rate: generated replies copied verbatim from the prompt (e.g. an exemplar)
- assistant-isms ("Sure!", "That sounds great", "I'm here for you")

style_distance sums the gaps; lower is closer. Used by lora_train.py to pick
between near-best checkpoints, and runnable on a finished persona:

    python eval_style.py --name hudson
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

EMOJI_RE = re.compile("[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF☀-➿⭐❤]")
END_PUNCT_RE = re.compile(r"[.!?]$")
ASSISTANTISM_RE = re.compile(
    r"\b(as an ai|language model|i'?m here (?:for you|to help)|how can i (?:help|assist)|"
    r"feel free to|let me know if|great question|i understand (?:how|that|your)|"
    r"that sounds (?:great|fun|amazing|lovely|wonderful)|sure thing!|absolutely!|of course!)",
    re.I,
)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
RATE_KEYS = ("lowercase_start", "end_punct", "emoji", "multi_bubble")


def _quantile(sorted_values, q):
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _distinct(texts, n):
    grams, total = set(), 0
    for t in texts:
        words = t.lower().split()
        for i in range(len(words) - n + 1):
            grams.add(tuple(words[i:i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def style_metrics(texts):
    texts = [t for t in texts if t is not None]
    n = len(texts) or 1
    words = sorted(len(t.split()) for t in texts)
    first_alpha = [next((c for c in t if c.isalpha()), "") for t in texts]
    return {
        "n": len(texts),
        "length_quantiles": [_quantile(words, q) for q in QUANTILES],
        "lowercase_start": sum(1 for c in first_alpha if c and c.islower()) / n,
        "end_punct": sum(1 for t in texts if END_PUNCT_RE.search(t.strip())) / n,
        "emoji": sum(1 for t in texts if EMOJI_RE.search(t)) / n,
        "multi_bubble": sum(1 for t in texts if "\n" in t.strip()) / n,
        "distinct2": _distinct(texts, 2),
        "assistantisms": sum(1 for t in texts if ASSISTANTISM_RE.search(t)) / n,
    }


def _normalize(text):
    return " ".join(text.lower().split())


def copy_rate(generated, contexts):
    copied = 0
    for gen, ctx in zip(generated, contexts):
        g = _normalize(gen)
        if len(g) >= 12 and g in _normalize(" ".join(m["content"] for m in ctx)):
            copied += 1
    return copied / (len(generated) or 1)


def compare(real, generated, contexts):
    r, g = style_metrics(real), style_metrics(generated)
    length_gap = min(1.0, sum(
        abs(math.log1p(a) - math.log1p(b)) for a, b in zip(r["length_quantiles"], g["length_quantiles"])
    ) / len(QUANTILES))
    rate_gap = sum(abs(r[k] - g[k]) for k in RATE_KEYS)
    distinct_deficit = max(0.0, r["distinct2"] - g["distinct2"])
    copies = copy_rate(generated, contexts)
    distance = length_gap + rate_gap + distinct_deficit + copies + g["assistantisms"]
    return {
        "style_distance": round(distance, 4),
        "length_gap": round(length_gap, 4),
        "rate_gap": round(rate_gap, 4),
        "distinct2_deficit": round(distinct_deficit, 4),
        "copy_rate": round(copies, 4),
        "assistantisms": round(g["assistantisms"], 4),
        "real": r,
        "generated": g,
    }


def sample_contexts(val_rows, n, seed=0):
    """(context messages, real reply) pairs: each row's last trained reply."""
    rows = list(val_rows)
    random.Random(seed).shuffle(rows)
    pairs = []
    for row in rows[:n]:
        idx = max(row["train_turns"])
        pairs.append((row["messages"][:idx], row["messages"][idx]["content"]))
    return pairs


def generate_for_contexts(model, tokenizer, pairs, settings, seed=0):
    import mlx.core as mx
    from generation import clean_reply, generate_reply

    mx.random.seed(seed)
    outputs = []
    for context, _ in pairs:
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True)
        text, finish, _ = generate_reply(model, tokenizer, prompt, settings)
        outputs.append(clean_reply(text, finish))
    return outputs


def main():
    ap = argparse.ArgumentParser(description="Score a trained persona's texting style against its validation set.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=64, help="Validation contexts to generate for")
    args = ap.parse_args()

    import chat_infer
    from generation import resolve_settings

    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data" / "processed" / args.name
    adapter_dir = root / "models" / "adapters" / args.name
    val_rows = [json.loads(l) for l in (data_dir / "valid.jsonl").read_text().splitlines() if l.strip()]
    meta, model, tokenizer = chat_infer.load_persona(adapter_dir)
    settings = resolve_settings(chat_infer.read_settings_file(adapter_dir), meta.get("reply_tokens_p95"))

    pairs = sample_contexts(val_rows, args.n)
    generated = generate_for_contexts(model, tokenizer, pairs, settings)
    report = compare([real for _, real in pairs], generated, [ctx for ctx, _ in pairs])
    report["examples"] = [
        {"context_last": ctx[-1]["content"] if len(ctx) > 1 else "", "real": real, "generated": gen}
        for (ctx, real), gen in list(zip(pairs, generated))[:12]
    ]
    (adapter_dir / "style_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    meta_path = adapter_dir / "persona_meta.json"
    persona_meta = json.loads(meta_path.read_text())
    persona_meta["style"] = {k: v for k, v in report.items() if k not in ("real", "generated", "examples")}
    meta_path.write_text(json.dumps(persona_meta, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
