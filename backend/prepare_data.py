"""
Turn a parsed chat export into training data.

Two stages, so the expensive, model-specific part can be redone per model:

1. prepare()        parse -> sessions -> train/val units -> sessions.json
                    (model-independent; runs once at upload)
2. build_dataset()  sessions.json + a tokenizer -> train.jsonl / valid.jsonl
                    (re-run at train time with the *selected* model's
                    tokenizer — Llama and Qwen templates differ in length)

Rows are {"messages": [...], "train_turns": [i, ...]}: the loss covers every
listed assistant message, not just the last one.

Prompt styles
-------------
- "minimal":   system prompt is only the persona's identity. Sessions are
               cut into chunks that fit the token budget; every persona reply
               in a chunk is trained, and each reply is trained exactly once
               (overlapping lead-in context is never re-trained).
- "retrieval": the system prompt adds 3 of the persona's real replies to
               messages similar to the one being answered (retrieval.py).
               The prompt differs per reply, so each row trains one reply.

The split is by whole sessions (or by contiguous chunks when a file has no
timestamps), so no conversation leaks between train and validation.
"""
import argparse
import json
import math
import random
from pathlib import Path

from parser import PLACEHOLDERS, URL_RE, build_sessions, looks_like_handle, parse_file
from retrieval import ExemplarIndex, build_rows

FORMAT_VERSION = 2
PROMPT_STYLES = ("minimal", "retrieval")
DEFAULT_CONTEXT_TURNS = 8
MAX_SEQ_LENGTH = 1024
CHUNK_TOKENS = 640         # target size for multi-reply chunks (keeps batches light)
LENGTH_MARGIN = 16
MAX_VAL_EXAMPLES = 250     # validation runs on the whole set each eval; keep it bounded
DEFAULT_TOKENIZER_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
EXEMPLARS_PER_PROMPT = 3
MAX_EXEMPLAR_CLIP = 80

SYSTEM_BASE = "You are {name}, texting a friend."
EXEMPLAR_HEADER = "Things {name} has said in similar moments:"


# ---------------------------------------------------------------- prompts

def base_system_prompt(display_name):
    return SYSTEM_BASE.format(name=display_name)


def _clip(text, n=MAX_EXEMPLAR_CLIP):
    text = text.replace("\n", " / ")
    return text if len(text) <= n else text[:n].rstrip() + "…"


def render_system_prompt(display_name, exemplars=()):
    base = base_system_prompt(display_name)
    if not exemplars:
        return base
    lines = [f'- "{_clip(e["prompt"])}" → "{_clip(e["reply"])}"' for e in exemplars]
    return base + "\n\n" + EXEMPLAR_HEADER.format(name=display_name) + "\n" + "\n".join(lines)


# ---------------------------------------------------------------- turns

def persona_speech(text):
    """Persona text with placeholders and URLs removed — what the model
    should learn to say. Empty for media-only turns."""
    lines = []
    for line in text.split("\n"):
        if line in PLACEHOLDERS:
            continue
        if URL_RE.search(line):
            line = " ".join(URL_RE.sub("", line).split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def prepare_turns(session):
    """parser.Turn list -> dicts. Media-only persona turns stay as context
    (showing their placeholders) but are never trained."""
    out = []
    for t in session:
        if t.role == "persona":
            speech = persona_speech(t.text)
            out.append({"role": "persona", "text": speech or t.text, "trainable": bool(speech)})
        else:
            out.append({"role": "me", "text": t.text, "trainable": False})
    return out


def make_units(sessions, min_units=8, max_units=40):
    """Split units: sessions when there are enough, otherwise contiguous
    chunks cut *within* sessions (e.g. a transcript with no timestamps)."""
    sessions = [s for s in sessions if s]
    if len(sessions) >= min_units:
        return sessions
    total = sum(len(s) for s in sessions)
    target = min(max_units, max(min_units, total // 40))
    size = max(20, math.ceil(total / max(target, 1)))
    return [s[i:i + size] for s in sessions for i in range(0, len(s), size)]


def choose_val_units(units, val_frac=0.1, seed=0):
    counts = [sum(1 for t in u if t["trainable"]) for u in units]
    total = sum(counts)
    if len(units) < 2 or total == 0:
        return set()
    target = max(1, val_frac * total)
    order = list(range(len(units)))
    random.Random(seed).shuffle(order)
    val, acc = set(), 0
    for idx in order:
        if acc >= target or len(val) >= len(units) - 1:
            break
        c = counts[idx]
        if c == 0 or acc + c > target * 1.5:
            continue
        val.add(idx)
        acc += c
    if not val:  # every unit was too large: take the smallest non-empty one
        candidates = [i for i in order if counts[i] > 0]
        val.add(min(candidates, key=lambda i: counts[i]))
    return val


# ---------------------------------------------------------------- tokens

def load_tokenizer(model_repo):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_repo)


class TokenCounter:
    """Estimates chat-template lengths from per-turn token counts, so trimming
    doesn't re-render the template on every step. Final rows are still
    checked once against the real template."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self._cache = {}
        x = "x"
        cx = self.count(x)
        sys_only = self.render([{"role": "system", "content": x}])
        with_user = self.render([{"role": "system", "content": x}, {"role": "user", "content": x}])
        with_asst = self.render([
            {"role": "system", "content": x}, {"role": "user", "content": x},
            {"role": "assistant", "content": x},
        ])
        self.base = sys_only - cx
        self.per_message = max(with_user - sys_only, with_asst - with_user) - cx + 2
        self.generation_prompt = self.render(
            [{"role": "system", "content": x}, {"role": "user", "content": x}], add_generation_prompt=True
        ) - with_user

    def count(self, text):
        n = self._cache.get(text)
        if n is None:
            n = len(self.tok.encode(text, add_special_tokens=False))
            self._cache[text] = n
        return n

    def render(self, messages, add_generation_prompt=False):
        ids = self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=add_generation_prompt, return_dict=False
        )
        return len(ids)

    def turn_cost(self, turn):
        return self.per_message + self.count(turn["text"])

    def system_cost(self, system):
        return self.base + self.count(system)


def _message(turn):
    return {"role": "user" if turn["role"] == "me" else "assistant", "content": turn["text"]}


def _prefix_sums(values):
    sums = [0]
    for v in values:
        sums.append(sums[-1] + v)
    return sums


# ---------------------------------------------------------------- examples

def build_chunked_examples(unit, system, counter, budget, context_turns):
    """Every trainable reply in the unit is trained exactly once, with at least
    `context_turns` of lead-in context (budget permitting)."""
    costs = [counter.turn_cost(t) for t in unit]
    sums = _prefix_sums(costs)
    sys_cost = counter.system_cost(system)
    examples = []
    i, n = 0, len(unit)
    while i < n:
        while i < n and not unit[i]["trainable"]:
            i += 1
        if i >= n:
            break
        if sys_cost + costs[i] > budget:
            i += 1  # a single reply longer than the whole budget
            continue
        lead = max(0, i - context_turns)
        while lead < i and sys_cost + sums[i + 1] - sums[lead] > budget:
            lead += 1
        total = sys_cost + sums[i + 1] - sums[lead]
        j = i + 1
        while j < n and total + costs[j] <= budget:
            total += costs[j]
            j += 1
        last = max(k for k in range(i, j) if unit[k]["trainable"])
        chunk = unit[lead:last + 1]
        messages = [{"role": "system", "content": system}] + [_message(t) for t in chunk]
        train_turns = [1 + k - lead for k in range(i, last + 1) if unit[k]["trainable"]]
        examples.append({"messages": messages, "train_turns": train_turns})
        i = last + 1
    return examples


def build_single_examples(unit, unit_id, display_name, counter, budget, context_turns,
                          index=None, exclude_units=()):
    costs = [counter.turn_cost(t) for t in unit]
    sums = _prefix_sums(costs)
    examples = []
    for i, turn in enumerate(unit):
        if not turn["trainable"]:
            continue
        lead = max(0, i - context_turns)
        exemplars = []
        if index is not None:
            query = next((unit[k]["text"] for k in range(i - 1, lead - 1, -1) if unit[k]["role"] == "me"), "")
            exemplars = index.search(query, k=EXEMPLARS_PER_PROMPT, exclude_units=exclude_units) if query else []
        system = render_system_prompt(display_name, exemplars)
        sys_cost = counter.system_cost(system)
        if sys_cost + costs[i] > budget:
            continue
        while lead < i and sys_cost + sums[i + 1] - sums[lead] > budget:
            lead += 1
        messages = [{"role": "system", "content": system}] + [_message(t) for t in unit[lead:i + 1]]
        examples.append({"messages": messages, "train_turns": [len(messages) - 1]})
    return examples


def _fit_exactly(example, counter, max_len):
    """One real template render per row; drop leading context if the estimate
    was off. Returns None if the row still doesn't fit."""
    messages, train_turns = list(example["messages"]), list(example["train_turns"])
    while counter.render(messages) > max_len:
        if len(messages) > 2 and 1 not in train_turns:
            messages.pop(1)
            train_turns = [t - 1 for t in train_turns]
        else:
            return None
    return {"messages": messages, "train_turns": train_turns}


# ---------------------------------------------------------------- stages

def resolve_display_name(display_name, persona_senders, name):
    if display_name and display_name.strip():
        return display_name.strip()
    label = persona_senders[0] if persona_senders else name
    return name if looks_like_handle(label) else label


def prepare(raw_path, data_dir, *, name, me_label="Me", persona_senders=None, display_name=None,
            prompt_style="minimal", context_turns=DEFAULT_CONTEXT_TURNS, val_frac=0.1, seed=0,
            tokenizer=None, tokenizer_model=DEFAULT_TOKENIZER_MODEL):
    """Parse the export and write sessions.json, then build the dataset.
    Returns the dataset meta (see build_dataset)."""
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"prompt_style must be one of {PROMPT_STYLES}")
    messages, counts = parse_file(raw_path, me_label=me_label)
    if not persona_senders:
        persona_senders = [counts.most_common(1)[0][0]]
    unknown = [s for s in persona_senders if s not in counts]
    if unknown:
        raise ValueError(f"Not senders in this file: {', '.join(unknown)}")

    sessions = build_sessions(messages, persona_senders, me_label=me_label)
    units = make_units([prepare_turns(s) for s in sessions])
    val_units = choose_val_units(units, val_frac=val_frac, seed=seed)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": FORMAT_VERSION,
        "name": name,
        "display_name": resolve_display_name(display_name, persona_senders, name),
        "me_label": me_label,
        "persona_senders": list(persona_senders),
        "sender_counts": dict(counts.most_common()),
        "prompt_style": prompt_style,
        "context_turns": context_turns,
        "n_messages": len(messages),
        "n_sessions": len(sessions),
        "units": units,
        "val_units": sorted(val_units),
    }
    (data_dir / "sessions.json").write_text(json.dumps(state, ensure_ascii=False))
    return build_dataset(data_dir, tokenizer=tokenizer, tokenizer_model=tokenizer_model, seed=seed)


def build_dataset(data_dir, tokenizer=None, tokenizer_model=DEFAULT_TOKENIZER_MODEL,
                  max_seq_length=MAX_SEQ_LENGTH, seed=0):
    data_dir = Path(data_dir)
    state = json.loads((data_dir / "sessions.json").read_text())
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_model)
    counter = TokenCounter(tokenizer)
    units, val_units = state["units"], set(state["val_units"])
    display_name, style = state["display_name"], state["prompt_style"]
    context_turns = state["context_turns"]
    budget = max_seq_length - LENGTH_MARGIN - counter.generation_prompt

    index = None
    if style == "retrieval":
        index = ExemplarIndex(build_rows(units))
        index.save(data_dir / "exemplars.jsonl")

    train_rows, val_rows = [], []
    system = base_system_prompt(display_name)
    for unit_id, unit in enumerate(units):
        is_val = unit_id in val_units
        if style == "retrieval":
            exclude = val_units | {unit_id}
            rows = build_single_examples(unit, unit_id, display_name, counter, budget, context_turns,
                                         index=index, exclude_units=exclude)
        else:
            rows = build_chunked_examples(unit, system, counter, min(budget, CHUNK_TOKENS), context_turns)
        rows = [r for r in (_fit_exactly(r, counter, max_seq_length - LENGTH_MARGIN) for r in rows) if r]
        (val_rows if is_val else train_rows).extend(rows)

    if len(val_rows) > MAX_VAL_EXAMPLES:
        val_rows = random.Random(seed).sample(val_rows, MAX_VAL_EXAMPLES)

    for fname, rows in (("train.jsonl", train_rows), ("valid.jsonl", val_rows)):
        with open(data_dir / fname, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    reply_lengths = sorted(
        counter.count(t["text"]) for u in units for t in u if t["trainable"]
    )
    p95 = reply_lengths[min(len(reply_lengths) - 1, int(0.95 * len(reply_lengths)))] if reply_lengths else None
    meta = {
        "format_version": FORMAT_VERSION,
        "persona": display_name,
        "display_name": display_name,
        "me_label": state["me_label"],
        "persona_senders": state["persona_senders"],
        "prompt_style": style,
        "context_turns": context_turns,
        "system_prompt": system,
        "n_messages": state["n_messages"],
        "n_sessions": state["n_sessions"],
        "n_turns": sum(len(u) for u in units),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_train_replies": sum(len(r["train_turns"]) for r in train_rows),
        "reply_tokens_p95": p95,
        "tokenizer_model": tokenizer_model,
        "max_seq_length": max_seq_length,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Uploaded chat export (.txt)")
    ap.add_argument("--out_dir", required=True, help="Where to write sessions.json / train.jsonl / valid.jsonl")
    ap.add_argument("--name", default=None, help="Persona id (defaults to the out_dir name)")
    ap.add_argument("--me_label", default="Me")
    ap.add_argument("--persona_senders", default="",
                    help="Comma-separated sender labels that are this person (default: most frequent)")
    ap.add_argument("--display_name", default="")
    ap.add_argument("--prompt_style", choices=PROMPT_STYLES, default="minimal")
    ap.add_argument("--context_turns", type=int, default=DEFAULT_CONTEXT_TURNS)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer_model", default=DEFAULT_TOKENIZER_MODEL,
                    help="Tokenizer used to size examples; train_local.py rebuilds with the trained model's.")
    args = ap.parse_args()

    senders = [s.strip() for s in args.persona_senders.split(",") if s.strip()] or None
    meta = prepare(
        args.input, args.out_dir, name=args.name or Path(args.out_dir).name, me_label=args.me_label,
        persona_senders=senders, display_name=args.display_name, prompt_style=args.prompt_style,
        context_turns=args.context_turns, val_frac=args.val_frac, seed=args.seed,
        tokenizer_model=args.tokenizer_model,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
