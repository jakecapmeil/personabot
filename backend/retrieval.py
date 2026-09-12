"""
Retrieval of the persona's own past replies, for the "retrieved examples"
prompt style.

Instead of baking one fixed list of phrases into every system prompt (which
an instruct model tends to parrot), each example gets a few of the person's
real replies to messages *similar to the one being answered*. The index is
plain BM25 over the preceding "me" message — no embedding model to download.

At train time the target's own unit and every validation unit are excluded,
so the model never sees its answer in the prompt and learns to use exemplars
as style cues rather than copy them. At chat time nothing is excluded.
"""
import json
import math
import re
from collections import Counter, defaultdict

from parser import PLACEHOLDERS, URL_RE

WORD_RE = re.compile(r"[a-z0-9']+")
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:\+?\d[\s().-]?){7,}")
ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9.'-]+\s+){0,3}"
    r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|ct|court|way|pl|place|pkwy|hwy|terrace|ter)\b\.?",
    re.I,
)
MAX_EXEMPLAR_CHARS = 160
MAX_EXEMPLAR_WORDS = 25


def tokenize(text):
    return WORD_RE.findall(text.lower())


def has_pii(text):
    return bool(URL_RE.search(text) or EMAIL_RE.search(text) or PHONE_RE.search(text)
                or ADDRESS_RE.search(text))


def is_spammy(text):
    """Repetition spam ("hahahaha…" hundreds of chars) and very long lines."""
    if len(text) > MAX_EXEMPLAR_CHARS:
        return True
    return len(text) >= 20 and len(set(text.lower())) / len(text) < 0.25


def usable_exemplar_text(text):
    lines = [l for l in text.split("\n") if l not in PLACEHOLDERS]
    if not lines:
        return None
    joined = "\n".join(lines)
    n_words = len(joined.split())
    if n_words == 0 or n_words > MAX_EXEMPLAR_WORDS or is_spammy(joined) or has_pii(joined):
        return None
    return joined


class ExemplarIndex:
    def __init__(self, rows, k1=1.2, b=0.75):
        """rows: [{"prompt": str, "reply": str, "unit": int}]"""
        self.rows = rows
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(r["prompt"]) for r in rows]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if rows else 0.0
        self.postings = defaultdict(list)
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        n = len(rows)
        self.idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def search(self, query, k=3, exclude_units=()):
        terms = set(tokenize(query))
        if not terms or not self.rows:
            return []
        exclude_units = set(exclude_units)
        scores = defaultdict(float)
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                if self.rows[i]["unit"] in exclude_units:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1))
                scores[i] += idf * tf * (self.k1 + 1) / norm
        results, seen_replies = [], set()
        for i, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            if score <= 0:
                break
            reply_key = self.rows[i]["reply"].lower()
            if reply_key in seen_replies:
                continue
            seen_replies.add(reply_key)
            results.append(self.rows[i])
            if len(results) >= k:
                break
        return results

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])


def build_rows(units):
    """units: list of lists of prepared turn dicts ({"role", "text", "trainable"}).
    One row per trainable persona reply that directly follows a "me" message."""
    rows = []
    for unit_id, unit in enumerate(units):
        for i in range(1, len(unit)):
            turn, prev = unit[i], unit[i - 1]
            if turn["role"] != "persona" or not turn["trainable"] or prev["role"] != "me":
                continue
            prompt = usable_exemplar_text(prev["text"])
            reply = usable_exemplar_text(turn["text"])
            if prompt and reply:
                rows.append({"prompt": prompt, "reply": reply, "unit": unit_id})
    return rows
