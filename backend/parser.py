"""
Turn an uploaded text file into clean alternating (sender, text) turns.

Two input formats are accepted, auto-detected:

1. Already-formatted transcript — most lines look like "Name: message",
   e.g. exports from this project's own corpus.txt, or anything someone
   hand-formatted as "Me: ...\\nJordan: ...".

2. Raw imessage-exporter .txt output — the format produced by
   `imessage-exporter -f txt`, with date-header / sender / body blocks
   and "Tapbacks:" sections.

Either way, the target persona is whatever sender label is NOT "Me" —
if a file has more than two speakers we keep "Me" plus only the most
frequent other sender to keep the model single-persona.
"""
import re
from collections import Counter

DATE_RE = re.compile(r"^[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2}\s?[AP]M")
# Anywhere-in-text, not anchored to the start: iMessage exports render sticker/
# attachment placeholders as e.g. "Normal Sticker from Hudson: /Users/.../
# StickerCache/<uuid>.heic", so a start-anchored check on the path alone missed
# these (they were slipping into the training corpus, and worse, occasionally
# getting mined as "signature phrases" and baked into the system prompt).
ATTACHMENT_RE = re.compile(r"/(Attachments|StickerCache)/")
APPLE_CASH_RE = re.compile(r"^Apple Cash transaction:")
TAPBACK_HEADER = "Tapbacks:"
PLAIN_TURN_RE = re.compile(r"^[^\s:][^:\n]{0,40}:\s?.+$")


def is_junk_text(text):
    return bool(ATTACHMENT_RE.search(text) or APPLE_CASH_RE.match(text))


def looks_preformatted(lines, sample=200):
    sample_lines = [l for l in lines[:sample] if l.strip()]
    if not sample_lines:
        return False
    hits = sum(1 for l in sample_lines if PLAIN_TURN_RE.match(l.strip()))
    return hits / len(sample_lines) > 0.6


def parse_preformatted(lines):
    turns = []
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = re.match(r"^([^\s:][^:\n]{0,40}):\s?(.*)$", line.strip())
        if not m:
            continue
        sender, text = m.group(1).strip(), m.group(2).strip()
        if text and not is_junk_text(text):
            turns.append((sender, text))
    return turns


def parse_imessage_export(lines):
    turns = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        if DATE_RE.match(line):
            i += 1
            if i >= n:
                break
            sender = lines[i].strip()
            i += 1
            body_lines = []
            while i < n:
                cur = lines[i].rstrip("\n")
                if cur.strip() == "" or cur.strip() == TAPBACK_HEADER or DATE_RE.match(cur):
                    break
                body_lines.append(cur)
                i += 1
            if i < n and lines[i].strip() == TAPBACK_HEADER:
                i += 1
                while i < n and lines[i].strip() != "":
                    i += 1
            text = " ".join(l.strip() for l in body_lines if l.strip())
            if not text or is_junk_text(text):
                continue
            turns.append((sender, text))
        else:
            i += 1
    return turns


def merge_consecutive(turns):
    merged = []
    for sender, text in turns:
        if merged and merged[-1][0] == sender:
            merged[-1] = (sender, merged[-1][1] + " " + text)
        else:
            merged.append((sender, text))
    return merged


def restrict_to_two_speakers(turns, me_label="Me"):
    """Keep 'Me' plus only the most common other sender."""
    others = Counter(s for s, _ in turns if s != me_label)
    if not others:
        return turns, None
    persona = others.most_common(1)[0][0]
    return [(s, t) for s, t in turns if s == me_label or s == persona], persona


def parse_file(path, me_label="Me"):
    """Returns (turns, persona_name) where turns is a merged (sender, text) list."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if looks_preformatted(lines):
        turns = parse_preformatted(lines)
    else:
        turns = parse_imessage_export(lines)

    turns = merge_consecutive(turns)
    turns, persona = restrict_to_two_speakers(turns, me_label)
    if persona is None:
        raise ValueError(
            "Could not find a second speaker in this file. Expected lines like "
            "'Me: ...' / 'Name: ...', or a raw imessage-exporter .txt export."
        )
    return turns, persona
