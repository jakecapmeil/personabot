"""
Turn an uploaded text file into clean messages, then into conversation
sessions of alternating "me" / "persona" turns.

Two input formats are accepted, auto-detected:

1. Already-formatted transcript — most lines look like "Name: message".
   Lines without a "Name:" prefix continue the previous message.

2. Raw imessage-exporter .txt output (`imessage-exporter -f txt`): a
   timestamp line, a sender line, then body lines until a blank line.

The exporter writes a lot into a message body that the person never typed.
All of it used to be space-joined straight into the training text, so this
module cleans the body line by line:

- Threaded (swipe-to-reply) messages are rendered *indented inside the
  parent's body*, with their own timestamp and sender. They're parsed as
  their own messages and re-sorted into chronological order.
- "This message responded to an earlier message.", "Sent with Confetti",
  deleted/unsent notices, location-sharing and SharePlay markers are dropped.
- "Edited 30 seconds later: …" keeps only the final edit.
- Link previews (URL, page title, page summary) become a single "[link]";
  attachments and stickers become "[image]" / "[sticker]" / … placeholders,
  which keep conversational context ("lmao" replying to a photo) without
  ever being used as something the persona should say.
- Group-conversation announcements ("X named the conversation …") are skipped.

Consecutive messages from the same sender are joined with "\n", not a space,
so the model learns real texting cadence (several short bubbles in a row).
Timestamps split the conversation into sessions at long gaps, so a reply is
never trained against context from days earlier.
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

ME_LABEL = "Me"
DEFAULT_SESSION_GAP = timedelta(hours=4)
MAX_BURST_MESSAGES = 15  # cap on same-sender messages merged into one turn

PLACEHOLDERS = frozenset({
    "[image]", "[video]", "[audio]", "[sticker]", "[link]", "[attachment]", "[payment]", "[app]",
})

HEADER_RE = re.compile(
    r"^(?P<date>[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2}\s?[AP]M)(?P<rest>.*)$"
)
PLAIN_TURN_RE = re.compile(r"^([^\s:][^:\n]{0,40}):\s?(.*)$")

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
BARE_URL_RE = re.compile(r"^(?:https?://|www\.)\S+$", re.I)
ATTACHMENT_DIR_RE = re.compile(r"/(Attachments|StickerCache)/")
PATH_RE = re.compile(r"^(?:~|/|\.{1,2}/|[\w.-]+/)\S.*\.(?P<ext>[A-Za-z0-9]{2,5})$")
STICKER_RE = re.compile(r"Sticker from .+?: \S")
EDIT_RE = re.compile(r"^Edited .+? later: (?P<text>.*)$")
UNSENT_RE = re.compile(r"^.+ unsent this message part(?: .+ after sending)?!$")
EXPRESSIVE_RE = re.compile(
    r"^Sent with (Slam|Loud|Gentle|Invisible Ink|Echo|Spotlight|Balloons|Confetti|"
    r"Love|Lasers|Fireworks|Celebration|Shooting Star|Sparkles)$"
)
APP_BALLOON_RE = re.compile(r"^[\w .'&-]{1,40} message:(?: .*)?$")
PAYMENT_RE = re.compile(r"^[\w .'-]{0,40}transaction: ")
CHECK_IN_RE = re.compile(r"^Check In(?::.*)?$")
DROP_LINES = frozenset({
    "Tapbacks:",
    "This message responded to an earlier message.",
    "This message was deleted from the conversation!",
    "Started sharing location!",
    "Stopped sharing location!",
})

MEDIA_TYPES = {
    "image": {"heic", "heif", "jpg", "jpeg", "png", "gif", "webp", "tif", "tiff", "bmp"},
    "video": {"mov", "mp4", "m4v", "avi", "3gp"},
    "audio": {"caf", "m4a", "mp3", "amr", "wav", "aac", "opus"},
}

PHONE_HANDLE_RE = re.compile(r"^\+?[\d\s().-]{7,}$")
EMAIL_HANDLE_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class Message:
    sender: str
    lines: list = field(default_factory=list)  # speech lines and placeholders
    ts: datetime | None = None


@dataclass
class Turn:
    role: str  # "me" or "persona"
    text: str  # lines joined with "\n"


def looks_like_handle(label):
    return bool(PHONE_HANDLE_RE.match(label) or EMAIL_HANDLE_RE.match(label))


def _placeholder_for_path(line):
    if "/StickerCache/" in line or STICKER_RE.search(line):
        return "[sticker]"
    m = PATH_RE.match(line)
    ext = m.group("ext").lower() if m else ""
    for kind, exts in MEDIA_TYPES.items():
        if ext in exts:
            return f"[{kind}]"
    return "[attachment]"


def _is_attachment_line(line):
    if ATTACHMENT_DIR_RE.search(line) or STICKER_RE.search(line):
        return True
    m = PATH_RE.match(line)
    if not m:
        return False
    ext = m.group("ext").lower()
    return any(ext in exts for exts in MEDIA_TYPES.values()) or ext in {"pdf", "vcf", "pkpass", "usdz"}


def clean_body(raw_lines):
    """Exporter body lines (indentation already removed) -> speech lines and
    placeholders. See the module docstring for what gets dropped."""
    out = []
    skip = 0
    drop_rest = False
    prev = None
    for raw in raw_lines:
        s = raw.strip()
        if drop_rest or not s:
            continue
        if skip:
            skip -= 1
            continue
        if s in DROP_LINES or UNSENT_RE.match(s) or EXPRESSIVE_RE.match(s):
            pass
        elif s == "Translated from:":
            skip = 1  # the original-language text follows; keep the translation only
        elif s == "SharePlay Message":
            skip = 1  # followed by "Ended"
        elif s == "Attachment missing!":
            out.append("[attachment]")
        elif s.startswith("Transcription: ") and prev in ("[audio]", "[attachment]"):
            pass
        elif (m := EDIT_RE.match(s)):
            text = m.group("text").strip()
            if out and out[-1] not in PLACEHOLDERS:
                out[-1] = text
            elif text:
                out.append(text)
        elif _is_attachment_line(s):
            out.append(_placeholder_for_path(s))
        elif BARE_URL_RE.match(s):
            out.append("[link]")
            drop_rest = True  # preview title / summary / metadata lines follow
        elif PAYMENT_RE.match(s):
            out.append("[payment]")
            drop_rest = True
        elif not out and (APP_BALLOON_RE.match(s) or CHECK_IN_RE.match(s)):
            if APP_BALLOON_RE.match(s):
                out.append("[app]")
            drop_rest = True
        elif s.startswith("Photo album: "):
            out.append("[image]")
        elif s.startswith("Digital Touch "):
            pass
        else:
            out.append(s)
        prev = out[-1] if out else None
    return [line for line in out if line]


def _header(line):
    """('message', ts) for a message header line, ('announcement', ts) for a
    group event line, None otherwise. Only matches unindented lines."""
    m = HEADER_RE.match(line)
    if not m:
        return None
    try:
        ts = datetime.strptime(" ".join(m.group("date").split()), "%b %d, %Y %I:%M:%S %p")
    except ValueError:
        ts = None
    rest = m.group("rest").strip()
    if not rest or (rest.startswith("(") and rest.endswith(")")):
        return "message", ts
    return "announcement", ts


def _parse_block(lines):
    messages = []
    i, n = 0, len(lines)
    while i < n:
        header = _header(lines[i])
        if header is None or header[0] == "announcement":
            i += 1
            continue
        ts = header[1]
        if i + 1 >= n:
            break
        sender = lines[i + 1].strip()
        i += 2

        body = []
        while i < n:
            cur = lines[i]
            if not cur.strip():
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j >= n or _header(lines[j]) is not None:
                    i = j
                    break
                body.append("")  # blank line inside a multi-paragraph message
                i = j
                continue
            if _header(cur) is not None:
                break
            body.append(cur)
            i += 1

        own, nested = [], []
        k = 0
        while k < len(body):
            if body[k].startswith("    "):
                block = []
                while k < len(body) and body[k].startswith("    "):
                    block.append(body[k][4:])
                    k += 1
                nested.append(block)
            else:
                own.append(body[k])
                k += 1

        cleaned = clean_body(own)
        if sender and cleaned:
            messages.append(Message(sender, cleaned, ts))
        for block in nested:
            # Threaded replies (and tapback entries, which contain no header
            # and are skipped) rendered inside this message.
            messages.extend(_parse_block(block))
    return messages


def parse_imessage_export(lines):
    lines = [l.rstrip("\r\n") for l in lines]
    messages = _parse_block(lines)
    seen, unique = set(), []
    for m in messages:
        key = (m.sender, m.ts, tuple(m.lines))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    if unique and all(m.ts is not None for m in unique):
        unique.sort(key=lambda m: m.ts)  # stable: replies land where they happened
    return unique


def looks_preformatted(lines, sample=200):
    sample_lines = [l for l in lines[:sample] if l.strip()]
    if not sample_lines:
        return False
    if any(HEADER_RE.match(l) for l in sample_lines):
        return False
    hits = sum(1 for l in sample_lines if PLAIN_TURN_RE.match(l.strip()))
    return hits / len(sample_lines) > 0.6


def parse_preformatted(lines, me_label=ME_LABEL):
    rows = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        m = PLAIN_TURN_RE.match(line.strip())
        rows.append((m.group(1).strip(), m.group(2)) if m else (None, line))

    # "note: bring snacks" inside a message shouldn't become a speaker called
    # "note" — only labels that recur are treated as speakers.
    label_counts = Counter(label for label, _ in rows if label)
    speakers = {l for l, c in label_counts.items() if c >= 3} | {me_label}
    if len(speakers) < 2:
        speakers = set(label_counts)

    messages = []
    for label, text in rows:
        if label in speakers:
            messages.append(Message(label, clean_body([text])))
        elif messages:
            continuation = f"{label}: {text}" if label else text
            messages[-1].lines.extend(clean_body([continuation]))
    return [m for m in messages if m.lines]


def parse_file(path, me_label=ME_LABEL):
    """Returns (messages, sender_counts) where sender_counts counts every
    sender other than me_label, most frequent first."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if looks_preformatted(lines):
        messages = parse_preformatted(lines, me_label)
    else:
        messages = parse_imessage_export(lines)

    counts = Counter(m.sender for m in messages if m.sender != me_label)
    if not counts or not any(m.sender == me_label for m in messages):
        raise ValueError(
            f"Couldn't find both '{me_label}' and another speaker in this file. Expected "
            f"lines like '{me_label}: ...' / 'Name: ...', or a raw imessage-exporter .txt export. "
            "If your own messages use a different label, set it in the upload form."
        )
    return messages, counts


def build_sessions(messages, persona_senders, me_label=ME_LABEL, gap=DEFAULT_SESSION_GAP):
    """Keep me_label plus the chosen persona senders (e.g. both their phone
    number and email handle), split into sessions at time gaps, and merge
    consecutive same-sender messages with "\n". Filtering happens before
    merging, so dropping a third speaker never leaves two "me" turns in a row."""
    persona_senders = set(persona_senders)
    kept = [m for m in messages if m.sender == me_label or m.sender in persona_senders]

    sessions, current, counts = [], [], []
    last_ts = None
    for m in kept:
        if current and m.ts is not None and last_ts is not None and m.ts - last_ts > gap:
            sessions.append(current)
            current, counts = [], []
        role = "me" if m.sender == me_label else "persona"
        if current and current[-1][0] == role and counts[-1] < MAX_BURST_MESSAGES:
            current[-1][1].extend(m.lines)
            counts[-1] += 1
        else:
            current.append((role, list(m.lines)))
            counts.append(1)
        if m.ts is not None:
            last_ts = m.ts
    if current:
        sessions.append(current)

    return [[Turn(role, "\n".join(lines)) for role, lines in s] for s in sessions]
