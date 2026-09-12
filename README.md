# personabot

Upload someone's texts. Talk to them.

Fine-tunes a small instruction-tuned LLM (LoRA) on one person's side of a chat
export, then serves a chat window that talks back in their style — fully
local, running on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Requirements

- **A Mac with Apple Silicon** (M1/M2/M3/M4). This is a hard requirement —
  [MLX](https://github.com/ml-explore/mlx) only runs on Apple Silicon, so
  Intel Macs, Windows, and Linux can't run the local training/chat path.
- Python 3.10+ and [Homebrew](https://brew.sh) (for the `.txt` export step
  below).
- ~5-10GB free disk for a base model (4-bit quantized).

## Get the code

```bash
git clone https://github.com/jakecapmeil/personabot.git
cd personabot
pip install -r requirements.txt
```

Nobody's chat data or trained personas are in this repo — `data/` and
`models/adapters/` are gitignored. Each person who clones it uploads their
own export and trains their own.

## Run

```bash
cd backend
uvicorn app:app --port 8000
```

Open http://localhost:8000

That binds to this Mac only. To let other devices on your network open a
persona's share link, bind all interfaces instead:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Other devices only ever get the share page and chat for trained personas —
uploads, training data, settings and deletion stay limited to requests made
to `localhost` from this Mac, and there's no CORS, so web pages you visit
can't read your uploaded messages through the local API.

Tests (no MLX needed): `cd backend && python3 -m unittest discover tests`

### Optional: a double-clickable Mac app

```bash
bash macapp/build.sh
```

Installs `PersonaBot.app` to `~/Applications` (build with
`PERSONABOT_HOST=0.0.0.0 bash macapp/build.sh` for network share links) — double-click (or Spotlight)
starts the server if it isn't running and opens it in your browser. Still a
browser tab under the hood (a real native window is a bigger project), but
a real Dock/Finder icon instead of a terminal command. Re-run the script any
time after moving the repo, since it bakes in an absolute path to it.

## Getting a `.txt` export

Also available in-app behind the **?** button.

For an iMessage conversation on a Mac:

1. Install Homebrew if you don't have it, then: `brew install imessage-exporter`
2. Grant access — the Messages database is protected: **System Settings →
   Privacy & Security → Full Disk Access → add Terminal**, then restart
   Terminal.
3. Export every conversation as text:
   `imessage-exporter -f txt -o ~/Desktop/exported-messages`
4. Open that folder — one `.txt` per conversation, named by phone
   number/email/contact. Find the one for this person — **check for more
   than one file for them** (see below).
5. Upload *that* file in the app. If the person shows up under several
   handles, personabot asks which senders are them — tick every one.

Already-formatted transcripts work too — any `Me: …` / `Name: …` per-line
text file.

**Got fewer messages than expected?** The exporter can only read what's
actually on this Mac — two separate, common causes:

1. **Not fully synced.** In Messages → Settings → iMessage, turn on
   **Messages in iCloud** and set **Keep Messages** to **Forever**, then
   give it time to finish syncing (can take a while for a long history)
   before re-exporting.
2. **Split across threads.** iMessage starts a *separate conversation per
   handle* — a different phone number, a changed number, or an email
   address each create their own thread with the same person. Look for
   more than one exported file matching them and merge before uploading:
   `cat person-number.txt person-email.txt > combined.txt`, then tick
   both handles when personabot asks which senders are this person.

([imessage-exporter](https://github.com/ReagentX/imessage-exporter) reads
your local Messages database directly; nothing leaves your machine.)

## How it works

1. **Upload** a chat export as `.txt` — a raw `imessage-exporter -f txt`
   export or an already-formatted `Me: …` / `Name: …` transcript
   (`backend/parser.py`). Exporter boilerplate is stripped: threaded replies
   are pulled out into their own messages, edits keep their final text, and
   reply notices, send effects, link-preview titles/summaries and
   deleted/unsent notices are dropped. Photos, links and stickers become
   `[image]`/`[link]`/`[sticker]` placeholders that give context but are
   never something the persona learns to say. Back-to-back messages are
   joined with line breaks, so the persona learns to double-text.
   Timestamps split the chat into conversations at gaps over 4 hours. If
   several people or handles appear, you pick which senders are this person.
2. **Prepare** (`backend/prepare_data.py`) builds training rows from whole
   conversations; the split into train/validation is by conversation, so
   nothing leaks between them. The system prompt is just
   *"You are {name}, texting a friend."* — the voice comes from training.
   Optionally, **style hints** add three of the person's real replies to
   messages similar to the one being answered (`backend/retrieval.py`,
   BM25 over their history, with contact details filtered out).
3. **Train** (`backend/train_local.py` → `backend/lora_train.py`) runs LoRA
   on a 4-bit base model with mlx-lm (pinned to 0.31.3). The loss covers
   every reply in a conversation chunk, each reply exactly once. It trains
   about one epoch with warmup and cosine decay, evaluates on the whole
   validation set ~10 times, and stops after 3 evals without improvement.
   Among checkpoints within 2% of the best validation loss, it keeps the
   one whose generated replies are closest to the person's real texting
   style (`backend/eval_style.py`: reply length, casing, punctuation,
   emoji, double-texting, repetition, copying, assistant-like phrases).
   Base models, picked per persona in Settings:
   - `Qwen2.5-3B-Instruct` / `Qwen2.5-7B-Instruct`
   - `Llama-3.2-3B-Instruct` / `Llama-3.1-8B-Instruct`
   - `Qwen2.5-3B` / `Qwen2.5-7B` base models — no assistant tuning to fight;
     worth comparing on the style score

   all 4-bit MLX builds from `mlx-community`. Examples are re-sized with the
   chosen model's own tokenizer before training.
4. **Chat** (`backend/chat_infer.py`) rebuilds the prompt exactly as in
   training, including a history window of the length the model was trained
   on, and reuses the previous turn's KV cache. Sampling uses min-p with a
   light repetition penalty on generated tokens only; the default reply
   length comes from the person's own messages. Multi-line replies show up
   as separate bubbles. Generation settings are editable per persona.
5. **Share** a trained persona two ways: a live link (`/p/<name>`, see
   *Run* for network access) or a downloadable adapter `.zip`.
6. **Cloud training (Colab + Unsloth)**: one click downloads a notebook for
   the selected model plus `train.jsonl`/`valid.jsonl`. It downloads just the
   LoRA adapter back, which `backend/import_colab.py` converts to mlx-lm's
   format and runs on the same 4-bit base as local training.

### Checking a persona's style

The style score shows on each persona card and in Settings. To re-run it
(e.g. after changing generation settings):

```bash
cd backend
python3 eval_style.py --name <persona-name>
```

It writes `models/adapters/<name>/style_eval.json`, including sample
real-vs-generated replies.

## Cloud training (Colab)

Faster than local (a free Colab GPU beats the M-series' unified-memory GPU
for this), at the cost of leaving your Mac.

1. In the app, pick the base model in the persona's **Settings**, then click
   **Download for Colab** — downloads the notebook, `train.jsonl` and
   `valid.jsonl` (an in-app popup walks through the rest). The notebook is
   plain code with no chat data in it; the two `.jsonl` files carry this
   person's texts, so treat them like the export they came from.
2. Go to [colab.research.google.com](https://colab.research.google.com) →
   **File → Upload notebook** → pick the `.ipynb`, set
   **Runtime → Change runtime type → T4 GPU**, then run the cells in order.
   The upload cell asks for **both** `train.jsonl` and `valid.jsonl`.

The notebook trains one epoch with early stopping, keeps the best checkpoint
by validation loss, and downloads `persona_adapter.zip` (the LoRA weights —
at most a few hundred MB). Back on your Mac, use **Import Colab adapter
(.zip)** on the persona's card, or run:
```bash
python3 backend/import_colab.py --name <persona-name> --zip ~/Downloads/persona_adapter.zip
```
`merged_model.zip` files from older notebooks still import (converted with
8-bit quantization).

The notebook hasn't been run end-to-end on Colab by this project; if an
Unsloth update breaks a cell, Unsloth's docs are the fastest fix.

## Notes on scale

LoRA fine-tuning a pretrained instruct model is a different regime from a
from-scratch transformer — the base model already knows English; the adapter
only has to steer style and content, so a few-thousand-message corpus goes a
long way. Still, a few hundred examples will sound thinner than a few
thousand. The upload step reports how many of their replies it can learn from —
treat under ~500 as "expect a rough sketch of their style," not a reliable
conversationalist.

Personas prepared or trained by earlier versions still chat (with their
original prompt and 4-message history), but need their export uploaded again
before retraining.
