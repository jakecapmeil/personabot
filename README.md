# personabot

Upload someone's texts. Talk to them.

Fine-tunes a small instruction-tuned LLM (LoRA) on one person's side of a chat
export, then serves a chat window that talks back in their style — fully
local, running on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
cd backend
uvicorn app:app --port 8000
```

Open http://localhost:8000

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
5. Upload *that* file in the app.

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
   `cat person-number.txt person-email.txt > combined.txt`

([imessage-exporter](https://github.com/ReagentX/imessage-exporter) reads
your local Messages database directly; nothing leaves your machine.)

## How it works

1. **Upload** a chat export as `.txt` — auto-detected format (already-formatted
   transcript, or a raw `imessage-exporter -f txt` export). If more than two
   speakers appear, only "Me" plus the single most frequent other sender are
   kept.
2. **Prepare** (`backend/prepare_data.py`) turns the conversation into
   windowed chat examples — each ends with one real reply from the person,
   with up to 4 prior turns of history so the model learns to track a
   conversation, not just isolated reply style. Low-content replies ("lol",
   "ok") are kept, not filtered, to reproduce their actual ratio of
   short/long replies. It also mines a few of the person's actual signature
   phrases (frequent short interjections + a couple of distinctive longer
   lines) into the system prompt used both at train and chat time — a free
   lever on top of LoRA training itself, filtered against spam/attachment/
   address junk. Train/val is a random-shard split (not a chronological tail
   split) so validation loss reflects fit rather than topic drift between old
   and recent messages.
3. **Train** (`backend/train_local.py`) wraps `mlx_lm.lora` with a real LoRA
   config (rank/layers — mlx_lm's own default of rank 8 is low headroom for
   shifting a person's actual voice, not just tone), masks the loss to the
   person's reply tokens only, evaluates periodically, promotes whichever
   checkpoint had the best validation loss, and stops itself once val loss
   hasn't improved for a few evals in a row — this corpus size overfits
   within well under one epoch. Four base models, picked per persona in
   Settings:
   - `Qwen2.5-3B-Instruct` / `Qwen2.5-7B-Instruct`
   - `Llama-3.2-3B-Instruct` / `Llama-3.1-8B-Instruct`

   all 4-bit MLX builds from `mlx-community`.
4. **Chat** (`backend/chat_infer.py`) loads the base model + adapter once and
   generates replies through the same chat template + system prompt used in
   training, with a repetition penalty and per-persona generation settings
   (temperature/top-p/max length) editable live in the Settings sheet.
5. **Share** a trained persona two ways: a live link (`/p/<name>`, works for
   anyone who can reach this Mac while personabot is running — no
   tunneling/public hosting is set up automatically) or a downloadable
   adapter `.zip` (portable to anyone else running personabot).
6. **Cloud training (Colab + Unsloth)**: export a self-contained notebook
   (no chat data baked in — you upload `train.jsonl`/`valid.jsonl` into your
   own Colab session) for a free-GPU training run. It merges the LoRA into
   the base model before you download it; `backend/import_colab.py` converts
   that merged model to MLX locally (`mlx_lm.convert`) and registers it as a
   normal, ready-to-chat persona — no adapter-format mismatch to resolve.

## Notes on scale

LoRA fine-tuning a pretrained instruct model is a different regime from a
from-scratch transformer — the base model already knows English; the adapter
only has to steer style and content, so a few-thousand-message corpus goes a
long way. Still, a few hundred examples will sound thinner than a few
thousand. The upload step reports how many training examples it produced —
treat under ~500 as "expect a rough sketch of their style," not a reliable
conversationalist.
