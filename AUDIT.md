# personabot audit — efficiency & natural speech

Branch: `audit/natural-speech` · base commit `5bec515` · audited 2026-09-12

Scope: the path from chat export → training data → LoRA → generation, looking
for (a) wasted compute/latency and (b) anything that makes the persona sound
less like a real person texting. Claims about mlx-lm behavior were checked
against `ml-explore/mlx-lm@main`. **Correction:** the unpinned
`requirements.txt` actually installs the latest *release* (0.31.3), not
main, and the two differ in how much prompt a logits processor sees — see
§1.3. Claims about the export format were checked against
imessage-exporter's current `txt` templates. Nothing was trained or run on a
GPU; the parser findings were reproduced with a synthetic export (see §1.1).

## Status: all findings implemented on this branch

| Finding | Where it's fixed |
|---|---|
| 1.1 exporter boilerplate, threaded replies | `backend/parser.py` (`clean_body`, `_parse_block`) + `tests/test_parser.py` |
| 1.2 bursts flattened | `parser.build_sessions` joins with `\n`; `frontend/app.js` renders one bubble per line |
| 1.3 repetition penalty / sampler / length | `generation.py`: penalty on generated tokens only (1.05), `min_p` 0.05, `max_tokens` from p95 reply length, clean trim on length stops; legacy settings migrated |
| 1.4 time gaps, history mismatch | sessions split at 4 h gaps; `context_turns` 8 at 1024 tokens; chat uses the trained window (`generation.sticky_window`) |
| 1.5 phone-number identity, dropped handles | display name field; sender picker (`POST /api/personas/{name}/prepare`); filter before merge |
| 1.6 system prompt | minimal identity prompt by default; optional retrieved exemplars (`retrieval.py`, own/val sessions excluded at train time) |
| 1.7 noisy checkpoint selection | `lora_train.py`: whole validation set (capped at 250 rows), ~10 evals, patience 3, warmup + cosine, ~1 epoch |
| 1.8 no style eval | `eval_style.py`; used to choose among near-best checkpoints; shown on persona cards |
| 1.9 Colab objective / export | notebook masks exactly `train_turns`, 1 epoch + early stopping, selected model, adapter-only export converted by `import_colab.py` |
| 1.10 base vs instruct | `qwen-3b-base` / `qwen-7b-base` options |
| 2.1–2.4 training/data prep | chunked multi-reply rows; per-turn token counts; re-sized with the selected model's tokenizer at train time |
| 2.5–2.8 server/jobs | sync upload handler; shared `prepare()`; per-job log callbacks; base-model-best warning |
| 2.9–2.15 inference/export | prompt-cache reuse; lock; model evicted during training (chat paused); `ZIP_STORED` + temp cleanup; adapter-only Colab export |
| §3 | CORS removed + localhost-only API with share-route allowlist and Origin check; `--host` docs/launcher option; regenerate fixed; no `innerHTML` with data; `trl` pin gone; docstring fixed; PII filter on exemplars |

Verification: `cd backend && python3 -m unittest discover tests` (25 tests,
no MLX needed), plus an end-to-end smoke test on Apple Silicon with
mlx-lm 0.31.3 and `mlx-community/Qwen2.5-0.5B-Instruct-4bit` on a synthetic
export:
- Both prompt styles trained. Validation loss went 5.16 → 0.56 (minimal) and
  3.43 → 0.56 (retrieval). Style selection picked a checkpoint other than
  the lowest-loss one when it scored closer on style. A too-high learning
  rate triggered early stopping and the base-model warning.
- The minimal style covered the same 257 replies with 54 chunked rows in 39 s,
  against 257 single-reply rows in 118 s.
- Multi-turn chat reused the KV cache for the shared prompt prefix. On
  mlx-lm 0.31.3 the repetition-penalty window matched exactly the generated
  tokens.
- A 32-check live API run covered the sender picker, handle merge, legacy
  settings migration, chat/share limits, and colab export. It also confirmed
  a failed import keeps the old adapter, and that access control blocks a
  rebound Host, cross-origin writes and share visitors, with no CORS headers.
  31 passed; the one failure was a wrong expectation in the test itself.
- A PEFT-format adapter imported and matched the PEFT LoRA delta to 4e-7.
- UI checked in a browser: sender picker, multi-bubble replies, regenerate,
  and the share-link view.

Not verified: the Colab notebook on Colab, real 3B/7B/8B memory use and
timing, Llama chat templates inside the trainer, and real (non-synthetic)
iMessage exports.

File:line references in the findings below point at the base commit `5bec515`.

---

## 0. Architecture overview

```
 .txt export
     │
     ▼
 backend/parser.py ── auto-detects "Name: msg" vs imessage-exporter txt,
     │                 drops attachments/tapbacks, merges consecutive
     │                 same-sender messages, keeps Me + top other sender
     ▼
 backend/prepare_data.py ── mines "signature phrases" into a fixed system
     │                       prompt; builds one example per persona reply
     │                       (≤4 turns history, trimmed to 488 Qwen tokens);
     │                       40-shard random train/val split
     ▼
 data/processed/<name>/{train,valid}.jsonl + meta.json
     │
     ├──► backend/train_local.py ── subprocess `mlx_lm lora` (4-bit base,
     │        rank 16, top 16/24 layers, --mask-prompt), parses stdout for
     │        val loss, early-stops, promotes best checkpoint
     │        → models/adapters/<name>/adapters.safetensors + persona_meta.json
     │
     └──► backend/colab_export.py ── Unsloth notebook (always Qwen2.5-7B,
              r=32, 3 epochs) → merged fp16 zip → backend/import_colab.py
              (`mlx_lm convert -q`) → models/adapters/<name>/mlx_model
     │
     ▼
 backend/chat_infer.py ── mlx_lm.load(base, adapter) cached for one persona;
     │                     chat template + stored system prompt + full
     │                     history; top-p sampler + custom repetition penalty
     ▼
 backend/app.py (FastAPI) ── upload / train / import / status / chat /
     │                        settings / zip download / share page; in-memory
     │                        job registry on background threads
     ▼
 frontend/ (vanilla JS SPA, polls every 4s; /p/<name> = chat-only view)
 macapp/  (bash-built .app that starts uvicorn and opens the browser)
```

---

## 1. Natural speech — ranked by expected impact

### 1.1 The iMessage parser folds non-speech text into the persona's replies (high)

`parse_imessage_export` joins every non-blank line after the sender line into
one message (`backend/parser.py:72-82`). The current exporter emits several
things inside that block that are not the person's words:

| Exporter output | What ends up in training text |
|---|---|
| Threaded (swipe-to-reply) messages, rendered **indented inside the parent's body** with their own timestamp + sender | `yeah May 17, 2022  5:31:00 PM Me what time` — the other person's reply, attributed to the persona |
| `This message responded to an earlier message.` | appended verbatim |
| `Edited 30 seconds later: …` | prefix kept |
| `Sent with Confetti` / other effects | appended |
| URL previews (URL, page title, page summary lines) | web-page title + description appended |
| `This message was deleted from the conversation!`, `… unsent this message part!`, `Attachment missing!` | appended |

Reproduced with a synthetic export; one persona turn came out as:

```
'yeah May 17, 2022  5:31:00 PM Me what time This message responded to an
earlier message. Edited 30 seconds later: like 9ish Sent with Confetti
https://youtube.com/watch?v=x Never Gonna Give You Up The official video for Rick Astley'
```

The model learns to emit timestamps, boilerplate, and link-preview copy, and
long contaminated rows skew its sense of reply length.

**Fix:** stop the body at lines starting with whitespace (replies) and
process them as their own turns; strip or skip known exporter boilerplate
(`This message responded…`, `Edited … later:` → keep only the final edit,
`Sent with …`, deleted/unsent/missing notices); keep the URL line and drop
the preview title/summary lines. Add a regression test using the
exporter's templates.

### 1.2 Message bursts are flattened into run-on sentences (high)

`merge_consecutive` joins back-to-back messages with a space
(`backend/parser.py:95`), and multi-line bodies are also space-joined
(`:82`). Real texting cadence ("wait" / "no way" / "who told u") becomes
one run-on line, so the model can't learn double-texting, and every reply
renders as a single bubble.

**Fix:** join with `"\n"` for both. In the frontend, split the reply on
`\n` and render one bubble per line with a short typing delay. This is
probably the single cheapest "feels like a person" change.

### 1.3 The repetition penalty suppresses short replies and echoing (high)

mlx-lm passes **fed prompt tokens plus generated tokens** to logits
processors, and the custom processor penalizes the last 24 of those
(`backend/chat_infer.py:71-79`). How much prompt is included depends on the
version: in mlx-lm ≤0.27 and on current main (the next release), the whole
prefilled prompt; in 0.28–0.31.3 (the release pip installs today), only the
final prompt token — the `\n` after the assistant header, which still
penalizes starting a line break. Where the full prompt is included, the
first ~19 generated tokens see a window containing:

- the end-of-turn token (`<|im_end|>` for Qwen, `<|eot_id|>` for Llama),
  which is also the stop token. At 1.15 a positive stop logit is divided by
  1.15, so "lol" / "ok" / "bet" become much less likely to end right there.
- the user's last message, so natural echoes ("you coming?" → "yeah
  coming") are penalized.

Combined with `max_tokens: 48` (`:27`), replies drift long and get cut off
mid-sentence (`generate` gives no finish reason, and nothing trims to a
sentence boundary).

**Fix:** penalize only generated tokens and never special tokens; start
around 1.05, or switch to a frequency/presence penalty. Add `min_p`
(≈0.05) to `make_sampler`, and set `max_tokens` from the persona's p95
reply length (in tokens) instead of a fixed 48. Use `stream_generate` so a
`length` finish can be trimmed cleanly.

### 1.4 Context windows ignore time gaps, and inference history doesn't match training (medium-high)

Timestamps are thrown away during parsing, so a 4-turn window can cover
"see u tmrw" (Tuesday) → "happy birthday!!" (Friday) and train the model
that this is a normal immediate reply. At inference the frontend sends the
**entire** session history (`backend/chat_infer.py:106`), but the adapter
was trained on ≤4 turns (often fewer after trimming). Long chats go
out-of-distribution and prefill keeps getting slower.

**Fix:** keep timestamps and split into sessions at gaps (e.g. >4h). Build
windows inside sessions only, and let session openers train with no
history (currently skipped, `backend/prepare_data.py:156-157`). Raise
`context_turns` to ~8–10 with `max_seq_length` 1024, and cap inference
history to the same window.

### 1.5 The persona's identity is a phone number, and extra handles are dropped (medium)

For iMessage exports the sender label is the handle, so the system prompt
reads *"You are +15551234567, texting a close friend"*. The name the user
typed in the form isn't used. `restrict_to_two_speakers` also keeps only the
single most frequent non-Me handle (`backend/parser.py:101-107`), so the
README's advice to `cat` the phone and email exports together drops the
smaller handle's messages. Because merging runs before the restriction
(`:120-121`), it also leaves back-to-back `Me` turns.

**Fix:** pass the display name into `build_system_prompt`. Return the
sender counts from upload and let the user pick which handles are the
persona (default: all non-Me senders in a 1:1 export), then merge after
restricting.

### 1.6 The system prompt pushes toward "assistant doing an impression" (medium)

`SYSTEM_TEMPLATE` (`backend/prepare_data.py:28-33`) is an instruction to an
assistant ("Reply exactly the way X actually texts… Never mention being an
AI"). On an instruct model that invites caricature: forced slang and
overused catchphrases. The "example longer messages" are simply the **three
longest** 6–18-word messages (`:77-81`), not distinctive ones, and they are
identical in every example. An adapter stopped at ~0.1 epoch (see 1.7)
still carries the instruct model's copy-from-context habit, so those lines
get parroted. The prompt also takes an estimated ~150–200 of the 488-token
budget in every example (see 2.1).

**Fix (A/B these, measured with 1.8's metrics):**
- Minimal identity prompt, e.g. `"{name} texting {me}."` Style should come
  from the adapter weights.
- If exemplars stay, **retrieve** 3–5 of the persona's real replies to
  messages similar to the current one (BM25 or small embeddings), per
  example at train time (excluding the target's shard) and per turn at
  inference. The model then learns to use exemplars as style cues instead
  of memorizing one fixed list. This likely has the most upside of any
  prompt-side change.

### 1.7 Checkpoint selection is driven by noise (medium-high)

`--val-batches 5` (`backend/train_local.py:110`) means each eval scores
**10 examples** (batch 2), or **5** on 7B/8B (batch 1). mlx-lm's
`iterate_batches` draws a fresh random permutation of length-bucketed
batches on every eval, so each "val loss" is measured on a different tiny
subset. Best-checkpoint promotion (`:145-149`) and 4-eval patience
early-stopping (`:150-158`) are therefore picking between random draws,
and a lucky early draw can end training long before the voice has moved.
The README's "overfits within 0.1 epoch" finding may be partly this noise.

Val loss is also a weak proxy for style: content NLL bottoms out early
while length, casing, and slang distributions keep improving.

**Fix:** use `--val-batches -1` (whole val set) and eval less often (e.g.
every ~10% of an epoch). Add a cosine schedule with warmup through the YAML
`lr_schedule`, budget ~1 epoch, and choose the checkpoint with val loss
**plus** 1.8's style metrics.

### 1.8 No style evaluation exists (medium, enables everything above)

Add `backend/eval_style.py`: generate replies for ~100 val contexts and
compare them with the persona's real replies on:
- reply length distribution (chars/words, KL or quantile gaps)
- lowercase-start rate, trailing-punctuation rate, emoji rate, `\n`-burst rate
- distinct-1/2
- verbatim copy rate from the system prompt
- assistant-isms ("Sure!", "I'm here", "That sounds", "As an AI")

Save the scores in `persona_meta.json` next to `best_val` and show them on
the persona card.

### 1.9 The Colab path trains a different, heavier-overfit model (medium)

- `train_on_responses_only` unmasks **every** assistant turn in a window
  (`backend/colab_export.py:142-146`). Persona messages in the history are
  trained again in later windows, roughly 2–3× duplication. On top of that:
  3 epochs at lr 2e-4 (`:124-125`), when local runs found overfitting
  within a fraction of one epoch.
- It always trains Qwen2.5-7B, whatever model was picked in Settings.
- Merging into fp16 and then running `mlx_lm convert -q` (4-bit)
  (`backend/import_colab.py:54-60`) quantizes the small LoRA delta together
  with the base weights, which can blur the learned style. The local path
  keeps the adapter at full precision on the quantized base.
- The Colab chat cell uses a different system prompt from the trained one
  (`:182-184`).

**Fix:** train only on the last assistant turn (or build non-overlapping
session examples), use 1 epoch plus `EarlyStoppingCallback`, and use the
model the user selected. Download the **PEFT adapter** (~100 MB) instead of
a ~15 GB merged zip, then convert it to mlx-lm adapter format
(`lora_a = A.T`, `lora_b = B.T`, `scale = alpha / r`) and apply it to the
same `mlx-community` 4-bit base. If merging stays, quantize to 8-bit.

### 1.10 Worth an A/B: base model choice (low-medium)

Instruct/RLHF models carry an assistant register: tidy capitalization,
exclamation marks, validation phrases. Try a **base** checkpoint with the
same chat template (the LoRA learns the template tokens quickly on a few
thousand examples) and newer small families, judged with 1.8's metrics.

---

## 2. Efficiency

### Training / data prep
| # | Issue | Where | Fix |
|---|---|---|---|
| 2.1 | Constant ~150–200-token system prompt (estimate) in every example, about a third of the 488-token budget. Forward/backward runs through it every step and it pushes history out | `prepare_data.py:90-99`, `:109-110` | Minimal prompt (1.6), or cache-and-mask a fixed prefix |
| 2.2 | Overlapping windows: each message is encoded ~3× but supervised once | `prepare_data.py:151-181` | Session-level examples. mlx-lm's `--mask-prompt` only unmasks the last turn, so train all persona turns with a small custom `ChatDataset` subclass returning per-turn masks |
| 2.3 | `_token_len` re-renders the Jinja chat template and re-tokenizes the whole window on every trim iteration (up to ~6× per example) | `prepare_data.py:122-130`, `:163-174` | Tokenize each turn once, add fixed per-message template overhead, trim arithmetically |
| 2.4 | Trimming always uses the **Qwen** tokenizer. Llama 3.x templates add a "Cutting Knowledge Date / Today Date" header (~25 tokens) and tokenize differently, so examples can exceed 512 once the 24-token margin runs out. With llama-8b's batch size 1 that is the NaN failure described in the code comments | `prepare_data.py:207`, `app.py:120` | Re-trim at train time with the selected model's tokenizer |
| 2.5 | Upload handler is `async def` but does CPU-heavy parsing and tokenization on the event loop, which freezes the whole server (status polls, chat) until it finishes | `app.py:99-144` | Make it `def` (runs in the threadpool) or use `run_in_threadpool` |
| 2.6 | Upload-prep logic is duplicated between `app.py` and `prepare_data.main` | `app.py:117-142`, `prepare_data.py:224-252` | One `prepare(...)` function |
| 2.7 | `contextlib.redirect_stdout` is process-global, so two jobs (or uvicorn's own output) mix into each other's logs | `app.py:180` | Pass a log callback / read the subprocess pipe per job |
| 2.8 | If the iter-1 (base model) val loss is best, no checkpoint `0000001` exists, and chat silently uses the last saved adapter | `train_local.py:167-171` | Handle explicitly (warn, or treat as "training didn't help") |

### Inference
| # | Issue | Where | Fix |
|---|---|---|---|
| 2.9 | Every turn re-prefills system prompt + full history from scratch | `chat_infer.py:106-121` | Keep an mlx-lm `prompt_cache` per chat session (or at least a cached system-prompt prefix) and prefill only the new turn |
| 2.10 | History is unbounded, so latency grows every turn (and see 1.4) | `chat_infer.py:106` | Cap to training window |
| 2.11 | Repetition processor calls `.tolist()` (GPU→CPU sync) every token | `chat_infer.py:75` | Vectorize, or use `mlx_lm.sample_utils.make_logits_processors` |
| 2.12 | No lock around the model cache / `generate`. Concurrent share-link chats with different personas can swap weights mid-request or briefly hold two 7B models (16 GB OOM) | `chat_infer.py:96-98`, `app.py:288` | `threading.Lock` around load+generate; queue requests |
| 2.13 | Chat stays loaded while local training runs, so two large models share unified memory | `app.py:212` | Evict `_cache` when a training job starts |
| 2.14 | Adapter download zips `mlx_model` (GBs of already-compressed safetensors) with `ZIP_DEFLATED`, blocking the request | `app.py:373` | `ZIP_STORED`, stream it |
| 2.15 | Colab round-trip downloads a ~15 GB merged model through `files.download` | `colab_export.py:167-172` | Adapter-only export (1.9) |

---

## 3. Other issues spotted along the way

- **Private texts readable by any website (security).** `CORSMiddleware(allow_origins=["*"])` with no auth (`backend/app.py:36-38`) lets any page open in the user's browser `fetch('http://localhost:8000/api/personas')`, then read `/api/personas/<name>/colab_train_jsonl` (the raw message corpus) or `DELETE` personas. The SPA is same-origin and doesn't need CORS. Remove it and check `Origin`/`Host`.
- **Share links only work on localhost.** Both the README and `macapp/build.sh` run uvicorn without `--host`, so it binds 127.0.0.1 and `/p/<name>` is unreachable from other devices. Binding `0.0.0.0` would expose the whole unauthenticated API, not just chat. Share mode needs its own restricted route set.
- **Regenerate bugs** (`frontend/app.js:527-534`): it re-adds a duplicate user bubble. After a failed reply (no assistant message was pushed), it pops the *user* message and re-sends the previous bot reply as a user message.
- **HTML injection:** persona labels from the uploaded file go into `innerHTML` (`frontend/app.js:291`).
- **Colab install pin:** `trl==0.9.6` (`backend/colab_export.py:55`) is very likely incompatible with current Unsloth. Unverified, since there was no Colab run.
- **Stale docstring:** mentions a `/colab_bundle` endpoint that doesn't exist (`backend/colab_export.py:15`).
- **PII in the system prompt:** the "example longer messages" filter only catches URLs and attachment markers, not addresses or phone numbers, despite the docstring.

---

## 4. Suggested order of work

1. Parser fixes + `\n` bursts + multi-bubble rendering (1.1, 1.2, 1.5)
2. Repetition penalty / sampler / max_tokens (1.3)
3. Remove CORS (§3)
4. Style eval script + full-val checkpoint selection (1.8, 1.7)
5. Session-aware windows + matched inference window + prompt cache (1.4, 2.9, 2.10)
6. Minimal vs. retrieved-exemplar system prompt A/B (1.6, 2.1)
7. Colab objective + adapter-only export (1.9)
