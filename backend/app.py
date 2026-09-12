"""
Personabot backend: upload a chat export, prepare training data, fine-tune a
persona locally (MLX LoRA) or export a Colab notebook for cloud training, and
chat with the result once an adapter exists.

Run: uvicorn app:app --reload --port 8000   (from the backend/ directory)
"""
import json
import re
import shutil
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import prepare_data
import train_local
import colab_export
import import_colab
import chat_infer

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
PROCESSED_DIR = ROOT / "data" / "processed"
ADAPTER_DIR = ROOT / "models" / "adapters"
for d in (UPLOAD_DIR, PROCESSED_DIR, ADAPTER_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="personabot")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def no_cache_frontend(request, call_next):
    """This is an actively-edited local app, not a CDN-fronted site — a
    browser silently serving a stale style.css/app.js after an edit (which
    happened mid-session and looked exactly like a real UI bug) is a worse
    default than always refetching."""
    response = await call_next(request)
    if request.url.path in ("/", "/style.css", "/app.js") or request.url.path.startswith("/p/"):
        response.headers["Cache-Control"] = "no-store"
    return response

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

# In-memory training job state: {name: {"state": ..., "log": [...], "error": ...}}
_jobs = {}
_jobs_lock = threading.Lock()


def _safe_name(name):
    if not NAME_RE.match(name):
        raise HTTPException(400, "Persona name must be alphanumeric/-/_ , max 40 chars.")
    return name


def _persona_status(name):
    data_dir = PROCESSED_DIR / name
    adapter_dir = ADAPTER_DIR / name
    job = _jobs.get(name, {})

    if job.get("state") in ("training", "importing"):
        return {"name": name, "state": job["state"], "log_tail": job["log"][-20:]}
    if job.get("state") == "error":
        return {"name": name, "state": "error", "error": job.get("error")}

    if (adapter_dir / "persona_meta.json").exists():
        meta = json.loads((adapter_dir / "persona_meta.json").read_text())
        return {"name": name, "state": "ready", **meta}

    if (data_dir / "meta.json").exists():
        meta = json.loads((data_dir / "meta.json").read_text())
        return {"name": name, "state": "uploaded", **meta}

    return {"name": name, "state": "unknown"}


@app.get("/api/personas")
def list_personas():
    names = set()
    for d in list(PROCESSED_DIR.iterdir()) if PROCESSED_DIR.exists() else []:
        if d.is_dir():
            names.add(d.name)
    for d in list(ADAPTER_DIR.iterdir()) if ADAPTER_DIR.exists() else []:
        if d.is_dir():
            names.add(d.name)
    names |= set(_jobs.keys())
    return [_persona_status(n) for n in sorted(names)]


@app.post("/api/personas")
async def upload_persona(name: str = Form(...), file: UploadFile = None,
                          me_label: str = Form("Me"), context_turns: int = Form(4)):
    name = _safe_name(name)
    if file is None:
        raise HTTPException(400, "Missing file upload.")

    raw_path = UPLOAD_DIR / f"{name}.txt"
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    data_dir = PROCESSED_DIR / name
    from parser import parse_file
    try:
        turns, persona = parse_file(str(raw_path), me_label=me_label)
    except ValueError as e:
        raise HTTPException(400, str(e))

    short_phrases, long_lines = prepare_data.mine_signature_phrases(turns, persona)
    system_content = prepare_data.build_system_prompt(persona, short_phrases, long_lines)

    tokenizer = prepare_data._get_tokenizer(prepare_data.DEFAULT_TOKENIZER_MODEL)
    train_ex, val_ex = prepare_data.shard_split(
        turns, persona, me_label, context_turns, 0.1, 0, tokenizer=tokenizer,
        system_content=system_content,
    )
    if len(train_ex) == 0:
        raise HTTPException(
            400,
            f"Parsed {len(turns)} turns but couldn't build any training examples "
            f"from them (need at least a few dozen back-and-forth turns). "
            f"Upload a longer chat export.",
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname, ex in [("train.jsonl", train_ex), ("valid.jsonl", val_ex)]:
        with open(data_dir / fname, "w", encoding="utf-8") as f:
            for row in ex:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "persona": persona, "me_label": me_label, "n_turns": len(turns),
        "n_train": len(train_ex), "n_val": len(val_ex), "context_turns": context_turns,
        "system_prompt": system_content,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return {"name": name, **meta}


class TrainRequest(BaseModel):
    backend: str = "local"        # "local" or "colab"
    model_key: str = "qwen-7b"    # one of train_local.MODELS
    learning_rate: float = 1e-5
    rank: int = 16           # LoRA rank — mlx_lm's own default (8) is low headroom
    num_layers: int | None = None  # None -> train_local picks per model tier


def _start_job(name, state_label, fn):
    """Runs fn() on a background thread, capturing its stdout into a
    per-persona log the UI polls, and recording done/error — shared by
    local training and Colab-model import so both get the same status/log
    UX for free."""
    with _jobs_lock:
        if _jobs.get(name, {}).get("state") in ("training", "importing"):
            raise HTTPException(409, f"Already {_jobs[name]['state']}.")
        _jobs[name] = {"state": state_label, "log": []}

    def _run():
        try:
            log = _jobs[name]["log"]

            import io
            import contextlib

            class LogWriter(io.StringIO):
                def write(self_inner, s):
                    if s.strip():
                        log.append(s.strip())
                        if len(log) > 500:
                            del log[:250]
                    return len(s)

            with contextlib.redirect_stdout(LogWriter()):
                fn()
            with _jobs_lock:
                _jobs[name]["state"] = "done"
        except Exception as e:
            with _jobs_lock:
                _jobs[name] = {"state": "error", "error": f"{e}\n{traceback.format_exc()}"}

    threading.Thread(target=_run, daemon=True).start()


@app.post("/api/personas/{name}/train")
def train_persona(name: str, req: TrainRequest):
    name = _safe_name(name)
    data_dir = PROCESSED_DIR / name
    if not (data_dir / "meta.json").exists():
        raise HTTPException(404, "Upload this persona's data first.")

    if req.backend == "colab":
        persona_meta = json.loads((data_dir / "meta.json").read_text())
        nb = colab_export.build_notebook(persona_meta["persona"])
        nb_path = data_dir / f"{name}_colab.ipynb"
        nb_path.write_text(json.dumps(nb, indent=1))
        return {
            "backend": "colab",
            "notebook_url": f"/api/personas/{name}/colab_notebook",
            "train_jsonl_url": f"/api/personas/{name}/colab_train_jsonl",
            "valid_jsonl_url": f"/api/personas/{name}/colab_valid_jsonl",
        }

    meta = json.loads((data_dir / "meta.json").read_text())
    adapter_dir = ADAPTER_DIR / name
    _start_job(name, "training", lambda: train_local.run_training(
        meta["persona"], data_dir, adapter_dir,
        model_key=req.model_key, learning_rate=req.learning_rate,
        rank=req.rank, num_layers=req.num_layers,
    ))
    return {"backend": "local", "state": "training"}


@app.post("/api/personas/{name}/import_colab")
async def import_colab_model(name: str, file: UploadFile):
    """Brings back a merged model trained in the exported Colab notebook
    (see colab_export.py step 5) — converts it to MLX format locally and
    registers it as a ready-to-chat persona."""
    name = _safe_name(name)
    data_dir = PROCESSED_DIR / name
    if not (data_dir / "meta.json").exists():
        raise HTTPException(404, "Upload this persona's data first (same name as in Colab).")

    zip_path = UPLOAD_DIR / f"{name}_merged_model.zip"
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    meta = json.loads((data_dir / "meta.json").read_text())
    adapter_dir = ADAPTER_DIR / name
    _start_job(name, "importing", lambda: import_colab.import_merged_model(
        name, meta["persona"], zip_path, adapter_dir,
        system_prompt=meta.get("system_prompt"),
    ))
    return {"state": "importing"}


@app.get("/api/personas/{name}/status")
def persona_status(name: str):
    return _persona_status(_safe_name(name))


@app.get("/api/personas/{name}/colab_notebook")
def download_notebook(name: str):
    name = _safe_name(name)
    path = PROCESSED_DIR / name / f"{name}_colab.ipynb"
    if not path.exists():
        raise HTTPException(404, "No notebook generated yet — POST /train with backend=colab first.")
    return FileResponse(path, filename=path.name, media_type="application/x-ipynb+json")


@app.get("/api/personas/{name}/colab_train_jsonl")
def download_colab_train_jsonl(name: str):
    """Served with the exact filename `train.jsonl` — the notebook's upload
    cell checks for that literal name."""
    name = _safe_name(name)
    path = PROCESSED_DIR / name / "train.jsonl"
    if not path.exists():
        raise HTTPException(404, "Upload this persona's data first.")
    return FileResponse(path, filename="train.jsonl", media_type="application/jsonl")


@app.get("/api/personas/{name}/colab_valid_jsonl")
def download_colab_valid_jsonl(name: str):
    name = _safe_name(name)
    path = PROCESSED_DIR / name / "valid.jsonl"
    if not path.exists():
        raise HTTPException(404, "Upload this persona's data first.")
    return FileResponse(path, filename="valid.jsonl", media_type="application/jsonl")


class ChatRequest(BaseModel):
    history: list  # [{"role": "user"|"assistant", "content": str}, ...]
    # Optional per-request overrides; unset fields fall back to this
    # persona's settings.json (see /settings endpoints below), then to
    # chat_infer.DEFAULT_SETTINGS.
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None


@app.post("/api/personas/{name}/chat")
def chat(name: str, req: ChatRequest):
    name = _safe_name(name)
    adapter_dir = ADAPTER_DIR / name
    meta_path = adapter_dir / "persona_meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "This persona hasn't finished training yet.")
    meta = json.loads(meta_path.read_text())

    try:
        text = chat_infer.reply(
            adapter_dir, meta["persona"], req.history,
            temperature=req.temperature, top_p=req.top_p,
            max_tokens=req.max_tokens, repetition_penalty=req.repetition_penalty,
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")
    return {"reply": text}


class SettingsRequest(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None


@app.get("/api/personas/{name}/settings")
def get_settings(name: str):
    name = _safe_name(name)
    return chat_infer.load_settings(ADAPTER_DIR / name)


@app.put("/api/personas/{name}/settings")
def put_settings(name: str, req: SettingsRequest):
    name = _safe_name(name)
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    return chat_infer.save_settings(ADAPTER_DIR / name, updates)


@app.delete("/api/personas/{name}")
def delete_persona(name: str):
    name = _safe_name(name)
    removed = []
    for path in (UPLOAD_DIR / f"{name}.txt", PROCESSED_DIR / name, ADAPTER_DIR / name):
        if path.exists():
            (path.unlink() if path.is_file() else shutil.rmtree(path))
            removed.append(str(path))
    with _jobs_lock:
        _jobs.pop(name, None)
    if _cache_matches(name):
        chat_infer._cache.update(persona=None, model=None, tokenizer=None, meta=None)
    return {"deleted": name, "removed_paths": removed}


def _cache_matches(name):
    p = chat_infer._cache.get("persona")
    return bool(p) and Path(p).name == name


@app.get("/api/personas/{name}/download")
def download_adapter(name: str):
    """Zips this persona's trained adapter (+ config/meta/settings) so it can
    be backed up, or handed to someone else running personabot themselves —
    the file-based complement to the live /p/<name> share link.

    Zips only what's actually needed to chat, not the whole adapter_dir:
    mlx_lm saves a full checkpoint at every eval (tens-hundreds of MB each,
    scaling with rank/num_layers), and run_training() normally cleans those
    up, but zipping selectively here is cheap insurance against stray
    checkpoints ballooning the download to hundreds of MB for no reason.
    """
    name = _safe_name(name)
    adapter_dir = ADAPTER_DIR / name
    if not (adapter_dir / "persona_meta.json").exists():
        raise HTTPException(404, "This persona hasn't finished training yet.")

    import tempfile
    import zipfile

    meta = json.loads((adapter_dir / "persona_meta.json").read_text())
    keep_names = ["persona_meta.json", "settings.json"]
    keep_names += ["mlx_model"] if meta.get("merged") else ["adapters.safetensors", "adapter_config.json"]

    zip_path = Path(tempfile.gettempdir()) / f"{name}_adapter.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item_name in keep_names:
            item = adapter_dir / item_name
            if not item.exists():
                continue
            if item.is_dir():
                for f in item.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(adapter_dir))
            else:
                zf.write(item, item.relative_to(adapter_dir))

    return FileResponse(zip_path, filename=f"{name}_adapter.zip", media_type="application/zip")


@app.get("/p/{name}")
def shared_chat_page(name: str):
    """A persona's shareable chat link — serves the same SPA; app.js detects
    the /p/<name> path and renders a simplified chat-only view. Only reaches
    whoever can hit this Mac over the network while personabot is running —
    no tunneling/public exposure is set up here, that's a deliberate choice
    left to you."""
    name = _safe_name(name)
    return FileResponse(ROOT / "frontend" / "index.html")


# Serve the frontend as static files at /
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
