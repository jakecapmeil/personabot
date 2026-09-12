"""
Personabot backend: upload a chat export, prepare training data, fine-tune a
persona locally (MLX LoRA) or export a Colab notebook for cloud training, and
chat with the result.

Run: uvicorn app:app --port 8000                  (this Mac only)
     uvicorn app:app --host 0.0.0.0 --port 8000   (share links on your network)

Access control: the full API only answers requests made to localhost from
this machine. Anything else — another device, or a web page trying to reach
the API through a rebound hostname — only gets share-link routes: the chat
page and chat for personas that are ready. Browser-sent writes from another
origin are rejected, and there is no CORS, so other sites can't read
responses (your uploaded messages included).
"""
import json
import re
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

import chat_infer
import colab_export
import import_colab
import prepare_data
import train_local
from parser import looks_like_handle, parse_file

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
PROCESSED_DIR = ROOT / "data" / "processed"
ADAPTER_DIR = ROOT / "models" / "adapters"
FRONTEND_DIR = ROOT / "frontend"
for d in (UPLOAD_DIR, PROCESSED_DIR, ADAPTER_DIR):
    d.mkdir(parents=True, exist_ok=True)

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
LOOPBACK_CLIENTS = {"127.0.0.1", "::1"}
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1"}
SHARE_PATH_RE = re.compile(r"^/(?:p/[A-Za-z0-9_-]{1,40}|style\.css|app\.js|api/share/[A-Za-z0-9_-]{1,40}(?:/chat)?)$")
SAFE_METHODS = {"GET", "HEAD"}
MAX_HISTORY_MESSAGES = 60
MAX_MESSAGE_CHARS = 2000

app = FastAPI(title="personabot")


def _hostname(host_header):
    return (urlsplit("//" + host_header).hostname or "").lower() if host_header else ""


@app.middleware("http")
async def access_control(request: Request, call_next):
    client = request.client.host if request.client else ""
    host = request.headers.get("host", "")
    is_local = client in LOOPBACK_CLIENTS and _hostname(host) in LOCAL_HOSTNAMES
    if not is_local and not SHARE_PATH_RE.match(request.url.path):
        return PlainTextResponse("Only share links are available from other devices.", status_code=403)
    if request.method not in SAFE_METHODS:
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != host:
            return PlainTextResponse("Cross-origin request blocked.", status_code=403)

    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/style.css", "/app.js") or path.startswith("/p/") or path.startswith("/api/"):
        # A stale app.js/style.css after an edit looks exactly like a UI bug.
        response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------- jobs

_jobs = {}  # name -> {"state": "training"|"importing"|"done"|"error", "log": [...], "error": str}
_jobs_lock = threading.Lock()
MAX_LOG_LINES = 400


def _safe_name(name):
    if not NAME_RE.match(name):
        raise HTTPException(400, "Persona name must be letters, numbers, - or _, max 40 characters.")
    return name


def _start_job(name, state_label, fn):
    """Run fn(log) on a background thread. Each job writes to its own log
    through the callback, so concurrent jobs never mix output."""
    with _jobs_lock:
        if _jobs.get(name, {}).get("state") in ("training", "importing"):
            raise HTTPException(409, f"Already {_jobs[name]['state']}.")
        job = {"state": state_label, "log": []}
        _jobs[name] = job

    def log(message):
        with _jobs_lock:
            for line in str(message).splitlines():
                if line.strip():
                    job["log"].append(line)
            del job["log"][:-MAX_LOG_LINES]

    def run():
        try:
            fn(log)
            with _jobs_lock:
                job["state"] = "done"
        except Exception as e:
            with _jobs_lock:
                job["state"] = "error"
                job["error"] = f"{e}\n{traceback.format_exc()}"

    threading.Thread(target=run, daemon=True).start()


def _local_training_active():
    return any(j.get("state") == "training" for j in _jobs.values())


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def _persona_status(name):
    job = _jobs.get(name, {})
    if job.get("state") in ("training", "importing"):
        return {"name": name, "state": job["state"], "log_tail": job["log"][-20:]}
    if job.get("state") == "error":
        return {"name": name, "state": "error", "error": job.get("error"), "log_tail": job["log"][-20:]}

    persona_meta = _read_json(ADAPTER_DIR / name / "persona_meta.json")
    if persona_meta:
        return {"name": name, "state": "ready", **persona_meta}
    data_meta = _read_json(PROCESSED_DIR / name / "meta.json")
    if data_meta:
        return {"name": name, "state": "uploaded", **data_meta}
    if (UPLOAD_DIR / f"{name}.txt").exists():
        return {"name": name, "state": "needs_senders"}
    return {"name": name, "state": "unknown"}


# ---------------------------------------------------------------- personas

@app.get("/api/personas")
def list_personas():
    names = {d.name for base in (PROCESSED_DIR, ADAPTER_DIR) for d in base.iterdir()
             if d.is_dir() and NAME_RE.match(d.name)}
    names |= set(_jobs)
    return [_persona_status(n) for n in sorted(names)]


def _prepare(name, me_label, persona_senders, display_name, prompt_style):
    try:
        meta = prepare_data.prepare(
            UPLOAD_DIR / f"{name}.txt", PROCESSED_DIR / name, name=name, me_label=me_label,
            persona_senders=persona_senders, display_name=display_name, prompt_style=prompt_style,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    if meta["n_train"] == 0:
        raise HTTPException(
            400,
            f"Parsed {meta['n_turns']} turns but couldn't build any training examples from them. "
            "Upload a longer chat export (a few dozen back-and-forth messages at minimum).",
        )
    return {"name": name, **meta}


@app.post("/api/personas")
def upload_persona(name: str = Form(...), file: UploadFile = File(...), me_label: str = Form("Me"),
                   display_name: str = Form(""), prompt_style: str = Form("minimal")):
    """Plain `def`: parsing and tokenizing are CPU-heavy, so FastAPI runs this
    in its threadpool instead of blocking the event loop."""
    name = _safe_name(name)
    if prompt_style not in prepare_data.PROMPT_STYLES:
        raise HTTPException(400, f"prompt_style must be one of {prepare_data.PROMPT_STYLES}")
    raw_path = UPLOAD_DIR / f"{name}.txt"
    with open(raw_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        _, counts = parse_file(raw_path, me_label=me_label)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if len(counts) > 1:
        # Same person under several handles, or a group chat: let the user pick.
        return {
            "name": name,
            "needs_sender_choice": True,
            "senders": [{"label": label, "count": c, "is_handle": looks_like_handle(label)}
                        for label, c in counts.most_common(20)],
            "me_label": me_label,
            "display_name": display_name,
            "prompt_style": prompt_style,
        }
    return _prepare(name, me_label, [next(iter(counts))], display_name, prompt_style)


class PrepareRequest(BaseModel):
    persona_senders: list[str] = Field(min_length=1)
    me_label: str = "Me"
    display_name: str = ""
    prompt_style: str = "minimal"


@app.post("/api/personas/{name}/prepare")
def prepare_persona(name: str, req: PrepareRequest):
    name = _safe_name(name)
    if not (UPLOAD_DIR / f"{name}.txt").exists():
        raise HTTPException(404, "Upload the chat export first.")
    if req.prompt_style not in prepare_data.PROMPT_STYLES:
        raise HTTPException(400, f"prompt_style must be one of {prepare_data.PROMPT_STYLES}")
    return _prepare(name, req.me_label, req.persona_senders, req.display_name, req.prompt_style)


class TrainRequest(BaseModel):
    backend: str = "local"      # "local" or "colab"
    model_key: str = "qwen-3b"  # one of train_local.MODELS
    learning_rate: float = 1e-5
    rank: int = Field(16, ge=2, le=128)
    num_layers: int | None = None


@app.post("/api/personas/{name}/train")
def train_persona(name: str, req: TrainRequest):
    name = _safe_name(name)
    data_dir = PROCESSED_DIR / name
    if not (data_dir / "sessions.json").exists():
        raise HTTPException(404, "Upload this persona's data first (re-upload personas prepared before this version).")
    if req.model_key not in train_local.MODELS:
        raise HTTPException(400, f"Unknown model '{req.model_key}'.")

    if req.backend == "colab":
        meta = prepare_data.build_dataset(data_dir, tokenizer_model=train_local.MODELS[req.model_key]["repo"])
        notebook = colab_export.build_notebook(meta, req.model_key, req.rank)
        (data_dir / f"{name}_colab.ipynb").write_text(json.dumps(notebook, indent=1))
        return {
            "backend": "colab",
            "notebook_url": f"/api/personas/{name}/colab_notebook",
            "train_jsonl_url": f"/api/personas/{name}/colab_train_jsonl",
            "valid_jsonl_url": f"/api/personas/{name}/colab_valid_jsonl",
        }

    chat_infer.unload()  # training and a loaded chat model don't both fit in 16 GB
    _start_job(name, "training", lambda log: train_local.run_training(
        name, data_dir, ADAPTER_DIR / name, model_key=req.model_key, learning_rate=req.learning_rate,
        rank=req.rank, num_layers=req.num_layers, log=log,
    ))
    return {"backend": "local", "state": "training"}


@app.post("/api/personas/{name}/import_colab")
def import_colab_model(name: str, file: UploadFile = File(...)):
    name = _safe_name(name)
    data_dir = PROCESSED_DIR / name
    if not (data_dir / "meta.json").exists():
        raise HTTPException(404, "Upload this persona's data first (same name as in Colab).")
    zip_path = UPLOAD_DIR / f"{name}_colab_import.zip"
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if chat_infer.loaded_adapter() and Path(chat_infer.loaded_adapter()).name == name:
        chat_infer.unload()
    _start_job(name, "importing", lambda log: import_colab.import_zip(
        name, zip_path, data_dir, ADAPTER_DIR / name, log=log,
    ))
    return {"state": "importing"}


@app.get("/api/personas/{name}/status")
def persona_status(name: str):
    return _persona_status(_safe_name(name))


def _data_file(name, filename, download_name, media_type):
    path = PROCESSED_DIR / _safe_name(name) / filename
    if not path.exists():
        raise HTTPException(404, "Not prepared yet.")
    return FileResponse(path, filename=download_name, media_type=media_type)


@app.get("/api/personas/{name}/colab_notebook")
def download_notebook(name: str):
    return _data_file(name, f"{name}_colab.ipynb", f"{name}_colab.ipynb", "application/x-ipynb+json")


@app.get("/api/personas/{name}/colab_train_jsonl")
def download_colab_train_jsonl(name: str):
    # Exact filename: the notebook's upload cell checks for it.
    return _data_file(name, "train.jsonl", "train.jsonl", "application/jsonl")


@app.get("/api/personas/{name}/colab_valid_jsonl")
def download_colab_valid_jsonl(name: str):
    return _data_file(name, "valid.jsonl", "valid.jsonl", "application/jsonl")


# ---------------------------------------------------------------- chat

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    history: list[ChatMessage] = Field(min_length=1, max_length=MAX_HISTORY_MESSAGES)
    temperature: float | None = None
    top_p: float | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None


def _chat(name, history, **overrides):
    adapter_dir = ADAPTER_DIR / name
    if not (adapter_dir / "persona_meta.json").exists():
        raise HTTPException(404, "This persona hasn't finished training yet.")
    if _local_training_active():
        raise HTTPException(503, "Chat is paused while a persona trains on this Mac — memory is needed for training.")
    if history[-1]["role"] != "user":
        raise HTTPException(400, "The last message must be from the user.")
    try:
        text = chat_infer.reply(adapter_dir, history, **overrides)
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {e}")
    return {"reply": text}


@app.post("/api/personas/{name}/chat")
def chat(name: str, req: ChatRequest):
    name = _safe_name(name)
    overrides = req.model_dump(exclude={"history"}, exclude_none=True)
    return _chat(name, [m.model_dump() for m in req.history], **overrides)


class SettingsRequest(BaseModel):
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)
    min_p: float | None = Field(None, ge=0.0, le=0.5)
    max_tokens: int | None = Field(None, ge=8, le=512)
    repetition_penalty: float | None = Field(None, ge=1.0, le=2.0)


@app.get("/api/personas/{name}/settings")
def get_settings(name: str):
    return chat_infer.load_settings(ADAPTER_DIR / _safe_name(name))


@app.put("/api/personas/{name}/settings")
def put_settings(name: str, req: SettingsRequest):
    name = _safe_name(name)
    if not (ADAPTER_DIR / name).exists():
        raise HTTPException(404, "This persona hasn't been trained yet.")
    return chat_infer.save_settings(ADAPTER_DIR / name, req.model_dump(exclude_none=True))


@app.delete("/api/personas/{name}")
def delete_persona(name: str):
    name = _safe_name(name)
    with _jobs_lock:
        if _jobs.get(name, {}).get("state") in ("training", "importing"):
            raise HTTPException(409, "Wait for training/import to finish before deleting.")
        _jobs.pop(name, None)
    loaded = chat_infer.loaded_adapter()
    if loaded and Path(loaded).name == name:
        chat_infer.unload()
    removed = []
    adapter_dir = ADAPTER_DIR / name
    for path in (UPLOAD_DIR / f"{name}.txt", UPLOAD_DIR / f"{name}_colab_import.zip", PROCESSED_DIR / name,
                 adapter_dir, train_local.staging_dir_for(adapter_dir)):
        if path.exists():
            path.unlink() if path.is_file() else shutil.rmtree(path)
            removed.append(str(path))
    return {"deleted": name, "removed_paths": removed}


@app.get("/api/personas/{name}/download")
def download_adapter(name: str):
    """Zips what's needed to chat with this persona on another personabot.
    Stored, not deflated: safetensors weights don't compress."""
    import zipfile

    name = _safe_name(name)
    adapter_dir = ADAPTER_DIR / name
    meta = _read_json(adapter_dir / "persona_meta.json")
    if not meta:
        raise HTTPException(404, "This persona hasn't finished training yet.")
    keep = ["persona_meta.json", "settings.json", "exemplars.jsonl", "selection.json"]
    keep += ["mlx_model"] if meta.get("merged") else ["adapters.safetensors", "adapter_config.json"]

    fd, tmp = tempfile.mkstemp(suffix=".zip")
    with open(fd, "wb") as handle, zipfile.ZipFile(handle, "w", zipfile.ZIP_STORED) as zf:
        for item_name in keep:
            item = adapter_dir / item_name
            if item.is_dir():
                for f in item.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(adapter_dir))
            elif item.exists():
                zf.write(item, item_name)
    return FileResponse(tmp, filename=f"{name}_adapter.zip", media_type="application/zip",
                        background=BackgroundTask(Path(tmp).unlink, missing_ok=True))


# ---------------------------------------------------------------- share links

@app.get("/api/share/{name}")
def share_info(name: str):
    name = _safe_name(name)
    meta = _read_json(ADAPTER_DIR / name / "persona_meta.json")
    if not meta:
        raise HTTPException(404, "Persona not found.")
    return {"name": name, "display_name": meta.get("display_name") or meta.get("persona") or name, "state": "ready"}


@app.post("/api/share/{name}/chat")
def share_chat(name: str, req: ChatRequest):
    name = _safe_name(name)
    return _chat(name, [m.model_dump() for m in req.history])  # visitors can't change generation settings


@app.get("/p/{name}")
def shared_chat_page(name: str):
    _safe_name(name)
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
