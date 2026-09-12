const API = "/api";
let activePersona = null;
let activePersonaDisplay = null; // display name (e.g. "Hudson"), for avatar consistency with the persona list
let chatHistory = [];
let pollTimer = null;
let sharedMode = false;

const $ = (sel) => document.querySelector(sel);

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || res.statusText);
  }
  return res.json();
}

// Copies text without ever falling back to a blocking dialog (prompt() is
// disallowed in some embedding contexts) — tries the async Clipboard API
// first, then a synchronous execCommand fallback via an offscreen textarea.
// Returns whether it actually worked, so callers can tell the user if not.
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }
}

function icon(id) {
  return document.getElementById(id).content.cloneNode(true);
}

const AVATAR_COLORS = ["#ff9f0a", "#ff375f", "#af52de", "#5856d6", "#007aff", "#34c759", "#00c7be"];
function avatarColor(name) {
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return AVATAR_COLORS[h % AVATAR_COLORS.length];
}
function makeAvatar(name, small = false) {
  const el = document.createElement("div");
  el.className = small ? "avatar small" : "avatar";
  el.style.background = avatarColor(name);
  el.textContent = (name[0] || "?").toUpperCase();
  return el;
}

const MODEL_OPTIONS = [
  { key: "qwen-3b", label: "Qwen2.5-3B (fast)" },
  { key: "qwen-7b", label: "Qwen2.5-7B (stronger, slower)" },
  { key: "llama-3b", label: "Llama-3.2-3B (fast)" },
  { key: "llama-8b", label: "Llama-3.1-8B (stronger, slower)" },
];
const DEFAULT_TRAIN_CONFIG = { model_key: "qwen-3b", rank: 16, num_layers: null };
const pendingTrainConfig = {}; // name -> {model_key, rank, num_layers}

function trainConfigFor(name) {
  if (!pendingTrainConfig[name]) pendingTrainConfig[name] = { ...DEFAULT_TRAIN_CONFIG };
  return pendingTrainConfig[name];
}

// ==================== Shared-link mode ====================
// A /p/<name> URL opens straight into chat for that persona, no upload/list UI.
function detectSharedMode() {
  const m = window.location.pathname.match(/^\/p\/([^/]+)/);
  if (!m) return null;
  return decodeURIComponent(m[1]);
}

// ==================== Upload ====================
$("#persona-file").addEventListener("change", () => {
  const f = $("#persona-file").files[0];
  $("#file-name").textContent = f ? f.name : "No file chosen";
});

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("#persona-name").value.trim();
  const meLabel = $("#me-label").value.trim() || "Me";
  const file = $("#persona-file").files[0];
  const statusEl = $("#upload-status");
  statusEl.textContent = "Uploading + preparing data…";

  const form = new FormData();
  form.append("name", name);
  form.append("me_label", meLabel);
  form.append("file", file);

  try {
    const meta = await api("/personas", { method: "POST", body: form });
    statusEl.textContent =
      `Parsed ${meta.n_turns} turns for "${meta.persona}" ` +
      `(${meta.n_train} train / ${meta.n_val} val examples).`;
    refreshPersonas();
  } catch (err) {
    statusEl.textContent = "Error: " + err.message;
  }
});

// ==================== Help sheet ====================
$("#help-btn").addEventListener("click", () => $("#help-overlay").classList.remove("hidden"));
$('[data-action="close-help"]').addEventListener("click", () => $("#help-overlay").classList.add("hidden"));
$("#help-overlay").addEventListener("click", (e) => { if (e.target.id === "help-overlay") e.target.classList.add("hidden"); });

// ==================== Settings sheet ====================
const settingsOverlay = $("#settings-overlay");
$('[data-action="close-settings"]').addEventListener("click", () => settingsOverlay.classList.add("hidden"));
settingsOverlay.addEventListener("click", (e) => { if (e.target === settingsOverlay) settingsOverlay.classList.add("hidden"); });

function openSettingsSheet(p) {
  $("#settings-title").textContent = `${p.persona || p.name} — Settings`;
  const body = $("#settings-body");
  body.innerHTML = "";

  if (p.state === "ready") {
    body.appendChild(buildGenerationSection(p));
  } else {
    body.appendChild(buildTrainConfigSection(p));
  }
  body.appendChild(buildDangerSection(p));
  settingsOverlay.classList.remove("hidden");
}

function settingsSection(title, cardHtml) {
  const sec = document.createElement("div");
  sec.className = "settings-section";
  sec.innerHTML = `<div class="settings-section-title">${title}</div><div class="settings-card">${cardHtml}</div>`;
  return sec;
}

function buildGenerationSection(p) {
  const fields = [
    { key: "temperature", label: "Temperature", min: 0.1, max: 1.5, step: 0.05 },
    { key: "top_p", label: "Top-p", min: 0.1, max: 1.0, step: 0.05 },
    { key: "max_tokens", label: "Max reply length", min: 12, max: 160, step: 4 },
    { key: "repetition_penalty", label: "Repetition penalty", min: 1.0, max: 1.6, step: 0.05 },
  ];
  const rowsHtml = fields.map(f => `
    <div class="settings-list-row" data-key="${f.key}">
      <span>${f.label}</span>
      <input type="range" min="${f.min}" max="${f.max}" step="${f.step}" />
      <span class="value">–</span>
    </div>
  `).join("");
  const sec = settingsSection("Generation", rowsHtml);

  api(`/personas/${p.name}/settings`).then((s) => {
    fields.forEach(f => {
      const row = sec.querySelector(`[data-key="${f.key}"]`);
      row.querySelector("input").value = s[f.key];
      row.querySelector(".value").textContent = s[f.key];
    });
  });

  fields.forEach(f => {
    const row = sec.querySelector(`[data-key="${f.key}"]`);
    const input = row.querySelector("input");
    const valueEl = row.querySelector(".value");
    input.addEventListener("input", () => { valueEl.textContent = input.value; });
    input.addEventListener("change", () => {
      api(`/personas/${p.name}/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [f.key]: Number(input.value) }),
      });
    });
  });
  return sec;
}

function buildTrainConfigSection(p) {
  const cfg = trainConfigFor(p.name);
  const optionsHtml = MODEL_OPTIONS.map(m =>
    `<option value="${m.key}" ${m.key === cfg.model_key ? "selected" : ""}>${m.label}</option>`
  ).join("");
  const html = `
    <div class="settings-list-row">
      <span>Base model</span>
      <select data-role="model_key">${optionsHtml}</select>
    </div>
    <div class="settings-list-row">
      <span>LoRA rank (voice capacity)</span>
      <input data-role="rank" type="number" value="${cfg.rank}" min="4" max="64" />
    </div>
    <div class="settings-list-row">
      <span>Layers adapted</span>
      <input data-role="num_layers" type="number" placeholder="auto" value="${cfg.num_layers ?? ""}" />
    </div>
  `;
  const sec = settingsSection("Model & training (applies on next train)", html);
  sec.querySelector('[data-role="model_key"]').addEventListener("change", (e) => { cfg.model_key = e.target.value; });
  sec.querySelector('[data-role="rank"]').addEventListener("change", (e) => { cfg.rank = Number(e.target.value) || 16; });
  sec.querySelector('[data-role="num_layers"]').addEventListener("change", (e) => {
    cfg.num_layers = e.target.value ? Number(e.target.value) : null;
  });
  return sec;
}

function buildDangerSection(p) {
  const html = `<div class="settings-list-row"><span>Delete persona</span>
    <button class="pill-button danger small" data-action="delete-in-sheet">Delete</button></div>`;
  const sec = settingsSection("Danger zone", html);
  sec.querySelector('[data-action="delete-in-sheet"]').addEventListener("click", async () => {
    if (!confirm(`Delete "${p.persona || p.name}"? This removes its data and trained adapter.`)) return;
    await api(`/personas/${p.name}`, { method: "DELETE" });
    settingsOverlay.classList.add("hidden");
    if (activePersona === p.name) {
      activePersona = null;
      activePersonaDisplay = null;
      $("#chat-section").classList.add("hidden");
    }
    refreshPersonas();
  });
  return sec;
}

// ==================== Persona list ====================
function badgeClass(state) {
  if (state === "ready") return "badge ready";
  if (state === "training" || state === "importing") return "badge training";
  if (state === "error") return "badge error";
  return "badge";
}

function metaLine(p) {
  if (p.state === "ready") {
    if (p.merged) return `imported from Colab · ${p.model_key || "colab"}`;
    const model = p.model_key || p.size; // p.size: back-compat with adapters trained before the model_key rename
    return `best val ${p.best_val?.toFixed(2)} · iter ${p.best_iter}` + (model ? ` · ${model}` : "");
  }
  if (p.state === "uploaded") return `${p.n_train} train examples · ${p.n_val} val`;
  if (p.state === "training") return "training…";
  if (p.state === "importing") return "importing from Colab…";
  if (p.state === "error") return "last run failed";
  return "";
}

function renderPersona(p) {
  const div = document.createElement("div");
  div.className = "persona-card";

  const header = document.createElement("div");
  header.className = "row";
  header.appendChild(makeAvatar(p.persona || p.name));

  const identity = document.createElement("div");
  identity.className = "identity";
  identity.innerHTML = `<div class="name">${p.persona || p.name}</div><div class="meta-line">${metaLine(p)}</div>`;
  header.appendChild(identity);

  const badge = document.createElement("span");
  badge.className = badgeClass(p.state);
  badge.textContent = p.state;
  header.appendChild(badge);
  div.appendChild(header);

  const actions = document.createElement("div");
  actions.className = "actions";

  if (p.state === "uploaded" || p.state === "error") {
    actions.innerHTML = `
      <button class="pill-button primary small" data-action="train-local">Train locally</button>
      <button class="pill-button small" data-action="train-colab">Export Colab notebook</button>
      <label class="pill-button small file-button" for="import-${p.name}">Import trained model (.zip)</label>
      <input id="import-${p.name}" data-role="import-file" type="file" accept=".zip" hidden />
    `;
  } else if (p.state === "ready") {
    actions.innerHTML = `<button class="pill-button primary small" data-action="chat">Chat</button>`;
  }

  const iconActions = document.createElement("div");
  iconActions.className = "icon-actions";
  const gearBtn = document.createElement("button");
  gearBtn.className = "icon-button";
  gearBtn.title = "Settings";
  gearBtn.appendChild(icon("icon-gear"));
  iconActions.appendChild(gearBtn);

  if (p.state === "ready") {
    const shareBtn = document.createElement("button");
    shareBtn.className = "icon-button";
    shareBtn.title = "Copy share link";
    shareBtn.appendChild(icon("icon-share"));
    shareBtn.addEventListener("click", async () => {
      const url = `${window.location.origin}/p/${encodeURIComponent(p.name)}`;
      const ok = await copyToClipboard(url);
      shareBtn.title = ok ? "Copied!" : "Copy failed — link: " + url;
      setTimeout(() => { shareBtn.title = "Copy share link"; }, 2500);
    });
    const downloadBtn = document.createElement("button");
    downloadBtn.className = "icon-button";
    downloadBtn.title = "Download adapter";
    downloadBtn.appendChild(icon("icon-download"));
    downloadBtn.addEventListener("click", () => {
      window.open(`${API}/personas/${p.name}/download`, "_blank");
    });
    iconActions.append(shareBtn, downloadBtn);
  }

  const trashBtn = document.createElement("button");
  trashBtn.className = "icon-button danger";
  trashBtn.title = "Delete persona";
  trashBtn.appendChild(icon("icon-trash"));
  iconActions.appendChild(trashBtn);

  actions.appendChild(iconActions);
  div.appendChild(actions);

  if (p.state === "ready") {
    const note = document.createElement("p");
    note.className = "share-note";
    note.textContent = "Share links only work while this Mac is running personabot and reachable on your network.";
    div.appendChild(note);
  }

  if (p.state === "training" || p.state === "importing") {
    const log = document.createElement("div");
    log.className = "log";
    log.textContent = (p.log_tail || []).join("\n");
    div.appendChild(log);
  }
  if (p.state === "error") {
    const log = document.createElement("div");
    log.className = "log";
    log.textContent = p.error || "";
    div.appendChild(log);
  }

  gearBtn.addEventListener("click", () => openSettingsSheet(p));

  trashBtn.addEventListener("click", async () => {
    if (!confirm(`Delete "${p.persona || p.name}"? This removes its data and trained adapter.`)) return;
    await api(`/personas/${p.name}`, { method: "DELETE" });
    if (activePersona === p.name) {
      activePersona = null;
      activePersonaDisplay = null;
      $("#chat-section").classList.add("hidden");
    }
    refreshPersonas();
  });

  div.querySelector('[data-role="import-file"]')?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    await api(`/personas/${p.name}/import_colab`, { method: "POST", body: form });
    refreshPersonas();
  });

  div.querySelector('[data-action="train-local"]')?.addEventListener("click", async () => {
    const cfg = trainConfigFor(p.name);
    const body = { backend: "local", model_key: cfg.model_key, rank: cfg.rank };
    if (cfg.num_layers) body.num_layers = cfg.num_layers;
    await api(`/personas/${p.name}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    refreshPersonas();
  });

  div.querySelector('[data-action="train-colab"]')?.addEventListener("click", async () => {
    const res = await api(`/personas/${p.name}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend: "colab" }),
    });
    window.open(API + res.notebook_url.replace("/api", ""), "_blank");
  });

  div.querySelector('[data-action="chat"]')?.addEventListener("click", () => openChat(p));

  return div;
}

async function refreshPersonas() {
  const list = await api("/personas");

  if (!sharedMode) {
    const container = $("#persona-list");
    container.innerHTML = "";
    if (list.length === 0) {
      container.innerHTML = `<p class="status">No personas yet — upload a chat export above.</p>`;
    }
    list.forEach((p) => container.appendChild(renderPersona(p)));
  }

  renderPersonaRow(list.filter((p) => p.state === "ready"));

  const anyActive = list.some((p) => p.state === "training" || p.state === "importing");
  if (anyActive && !pollTimer) {
    pollTimer = setInterval(refreshPersonas, 4000);
  } else if (!anyActive && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }

  return list;
}

function renderPersonaRow(readyPersonas) {
  const row = $("#persona-row");
  if (sharedMode || readyPersonas.length < 2) {
    row.innerHTML = "";
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  row.innerHTML = "";
  readyPersonas.forEach((p) => {
    const item = document.createElement("div");
    item.className = "persona-row-item" + (p.name === activePersona ? " active" : "");
    item.appendChild(makeAvatar(p.persona || p.name));
    const label = document.createElement("span");
    label.className = "row-label";
    label.textContent = p.persona || p.name;
    item.appendChild(label);
    item.addEventListener("click", () => openChat(p));
    row.appendChild(item);
  });
}

// ==================== Chat ====================
function openChat(p) {
  activePersona = p.name;
  activePersonaDisplay = p.persona || p.name;
  chatHistory = [];
  $("#chat-section").classList.remove("hidden");
  $("#chat-title").textContent = `Chat with ${p.persona || p.name}`;
  $("#chat-window").innerHTML = "";
  if (!sharedMode) $("#chat-section").scrollIntoView({ behavior: "smooth" });
  refreshPersonas().then((list) => renderPersonaRow(list.filter((x) => x.state === "ready")));
}

function addBubble(text, who, personaLabel, onRegenerate) {
  const win = $("#chat-window");
  const rowEl = document.createElement("div");
  rowEl.className = `msg-row ${who}`;

  if (who === "them") rowEl.appendChild(makeAvatar(personaLabel || "?", true));

  const col = document.createElement("div");
  col.className = "msg-col";
  const bubble = document.createElement("div");
  bubble.className = `bubble ${who}`;
  bubble.textContent = text;
  col.appendChild(bubble);

  if (who === "them") {
    const actionsEl = document.createElement("div");
    actionsEl.className = "msg-actions";
    const copyBtn = document.createElement("button");
    copyBtn.title = "Copy";
    copyBtn.appendChild(icon("icon-copy"));
    copyBtn.addEventListener("click", () => copyToClipboard(bubble.textContent));
    const regenBtn = document.createElement("button");
    regenBtn.title = "Regenerate";
    regenBtn.appendChild(icon("icon-refresh"));
    regenBtn.dataset.role = "regenerate";
    if (onRegenerate) regenBtn.addEventListener("click", onRegenerate);
    actionsEl.append(copyBtn, regenBtn);
    col.appendChild(actionsEl);
  }

  rowEl.appendChild(col);
  win.appendChild(rowEl);
  win.scrollTop = win.scrollHeight;
  return { bubble, row: rowEl };
}

// Only the most recent bot reply is regeneratable — regenerating an older
// one would desync it from everything the user said after it.
function refreshRegenerateButtons() {
  const buttons = document.querySelectorAll('#chat-window .msg-row.them [data-role="regenerate"]');
  buttons.forEach((b, i) => { b.style.display = i === buttons.length - 1 ? "" : "none"; });
}

async function sendToPersona(text) {
  addBubble(text, "me");
  chatHistory.push({ role: "user", content: text });

  const { bubble: placeholder, row } = addBubble("…", "them", activePersonaDisplay, async () => {
    chatHistory.pop(); // drop the assistant reply being regenerated
    const lastUser = chatHistory[chatHistory.length - 1]?.content;
    if (!lastUser) return;
    chatHistory.pop();
    row.remove();
    await sendToPersona(lastUser);
  });
  refreshRegenerateButtons();

  try {
    const res = await api(`/personas/${activePersona}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: chatHistory }),
    });
    placeholder.textContent = res.reply;
    chatHistory.push({ role: "assistant", content: res.reply });
  } catch (err) {
    placeholder.textContent = "Error: " + err.message;
  }
  refreshRegenerateButtons();
}

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!activePersona) return;
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  await sendToPersona(text);
});

$("#clear-chat-btn").addEventListener("click", () => {
  chatHistory = [];
  $("#chat-window").innerHTML = "";
});

// ==================== Boot ====================
async function boot() {
  const sharedName = detectSharedMode();
  if (sharedName) {
    sharedMode = true;
    $("#upload-section").classList.add("hidden");
    $("#personas-section").classList.add("hidden");
    try {
      const list = await api("/personas");
      const p = list.find((x) => x.name === sharedName);
      if (p && p.state === "ready") {
        openChat(p);
      } else {
        $("#chat-section").classList.remove("hidden");
        $("#chat-title").textContent = p ? `${p.persona || p.name} isn't ready to chat yet` : "Persona not found";
      }
    } catch {
      $("#chat-section").classList.remove("hidden");
      $("#chat-title").textContent = "Couldn't load this persona";
    }
    return;
  }
  refreshPersonas();
}

boot();
