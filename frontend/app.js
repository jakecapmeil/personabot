const API = "/api";
let activePersona = null;
let activePersonaDisplay = null; // display name, for avatar consistency with the persona list
let chatHistory = [];
let pollTimer = null;
let sharedMode = false;
let chatBusy = false;

const $ = (sel) => document.querySelector(sel);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function api(path, opts) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || res.statusText);
  }
  return res.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Copies text without ever falling back to a blocking dialog (prompt() is
// disallowed in some embedding contexts) — tries the async Clipboard API
// first, then a synchronous execCommand fallback via an offscreen textarea.
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
  const node = el("div", small ? "avatar small" : "avatar", (name[0] || "?").toUpperCase());
  node.style.background = avatarColor(name);
  return node;
}

const MODEL_OPTIONS = [
  { key: "qwen-3b", label: "Qwen2.5-3B (fast)" },
  { key: "qwen-7b", label: "Qwen2.5-7B (stronger, slower)" },
  { key: "llama-3b", label: "Llama-3.2-3B (fast)" },
  { key: "llama-8b", label: "Llama-3.1-8B (stronger, slower)" },
  { key: "qwen-3b-base", label: "Qwen2.5-3B base (less assistant-like)" },
  { key: "qwen-7b-base", label: "Qwen2.5-7B base (less assistant-like)" },
];
const DEFAULT_TRAIN_CONFIG = { model_key: "qwen-3b", rank: 16, num_layers: null };
const pendingTrainConfig = {}; // name -> {model_key, rank, num_layers}

function trainConfigFor(name) {
  if (!pendingTrainConfig[name]) pendingTrainConfig[name] = { ...DEFAULT_TRAIN_CONFIG };
  return pendingTrainConfig[name];
}

function displayNameOf(p) {
  return p.display_name || p.persona || p.name;
}

// ==================== Shared-link mode ====================
// A /p/<name> URL opens straight into chat for that persona, no upload/list UI.
function detectSharedMode() {
  const m = window.location.pathname.match(/^\/p\/([^/]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

// ==================== Upload ====================
$("#persona-file").addEventListener("change", () => {
  const f = $("#persona-file").files[0];
  $("#file-name").textContent = f ? f.name : "No file chosen";
});

function uploadSummary(meta) {
  return `Prepared "${displayNameOf(meta)}": ${meta.n_sessions} conversations, ` +
    `${meta.n_train_replies} replies to learn from (${meta.n_train} training rows / ${meta.n_val} validation).`;
}

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = $("#persona-name").value.trim();
  const statusEl = $("#upload-status");
  statusEl.textContent = "Uploading + preparing data…";
  $("#sender-choice").classList.add("hidden");

  const form = new FormData();
  form.append("name", name);
  form.append("me_label", $("#me-label").value.trim() || "Me");
  form.append("display_name", $("#display-name").value.trim());
  form.append("prompt_style", $("#prompt-style").value);
  form.append("file", $("#persona-file").files[0]);

  try {
    const res = await api("/personas", { method: "POST", body: form });
    if (res.needs_sender_choice) {
      statusEl.textContent = "";
      showSenderChoice(res);
      return;
    }
    statusEl.textContent = uploadSummary(res);
    refreshPersonas();
  } catch (err) {
    statusEl.textContent = "Error: " + err.message;
  }
});

// More than one other sender: the same person under several handles (phone
// number + email), or a group chat. The user ticks which senders are them.
function showSenderChoice(res) {
  const box = $("#sender-choice");
  box.replaceChildren();
  box.appendChild(el("p", "sender-choice-title", "Which of these senders are this person?"));
  box.appendChild(el("p", "status", "Tick every handle they text from (e.g. both their number and their email). Leave group-chat members unticked."));

  const list = el("div", "sender-list");
  res.senders.forEach((s, i) => {
    const row = el("label", "sender-row");
    const box_ = document.createElement("input");
    box_.type = "checkbox";
    box_.value = s.label;
    box_.checked = i === 0;
    row.append(box_, el("span", "sender-label", s.label), el("span", "sender-count", `${s.count} messages`));
    list.appendChild(row);
  });
  box.appendChild(list);

  const button = el("button", "pill-button primary small", "Continue");
  button.type = "button";
  button.addEventListener("click", async () => {
    const chosen = [...list.querySelectorAll("input:checked")].map((c) => c.value);
    const statusEl = $("#upload-status");
    if (!chosen.length) {
      statusEl.textContent = "Tick at least one sender.";
      return;
    }
    button.disabled = true;
    statusEl.textContent = "Preparing data…";
    try {
      const meta = await api(`/personas/${res.name}/prepare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          persona_senders: chosen,
          me_label: res.me_label,
          display_name: res.display_name,
          prompt_style: res.prompt_style,
        }),
      });
      box.classList.add("hidden");
      statusEl.textContent = uploadSummary(meta);
      refreshPersonas();
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
    } finally {
      button.disabled = false;
    }
  });
  box.appendChild(button);
  box.classList.remove("hidden");
}

// ==================== Help sheet ====================
$("#help-btn").addEventListener("click", () => $("#help-overlay").classList.remove("hidden"));
$('[data-action="close-help"]').addEventListener("click", () => $("#help-overlay").classList.add("hidden"));
$("#help-overlay").addEventListener("click", (e) => { if (e.target.id === "help-overlay") e.target.classList.add("hidden"); });

// ==================== Colab instructions sheet ====================
$('[data-action="close-colab"]').addEventListener("click", () => $("#colab-overlay").classList.add("hidden"));
$("#colab-overlay").addEventListener("click", (e) => { if (e.target.id === "colab-overlay") e.target.classList.add("hidden"); });

// Same-tab download per URL via a throwaway <a download>; a short stagger
// keeps browsers from treating several saves in a row as spam.
function downloadFile(url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}
async function downloadSequentially(urls) {
  for (const url of urls) {
    downloadFile(url);
    await sleep(400);
  }
}

// ==================== Settings sheet ====================
const settingsOverlay = $("#settings-overlay");
$('[data-action="close-settings"]').addEventListener("click", () => settingsOverlay.classList.add("hidden"));
settingsOverlay.addEventListener("click", (e) => { if (e.target === settingsOverlay) settingsOverlay.classList.add("hidden"); });

function openSettingsSheet(p) {
  $("#settings-title").textContent = `${displayNameOf(p)} — Settings`;
  const body = $("#settings-body");
  body.replaceChildren();
  if (p.state === "ready") body.appendChild(buildGenerationSection(p));
  if (p.state === "ready" && p.style) body.appendChild(buildStyleSection(p));
  if (p.state !== "training" && p.state !== "importing") body.appendChild(buildTrainConfigSection(p));
  body.appendChild(buildDangerSection(p));
  settingsOverlay.classList.remove("hidden");
}

function settingsSection(title) {
  const sec = el("div", "settings-section");
  sec.appendChild(el("div", "settings-section-title", title));
  const card = el("div", "settings-card");
  sec.appendChild(card);
  return { sec, card };
}

function settingsRow(label) {
  const row = el("div", "settings-list-row");
  row.appendChild(el("span", "", label));
  return row;
}

function buildGenerationSection(p) {
  const fields = [
    { key: "temperature", label: "Temperature", min: 0.1, max: 1.5, step: 0.05 },
    { key: "min_p", label: "Min-p", min: 0, max: 0.3, step: 0.01 },
    { key: "top_p", label: "Top-p (1 = off)", min: 0.1, max: 1.0, step: 0.05 },
    { key: "max_tokens", label: "Max reply length", min: 16, max: 256, step: 8 },
    { key: "repetition_penalty", label: "Repetition penalty", min: 1.0, max: 1.3, step: 0.01 },
  ];
  const { sec, card } = settingsSection("Generation");
  const inputs = {};
  fields.forEach((f) => {
    const row = settingsRow(f.label);
    const input = document.createElement("input");
    Object.assign(input, { type: "range", min: f.min, max: f.max, step: f.step });
    const value = el("span", "value", "–");
    input.addEventListener("input", () => { value.textContent = input.value; });
    input.addEventListener("change", () => {
      api(`/personas/${p.name}/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [f.key]: Number(input.value) }),
      }).catch((err) => { value.textContent = "!"; value.title = err.message; });
    });
    row.append(input, value);
    card.appendChild(row);
    inputs[f.key] = { input, value };
  });
  api(`/personas/${p.name}/settings`).then((s) => {
    fields.forEach((f) => {
      inputs[f.key].input.value = s[f.key];
      inputs[f.key].value.textContent = s[f.key];
    });
  });
  return sec;
}

function buildStyleSection(p) {
  const s = p.style;
  const { sec, card } = settingsSection("Style check (lower is closer to their real texts)");
  const rows = [
    ["Overall distance", s.style_distance],
    ["Reply length gap", s.length_gap],
    ["Casing / punctuation / emoji gap", s.rate_gap],
    ["Copied from prompt", s.copy_rate],
    ["Assistant-like phrases", s.assistantisms],
  ];
  rows.forEach(([label, v]) => {
    const row = settingsRow(label);
    row.appendChild(el("span", "value", typeof v === "number" ? v.toFixed(2) : "–"));
    card.appendChild(row);
  });
  return sec;
}

function buildTrainConfigSection(p) {
  const cfg = trainConfigFor(p.name);
  const title = p.state === "ready" ? "Retrain with (applies on next train)" : "Model & training (applies on next train)";
  const { sec, card } = settingsSection(title);

  const modelRow = settingsRow("Base model");
  const select = document.createElement("select");
  MODEL_OPTIONS.forEach((m) => {
    const opt = el("option", "", m.label);
    opt.value = m.key;
    opt.selected = m.key === cfg.model_key;
    select.appendChild(opt);
  });
  select.addEventListener("change", () => { cfg.model_key = select.value; });
  modelRow.appendChild(select);

  const rankRow = settingsRow("LoRA rank (voice capacity)");
  const rank = document.createElement("input");
  Object.assign(rank, { type: "number", min: 4, max: 64, value: cfg.rank });
  rank.addEventListener("change", () => { cfg.rank = Number(rank.value) || 16; });
  rankRow.appendChild(rank);

  const layersRow = settingsRow("Layers adapted");
  const layers = document.createElement("input");
  Object.assign(layers, { type: "number", placeholder: "auto", value: cfg.num_layers ?? "" });
  layers.addEventListener("change", () => { cfg.num_layers = layers.value ? Number(layers.value) : null; });
  layersRow.appendChild(layers);

  card.append(modelRow, rankRow, layersRow);
  if (p.state === "ready") {
    const retrainRow = settingsRow("Train again with these settings");
    const button = el("button", "pill-button small", "Retrain");
    button.addEventListener("click", async () => {
      settingsOverlay.classList.add("hidden");
      await startLocalTraining(p);
    });
    retrainRow.appendChild(button);
    card.appendChild(retrainRow);
  }
  return sec;
}

function buildDangerSection(p) {
  const { sec, card } = settingsSection("Danger zone");
  const row = settingsRow("Delete persona");
  const button = el("button", "pill-button danger small", "Delete");
  button.addEventListener("click", async () => {
    if (await deletePersona(p)) settingsOverlay.classList.add("hidden");
  });
  row.appendChild(button);
  card.appendChild(row);
  return sec;
}

async function deletePersona(p) {
  if (!confirm(`Delete "${displayNameOf(p)}"? This removes its data and trained adapter.`)) return false;
  try {
    await api(`/personas/${p.name}`, { method: "DELETE" });
  } catch (err) {
    alert("Couldn't delete: " + err.message);
    return false;
  }
  if (activePersona === p.name) {
    activePersona = null;
    activePersonaDisplay = null;
    $("#chat-section").classList.add("hidden");
  }
  refreshPersonas();
  return true;
}

async function startLocalTraining(p) {
  const cfg = trainConfigFor(p.name);
  const body = { backend: "local", model_key: cfg.model_key, rank: cfg.rank };
  if (cfg.num_layers) body.num_layers = cfg.num_layers;
  try {
    await api(`/personas/${p.name}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    alert("Couldn't start training: " + err.message);
  }
  refreshPersonas();
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
    const parts = [];
    if (p.trained_on === "colab" || p.merged) parts.push("trained on Colab");
    if (typeof p.best_val === "number") parts.push(`val ${p.best_val.toFixed(2)}`);
    if (p.style && typeof p.style.style_distance === "number") parts.push(`style ${p.style.style_distance.toFixed(2)}`);
    const model = p.model_key || p.size; // p.size: adapters trained before the model_key rename
    if (model) parts.push(model);
    return parts.join(" · ");
  }
  if (p.state === "uploaded") {
    const replies = p.n_train_replies ?? p.n_train;
    return `${replies} replies to learn from · ${p.prompt_style === "retrieval" ? "retrieved examples" : "minimal prompt"}`;
  }
  if (p.state === "training") return "training…";
  if (p.state === "importing") return "importing from Colab…";
  if (p.state === "error") return "last run failed";
  return "";
}

function renderPersona(p) {
  const card = el("div", "persona-card");

  const header = el("div", "row");
  header.appendChild(makeAvatar(displayNameOf(p)));
  const identity = el("div", "identity");
  identity.append(el("div", "name", displayNameOf(p)), el("div", "meta-line", metaLine(p)));
  header.appendChild(identity);
  header.appendChild(el("span", badgeClass(p.state), p.state));
  card.appendChild(header);

  const actions = el("div", "actions");
  if (p.state === "uploaded" || p.state === "error") {
    const train = el("button", "pill-button primary small", "Train locally");
    train.addEventListener("click", () => startLocalTraining(p));

    const colab = el("button", "pill-button small", "Download for Colab");
    colab.addEventListener("click", async () => {
      const cfg = trainConfigFor(p.name);
      colab.disabled = true;
      try {
        const res = await api(`/personas/${p.name}/train`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ backend: "colab", model_key: cfg.model_key, rank: cfg.rank }),
        });
        await downloadSequentially([res.notebook_url, res.train_jsonl_url, res.valid_jsonl_url]);
        $("#colab-overlay").classList.remove("hidden");
      } catch (err) {
        alert("Couldn't build the Colab notebook: " + err.message);
      } finally {
        colab.disabled = false;
      }
    });

    const importInput = document.createElement("input");
    Object.assign(importInput, { type: "file", accept: ".zip", hidden: true, id: `import-${p.name}` });
    const importLabel = el("label", "pill-button small file-button", "Import Colab adapter (.zip)");
    importLabel.htmlFor = importInput.id;
    importInput.addEventListener("change", async () => {
      const file = importInput.files[0];
      if (!file) return;
      const form = new FormData();
      form.append("file", file);
      try {
        await api(`/personas/${p.name}/import_colab`, { method: "POST", body: form });
      } catch (err) {
        alert("Import failed: " + err.message);
      }
      refreshPersonas();
    });
    actions.append(train, colab, importLabel, importInput);
  } else if (p.state === "ready") {
    const chat = el("button", "pill-button primary small", "Chat");
    chat.addEventListener("click", () => openChat(p));
    actions.appendChild(chat);
  }

  const iconActions = el("div", "icon-actions");
  const gearBtn = el("button", "icon-button");
  gearBtn.title = "Settings";
  gearBtn.appendChild(icon("icon-gear"));
  gearBtn.addEventListener("click", () => openSettingsSheet(p));
  iconActions.appendChild(gearBtn);

  if (p.state === "ready") {
    const shareBtn = el("button", "icon-button");
    shareBtn.title = "Copy share link";
    shareBtn.appendChild(icon("icon-share"));
    shareBtn.addEventListener("click", async () => {
      const url = `${window.location.origin}/p/${encodeURIComponent(p.name)}`;
      const ok = await copyToClipboard(url);
      shareBtn.title = ok ? "Copied!" : "Copy failed — link: " + url;
      setTimeout(() => { shareBtn.title = "Copy share link"; }, 2500);
    });
    const downloadBtn = el("button", "icon-button");
    downloadBtn.title = "Download adapter";
    downloadBtn.appendChild(icon("icon-download"));
    downloadBtn.addEventListener("click", () => downloadFile(`${API}/personas/${p.name}/download`));
    iconActions.append(shareBtn, downloadBtn);
  }

  const trashBtn = el("button", "icon-button danger");
  trashBtn.title = "Delete persona";
  trashBtn.appendChild(icon("icon-trash"));
  trashBtn.addEventListener("click", () => deletePersona(p));
  iconActions.appendChild(trashBtn);

  actions.appendChild(iconActions);
  card.appendChild(actions);

  if (p.state === "ready") {
    card.appendChild(el("p", "share-note",
      "Share links work for other devices only while personabot runs with --host 0.0.0.0 on a network they can reach."));
    (p.warnings || []).forEach((w) => card.appendChild(el("p", "warn-note", w)));
  }
  if (p.state === "training" || p.state === "importing") {
    card.appendChild(el("div", "log", (p.log_tail || []).join("\n")));
  }
  if (p.state === "error") {
    card.appendChild(el("div", "log", p.error || ""));
  }
  return card;
}

async function refreshPersonas() {
  const list = await api("/personas");
  if (!sharedMode) {
    const container = $("#persona-list");
    container.replaceChildren();
    if (list.length === 0) container.appendChild(el("p", "status", "No personas yet — upload a chat export above."));
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
  row.replaceChildren();
  if (sharedMode || readyPersonas.length < 2) {
    row.classList.add("hidden");
    return;
  }
  row.classList.remove("hidden");
  readyPersonas.forEach((p) => {
    const item = el("div", "persona-row-item" + (p.name === activePersona ? " active" : ""));
    item.append(makeAvatar(displayNameOf(p)), el("span", "row-label", displayNameOf(p)));
    item.addEventListener("click", () => openChat(p));
    row.appendChild(item);
  });
}

// ==================== Chat ====================
function openChat(p) {
  activePersona = p.name;
  activePersonaDisplay = displayNameOf(p);
  chatHistory = [];
  $("#chat-section").classList.remove("hidden");
  $("#chat-title").textContent = `Chat with ${activePersonaDisplay}`;
  $("#chat-window").replaceChildren();
  if (!sharedMode) {
    $("#chat-section").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
    refreshPersonas();
  }
}

function scrollChat() {
  const win = $("#chat-window");
  win.scrollTop = win.scrollHeight;
}

function addUserBubble(text) {
  const row = el("div", "msg-row me");
  const col = el("div", "msg-col");
  col.appendChild(el("div", "bubble me", text));
  row.appendChild(col);
  $("#chat-window").appendChild(row);
  scrollChat();
}

// One reply = a group of bubbles (the persona can double-text). Actions
// (copy/regenerate) sit under the group's last bubble.
function addReplyGroup() {
  const row = el("div", "msg-row them");
  row.appendChild(makeAvatar(activePersonaDisplay || "?", true));
  const col = el("div", "msg-col");
  const typing = el("div", "bubble them typing");
  typing.setAttribute("aria-label", "typing");
  typing.append(el("span"), el("span"), el("span"));
  col.appendChild(typing);
  row.appendChild(col);
  $("#chat-window").appendChild(row);
  scrollChat();
  return { row, col, typing };
}

async function renderReply(group, text) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  if (!lines.length) lines.push("…");
  group.typing.remove();
  for (let i = 0; i < lines.length; i++) {
    if (i > 0 && !reducedMotion) {
      const typing = el("div", "bubble them typing");
      typing.append(el("span"), el("span"), el("span"));
      group.col.appendChild(typing);
      scrollChat();
      await sleep(Math.min(1400, 350 + lines[i].length * 25));
      typing.remove();
    }
    group.col.appendChild(el("div", "bubble them", lines[i]));
    scrollChat();
  }
}

function addReplyActions(group, text, onRegenerate) {
  const actions = el("div", "msg-actions");
  const copyBtn = el("button");
  copyBtn.title = "Copy";
  copyBtn.appendChild(icon("icon-copy"));
  copyBtn.addEventListener("click", () => copyToClipboard(text));
  const regenBtn = el("button");
  regenBtn.title = "Regenerate";
  regenBtn.dataset.role = "regenerate";
  regenBtn.appendChild(icon("icon-refresh"));
  regenBtn.addEventListener("click", onRegenerate);
  actions.append(copyBtn, regenBtn);
  group.col.appendChild(actions);
}

// Only the most recent reply is regeneratable — regenerating an older one
// would desync it from everything said after it.
function refreshRegenerateButtons() {
  const buttons = document.querySelectorAll('#chat-window .msg-row.them [data-role="regenerate"]');
  buttons.forEach((b, i) => { b.hidden = i !== buttons.length - 1; });
}

function chatPath() {
  return sharedMode ? `/share/${activePersona}/chat` : `/personas/${activePersona}/chat`;
}

// chatHistory always ends with the user message this reply answers; the
// assistant message is appended only once a reply actually arrives.
async function requestReply() {
  if (chatBusy) return;
  chatBusy = true;
  $("#chat-form button").disabled = true;
  const group = addReplyGroup();
  try {
    const res = await api(chatPath(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: chatHistory }),
    });
    await renderReply(group, res.reply);
    chatHistory.push({ role: "assistant", content: res.reply });
    addReplyActions(group, res.reply, () => regenerate(group));
  } catch (err) {
    group.typing.remove();
    group.col.appendChild(el("div", "bubble them error", "Error: " + err.message));
    addReplyActions(group, "", () => regenerate(group));
  } finally {
    chatBusy = false;
    $("#chat-form button").disabled = false;
    refreshRegenerateButtons();
  }
}

async function regenerate(group) {
  if (chatBusy) return;
  if (chatHistory.length && chatHistory[chatHistory.length - 1].role === "assistant") chatHistory.pop();
  group.row.remove();
  if (!chatHistory.length) return;
  await requestReply();
}

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!activePersona || chatBusy) return;
  const input = $("#chat-input");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  addUserBubble(text);
  chatHistory.push({ role: "user", content: text });
  await requestReply();
});

$("#clear-chat-btn").addEventListener("click", () => {
  if (chatBusy) return;
  chatHistory = [];
  $("#chat-window").replaceChildren();
});

// ==================== Boot ====================
async function boot() {
  const sharedName = detectSharedMode();
  if (!sharedName) {
    refreshPersonas();
    return;
  }
  sharedMode = true;
  $("#upload-section").classList.add("hidden");
  $("#personas-section").classList.add("hidden");
  try {
    const p = await api(`/share/${encodeURIComponent(sharedName)}`);
    openChat(p);
  } catch {
    $("#chat-section").classList.remove("hidden");
    $("#chat-title").textContent = "This persona isn't available";
  }
}

boot();
