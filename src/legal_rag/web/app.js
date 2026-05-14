const STORAGE_KEY = "legal-rag-evidence-chat-v1";

const state = {
  mode: "hybrid",
  messages: loadMessages(),
};

const els = {
  apiStatus: document.querySelector("#apiStatus"),
  collectionName: document.querySelector("#collectionName"),
  ingestBtn: document.querySelector("#ingestBtn"),
  clearBtn: document.querySelector("#clearBtn"),
  topK: document.querySelector("#topK"),
  topKValue: document.querySelector("#topKValue"),
  modeReadout: document.querySelector("#modeReadout"),
  filterDocType: document.querySelector("#filterDocType"),
  filterJurisdiction: document.querySelector("#filterJurisdiction"),
  filterCourt: document.querySelector("#filterCourt"),
  filterDateFrom: document.querySelector("#filterDateFrom"),
  filterDateTo: document.querySelector("#filterDateTo"),
  questionInput: document.querySelector("#questionInput"),
  queryForm: document.querySelector("#queryForm"),
  sendBtn: document.querySelector("#sendBtn"),
  emptyState: document.querySelector("#emptyState"),
  chatLog: document.querySelector("#chatLog"),
};

function loadMessages() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function saveMessages() {
  const settledMessages = state.messages.filter((message) => !message.loading);
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settledMessages.slice(-20)));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatScore(value) {
  if (value === null || value === undefined) return "--";
  return Number(value).toFixed(4);
}

function messageId() {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    const message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data);
    throw new Error(message);
  }
  return data;
}

function setBusy(isBusy) {
  els.sendBtn.disabled = isBusy;
  els.ingestBtn.disabled = isBusy;
  document.body.classList.toggle("is-busy", isBusy);
}

function render() {
  const hasMessages = state.messages.length > 0;
  document.body.classList.toggle("has-chat", hasMessages);
  els.emptyState.hidden = hasMessages;
  els.chatLog.hidden = !hasMessages;
  els.chatLog.innerHTML = state.messages.map(renderMessage).join("");
}

function renderMessage(message) {
  if (message.role === "user") {
    return `
      <article class="message-row user-row">
        <div class="message-bubble user-bubble">${escapeHtml(message.content)}</div>
      </article>
    `;
  }

  if (message.role === "system") {
    return `
      <article class="system-note ${message.error ? "system-error" : ""}">
        ${escapeHtml(message.content)}
      </article>
    `;
  }

  if (message.loading) {
    return `
      <article class="message-row assistant-row">
        <div class="assistant-avatar">R</div>
        <div class="message-bubble assistant-bubble">
          <div class="loading-line"><span></span><span></span><span></span></div>
          <p>正在检索相关证据...</p>
        </div>
      </article>
    `;
  }

  if (message.error) {
    return `
      <article class="message-row assistant-row">
        <div class="assistant-avatar">R</div>
        <div class="message-bubble assistant-bubble error-bubble">
          <strong>检索失败</strong>
          <p>${escapeHtml(message.error)}</p>
        </div>
      </article>
    `;
  }

  return `
    <article class="message-row assistant-row">
      <div class="assistant-avatar">R</div>
      <div class="message-bubble assistant-bubble">
        ${renderEvidencePack(message.pack)}
      </div>
    </article>
  `;
}

function renderEvidencePack(pack) {
  const count = Number(pack.hit_count || 0);
  const summary = count > 0 ? `找到 ${count} 条相关证据` : "没有找到相关证据";
  return `
    <div class="pack-heading">
      <div>
        <p class="pack-kicker">RAG evidence pack</p>
        <h2>${summary}</h2>
      </div>
      <div class="pack-meta">
        <span>${escapeHtml(pack.mode || state.mode).toUpperCase()}</span>
        <span>Top ${escapeHtml(pack.top_k)}</span>
      </div>
    </div>
    <p class="pack-query">${escapeHtml(pack.query)}</p>
    ${renderQueryRewrite(pack.query_rewrite)}
    ${renderQueryExtensions(pack.query_extensions)}
    ${renderMetadataFilters(pack.metadata_filters)}
    <div class="evidence-list">
      ${(pack.hits || []).map(renderEvidenceCard).join("")}
    </div>
  `;
}

function renderQueryRewrite(rewrite) {
  if (!rewrite) return "";
  const hasRewrite = Boolean(rewrite.applied);
  const hasError = Boolean(rewrite.error);
  if (!hasRewrite && !hasError) return "";

  const queries = Array.isArray(rewrite.search_queries) ? rewrite.search_queries : [];
  const aliases = Array.isArray(rewrite.aliases) ? rewrite.aliases : [];
  const title = hasRewrite ? "模型已改写检索词" : "模型改写不可用，已使用原始问题检索";
  return `
    <details class="rewrite-panel" ${hasError ? "" : "open"}>
      <summary>
        <span>${escapeHtml(title)}</span>
        <strong>${escapeHtml(rewrite.backend || "rewrite")}</strong>
      </summary>
      <div class="rewrite-body">
        ${hasError ? `<p class="rewrite-error">${escapeHtml(rewrite.error)}</p>` : ""}
        ${rewrite.canonical_query ? `<p><span>Canonical</span>${escapeHtml(rewrite.canonical_query)}</p>` : ""}
        <p><span>Retrieval query</span>${escapeHtml(rewrite.retrieval_query || "")}</p>
        ${queries.length ? `<p><span>Search queries</span>${queries.map(escapeHtml).join(" · ")}</p>` : ""}
        ${aliases.length ? `<p><span>Aliases</span>${aliases.map(escapeHtml).join(" · ")}</p>` : ""}
      </div>
    </details>
  `;
}

function renderQueryExtensions(extensions) {
  if (!extensions) return "";
  const accepted = Array.isArray(extensions.accepted) ? extensions.accepted : [];
  const rejected = Array.isArray(extensions.rejected) ? extensions.rejected : [];
  if (!accepted.length && !rejected.length && !extensions.error) return "";

  return `
    <details class="rewrite-panel">
      <summary>
        <span>扩展 query 语义过滤</span>
        <strong>${accepted.length} kept / ${rejected.length} dropped</strong>
      </summary>
      <div class="rewrite-body">
        <p><span>Threshold</span>${formatScore(extensions.min_similarity)}</p>
        ${extensions.error ? `<p class="rewrite-error">${escapeHtml(extensions.error)}</p>` : ""}
        ${accepted.length ? `<p><span>Accepted</span>${accepted.map(renderExtensionChip).join(" ")}</p>` : ""}
        ${rejected.length ? `<p><span>Rejected</span>${rejected.map(renderRejectedExtension).join(" ")}</p>` : ""}
      </div>
    </details>
  `;
}

function renderExtensionChip(item) {
  return `<em class="query-token">${escapeHtml(item.query)} · ${formatScore(item.similarity)}</em>`;
}

function renderRejectedExtension(item) {
  return `<em class="query-token rejected-token">${escapeHtml(item.query)} · ${formatScore(item.similarity)} · ${escapeHtml(item.reason)}</em>`;
}

function renderMetadataFilters(metadata) {
  if (!metadata) return "";
  const effective = metadata.effective_filters || {};
  const discarded = Array.isArray(metadata.discarded_filters) ? metadata.discarded_filters : [];
  const effectiveTokens = Object.entries(effective).flatMap(([key, value]) => {
    const values = Array.isArray(value) ? value : [value];
    return values.map((item) => `${key}: ${item}`);
  });
  if (!effectiveTokens.length && !discarded.length) return "";

  return `
    <details class="rewrite-panel">
      <summary>
        <span>元数据过滤</span>
        <strong>${effectiveTokens.length} active / ${discarded.length} dropped</strong>
      </summary>
      <div class="rewrite-body">
        ${effectiveTokens.length ? `<p><span>Effective</span>${effectiveTokens.map(renderFilterToken).join(" ")}</p>` : ""}
        ${discarded.length ? `<p><span>Discarded</span>${discarded.map(renderDiscardedFilter).join(" ")}</p>` : ""}
      </div>
    </details>
  `;
}

function renderFilterToken(label) {
  return `<em class="query-token filter-token">${escapeHtml(label)}</em>`;
}

function renderDiscardedFilter(item) {
  return `<em class="query-token rejected-token">${escapeHtml(item.key)} · ${escapeHtml(item.reason)}</em>`;
}

function renderEvidenceCard(hit) {
  return `
    <details class="evidence-card">
      <summary>
        <span class="rank-badge">${escapeHtml(hit.rank)}</span>
        <span class="card-title">
          <strong>${escapeHtml(hit.title)}</strong>
          <span>${escapeHtml(hit.citation)}</span>
        </span>
        <span class="doc-type">${escapeHtml(hit.doc_type)}</span>
        <span class="final-score">${formatScore(hit.final_score ?? hit.scores?.final)}</span>
      </summary>
      <div class="card-body">
        <p class="snippet">${escapeHtml(hit.snippet)}</p>
        <div class="meta-grid">
          <div><span>Source</span><strong>${escapeHtml(hit.source_id)}</strong></div>
          <div><span>Jurisdiction</span><strong>${escapeHtml(hit.jurisdiction)}</strong></div>
          <div><span>Court</span><strong>${escapeHtml(hit.court || "--")}</strong></div>
          <div><span>Section</span><strong>${escapeHtml(hit.section || "--")}</strong></div>
        </div>
        ${renderScores(hit.scores || {})}
        <details class="raw-details">
          <summary>完整 chunk</summary>
          <p>${escapeHtml(hit.text)}</p>
        </details>
        <details class="raw-details">
          <summary>排序解释</summary>
          <pre>${escapeHtml(JSON.stringify(hit.rank_explanation || {}, null, 2))}</pre>
        </details>
      </div>
    </details>
  `;
}

function renderScores(scores) {
  return `
    <div class="score-grid">
      <div><span>Dense</span><strong>${formatScore(scores.dense)}</strong></div>
      <div><span>BM25</span><strong>${formatScore(scores.bm25)}</strong></div>
      <div><span>Fusion</span><strong>${formatScore(scores.fusion)}</strong></div>
      <div><span>Rerank</span><strong>${formatScore(scores.rerank)}</strong></div>
    </div>
  `;
}

async function refreshHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    els.apiStatus.textContent = data.status || "ok";
    els.collectionName.textContent = data.collection || "legal_rag";
    document.body.classList.remove("api-offline");
  } catch {
    els.apiStatus.textContent = "offline";
    document.body.classList.add("api-offline");
  }
}

async function ingestCorpus() {
  setBusy(true);
  try {
    const data = await postJson("/ingest", {});
    renderSystemNote(`已导入 ${data.chunks} 个 chunks`);
  } catch (error) {
    renderSystemNote(`导入失败：${error.message}`, true);
  } finally {
    setBusy(false);
  }
}

function renderSystemNote(text, isError = false) {
  state.messages.push({
    id: messageId(),
    role: "system",
    content: text,
    error: isError,
  });
  saveMessages();
  render();
}

async function submitQuery() {
  const query = els.questionInput.value.trim();
  if (!query) return;

  const userMessage = { id: messageId(), role: "user", content: query };
  const assistantId = messageId();
  state.messages.push(userMessage, { id: assistantId, role: "assistant", loading: true });
  els.questionInput.value = "";
  autosizeInput();
  setBusy(true);
  render();

  try {
    const pack = await postJson("/evidence", {
      query,
      top_k: Number(els.topK.value),
      mode: state.mode,
      filters: collectMetadataFilters(),
    });
    const index = state.messages.findIndex((message) => message.id === assistantId);
    state.messages[index] = { id: assistantId, role: "assistant", pack };
  } catch (error) {
    const index = state.messages.findIndex((message) => message.id === assistantId);
    state.messages[index] = { id: assistantId, role: "assistant", error: error.message };
  } finally {
    setBusy(false);
    saveMessages();
    render();
    scrollPageToLatest();
  }
}

function collectMetadataFilters() {
  const filters = {};
  const docType = els.filterDocType.value.trim();
  const jurisdiction = els.filterJurisdiction.value.trim();
  const court = els.filterCourt.value.trim();
  const dateFrom = els.filterDateFrom.value.trim();
  const dateTo = els.filterDateTo.value.trim();
  if (docType) filters.doc_type = docType;
  if (jurisdiction) filters.jurisdiction = jurisdiction;
  if (court) filters.court = court;
  if (dateFrom) filters.date_from = dateFrom;
  if (dateTo) filters.date_to = dateTo;
  return filters;
}

function scrollPageToLatest() {
  window.requestAnimationFrame(() => {
    window.scrollTo({
      top: document.documentElement.scrollHeight,
      behavior: "smooth",
    });
  });
}

function autosizeInput() {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(180, els.questionInput.scrollHeight)}px`;
}

function initControls() {
  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      state.mode = button.dataset.mode;
      els.modeReadout.textContent = button.textContent;
      document.querySelectorAll(".segment").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
    });
  });

  document.querySelectorAll(".sample-chip").forEach((button) => {
    button.addEventListener("click", () => {
      els.questionInput.value = button.dataset.sample;
      autosizeInput();
      els.questionInput.focus();
    });
  });

  els.topK.addEventListener("input", () => {
    els.topKValue.textContent = els.topK.value;
  });

  els.questionInput.addEventListener("input", autosizeInput);
  els.questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitQuery();
    }
  });

  els.queryForm.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuery();
  });

  els.ingestBtn.addEventListener("click", ingestCorpus);
  els.clearBtn.addEventListener("click", () => {
    state.messages = [];
    saveMessages();
    render();
    els.questionInput.focus();
  });
}

initControls();
refreshHealth();
render();
autosizeInput();
