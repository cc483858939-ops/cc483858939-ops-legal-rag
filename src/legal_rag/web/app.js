const STORAGE_KEY = "legal-rag-evidence-chat-v1";
const LLM_CONFIG_KEY = "legal-rag-llm-config-v1";
const ANSWER_MODE_VERSION = 2;
const DEFAULT_LLM_PRESET = "ollama-qwen35";
const DEFAULT_LLM_TIMEOUT_SECONDS = 45;
const THINKING_LLM_TIMEOUT_SECONDS = 180;

const LLM_PRESETS = {
  "ollama-qwen35": {
    provider: "openai_compatible",
    baseUrl: "http://ollama:11434/v1",
    model: "qwen3.5:9b",
  },
  "deepseek-chat": {
    provider: "openai_compatible",
    baseUrl: "https://api.deepseek.com/v1",
    model: "deepseek-chat",
  },
  "openai-compatible": {
    provider: "openai_compatible",
    baseUrl: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
  },
  openrouter: {
    provider: "openai_compatible",
    baseUrl: "https://openrouter.ai/api/v1",
    model: "openai/gpt-4o-mini",
  },
  siliconflow: {
    provider: "openai_compatible",
    baseUrl: "https://api.siliconflow.cn/v1",
    model: "Qwen/Qwen2.5-7B-Instruct",
  },
  lmstudio: {
    provider: "openai_compatible",
    baseUrl: "http://host.docker.internal:1234/v1",
    model: "local-model",
  },
  custom: {
    provider: "openai_compatible",
    baseUrl: "",
    model: "",
  },
};

const state = {
  mode: "hybrid",
  messages: loadMessages(),
  llmConfig: loadLLMConfig(),
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
  filterDomain: document.querySelector("#filterDomain"),
  filterProduct: document.querySelector("#filterProduct"),
  filterCategory: document.querySelector("#filterCategory"),
  filterLocale: document.querySelector("#filterLocale"),
  answerEnabled: document.querySelector("#answerEnabled"),
  llmPreset: document.querySelector("#llmPreset"),
  llmBaseUrl: document.querySelector("#llmBaseUrl"),
  llmModel: document.querySelector("#llmModel"),
  llmApiKey: document.querySelector("#llmApiKey"),
  llmTemperature: document.querySelector("#llmTemperature"),
  llmMaxTokens: document.querySelector("#llmMaxTokens"),
  thinkingToggle: document.querySelector("#thinkingToggle"),
  questionInput: document.querySelector("#questionInput"),
  queryForm: document.querySelector("#queryForm"),
  sendBtn: document.querySelector("#sendBtn"),
  emptyState: document.querySelector("#emptyState"),
  chatLog: document.querySelector("#chatLog"),
  composerShell: document.querySelector(".composer-shell"),
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

function loadLLMConfig() {
  try {
    const raw = window.localStorage.getItem(LLM_CONFIG_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    if (!parsed || typeof parsed !== "object") return {};
    if (parsed.answerModeVersion !== ANSWER_MODE_VERSION) {
      return { ...parsed, enabled: true, answerModeVersion: ANSWER_MODE_VERSION };
    }
    return parsed;
  } catch {
    return {};
  }
}

function saveLLMConfig() {
  const config = {
    answerModeVersion: ANSWER_MODE_VERSION,
    enabled: Boolean(els.answerEnabled.checked),
    preset: els.llmPreset.value,
    baseUrl: els.llmBaseUrl.value.trim(),
    model: els.llmModel.value.trim(),
    temperature: Number(els.llmTemperature.value || 0.2),
    maxTokens: Number(els.llmMaxTokens.value || 700),
    thinkingEnabled: isThinkingEnabled(),
    timeoutSeconds: answerTimeoutSeconds(),
  };
  window.localStorage.setItem(LLM_CONFIG_KEY, JSON.stringify(config));
}

function normalizeLLMPreset(preset) {
  if (preset === "ollama-gemma4") return DEFAULT_LLM_PRESET;
  return LLM_PRESETS[preset] ? preset : DEFAULT_LLM_PRESET;
}

function migrateLegacyLLMConfig(config) {
  const normalizedPreset = normalizeLLMPreset(config.preset || DEFAULT_LLM_PRESET);
  if (normalizedPreset !== config.preset) {
    const preset = LLM_PRESETS[normalizedPreset];
    return { ...config, preset: normalizedPreset, baseUrl: preset.baseUrl, model: preset.model };
  }
  return { ...config, preset: normalizedPreset };
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
  updateComposerMetrics();
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
          <p>${message.answer ? "正在检索证据并生成回答..." : "正在检索相关证据..."}</p>
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
  if (pack.retrieval_status === "not_run") {
    return `${renderRouteDebug(pack)}${renderGeneratedAnswer(pack.answer)}`;
  }
  const count = Number(pack.hit_count || 0);
  let summary =
    pack.retrieval_status === "model_fallback"
      ? "个人库未命中，模型通用知识回答"
      : pack.retrieval_status === "model_answer"
        ? "模型通用知识回答"
        : pack.retrieval_status === "no_relevant_evidence" && pack.kb_required
          ? "严格知识库问题，证据不足"
          : pack.retrieval_status === "no_relevant_evidence"
            ? "知识库没有找到直接相关材料"
            : count > 0 && pack.answer_source === "personal_kb"
              ? `个人库命中 ${count} 条证据`
      : count > 0
        ? `找到 ${count} 条相关证据`
        : "没有找到相关证据";
  summary = packSummaryLabel(pack);
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
    ${renderRouteDebug(pack)}
    ${renderQueryRewrite(pack.query_rewrite)}
    ${renderQueryExtensions(pack.query_extensions)}
    ${renderMetadataFilters(pack.metadata_filters)}
    ${renderGeneratedAnswer(pack.answer)}
    <div class="evidence-list">
      ${(pack.hits || []).map(renderEvidenceCard).join("")}
    </div>
  `;
}

function packSummaryLabel(pack) {
  const relevantCount = Number(pack.relevant_hit_count ?? pack.hit_count ?? 0);
  const candidateCount = Number(pack.candidate_hit_count ?? pack.hit_count ?? 0);
  if (pack.retrieval_status === "model_fallback") {
    return "个人库未命中，模型按通识回答";
  }
  if (pack.retrieval_status === "model_answer") {
    return "模型通识回答";
  }
  if (pack.retrieval_status === "no_relevant_evidence" && pack.kb_required) {
    return candidateCount > 0
      ? `严格知识库问题，${candidateCount} 条候选均无直接证据`
      : "严格知识库问题，证据不足";
  }
  if (pack.retrieval_status === "no_relevant_evidence") {
    return candidateCount > 0
      ? `个人库返回 ${candidateCount} 条候选，未发现相关证据`
      : "知识库没有找到直接相关材料";
  }
  if (relevantCount > 0 && pack.answer_source === "personal_kb") {
    return `个人库命中 ${relevantCount} 条相关证据`;
  }
  if (candidateCount > 0) {
    return `个人库返回 ${candidateCount} 条候选`;
  }
  return "没有找到相关证据";
}

function renderGeneratedAnswer(answer) {
  if (!answer) return "";
  const llm = answer.llm || {};
  const modelLabel = llm.model ? `${llm.model}` : "system route";
  const status = answer.answer_status || (answer.error ? "error" : answer.refused ? "refused" : "answered");
  const heading = answer.error
    ? "回答模型调用失败"
    : {
        chat: "普通对话",
        clarification: "需要澄清",
        acknowledged: "已收到",
        memory_candidate: "待保存信息",
        answered: "模型回答",
        model_fallback: "个人库未命中，模型回答",
        partial: "基于现有材料",
        refused: "证据不足",
        llm_refused: "模型未能回答",
        error: "回答模型调用失败",
      }[status] || "模型回答";
  const content = answer.answer || answerErrorMessage(answer);
  const citations = Array.isArray(answer.citations) ? answer.citations : [];
  return `
    <section class="answer-panel ${answer.error ? "answer-error" : ""} answer-status-${escapeHtml(status)}">
      <div class="answer-heading">
        <p class="pack-kicker">${escapeHtml(modelLabel)}</p>
        <h3>${escapeHtml(heading)}</h3>
      </div>
      <p class="answer-text">${escapeHtml(content)}</p>
      ${renderGrounding(answer.grounding)}
      ${
        citations.length
          ? `<div class="answer-citations">${citations
              .map((item) => `<em class="query-token">[${escapeHtml(item.rank)}] ${escapeHtml(item.title)}</em>`)
              .join(" ")}</div>`
          : ""
      }
    </section>
  `;
}

function answerErrorMessage(answer) {
  if (answer?.reason === "empty_model_response") {
    return "模型只返回了思考内容或空正文，系统未拿到可展示答案。";
  }
  const error = String(answer?.error || "");
  if (error.includes("ReadTimeout")) {
    return Number(answer?.retry_count || 0) > 0
      ? "模型生成超时，已尝试关闭 Thinking 重试；仍失败请关闭 Thinking 或稍后重试。"
      : "模型生成超时，请关闭 Thinking 或稍后重试。";
  }
  return error || "模型没有返回可展示内容。";
}

function renderRouteDebug(pack) {
  if (!pack || !pack.intent) return "";
  return `
    <details class="rewrite-panel">
      <summary>
        <span>路由决策</span>
        <strong>${escapeHtml(pack.intent)} · ${formatScore(pack.intent_confidence)}</strong>
      </summary>
      <div class="rewrite-body">
        <p><span>Need retrieval</span>${escapeHtml(pack.need_retrieval)}</p>
        <p><span>Answer source</span>${escapeHtml(pack.answer_source || "personal_kb")}</p>
        <p><span>KB required</span>${escapeHtml(pack.kb_required)}</p>
        <p><span>Model fallback</span>${escapeHtml(pack.allow_model_fallback)}</p>
        <p><span>Domain</span>${escapeHtml(pack.domain || "unknown")}</p>
        <p><span>Query type</span>${escapeHtml(pack.query_type || "unknown")}</p>
        <p><span>Strategy</span>${escapeHtml(pack.retrieval_strategy || "hybrid")}</p>
        <p><span>Candidates</span>${escapeHtml(pack.candidate_hit_count ?? pack.hit_count ?? 0)}</p>
        <p><span>Relevant</span>${escapeHtml(pack.relevant_hit_count ?? pack.hit_count ?? 0)}</p>
        <p><span>Priority</span>${escapeHtml(pack.intent_priority_reason || pack.intent_reason || "--")}</p>
        <p><span>Source</span>${escapeHtml(pack.intent_source || "--")}</p>
        ${pack.requires_clarification ? `<p><span>Clarify</span>${escapeHtml(pack.clarification_question || "")}</p>` : ""}
        ${pack.intent_error ? `<p class="rewrite-error">${escapeHtml(pack.intent_error)}</p>` : ""}
      </div>
    </details>
  `;
}

function renderGrounding(grounding) {
  if (!grounding) return "";
  const cues = Array.isArray(grounding.matched_cues) ? grounding.matched_cues : [];
  const missing = Array.isArray(grounding.missing_evidence) ? grounding.missing_evidence : [];
  return `
    <div class="answer-grounding">
      <p><span>问题类型</span>${escapeHtml(grounding.question_type || "--")}</p>
      <p><span>回答模式</span>${escapeHtml(grounding.answer_mode || "--")}</p>
      <p><span>需要证据</span>${escapeHtml(grounding.required_evidence || "--")}</p>
      ${
        cues.length
          ? `<p><span>命中信号</span>${cues.map((cue) => `<em class="query-token">${escapeHtml(cue)}</em>`).join(" ")}</p>`
          : ""
      }
      ${
        missing.length
          ? `<p><span>缺少证据</span>${missing.map((item) => `<em class="query-token rejected-token">${escapeHtml(item)}</em>`).join(" ")}</p>`
          : ""
      }
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
    const manifestName = data.corpus_manifest ? data.corpus_manifest.split(/[\\/]/).pop() : "";
    els.collectionName.textContent =
      data.store_backend === "memory" && manifestName
        ? `memory / ${manifestName}`
        : data.collection || "legal_rag";
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

  const context = collectConversationContext();
  const userMessage = { id: messageId(), role: "user", content: query };
  const assistantId = messageId();
  const answerEnabled = Boolean(els.answerEnabled.checked);
  state.messages.push(userMessage, {
    id: assistantId,
    role: "assistant",
    loading: true,
    answer: answerEnabled,
  });
  els.questionInput.value = "";
  autosizeInput();
  setBusy(true);
  render();
  scrollPageToLatest("auto");

  try {
    const payload = {
      query,
      top_k: Number(els.topK.value),
      mode: state.mode,
      filters: collectMetadataFilters(),
      context,
    };
    const endpoint = answerEnabled ? "/answer" : "/evidence";
    if (answerEnabled) {
      payload.llm = collectLLMConfig();
      saveLLMConfig();
    }
    const pack = await postJson(endpoint, payload);
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

function collectConversationContext() {
  const turns = [];
  for (const message of state.messages.slice(-10)) {
    if (message.loading || message.error) continue;
    if (message.role === "user") {
      turns.push({ role: "user", content: message.content || "" });
    } else if (message.role === "assistant" && message.pack) {
      const answer = message.pack.answer?.answer || "";
      const query = message.pack.query || "";
      turns.push({ role: "assistant", content: answer || `检索问题：${query}` });
    } else if (message.role === "system") {
      turns.push({ role: "system", content: message.content || "" });
    }
  }
  return turns.slice(-5);
}

function collectLLMConfig() {
  return {
    provider: "openai_compatible",
    base_url: els.llmBaseUrl.value.trim(),
    api_key: els.llmApiKey.value.trim() || null,
    model: els.llmModel.value.trim(),
    temperature: Number(els.llmTemperature.value || 0.2),
    max_tokens: Number(els.llmMaxTokens.value || 700),
    thinking_enabled: isThinkingEnabled(),
    timeout_seconds: answerTimeoutSeconds(),
  };
}

function collectMetadataFilters() {
  const filters = {};
  const docType = els.filterDocType.value.trim();
  const jurisdiction = els.filterJurisdiction.value.trim();
  const court = els.filterCourt.value.trim();
  const dateFrom = els.filterDateFrom.value.trim();
  const dateTo = els.filterDateTo.value.trim();
  const domain = els.filterDomain.value.trim();
  const product = els.filterProduct.value.trim();
  const category = els.filterCategory.value.trim();
  const locale = els.filterLocale.value.trim();
  if (docType) filters.doc_type = docType;
  if (jurisdiction) filters.jurisdiction = jurisdiction;
  if (court) filters.court = court;
  if (dateFrom) filters.date_from = dateFrom;
  if (dateTo) filters.date_to = dateTo;
  if (domain) filters.domain = domain;
  if (product) filters.product = product;
  if (category) filters.category = category;
  if (locale) filters.locale = locale;
  return filters;
}

function scrollPageToLatest(behavior = "smooth") {
  scrollChatToBottom(behavior);
}

function scrollChatToBottom(behavior = "smooth") {
  const scrollToBottom = () => {
    updateComposerMetrics();
    if (els.chatLog && !els.chatLog.hidden) {
      const top = els.chatLog.scrollHeight;
      els.chatLog.scrollTo({ top, behavior });
      if (behavior === "auto") {
        els.chatLog.scrollTop = top;
      }
      return;
    }
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior });
  };
  window.requestAnimationFrame(() => {
    scrollToBottom();
    window.requestAnimationFrame(scrollToBottom);
    window.setTimeout(scrollToBottom, 80);
  });
}

function autosizeInput() {
  els.questionInput.style.height = "auto";
  els.questionInput.style.height = `${Math.min(180, els.questionInput.scrollHeight)}px`;
  updateComposerMetrics();
}

function updateComposerMetrics() {
  if (!els.composerShell) return;
  window.requestAnimationFrame(() => {
    const composerHeight = Math.ceil(els.composerShell.getBoundingClientRect().height);
    const space = Math.max(140, composerHeight + 28);
    document.documentElement.style.setProperty("--composer-space", `${space}px`);
  });
}

function initControls() {
  initLLMControls();

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

  document.querySelectorAll(".advanced-panel").forEach((panel) => {
    panel.addEventListener("toggle", updateComposerMetrics);
  });

  window.addEventListener("resize", updateComposerMetrics);
}

function initLLMControls() {
  const config = migrateLegacyLLMConfig(state.llmConfig);
  state.llmConfig = config;
  if (
    (config.preset || DEFAULT_LLM_PRESET) === DEFAULT_LLM_PRESET &&
    config.baseUrl === "http://host.docker.internal:11434/v1"
  ) {
    config.baseUrl = LLM_PRESETS[DEFAULT_LLM_PRESET].baseUrl;
  }
  els.answerEnabled.checked = config.enabled !== false;
  els.llmPreset.value = config.preset || DEFAULT_LLM_PRESET;
  applyLLMPreset({ preserveCustom: true });
  if (config.baseUrl) els.llmBaseUrl.value = config.baseUrl;
  if (config.model) els.llmModel.value = config.model;
  if (config.temperature !== undefined) els.llmTemperature.value = config.temperature;
  if (config.maxTokens !== undefined) els.llmMaxTokens.value = config.maxTokens;
  setThinkingEnabled(config.thinkingEnabled === true);

  els.llmPreset.addEventListener("change", () => {
    applyLLMPreset();
    saveLLMConfig();
  });
  els.thinkingToggle.addEventListener("click", () => {
    setThinkingEnabled(!isThinkingEnabled());
    saveLLMConfig();
  });
  [els.answerEnabled, els.llmBaseUrl, els.llmModel, els.llmTemperature, els.llmMaxTokens].forEach((item) => {
    item.addEventListener("input", saveLLMConfig);
    item.addEventListener("change", saveLLMConfig);
  });
}

function isThinkingEnabled() {
  return els.thinkingToggle?.getAttribute("aria-pressed") === "true";
}

function setThinkingEnabled(enabled) {
  if (!els.thinkingToggle) return;
  els.thinkingToggle.setAttribute("aria-pressed", enabled ? "true" : "false");
  els.thinkingToggle.title = enabled
    ? `回答时启用模型思考，超时上限 ${THINKING_LLM_TIMEOUT_SECONDS} 秒`
    : "回答时启用模型思考";
}

function answerTimeoutSeconds() {
  return isThinkingEnabled() ? THINKING_LLM_TIMEOUT_SECONDS : DEFAULT_LLM_TIMEOUT_SECONDS;
}

function applyLLMPreset({ preserveCustom = false } = {}) {
  const preset = LLM_PRESETS[els.llmPreset.value] || LLM_PRESETS.custom;
  if (preserveCustom && state.llmConfig.baseUrl && state.llmConfig.model) return;
  els.llmBaseUrl.value = preset.baseUrl;
  els.llmModel.value = preset.model;
}

initControls();
refreshHealth();
render();
autosizeInput();
