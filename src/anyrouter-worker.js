const DEFAULT_UPSTREAM_BASE_URL = "https://anyrouter.top";
const DEFAULT_MAX_TOKENS = 4096;

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
  "Access-Control-Allow-Headers": "authorization,content-type,x-api-key,api-key,anthropic-api-key,anthropic-version,anthropic-beta,openai-beta",
  "Access-Control-Expose-Headers": "content-type,request-id,x-request-id",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = normalizePath(url.pathname);

    try {
      const authError = validateWorkerApiKey(request, env);
      if (authError) return authError;

      if (path === "/" || path === "/health" || path === "/v1/health") {
        return jsonResponse(serviceInfo(env));
      }

      if ((path === "/v1/models" || path === "/anthropic/v1/models") && request.method === "GET") {
        const anthropic = path.startsWith("/anthropic/") || looksLikeAnthropicRequest(request);
        return handleModels(request, env, anthropic);
      }

      if (path === "/v1/chat/completions" && request.method === "POST") {
        const body = await readJson(request);
        return handleOpenAIChat(request, env, body);
      }

      if (path === "/v1/responses" && request.method === "POST") {
        const body = await readJson(request);
        return handleOpenAIResponses(request, env, body);
      }

      if ((path === "/v1/messages" || path === "/anthropic/v1/messages") && request.method === "POST") {
        const body = await readJson(request);
        return handleAnthropicMessages(request, env, body);
      }

      return errorResponse(404, "not_found", `No route for ${path}`);
    } catch (error) {
      return toErrorResponse(error);
    }
  },
};

async function handleModels(request, env, anthropicFormat) {
  const upstream = await upstreamJson(request, env, "/v1/models", {
    method: "GET",
    headers: buildUpstreamHeaders(request, env, { Accept: "application/json" }),
  });

  const models = normalizeModelObjects(upstream.data);
  const filteredModels = applyModelFilters(models, env);

  if (anthropicFormat) {
    return jsonResponse({
      data: filteredModels.map((model) => ({
        type: "model",
        id: model.id,
        display_name: model.id,
        created_at: "2025-01-01T00:00:00Z",
      })),
      has_more: false,
      first_id: filteredModels[0]?.id || null,
      last_id: filteredModels[filteredModels.length - 1]?.id || null,
    });
  }

  return jsonResponse({
    object: "list",
    data: filteredModels.map((model) => ({
      object: "model",
      created: 1704067200,
      owned_by: "anyrouter-worker",
      ...model,
    })),
  });
}

async function handleOpenAIChat(request, env, body) {
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return errorResponse(400, "invalid_request_error", "messages must be a non-empty array");
  }

  const requestedModel = body.model || env.DEFAULT_MODEL || null;
  const allowSyntheticStream = body.stream === true;
  const preferAnthropic = shouldPreferAnthropic(requestedModel, env);
  const attempts = preferAnthropic
    ? [chatViaAnthropic, chatViaOpenAI]
    : [chatViaOpenAI, ...(looksLikeClaudeModel(requestedModel) ? [chatViaAnthropic] : [])];

  let lastError = null;
  for (const attempt of attempts) {
    try {
      const completion = await attempt(request, env, body);
      if (allowSyntheticStream) {
        return openAICompletionStreamResponse(completion);
      }
      return jsonResponse(completion);
    } catch (error) {
      lastError = error;
      if (!shouldTryFallback(error)) {
        throw error;
      }
    }
  }

  throw lastError || createHttpError(502, "bad_gateway", "Upstream request failed");
}

async function handleOpenAIResponses(request, env, body) {
  if (body.stream) {
    return errorResponse(501, "unsupported_stream", "/v1/responses stream mode is not enabled in this Worker");
  }

  const chatBody = responsesToChatBody(body, env);
  const completion = await handleChatAttempt(request, env, chatBody);
  return jsonResponse({
    id: `resp_${randomId()}`,
    object: "response",
    created_at: nowSeconds(),
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: body.instructions || null,
    max_output_tokens: body.max_output_tokens || body.max_tokens || null,
    model: completion.model,
    output: [
      {
        id: `msg_${randomId()}`,
        type: "message",
        status: "completed",
        role: "assistant",
        content: [{ type: "output_text", text: completion.choices[0].message.content, annotations: [] }],
      },
    ],
    output_text: completion.choices[0].message.content,
    parallel_tool_calls: true,
    previous_response_id: body.previous_response_id || null,
    reasoning: body.reasoning || null,
    store: body.store || false,
    temperature: body.temperature || null,
    text: body.text || { format: { type: "text" } },
    tool_choice: body.tool_choice || "auto",
    tools: body.tools || [],
    top_p: body.top_p || null,
    truncation: body.truncation || "disabled",
    usage: {
      input_tokens: completion.usage?.prompt_tokens || 0,
      output_tokens: completion.usage?.completion_tokens || 0,
      total_tokens: completion.usage?.total_tokens || 0,
    },
    user: body.user || null,
  });
}

async function handleAnthropicMessages(request, env, body) {
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    return errorResponse(400, "invalid_request_error", "messages must be a non-empty array");
  }

  const normalizedBody = {
    model: body.model || env.DEFAULT_MODEL || null,
    max_tokens: body.max_tokens || Number(env.DEFAULT_MAX_TOKENS) || DEFAULT_MAX_TOKENS,
    temperature: body.temperature,
    top_p: body.top_p,
    system: body.system,
    messages: normalizeAnthropicMessages(body.messages),
    stop_sequences: Array.isArray(body.stop_sequences) ? body.stop_sequences : undefined,
    metadata: body.metadata,
    tools: Array.isArray(body.tools) ? body.tools : undefined,
    tool_choice: body.tool_choice,
    stream: false,
  };

  const message = await callAnthropicMessagesUpstream(request, env, normalizedBody);
  if (body.stream) {
    return anthropicMessageStreamResponse(message);
  }
  return jsonResponse(message);
}

async function handleChatAttempt(request, env, body) {
  const requestedModel = body.model || env.DEFAULT_MODEL || null;
  const preferAnthropic = shouldPreferAnthropic(requestedModel, env);
  const attempts = preferAnthropic
    ? [chatViaAnthropic, chatViaOpenAI]
    : [chatViaOpenAI, ...(looksLikeClaudeModel(requestedModel) ? [chatViaAnthropic] : [])];

  let lastError = null;
  for (const attempt of attempts) {
    try {
      return await attempt(request, env, body);
    } catch (error) {
      lastError = error;
      if (!shouldTryFallback(error)) {
        throw error;
      }
    }
  }

  throw lastError || createHttpError(502, "bad_gateway", "Upstream request failed");
}

async function chatViaOpenAI(request, env, body) {
  const upstreamBody = {
    ...body,
    model: body.model || env.DEFAULT_MODEL || null,
    stream: false,
  };

  const upstream = await upstreamJson(request, env, "/v1/chat/completions", {
    method: "POST",
    headers: buildUpstreamHeaders(request, env, { "Content-Type": "application/json", Accept: "application/json" }),
    body: JSON.stringify(upstreamBody),
  });

  if (!upstream.data || !Array.isArray(upstream.data.choices)) {
    throw createHttpError(502, "bad_gateway", "Upstream OpenAI response was not a valid chat completion", upstream.data);
  }

  return upstream.data;
}

async function chatViaAnthropic(request, env, body) {
  const anthropicRequest = openAIChatToAnthropicBody(body, env);
  const message = await callAnthropicMessagesUpstream(request, env, anthropicRequest);
  return anthropicMessageToOpenAICompletion(message, body.model || message.model || env.DEFAULT_MODEL || "claude");
}

async function callAnthropicMessagesUpstream(request, env, body) {
  const upstream = await upstreamJson(request, env, "/v1/messages", {
    method: "POST",
    headers: buildUpstreamHeaders(request, env, {
      "Content-Type": "application/json",
      Accept: "application/json",
      "anthropic-version": request.headers.get("anthropic-version") || "2023-06-01",
    }),
    body: JSON.stringify(body),
  });

  if (!upstream.data || upstream.data.type !== "message" || !Array.isArray(upstream.data.content)) {
    throw createHttpError(502, "bad_gateway", "Upstream Anthropic response was not a valid message", upstream.data);
  }

  return upstream.data;
}

async function upstreamJson(request, env, path, init) {
  const response = await fetch(buildUpstreamUrl(env, path), init);
  const contentType = response.headers.get("content-type") || "";
  const rawText = await response.text();

  if (isHtmlChallenge(contentType, rawText)) {
    throw createHttpError(502, "upstream_html_challenge", "Upstream returned an HTML challenge page instead of JSON");
  }

  const data = parseMaybeJson(rawText);
  if (!response.ok) {
    throw createHttpError(response.status, inferErrorCode(data, response.status), extractErrorMessage(data, response.status), data);
  }

  if (data && typeof data === "object" && data.error && !Array.isArray(data.choices) && data.type === "error") {
    throw createHttpError(502, inferErrorCode(data, 502), extractErrorMessage(data, 502), data);
  }

  return { response, data, rawText };
}

function openAIChatToAnthropicBody(body, env) {
  const normalized = normalizeOpenAIMessages(body.messages || []);
  return {
    model: body.model || env.DEFAULT_MODEL || null,
    max_tokens: body.max_completion_tokens || body.max_tokens || Number(env.DEFAULT_MAX_TOKENS) || DEFAULT_MAX_TOKENS,
    temperature: body.temperature,
    top_p: body.top_p,
    system: normalized.system || undefined,
    messages: normalized.messages,
    stop_sequences: Array.isArray(body.stop) ? body.stop : typeof body.stop === "string" ? [body.stop] : undefined,
    metadata: body.metadata,
    stream: false,
  };
}

function normalizeOpenAIMessages(messages) {
  const systemParts = [];
  const normalized = [];

  for (const message of messages) {
    const role = message?.role;
    const text = extractTextContent(message?.content);

    if (role === "system") {
      if (text) systemParts.push(text);
      continue;
    }

    let targetRole = role === "assistant" ? "assistant" : "user";
    let targetText = text;

    if (role === "tool") {
      targetRole = "user";
      targetText = `Tool result${message.name ? ` (${message.name})` : ""}: ${text}`;
    }

    if (Array.isArray(message?.tool_calls) && message.tool_calls.length > 0) {
      const toolSummary = JSON.stringify(message.tool_calls);
      targetText = targetText ? `${targetText}\n\nTool calls: ${toolSummary}` : `Tool calls: ${toolSummary}`;
    }

    pushMergedMessage(normalized, targetRole, targetText || "");
  }

  if (normalized.length === 0) {
    normalized.push({ role: "user", content: [{ type: "text", text: "Hello" }] });
  }

  return {
    system: systemParts.join("\n\n") || null,
    messages: normalized,
  };
}

function normalizeAnthropicMessages(messages) {
  const normalized = [];
  for (const message of messages) {
    const role = message?.role === "assistant" ? "assistant" : "user";
    const text = extractTextContent(message?.content);
    pushMergedMessage(normalized, role, text || "");
  }
  return normalized.length > 0 ? normalized : [{ role: "user", content: [{ type: "text", text: "Hello" }] }];
}

function pushMergedMessage(target, role, text) {
  const trimmed = typeof text === "string" ? text : String(text || "");
  const last = target[target.length - 1];
  if (last && last.role === role) {
    last.content[0].text += `${last.content[0].text ? "\n\n" : ""}${trimmed}`;
    return;
  }
  target.push({ role, content: [{ type: "text", text: trimmed }] });
}

function anthropicMessageToOpenAICompletion(message, fallbackModel) {
  const text = extractTextContent(message.content);
  const model = message.model || fallbackModel;
  const promptTokens = message.usage?.input_tokens || 0;
  const completionTokens = message.usage?.output_tokens || 0;
  return {
    id: message.id || `chatcmpl_${randomId()}`,
    object: "chat.completion",
    created: nowSeconds(),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        logprobs: null,
        finish_reason: mapAnthropicStopReason(message.stop_reason),
      },
    ],
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    },
    system_fingerprint: "anyrouter-worker",
  };
}

function responsesToChatBody(body, env) {
  return {
    model: body.model || env.DEFAULT_MODEL || null,
    messages: [{ role: "user", content: extractResponsesInput(body.input) }],
    max_tokens: body.max_output_tokens || body.max_tokens,
    temperature: body.temperature,
    top_p: body.top_p,
    stream: false,
    metadata: body.metadata,
  };
}

function extractResponsesInput(input) {
  if (typeof input === "string") return input;
  if (Array.isArray(input)) {
    return input
      .map((item) => {
        if (typeof item === "string") return item;
        if (item?.content) return extractTextContent(item.content);
        if (item?.text) return item.text;
        return JSON.stringify(item);
      })
      .join("\n\n");
  }
  if (input && typeof input === "object") {
    if (input.content) return extractTextContent(input.content);
    return JSON.stringify(input);
  }
  return "Hello";
}

function normalizeModelObjects(data) {
  const items = Array.isArray(data?.data) ? data.data : Array.isArray(data) ? data : [];
  return items
    .map((item) => {
      if (typeof item === "string") return { id: item };
      if (item && typeof item.id === "string") return item;
      if (item && typeof item.name === "string") return { ...item, id: item.name };
      return null;
    })
    .filter(Boolean);
}

function applyModelFilters(models, env) {
  const allowlist = parseListEnv(env.MODEL_ALLOWLIST);
  const blocklist = parseListEnv(env.MODEL_BLOCKLIST);
  return models.filter((model) => {
    if (allowlist.length > 0 && !allowlist.includes(model.id)) return false;
    if (blocklist.includes(model.id)) return false;
    return true;
  });
}

function parseListEnv(value) {
  if (!value || typeof value !== "string") return [];
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function validateWorkerApiKey(request, env) {
  const configured = env.WORKER_API_KEY;
  if (!configured) return null;

  const clientKey = getClientApiKey(request);
  if (clientKey === configured) return null;
  return errorResponse(401, "invalid_api_key", "Invalid Worker API key");
}

function buildUpstreamHeaders(request, env, extra = {}) {
  const upstreamKey = env.ANYROUTER_API_KEY || getClientApiKey(request);
  if (!upstreamKey) {
    throw createHttpError(401, "missing_api_key", "Missing upstream API key. Set ANYROUTER_API_KEY or pass a client key.");
  }

  const headers = {
    Authorization: `Bearer ${upstreamKey}`,
    "x-api-key": upstreamKey,
    "api-key": upstreamKey,
    "anthropic-api-key": upstreamKey,
    "anthropic-version": request.headers.get("anthropic-version") || "2023-06-01",
    Accept: "application/json",
    ...extra,
  };

  const anthropicBeta = request.headers.get("anthropic-beta");
  if (anthropicBeta) headers["anthropic-beta"] = anthropicBeta;

  const openAIBeta = request.headers.get("openai-beta");
  if (openAIBeta) headers["openai-beta"] = openAIBeta;

  return headers;
}

function getClientApiKey(request) {
  const auth = request.headers.get("authorization") || "";
  const bearer = auth.match(/^Bearer\s+(.+)$/i);
  if (bearer) return bearer[1].trim();

  return (
    request.headers.get("x-api-key") ||
    request.headers.get("api-key") ||
    request.headers.get("anthropic-api-key") ||
    ""
  ).trim();
}

function shouldPreferAnthropic(model, env) {
  if (env.PREFER_ANTHROPIC_FOR_CLAUDE === "0") return false;
  return looksLikeClaudeModel(model);
}

function looksLikeClaudeModel(model) {
  return typeof model === "string" && /claude/i.test(model);
}

function shouldTryFallback(error) {
  const status = error?.status || 0;
  const message = String(error?.message || "");
  const code = String(error?.code || "");

  if ([404, 409, 429, 500, 502, 503, 504].includes(status)) return true;
  if (/unsupported|not support|不支持|service unavailable/i.test(message)) return true;
  if (/unsupported|unavailable/i.test(code)) return true;
  return false;
}

function looksLikeAnthropicRequest(request) {
  return Boolean(request.headers.get("anthropic-version") || request.headers.get("anthropic-api-key"));
}

function openAICompletionStreamResponse(completion) {
  const created = completion.created || nowSeconds();
  const model = completion.model || "unknown";
  const id = completion.id || `chatcmpl_${randomId()}`;
  const text = completion.choices?.[0]?.message?.content || "";

  const chunks = [
    { id, object: "chat.completion.chunk", created, model, choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }] },
    { id, object: "chat.completion.chunk", created, model, choices: [{ index: 0, delta: { content: text }, finish_reason: null }] },
    { id, object: "chat.completion.chunk", created, model, choices: [{ index: 0, delta: {}, finish_reason: completion.choices?.[0]?.finish_reason || "stop" }] },
  ];

  return sseResponse([
    ...chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`),
    "data: [DONE]\n\n",
  ]);
}

function anthropicMessageStreamResponse(message) {
  const text = extractTextContent(message.content);
  const startEvent = {
    type: "message_start",
    message: {
      id: message.id || `msg_${randomId()}`,
      type: "message",
      role: "assistant",
      model: message.model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: message.usage?.input_tokens || 0, output_tokens: 0 },
    },
  };
  const blockStart = { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } };
  const blockDelta = { type: "content_block_delta", index: 0, delta: { type: "text_delta", text } };
  const blockStop = { type: "content_block_stop", index: 0 };
  const messageDelta = {
    type: "message_delta",
    delta: { stop_reason: message.stop_reason || "end_turn", stop_sequence: message.stop_sequence || null },
    usage: { output_tokens: message.usage?.output_tokens || 0 },
  };
  const messageStop = { type: "message_stop" };

  return sseResponse([
    `event: message_start\ndata: ${JSON.stringify(startEvent)}\n\n`,
    `event: content_block_start\ndata: ${JSON.stringify(blockStart)}\n\n`,
    `event: content_block_delta\ndata: ${JSON.stringify(blockDelta)}\n\n`,
    `event: content_block_stop\ndata: ${JSON.stringify(blockStop)}\n\n`,
    `event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`,
    `event: message_stop\ndata: ${JSON.stringify(messageStop)}\n\n`,
  ]);
}

function sseResponse(chunks) {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      },
    }),
    {
      status: 200,
      headers: {
        ...CORS_HEADERS,
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    },
  );
}

function normalizePath(pathname) {
  const normalized = pathname.replace(/\/+/g, "/");
  if (normalized.length > 1 && normalized.endsWith("/")) {
    return normalized.slice(0, -1);
  }
  return normalized || "/";
}

function buildUpstreamUrl(env, path) {
  const base = (env.ANYROUTER_BASE_URL || DEFAULT_UPSTREAM_BASE_URL).replace(/\/+$/, "");
  return `${base}${path}`;
}

function serviceInfo(env) {
  return {
    ok: true,
    service: "anyrouter-cli-proxy-worker",
    upstream_base_url: env.ANYROUTER_BASE_URL || DEFAULT_UPSTREAM_BASE_URL,
    default_model: env.DEFAULT_MODEL || null,
    default_max_tokens: Number(env.DEFAULT_MAX_TOKENS) || DEFAULT_MAX_TOKENS,
    has_worker_api_key: Boolean(env.WORKER_API_KEY),
    has_upstream_api_key: Boolean(env.ANYROUTER_API_KEY),
    model_allowlist: parseListEnv(env.MODEL_ALLOWLIST),
    model_blocklist: parseListEnv(env.MODEL_BLOCKLIST),
    supports: ["/v1/models", "/v1/chat/completions", "/v1/responses", "/v1/messages", "/anthropic/v1/models", "/anthropic/v1/messages"],
  };
}

function isHtmlChallenge(contentType, text) {
  if (typeof text !== "string") return false;
  if (/text\/html/i.test(contentType)) return true;
  return /^\s*<html/i.test(text) || /acw_sc__v2|<script>var arg1=/i.test(text);
}

function parseMaybeJson(text) {
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function inferErrorCode(data, status) {
  if (typeof data?.type === "string") return data.type;
  if (typeof data?.error?.type === "string") return data.error.type;
  if (status === 429) return "rate_limited";
  if (status === 404) return "not_found";
  return "upstream_error";
}

function extractErrorMessage(data, status) {
  if (typeof data === "string" && data.trim()) return data.trim();
  if (typeof data?.error === "string") return data.error;
  if (typeof data?.message === "string") return data.message;
  if (typeof data?.error?.message === "string") return data.error.message;
  return `Upstream request failed with status ${status}`;
}

function extractTextContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    if (content && typeof content.text === "string") return content.text;
    if (content == null) return "";
    return JSON.stringify(content);
  }

  return content
    .map((item) => {
      if (typeof item === "string") return item;
      if (!item || typeof item !== "object") return "";
      if (typeof item.text === "string") return item.text;
      if (typeof item.input_text === "string") return item.input_text;
      if (typeof item.output_text === "string") return item.output_text;
      if (item.type === "image" || item.type === "image_url") return "[image omitted]";
      if (item.type === "tool_result" || item.type === "tool_use") return JSON.stringify(item);
      return JSON.stringify(item);
    })
    .join("\n");
}

function mapAnthropicStopReason(stopReason) {
  if (stopReason === "max_tokens") return "length";
  if (stopReason === "tool_use") return "tool_calls";
  return "stop";
}

async function readJson(request) {
  const text = await request.text();
  if (!text) return {};

  try {
    return JSON.parse(text);
  } catch {
    throw createHttpError(400, "invalid_json", "Request body must be valid JSON");
  }
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload, null, 2), {
    status,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json; charset=utf-8",
    },
  });
}

function errorResponse(status, code, message, details) {
  const payload = {
    error: {
      type: code,
      message,
      ...(details === undefined ? {} : { details }),
    },
    type: "error",
  };
  return jsonResponse(payload, status);
}

function createHttpError(status, code, message, details) {
  const error = new Error(message);
  error.status = status;
  error.code = code;
  error.details = details;
  return error;
}

function toErrorResponse(error) {
  const status = Number.isInteger(error?.status) ? error.status : 500;
  const code = error?.code || "internal_error";
  const message = error?.message || "Unexpected error";
  return errorResponse(status, code, message, error?.details);
}

function randomId() {
  const bytes = crypto.getRandomValues(new Uint8Array(8));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function nowSeconds() {
  return Math.floor(Date.now() / 1000);
}
