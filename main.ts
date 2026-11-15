// free-model-aggregator-02: one server, two client wires, real free OpenRouter models behind every answer.
//
// The entry listens on a port it picks, registers with the env, and serves the OpenAI Chat
// Completions API and the Anthropic Messages API (JSON and server-sent events). Every prompt is
// answered by a REAL completion from one of the fourteen free OpenRouter models the brief lists,
// dialled through the grader's gateway on the trial's key. There is no simulated provider and no
// local model: the gateway is the only way out, and every model call goes through it as a Chat
// Completions request (the gateway refuses these ids on its Anthropic route).
//
// How it wins:
//   - quality: candidates that meet the prompt's requirements are ranked by the brief's published
//     rating_milli (best first), so each prompt is served by the strongest free model that can take
//     it. The best overall model (thinkingmachines/inkling-2-small, 1475) cannot do JSON mode, so
//     JSON prompts go to the best JSON-capable model (nvidia/nemotron-3.5-super-vl, 1303).
//   - failover: a gateway refusal (4xx/5xx or a broken connection) is never an answer. The prompt
//     is retried on the next-ranked model that meets it, and the refusals are reported on the
//     x-makefun-upstream-failures header so the failover column can see them. Refusals are
//     classified: a 429, a 5xx, or a broken answer cools the model down for the rest of the trial;
//     a 400 or a transport error only bars it from this prompt. A 401 means the entrant pushed no
//     working key, so the prompt is abandoned at once and later prompts fail fast instead of
//     burning calls.
//   - rate limits: a model that refuses once is not dialled again for the rest of the trial. A
//     rate-limited free model does not recover inside one trial, and every extra call costs the
//     tie-break. Attempts per prompt are capped at six, so a trial spends at most 36 model calls
//     against the gateway's 60.
//   - catalog: exactly the fourteen free ids the brief lists, once each, on one openrouter provider.
//     The same document is printed by --list, carried in the register body, and listed by
//     GET /v1/models on both auth headers.
//   - the model field: the provider's answer spells the model without the gateway's openrouter/
//     prefix. This server puts the id it dialled back on every answer and every streamed chunk,
//     because the model a response names must be the model the gateway record shows really served it.
//   - determinism: routing is a pure function of the brief and the requests seen so far. No wall
//     clock, no unseeded randomness, no timestamps that move. A replay of the same trial makes the
//     same gateway requests and leaves the same export, byte for byte.
//
// The gateway refuses a request with no output cap or one over 8192 tokens, so every upstream body
// carries max_tokens (the client's, or 1024 when it sends none, clamped to 8192). The trial's dummy
// key is sent as the vendor's SDK sends it; the gateway swaps it for the entrant's own OpenRouter
// key, which never enters the sandbox.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asStr = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNum = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const asArr = (v: unknown): unknown[] => (Array.isArray(v) ? v : []);

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';
const PROTOCOL_VERSION = 1;
const POLL_MS = 100;
const MAX_BODY = 8 * 1024 * 1024;
const DEFAULT_MAX_TOKENS = 1024;
const GATEWAY_MAX_TOKENS = 8192;
const MAX_ATTEMPTS_PER_PROMPT = 6;
const MAX_TRAIL_ROWS = 8;
const DIAL_TIMEOUT_MS = 90000;
const OPENROUTER_BASE = 'https://openrouter.ai/api/v1';
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

function log(msg: string): void {
  process.stderr.write(`aggregator02: ${msg}\n`);
}

// ---------------------------------------------------------------------------
// The brief: two JSON documents (the free model list and the six prompts)
// embedded in prose, found by their `schema` member under this contest's prefix.
// ---------------------------------------------------------------------------

interface ModelRow {
  id: string;
  provider: string;
  displayName: string;
  rating: number;
  context: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: { tools?: boolean; vision?: boolean; json?: boolean; minContext?: number };
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

const SCHEMA_PREFIX = 'free-model-aggregator-02/';

function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inStr = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function docsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const v: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(v) && typeof v.schema === 'string') out.push(v);
    } catch {
      // Not a JSON object on its own; look on.
    }
  }
  return out;
}

function arrayDoc(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!asStr(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

function modelOf(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = asStr(v.id) !== '' ? asStr(v.id) : asStr(v.model);
  if (raw === '') return null;
  // Resolve the provider first (default openrouter), then normalize the id: a row that gives the
  // bare OpenRouter id (vendor/model:free) is routed as the gateway spells it (openrouter/...).
  const named = asStr(v.provider) !== '' ? asStr(v.provider) : asStr(v.wire) !== '' ? asStr(v.wire) : 'openrouter';
  const id = named === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  return {
    id,
    provider: named,
    displayName: asStr(v.display_name) !== '' ? asStr(v.display_name) : raw,
    rating: asNum(v.rating_milli) ?? 0,
    context: asNum(v.context_length),
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
    free: v.free !== false,
  };
}

function promptOf(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  const id = asStr(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      ...(req.tools === true ? { tools: true } : {}),
      ...(req.vision === true ? { vision: true } : {}),
      ...(req.json === true ? { json: true } : {}),
      ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context) ? { minContext: req.min_context } : {}),
    },
  };
}

function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = docsIn(text);
  const modelsDoc = arrayDoc(docs, ['models', 'ratings']);
  if (modelsDoc === null) return { models: [], prompts: [], error: `the brief at ${path} holds no model list` };
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const m = modelOf(v);
    return m === null ? [] : [m];
  });
  const promptsDoc = arrayDoc(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) return { models, prompts: [], error: `the brief at ${path} holds no prompt list` };
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const p = promptOf(v);
    return p === null ? [] : [p];
  });
  return { models, prompts, error: null };
}

const brief = readBrief(PROMPT_FILE);
const freeModels: ModelRow[] = brief.models.filter((m) => m.free);

// The catalog: one openrouter provider, exactly the brief's free models, once each. The same
// document is printed by --list and carried in the register body; GET /v1/models lists the same
// model-id set in the shape of the auth header it got (OpenAI or Anthropic).
function buildCatalog(): Json {
  const seen = new Set<string>();
  const models: Json[] = [];
  for (const m of freeModels) {
    if (seen.has(m.id)) continue;
    seen.add(m.id);
    models.push({ id: m.id, free: true, context_length: m.context });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [{ id: 'openrouter', base_url: OPENROUTER_BASE, key_env: OPENROUTER_KEY_ENV, wire: 'openai', models }],
  };
}
const catalog = buildCatalog();

// ---------------------------------------------------------------------------
// Routing: rank the free models that meet a prompt's requirements by the
// brief's published rating (best first), then by context length, then by id.
// A model that refuses once is not dialled again for the rest of the trial:
// a rate-limited free model does not recover inside one trial, and every
// extra call costs the tie-break. Routing is a pure function of the brief,
// the request, and the refusals seen so far, so a replay makes the same calls.
// ---------------------------------------------------------------------------

interface Requirement {
  tools: boolean;
  vision: boolean;
  json: boolean;
  minContext: number;
}

function meets(m: ModelRow, r: Requirement): boolean {
  if (r.tools && m.tools !== true) return false;
  if (r.vision && m.vision !== true) return false;
  if (r.json && m.json !== true) return false;
  if (r.minContext > 0 && (m.context === null || m.context < r.minContext)) return false;
  return true;
}

function rankCandidates(r: Requirement, refused: ReadonlySet<string>): ModelRow[] {
  return freeModels
    .filter((m) => meets(m, r) && !refused.has(m.id))
    .sort((a, b) => b.rating - a.rating || (b.context ?? 0) - (a.context ?? 0) || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

// The requirements of one arriving request, read off its own body: a tools prompt declares tools,
// a JSON-mode prompt carries a response_format, a vision prompt carries image blocks. The body's
// own fields are the ground truth (a manifest row is only a cross-check for a minimum context,
// which a body cannot state), so a prompt is never routed by a row that belongs to another one.
let arrivalIndex = 0;
function requirementsOf(body: Json): Requirement {
  const req: Requirement = { tools: false, vision: false, json: false, minContext: 0 };
  const manifest = brief.prompts[arrivalIndex];
  arrivalIndex += 1;
  if (manifest !== undefined && manifest.requires.minContext !== undefined) req.minContext = manifest.requires.minContext;
  const messages = asArr(body.messages);
  for (const raw of messages) {
    if (!isRecord(raw) || !Array.isArray(raw.content)) continue;
    for (const block of raw.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) req.vision = true;
    }
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (fmt === 'json_object' || fmt === 'json_schema') req.json = true;
  return req;
}

// ---------------------------------------------------------------------------
// The gateway: every model call is a Chat Completions request to the gateway's
// OpenAI-compatible route, naming the dialled model id, with the trial's dummy
// key as Bearer (the gateway swaps it for the entrant's own OpenRouter key).
// The Anthropic route refuses these ids, so it is never used upstream.
// ---------------------------------------------------------------------------

function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

// The output cap every upstream body must carry: the client's, or 1024 when it sends none,
// clamped to the gateway's 8192. A request with no cap, or one over the cap, is refused.
function outputCap(body: Json): number {
  const cap = asNum(body.max_tokens) ?? asNum(body.max_completion_tokens) ?? null;
  if (cap === null) return DEFAULT_MAX_TOKENS;
  if (cap < 1) return 1;
  if (cap > GATEWAY_MAX_TOKENS) return GATEWAY_MAX_TOKENS;
  return Math.floor(cap);
}

function dialBody(body: Json, model: string): Json {
  const out: Json = { ...body, model, max_tokens: outputCap(body) };
  delete out.max_completion_tokens;
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

interface DialResult {
  ok: boolean;
  status: number;
  text: string;
  json: Json | null;
  body: ReadableStream<Uint8Array> | null;
  failures: UpstreamFailure[];
}

function failureCode(status: number, text: string): string {
  if (status === 429) return 'rate_limited';
  if (status === 401) return 'unauthorized';
  if (status === 400) {
    try {
      const parsed: unknown = JSON.parse(text);
      if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string') return parsed.error.code;
      if (isRecord(parsed) && typeof parsed.code === 'string') return parsed.code;
    } catch {
      // Not JSON: fall through to the generic code.
    }
    return 'bad_request';
  }
  if (status >= 500) return 'upstream_error';
  return 'request_failed';
}

async function dialOnce(model: string, body: Json): Promise<DialResult> {
  const url = chatCompletionsUrl();
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(dialBody(body, model)), signal: AbortSignal.timeout(DIAL_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, status: 0, text: (err as Error).message, json: null, body: null, failures: [{ status: 0, code: 'transport_error', model }] };
  }
  if (!res.ok || res.body === null) {
    let text = '';
    try {
      text = await res.text();
    } catch (err) {
      text = (err as Error).message;
    }
    return { ok: false, status: res.status, text, json: null, body: null, failures: [{ status: res.status, code: failureCode(res.status, text), model }] };
  }
  if (body.stream === true) {
    return { ok: true, status: res.status, text: '', json: null, body: res.body, failures: [] };
  }
  let text = '';
  try {
    text = await res.text();
  } catch (err) {
    return { ok: false, status: res.status, text: (err as Error).message, json: null, body: null, failures: [{ status: res.status, code: 'transport_error', model }] };
  }
  let parsed: Json | null = null;
  try {
    const v: unknown = JSON.parse(text);
    if (isRecord(v)) parsed = v;
  } catch {
    // Not JSON: treated as a broken answer below.
  }
  return { ok: true, status: res.status, text, json: parsed, body: null, failures: [] };
}

// A non-streaming answer is usable only when it carries a first choice with a message that has
// text or tool calls. An empty envelope, an error object, or a choice list with nothing in it is
// a broken answer, not a completion, and the prompt is failed over to the next model.
function usableAnswer(answer: Json | null): boolean {
  if (answer === null) return false;
  const choices = asArr(answer.choices);
  if (choices.length === 0 || !isRecord(choices[0])) return false;
  const message = choices[0].message;
  if (!isRecord(message)) return false;
  if (typeof message.content === 'string') return true;
  if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'text' || block.type === 'tool_use')) return true;
    }
  }
  return Array.isArray(message.tool_calls) && message.tool_calls.length > 0;
}

// Dial a prompt on the best model that meets it, failing over to the next-ranked model on any
// refusal. Refusals are classified: a 429 or 5xx (or a broken answer) cools the model down for
// the rest of the trial, because a rate-limited free model does not recover inside one trial;
// a 400 or a transport error is request-specific, so it only bars the model from this prompt.
// A 401 means the entrant pushed no working key, so every model would refuse: the prompt is
// abandoned at once and later prompts fail fast instead of burning calls. At most
// MAX_ATTEMPTS_PER_PROMPT calls are spent on one prompt. Returns the model that served it, or
// null with the failure trail when none did.
const refusedModels = new Set<string>();
let gatewayUnauthorized = false;
async function serve(body: Json, req: Requirement): Promise<{ model: string | null; result: DialResult | null; failures: UpstreamFailure[] }> {
  const failures: UpstreamFailure[] = [];
  if (gatewayUnauthorized) {
    failures.push({ status: 401, code: 'unauthorized', model: null });
    return { model: null, result: null, failures };
  }
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_PROMPT; attempt++) {
    const candidates = rankCandidates(req, refusedModels).filter((m) => !tried.has(m.id));
    if (candidates.length === 0) break;
    const model = candidates[0];
    tried.add(model.id);
    const result = await dialOnce(model.id, body);
    if (result.ok && (body.stream === true || usableAnswer(result.json))) {
      return { model: model.id, result, failures };
    }
    for (const f of result.failures) failures.push(f);
    if (result.ok && body.stream !== true && !usableAnswer(result.json)) {
      failures.push({ status: result.status, code: 'invalid_response', model: model.id });
    }
    if (result.failures.some((f) => f.status === 401)) {
      gatewayUnauthorized = true;
      break;
    }
    // Cool the model down for the trial only on a model-side refusal (429/5xx) or a broken
    // answer; a 400 or a transport error is local to this prompt.
    const modelSide = result.failures.some((f) => f.status === 429 || f.status >= 500) || (result.ok && !usableAnswer(result.json));
    if (modelSide) refusedModels.add(model.id);
  }
  return { model: null, result: null, failures };
}

function trailHeader(failures: UpstreamFailure[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, MAX_TRAIL_ROWS).map((f) => ({ status: f.status, code: f.code, model: f.model }));
  const text = JSON.stringify(rows);
  if (text.length > 8192) return null;
  return text;
}

// ---------------------------------------------------------------------------
// The wires. The client speaks OpenAI Chat Completions or Anthropic Messages;
// upstream is always Chat Completions. The provider's answer spells the model
// without the gateway's openrouter/ prefix, so every answer and every streamed
// chunk gets the dialled id put back on it. Fixed timestamps in the model
// listings keep them identical on a live pass and its replay.
// ---------------------------------------------------------------------------

const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

function openAIModelList(): Json {
  return {
    object: 'list',
    data: freeModels.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

function anthropicModelList(): Json {
  return {
    data: freeModels.map((m) => ({ type: 'model', id: m.id, display_name: m.displayName, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function openAIError(message: string, type: string): Json {
  return { error: { message, type, code: null } };
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && (b.type === 'text' || b.text !== undefined) && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

function imagePart(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const src = block.source;
  if (src.type === 'base64' && typeof src.data === 'string') {
    const media = typeof src.media_type === 'string' ? src.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${src.data}` } };
  }
  if (src.type === 'url' && typeof src.url === 'string') return { type: 'image_url', image_url: { url: src.url } };
  return null;
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function argsObject(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

// An Anthropic Messages request as a Chat Completions request: the system prompt becomes a system
// message, image blocks become image_url parts, tool_use blocks become tool_calls, tool_result
// blocks become tool messages, tools and tool_choice are rewritten, and the output cap,
// temperature, top_p, stop sequences and streaming pass through.
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of asArr(body.messages)) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: asStr(block.id),
          type: 'function',
          function: { name: asStr(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: asStr(block.tool_use_id), content: toolResultText(block.content) });
        continue;
      }
      const part = imagePart(block);
      if (part !== null) {
        parts.push(part);
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: asStr(raw.description),
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

// A Chat Completions answer as an Anthropic Messages object. `model` is the id dialled.
function openAIToAnthropic(body: Json, model: string): Json {
  const choices = asArr(body.choices);
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  for (const call of asArr(message.tool_calls)) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: asStr(call.id) !== '' ? asStr(call.id) : 'toolu_unknown',
      name: asStr(call.function.name),
      input: argsObject(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${asStr(body.id) !== '' ? asStr(body.id) : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numOrZero(usage.prompt_tokens), output_tokens: numOrZero(usage.completion_tokens) },
  };
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

// One SSE frame with every data payload's `model` set to the dialled id; comments, `event:`
// lines and non-JSON payloads pass through untouched.
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

// A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id
// dialled. Frames split on the blank line that ends them; a frame cut across reads is held until
// its end arrives, and a body that ends without one is relayed as it is.
async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      // Normalize the accumulated buffer, not each read: a CR at the end of one read and the LF
      // at the start of the next are one line ending, and replacing per read would miss it.
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        yield frameWithModel(buffer.slice(0, at), model);
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield frameWithModel(buffer, model);
  } finally {
    reader.releaseLock();
  }
}

// A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first
// chunk, content_block_start/delta/stop per text block or tool call, then message_delta with the
// stop reason and usage, then message_stop. The stop reason is held until the stream's usage
// chunk has arrived: OpenAI sends usage after the finish_reason, and Anthropic's message_delta
// carries the output tokens.
async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numOrZero(chunk.usage.prompt_tokens);
        outputTokens = numOrZero(chunk.usage.completion_tokens);
      }
      const choices = asArr(chunk.choices);
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      for (const call of asArr(delta.tool_calls)) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: { type: 'tool_use', id: asStr(call.id) !== '' ? asStr(call.id) : `toolu_${index}`, name: asStr(fn.name), input: {} },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      // Normalize the accumulated buffer, not each read: a CR at the end of one read and the LF
      // at the start of the next are one line ending, and replacing per read would miss it.
      buffer = (buffer + decoder.decode(value, { stream: true })).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        for (const event of eventsOfFrame(buffer.slice(0, at))) yield event;
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// The HTTP server: GET /v1/models, POST /v1/chat/completions, POST /v1/messages.
// ---------------------------------------------------------------------------

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buf.length;
    if (size > MAX_BODY) return null;
    chunks.push(buf);
  }
  if (chunks.length === 0) return null;
  try {
    const v: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extraHeaders: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extraHeaders);
  else sendJson(res, status, openAIError(message, type), extraHeaders);
}

function trailOf(failures: UpstreamFailure[]): Record<string, string> {
  const header = trailHeader(failures);
  return header === null ? {} : { 'x-makefun-upstream-failures': header };
}

function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList());
  else sendJson(res, 200, openAIModelList());
}

async function chatRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve(body, requirementsOf(body));
  if (served.model === null || served.result === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model that meets this request refused it', trailOf(served.failures));
  }
  const { model, result, failures } = served;
  const extra = trailOf(failures);
  if (body.stream === true && result.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...extra });
    try {
      for await (const frame of openAIStreamWithModel(result.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer = result.json ?? {};
  sendJson(res, 200, { ...answer, model }, extra);
}

async function messagesRoute(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const upstreamBody = anthropicToOpenAI(body, 'auto');
  const served = await serve(upstreamBody, requirementsOf(body));
  if (served.model === null || served.result === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model that meets this request refused it', trailOf(served.failures));
  }
  const { model, result, failures } = served;
  const extra = trailOf(failures);
  if (body.stream === true && result.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...extra });
    try {
      for await (const event of anthropicEventsFromOpenAI(result.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer = result.json ?? {};
  sendJson(res, 200, openAIToAnthropic(answer, model), extra);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chatRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messagesRoute(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) throw new Error(`the brief could not be read: ${brief.error}`);
  if (freeModels.length === 0) throw new Error('the brief lists no free models');
  log(`brief: ${freeModels.length} free models, ${brief.prompts.length} prompts`);
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${freeModels.length} free models`);
  await register(port);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
