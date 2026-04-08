import { env } from "../config/env.js";

async function postJson(path, payload, timeoutMs = env.RAG_TIMEOUT_MS, method = "POST") {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${env.PY_RAG_URL}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
      body: method === "GET" ? undefined : JSON.stringify(payload),
      signal: controller.signal,
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const message = data?.detail || data?.error || `RAG service error ${response.status}`;
      throw new Error(message);
    }

    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(
        `RAG timeout: Python service at ${env.PY_RAG_URL} took too long to respond.`
      );
    }

    if (String(error.message || "").toLowerCase().includes("fetch failed")) {
      throw new Error(
        `RAG service unavailable at ${env.PY_RAG_URL}. Please run: python -m uvicorn rag_service.app:app --host 127.0.0.1 --port 8001`
      );
    }

    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function ragIngestSheet({ sheetUrl, gid }) {
  return postJson("/ingest-sheet", {
    sheet_url: sheetUrl || env.DEFAULT_SHEET_URL,
    gid: gid || env.DEFAULT_SHEET_GID,
    embedding_model: env.EMBEDDING_MODEL_NAME,
    index_path: env.FAISS_INDEX_PATH,
    meta_path: env.FAISS_META_PATH,
  }, env.RAG_INGEST_TIMEOUT_MS);
}

export async function ragAnswer({ question, topK }) {
  return postJson("/chat", {
    question,
    top_k: topK || env.TOP_K,
    qwen_model: env.QWEN_MODEL_NAME,
    embedding_model: env.EMBEDDING_MODEL_NAME,
    index_path: env.FAISS_INDEX_PATH,
    meta_path: env.FAISS_META_PATH,
  }, env.RAG_CHAT_TIMEOUT_MS);
}

export async function ragWarmup() {
  return postJson("/warmup", {}, env.RAG_INGEST_TIMEOUT_MS);
}

export async function ragStatus() {
  return postJson("/status", {}, 8000, "GET");
}
