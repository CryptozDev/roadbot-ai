import dotenv from "dotenv";
import { z } from "zod";

dotenv.config();

const envSchema = z.object({
  PORT: z.coerce.number().default(4000),
  NODE_ENV: z.string().default("development"),
  FRONTEND_ORIGIN: z.string().default("*"),
  BOT_API_TOKEN: z.string().min(16),

  PY_RAG_URL: z.string().default("http://127.0.0.1:8001"),
  OPENAI_API_KEY: z.string().optional(),
  OPENAI_MODEL_NAME: z.string().default("gpt-4o-mini"),
  LONGDO_EVENT_URL: z.string().default("https://event.longdo.com/feed/json"),
  EXAT_API_BASE_URL: z.string().default("https://exat-man.web.app/api"),
  OPENAI_WEB_SEARCH_ENABLED: z.string().default("true"),
  OPENAI_WEB_MODEL: z.string().default("gpt-4.1-mini"),
  QWEN_MODEL_NAME: z.string().default("Qwen/Qwen2.5-1.5B-Instruct"),
  EMBEDDING_MODEL_NAME: z.string().default(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  ),
  FAISS_INDEX_PATH: z.string().default("./data/faiss.index"),
  FAISS_META_PATH: z.string().default("./data/faiss_meta.json"),
  RAG_TIMEOUT_MS: z.coerce.number().default(180000),
  RAG_CHAT_TIMEOUT_MS: z.coerce.number().default(45000),
  RAG_INGEST_TIMEOUT_MS: z.coerce.number().default(300000),

  ADMIN_INGEST_TOKEN: z.string().optional(),
  DEFAULT_SHEET_URL: z.string().optional(),
  DEFAULT_SHEET_GID: z.string().default("0"),
  TOP_K: z.coerce.number().default(6),
});

const parsed = envSchema.safeParse(process.env);

if (!parsed.success) {
  console.error("Invalid environment variables", parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const env = parsed.data;
