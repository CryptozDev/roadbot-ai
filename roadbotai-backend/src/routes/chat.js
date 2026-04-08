import { Router } from "express";
import { z } from "zod";
import { env } from "../config/env.js";
import { ragAnswer, ragIngestSheet, ragStatus } from "../services/ragClient.js";

const router = Router();

const botChatSchema = z.object({
  message: z.string().optional(),
  content: z.string().optional(),
  question: z.string().optional(),
  source: z.string().optional(),
  channelId: z.union([z.string(), z.number()]).optional(),
  userId: z.union([z.string(), z.number()]).optional(),
  username: z.string().optional(),
  body: z.record(z.any()).optional(),
});

function pickMessage(payload) {
  const candidates = [
    payload?.message,
    payload?.content,
    payload?.question,
    payload?.body?.message,
    payload?.body?.content,
    payload?.body?.question,
  ];

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }

  return "";
}

let bootstrapPromise = null;
let bootstrapLastError = null;
let lastKnownRagStatus = {
  ready: false,
  embedder_ready: false,
  index_ready: false,
  meta_ready: false,
  updated_at: 0,
};
const RAG_HARD_TIMEOUT_MS = Math.max(20000, Number(env.RAG_CHAT_TIMEOUT_MS || 45000));
const REALTIME_ONLY_MESSAGE = "ขออภัย ระบบนี้ให้บริการเฉพาะข้อมูลสถิติอุบัติเหตุย้อนหลังเท่านั้น";
const HISTORY_ONLY_MESSAGE = "ขออภัย ระบบนี้ให้บริการเฉพาะข้อมูลสถิติอุบัติเหตุย้อนหลัง";

const realtimeKeywords = [
  "realtime",
  "real-time",
  "เรียลไทม์",
  "เรียลไทม",
  "สด",
  "ตอนนี้",
  "ขณะนี้",
  "ปัจจุบัน",
  "ล่าสุด",
  "วันนี้",
  "ตอนนี้มี",
  "ตอนนี้บนถนน",
  "ตอนนี้เส้นทาง",
  "ตอนนี้รถติด",
  "ถ่ายทอดสด",
  "live",
  "now",
];

const historicalScopeKeywords = [
  "อุบัติเหตุ",
  "จุดเสี่ยง",
  "เสี่ยง",
  "จุดอันตราย",
  "สถิติ",
  "ย้อนหลัง",
  "ถนน",
  "สายทาง",
  "เส้นทาง",
  "จังหวัด",
  "กม",
  "km",
  "ระวัง",
  "เดินทาง",
  "จาก",
  "ไป",
];

function normalizeQuestion(text) {
  return String(text || "").toLowerCase().replace(/\s+/g, "").trim();
}

function isRealtimeQuestion(text) {
  const normalized = normalizeQuestion(text);
  return realtimeKeywords.some((keyword) => normalized.includes(keyword));
}

function isHistoricalAccidentQuestion(text) {
  const normalized = normalizeQuestion(text);
  const hasRoutePattern = /จาก.+ไป.+/.test(String(text || ""));
  const hasScopeKeyword = historicalScopeKeywords.some((keyword) => normalized.includes(keyword));
  const hasAccidentIntent = ["อุบัติเหตุ", "จุดเสี่ยง", "ระวัง", "สถิติ", "ย้อนหลัง"].some((keyword) =>
    normalized.includes(keyword)
  );

  if (hasRoutePattern && hasAccidentIntent) {
    return true;
  }

  return hasScopeKeyword && hasAccidentIntent;
}

function getGuardrailReply(text) {
  const message = String(text || "").trim();
  if (!message) {
    return HISTORY_ONLY_MESSAGE;
  }

  const looksLikeRouteQuestion = ["จาก", "ไป", "ผ่าน", "แวะ", "route", "พระราม", "บางแสน", "พัทยา"].some(
    (keyword) => message.includes(keyword)
  );

  if (isRealtimeQuestion(message) || isHistoricalAccidentQuestion(message) || looksLikeRouteQuestion) {
    return null;
  }

  return HISTORY_ONLY_MESSAGE;
}

function hasIndexMissingMessage(text) {
  return String(text || "").includes("Index/meta files not found");
}

function isTransientRagIssue(text) {
  const message = String(text || "");
  return message.includes("RAG timeout") || message.includes("RAG service unavailable");
}

async function refreshRagStatus() {
  try {
    const status = await Promise.race([
      ragStatus(),
      new Promise((_, reject) => setTimeout(() => reject(new Error("status timeout")), 1200)),
    ]);

    if (status && typeof status === "object") {
      lastKnownRagStatus = {
        ...lastKnownRagStatus,
        ...status,
      };
    }
  } catch {
    // Keep previous known status.
  }
}

function getWarmupPercent() {
  if (lastKnownRagStatus.ready) {
    return 100;
  }

  let score = 0;
  if (lastKnownRagStatus.embedder_ready) score += 40;
  if (lastKnownRagStatus.meta_ready) score += 40;
  if (lastKnownRagStatus.index_ready) score += 20;
  return score;
}

function warmupMessage() {
  const modelState = lastKnownRagStatus.embedder_ready ? "พร้อม" : "กำลังโหลด";
  const metaState = lastKnownRagStatus.meta_ready ? "พร้อม" : "กำลังโหลด";
  const indexState = lastKnownRagStatus.index_ready ? "พร้อม" : "กำลังสร้าง";
  const percent = getWarmupPercent();

  const statusText = `สถานะระบบ ${percent}% (model:${modelState}, meta:${metaState}, index:${indexState})`;

  if (bootstrapLastError) {
    return `ระบบกำลังเตรียมฐานข้อมูลอยู่ และพบการเชื่อมต่อช้า\n${statusText}\nกรุณารอประมาณ 1-2 นาทีแล้วส่งคำถามเดิมอีกครั้ง`;
  }

  return `ระบบกำลังโหลดโมเดลและฐานข้อมูลครั้งแรกอยู่\n${statusText}\nกรุณารอประมาณ 1-2 นาทีแล้วส่งคำถามเดิมอีกครั้ง`;
}

function triggerBootstrapInBackground() {
  if (bootstrapPromise) {
    return;
  }

  bootstrapLastError = null;
  bootstrapPromise = (async () => {
    try {
      await refreshRagStatus();
      if (!lastKnownRagStatus.meta_ready || !lastKnownRagStatus.index_ready) {
        await ragIngestSheet({});
      }
      await refreshRagStatus();
    } catch (error) {
      bootstrapLastError = String(error?.message || "bootstrap failed");
      await refreshRagStatus();
    } finally {
      bootstrapPromise = null;
    }
  })();
}

async function withHardTimeout(promise, timeoutMs, timeoutMessage) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function getRagAnswerWithBootstrap(question) {
  try {
    if (bootstrapPromise) {
      await refreshRagStatus();

      if (lastKnownRagStatus.ready) {
        const readyResult = await withHardTimeout(
          ragAnswer({ question }),
          RAG_HARD_TIMEOUT_MS,
          "RAG timeout: ready-but-bootstrap-running"
        );

        const readyAnswerText = String(readyResult?.answer || "");
        if (!hasIndexMissingMessage(readyAnswerText)) {
          return readyResult;
        }
      }

      try {
        const quickResult = await withHardTimeout(
          ragAnswer({ question }),
          6000,
          "RAG timeout: bootstrap quick-answer timeout"
        );

        const quickAnswerText = String(quickResult?.answer || "");
        if (!hasIndexMissingMessage(quickAnswerText)) {
          return quickResult;
        }
      } catch {
        // Keep warmup fallback below while background bootstrap continues.
      }

      return {
        ok: true,
        answer: warmupMessage(),
        references: [],
      };
    }

    const result = await withHardTimeout(
      ragAnswer({ question }),
      RAG_HARD_TIMEOUT_MS,
      "RAG timeout: hard-timeout"
    );
    const answerText = String(result?.answer || "");

    if (hasIndexMissingMessage(answerText)) {
      triggerBootstrapInBackground();
      return {
        ok: true,
        answer: warmupMessage(),
        references: [],
      };
    }

    return result;
  } catch (error) {
    const message = String(error?.message || "");

    if (message.includes("Index/meta files not found")) {
      triggerBootstrapInBackground();
      await refreshRagStatus();
      return {
        ok: true,
        answer: warmupMessage(),
        references: [],
      };
    }

    if (isTransientRagIssue(message)) {
      await refreshRagStatus();

      if (!lastKnownRagStatus.ready) {
        triggerBootstrapInBackground();
        return {
          ok: true,
          answer: warmupMessage(),
          references: [],
        };
      }

      return {
        ok: true,
        answer: "RoadBot AI กำลังประมวลผลคำตอบนานกว่าปกติ กรุณาลองส่งคำถามเดิมอีกครั้งในอีกไม่กี่วินาที",
        references: [],
      };
    }

    if (message.includes("ready-but-bootstrap-running")) {
      return {
        ok: true,
        answer: "RoadBot AI กำลังประมวลผลคำตอบอยู่ กรุณาลองส่งอีกครั้งภายในไม่กี่วินาที",
        references: [],
      };
    }

    throw error;
  }
}

function getBotTokenFromRequest(req) {
  const rawBotHeader = req.headers["x-bot-token"];
  const botHeader = Array.isArray(rawBotHeader) ? rawBotHeader[0] : rawBotHeader;

  if (botHeader) {
    return String(botHeader).trim();
  }

  const authHeader = String(req.headers.authorization || "").trim();
  const [scheme, token] = authHeader.split(" ");
  if (scheme === "Bearer" && token) {
    return token.trim();
  }

  return "";
}

router.post("/chat/bot", async (req, res) => {
  try {
    const token = getBotTokenFromRequest(req);
    if (!env.BOT_API_TOKEN) {
      return res.status(503).json({ error: "BOT_API_TOKEN is not configured" });
    }

    if (token !== env.BOT_API_TOKEN) {
      return res.status(401).json({ error: "Unauthorized bot" });
    }

    const payload = botChatSchema.parse(req.body || {});
    const message = pickMessage(payload);
    if (!message) {
      return res.status(400).json({
        error: "Missing message/content/question in request body",
      });
    }

    const guardrailReply = getGuardrailReply(message);
    if (guardrailReply) {
      return res.json({
        ok: true,
        answer: guardrailReply,
        references: [],
      });
    }

    const rag = await getRagAnswerWithBootstrap(message);
    return res.json({
      ok: true,
      answer: rag.answer,
      references: rag.references || [],
    });
  } catch (error) {
    const message = String(error?.message || "Unknown error");
    const status = message.includes("RAG") ? 502 : 400;
    return res.status(status).json({ error: message });
  }
});

export default router;
