import { Router } from "express";
import { ragStatus } from "../services/ragClient.js";

const router = Router();

router.get("/health", async (_req, res) => {
  let rag = null;
  try {
    rag = await ragStatus();
  } catch {
    rag = { ok: false, ready: false };
  }

  res.json({ ok: true, service: "roadbotai-backend", rag });
});

export default router;
