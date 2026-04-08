import { Router } from "express";
import { z } from "zod";
import { ragIngestSheet } from "../services/ragClient.js";

const router = Router();

const ingestSchema = z.object({
  sheetUrl: z.string().url().optional(),
  gid: z.string().optional(),
  token: z.string().optional(),
});

router.post("/ingest/sheet", async (req, res) => {
  try {
    const parsed = ingestSchema.parse(req.body || {});

    const result = await ragIngestSheet({
      sheetUrl: parsed.sheetUrl,
      gid: parsed.gid,
    });

    return res.json({ ok: true, ...result });
  } catch (error) {
    return res.status(400).json({ error: error.message });
  }
});

export default router;
