import cors from "cors";
import express from "express";
import { env } from "./config/env.js";
import healthRoutes from "./routes/health.js";
import chatRoutes from "./routes/chat.js";
import ingestRoutes from "./routes/ingest.js";
import { ragStatus, ragWarmup } from "./services/ragClient.js";

const app = express();

app.use(cors({ origin: env.FRONTEND_ORIGIN === "*" ? true : env.FRONTEND_ORIGIN }));
app.use(express.json({ limit: "2mb" }));

app.use("/api", healthRoutes);
app.use("/api", chatRoutes);
app.use("/api", ingestRoutes);

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(500).json({ error: "Internal server error" });
});

async function warmupRagOnBoot() {
  for (let attempt = 1; attempt <= 24; attempt += 1) {
    try {
      const status = await ragStatus();
      if (status?.ready) {
        console.log("RAG already ready");
        return;
      }

      await ragWarmup();
      const after = await ragStatus();
      if (after?.ready) {
        console.log("RAG warmup completed");
        return;
      }
      console.log(`RAG warming up... attempt ${attempt}`);
    } catch {
      // RAG may still be booting; keep retrying quietly.
    }

    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  console.warn("RAG still not ready after startup polling; chat will continue with fallback until ready");
}

app.listen(env.PORT, () => {
  console.log(`Roadbot backend running at http://localhost:${env.PORT}`);
  warmupRagOnBoot();
});
