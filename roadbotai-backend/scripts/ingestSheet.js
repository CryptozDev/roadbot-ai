import { env } from "../src/config/env.js";
import { ragIngestSheet } from "../src/services/ragClient.js";

async function run() {
  try {
    const result = await ragIngestSheet({
      sheetUrl: process.argv[2] || env.DEFAULT_SHEET_URL,
      gid: process.argv[3] || env.DEFAULT_SHEET_GID,
    });

    console.log("Ingest success:", result);
  } catch (error) {
    console.error("Ingest failed:", error.message);
    process.exit(1);
  }
}

run();
