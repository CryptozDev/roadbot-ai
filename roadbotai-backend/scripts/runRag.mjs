import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectDir = path.resolve(__dirname, "..");
const workspaceDir = path.resolve(projectDir, "..");

const candidates = process.platform === "win32"
  ? [
      path.join(workspaceDir, ".venv", "Scripts", "python.exe"),
      "python",
      "py",
    ]
  : [
      path.join(workspaceDir, ".venv", "bin", "python"),
      "python3",
      "python",
    ];

const pythonCommand = candidates.find((candidate) =>
  candidate.includes(path.sep) ? fs.existsSync(candidate) : true,
);

const args = [
  "-m",
  "uvicorn",
  "rag_service.app:app",
  "--host",
  "127.0.0.1",
  "--port",
  "8001",
];

const child = spawn(pythonCommand, args, {
  cwd: projectDir,
  stdio: "inherit",
});

child.on("error", (error) => {
  console.error(`Failed to start RAG service with '${pythonCommand}':`, error.message);
  process.exit(1);
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
