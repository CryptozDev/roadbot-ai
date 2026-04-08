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

const child = spawn(pythonCommand, ["bot.py"], {
  cwd: projectDir,
  stdio: "inherit",
});

child.on("error", (error) => {
  console.error(`Failed to start Discord bot with '${pythonCommand}':`, error.message);
  process.exit(1);
});

child.on("exit", (code) => {
  process.exit(code ?? 0);
});
