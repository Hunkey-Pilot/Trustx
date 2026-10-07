import { spawn } from "node:child_process";
import { createConnection } from "node:net";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const frontendDirectory = path.resolve(scriptDirectory, "..");
const backendDirectory = path.resolve(frontendDirectory, "..", "backend");
const pythonExecutable = path.join(
  backendDirectory,
  ".venv",
  "Scripts",
  "python.exe",
);
const nextExecutable = path.join(
  frontendDirectory,
  "node_modules",
  "next",
  "dist",
  "bin",
  "next",
);
const host = "127.0.0.1";
const backendPort = Number(process.env.TRUSTX_API_PORT || 8000);
const frontendPort = Number(process.env.PORT || 3000);
const children = new Set();
let shuttingDown = false;

function isPortListening(port) {
  return new Promise((resolve) => {
    const socket = createConnection({ host, port });
    const finish = (listening) => {
      socket.destroy();
      resolve(listening);
    };
    socket.setTimeout(800);
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
    socket.once("timeout", () => finish(false));
  });
}

function stopProcessTree(child) {
  if (!child.pid || child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      child.kill();
      resolve();
    }, 5000);

    child.once("close", () => {
      clearTimeout(timer);
      resolve();
    });

    if (process.platform === "win32") {
      const killer = spawn(
        "taskkill.exe",
        ["/PID", String(child.pid), "/T", "/F"],
        { stdio: "ignore", windowsHide: true },
      );
      killer.once("error", () => child.kill());
      killer.once("close", (code) => {
        if (code !== 0) child.kill();
      });
      return;
    }

    try {
      process.kill(-child.pid, "SIGTERM");
    } catch {
      child.kill("SIGTERM");
    }
  });
}

async function shutdown(exitCode) {
  if (shuttingDown) return;
  shuttingDown = true;
  process.exitCode = exitCode;
  await Promise.all([...children].map(stopProcessTree));
}

function startChild(name, executable, args, cwd, env = process.env) {
  const child = spawn(executable, args, {
    cwd,
    env,
    stdio: "inherit",
    windowsHide: true,
    detached: process.platform !== "win32",
  });
  children.add(child);

  child.once("error", (error) => {
    console.error(`[${name}] Failed to start: ${error.message}`);
    void shutdown(1);
  });
  child.once("close", (code, signal) => {
    children.delete(child);
    if (!shuttingDown) {
      console.error(
        `[${name}] Exited${signal ? ` after ${signal}` : ` with code ${code}`}; stopping the other service.`,
      );
      void shutdown(code || 1);
    }
  });

  return child;
}

async function startDevelopment() {
  if (await isPortListening(frontendPort)) {
    throw new Error(
      `Frontend port ${frontendPort} is already in use. Stop the existing TrustX frontend before running npm run dev again.`,
    );
  }

  if (!existsSync(nextExecutable)) {
    throw new Error(`Next.js executable not found: ${nextExecutable}. Run npm install first.`);
  }

  if (await isPortListening(backendPort)) {
    console.log(`[backend] Reusing existing API listener at http://${host}:${backendPort}`);
  } else {
    if (!existsSync(pythonExecutable)) {
      throw new Error(`Backend Python environment not found: ${pythonExecutable}`);
    }
    console.log(`[backend] Starting FastAPI at http://${host}:${backendPort}`);
    startChild(
      "backend",
      pythonExecutable,
      [
        "-m",
        "uvicorn",
        "app.main:app",
        "--reload",
        "--host",
        host,
        "--port",
        String(backendPort),
      ],
      backendDirectory,
    );
  }

  console.log(`[frontend] Starting Next.js at http://localhost:${frontendPort}`);
  startChild(
    "frontend",
    process.execPath,
    [nextExecutable, "dev", "--port", String(frontendPort)],
    frontendDirectory,
    {
      ...process.env,
      // This script starts/reuses the local API, so point the frontend proxy at it unless the
      // shell explicitly overrides the URL (a value in .env.local alone must not win here).
      NEXT_PUBLIC_API_BASE_URL:
        process.env.NEXT_PUBLIC_API_BASE_URL || `http://${host}:${backendPort}`,
    },
  );
}

process.once("SIGINT", () => void shutdown(130));
process.once("SIGTERM", () => void shutdown(143));

startDevelopment().catch((error) => {
  console.error(`[trustx-dev] ${error.message}`);
  void shutdown(1);
});