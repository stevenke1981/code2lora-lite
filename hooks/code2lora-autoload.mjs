import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const MARKER = "code2lora-lite-autoload";

export const Code2LoRAAutoloadPlugin = async ({ directory, worktree }, options = {}) => {
  const repoRoot = resolvePath(options.repoPath || ".", worktree || directory || process.cwd());
  const contextDir = String(options.contextDir || ".code2lora/agent-context");
  const contextPath = resolvePath(join(contextDir, "context.md"), repoRoot);
  const statusPath =
    options.statusPath === false
      ? null
      : resolvePath(String(options.statusPath || join(contextDir, "autoload-status.json")), repoRoot);
  const maxChars = Number(options.maxChars || 24000);
  const refresh = String(options.refresh || "missing").toLowerCase();
  const minReduction = String(options.minReduction || "0.80");
  const maxFiles = String(options.maxFiles || "24");
  const strict = Boolean(options.strict || false);
  const refreshTimeoutMs = Number(options.refreshTimeoutMs || 120000);
  const cargoTargetDir =
    options.cargoTargetDir === false
      ? null
      : resolvePath(
          String(options.cargoTargetDir || join(tmpdir(), "code2lora-autoload-target", hashPath(repoRoot))),
          repoRoot,
        );

  let cachedContext = null;
  let lastError = null;
  let lastRefresh = null;

  return {
    async "experimental.chat.system.transform"(_input, output) {
      if (!Array.isArray(output.system)) {
        output.system = [];
      }

      if (output.system?.some((entry) => String(entry).includes(MARKER))) {
        return;
      }

      const context = loadContext();
      if (context) {
        output.system.push(formatContext(context));
        writeStatus({ injected: true, contextChars: context.length });
      } else if (strict) {
        writeStatus({ injected: false, contextChars: 0 });
        throw new Error(lastError || "Code2LoRA autoload context is unavailable");
      } else {
        writeStatus({ injected: false, contextChars: 0 });
      }
    },
  };

  function loadContext() {
    if (cachedContext && refresh !== "always") {
      return cachedContext;
    }

    if (refresh === "always" || !existsSync(contextPath)) {
      const result = refreshContext();
      if (!result.ok && strict) {
        throw new Error(result.error);
      }
    }

    try {
      if (!existsSync(contextPath)) {
        lastError = `Context not found at ${contextPath}`;
        return null;
      }
      const raw = readFileSync(contextPath, "utf8");
      cachedContext = raw.length > maxChars ? `${raw.slice(0, maxChars)}\n\n[truncated by ${MARKER}]` : raw;
      return cachedContext;
    } catch (error) {
      lastError = `Failed to read ${contextPath}: ${error.message}`;
      return null;
    }
  }

  function refreshContext() {
    const script = join(repoRoot, "scripts", "agent-context.ps1");
    if (!existsSync(script)) {
      lastError = `agent-context script not found at ${script}`;
      return { ok: false, error: lastError };
    }

    const shell = process.platform === "win32" ? "powershell" : "pwsh";
    const args = [
      "-NoProfile",
      ...(process.platform === "win32" ? ["-ExecutionPolicy", "Bypass"] : []),
      "-File",
      script,
      "-RepoPath",
      repoRoot,
      "-OutputDir",
      contextDir,
      "-MinReduction",
      minReduction,
      "-MaxFiles",
      maxFiles,
    ];

    const child = spawnSync(shell, args, {
      cwd: repoRoot,
      encoding: "utf8",
      env: {
        ...process.env,
        ...(cargoTargetDir ? { CARGO_TARGET_DIR: cargoTargetDir } : {}),
      },
      stdio: ["ignore", "pipe", "pipe"],
      timeout: refreshTimeoutMs,
    });

    if (child.status === 0) {
      lastError = null;
      lastRefresh = { attempted: true, ok: true, timedOut: false };
      return { ok: true };
    }

    const timedOut = child.error?.code === "ETIMEDOUT";
    lastError = [
      `Failed to refresh Code2LoRA context with ${shell}`,
      timedOut ? `Timed out after ${refreshTimeoutMs} ms` : null,
      child.stdout?.trim(),
      child.stderr?.trim(),
      child.error?.message,
    ]
      .filter(Boolean)
      .join("\n");
    lastRefresh = { attempted: true, ok: false, timedOut };
    return { ok: false, error: lastError };
  }

  function writeStatus(update) {
    if (!statusPath) {
      return;
    }

    try {
      mkdirSync(dirname(statusPath), { recursive: true });
      writeFileSync(
        statusPath,
        `${JSON.stringify(
          {
            marker: MARKER,
            repoRoot,
            contextDir,
            contextPath,
            statusPath,
            injected: Boolean(update.injected),
            contextChars: Number(update.contextChars || 0),
            maxChars,
            refresh,
            maxFiles: Number(maxFiles),
            minReduction: Number(minReduction),
            cargoTargetDir,
            refreshTimeoutMs,
            lastRefresh,
            lastError,
            generatedAtUnix: Math.floor(Date.now() / 1000),
          },
          null,
          2,
        )}\n`,
        "utf8",
      );
    } catch {
      // Status is diagnostic only; never make chat startup fail because of it.
    }
  }
};

function formatContext(context) {
  return [
    `<!-- ${MARKER} -->`,
    "The following repository context was auto-loaded from code2lora-lite. Use it before opening broad source files.",
    "",
    context,
  ].join("\n");
}

function resolvePath(value, base) {
  const text = String(value);
  return isAbsolute(text) ? text : resolve(base, text);
}

function hashPath(value) {
  return createHash("sha256").update(value).digest("hex").slice(0, 16);
}

export default Code2LoRAAutoloadPlugin;
export const server = Code2LoRAAutoloadPlugin;
