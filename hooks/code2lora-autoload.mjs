import { existsSync, readFileSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const MARKER = "code2lora-lite-autoload";

export const Code2LoRAAutoloadPlugin = async ({ directory, worktree }, options = {}) => {
  const repoRoot = resolvePath(options.repoPath || ".", worktree || directory || process.cwd());
  const contextDir = String(options.contextDir || ".code2lora/agent-context");
  const contextPath = resolvePath(join(contextDir, "context.md"), repoRoot);
  const maxChars = Number(options.maxChars || 24000);
  const refresh = String(options.refresh || "missing").toLowerCase();
  const minReduction = String(options.minReduction || "0.80");
  const maxFiles = String(options.maxFiles || "24");
  const strict = Boolean(options.strict || false);

  let cachedContext = null;
  let lastError = null;

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
      } else if (strict) {
        throw new Error(lastError || "Code2LoRA autoload context is unavailable");
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
      stdio: ["ignore", "pipe", "pipe"],
    });

    if (child.status === 0) {
      lastError = null;
      return { ok: true };
    }

    lastError = [
      `Failed to refresh Code2LoRA context with ${shell}`,
      child.stdout?.trim(),
      child.stderr?.trim(),
    ]
      .filter(Boolean)
      .join("\n");
    return { ok: false, error: lastError };
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

export default Code2LoRAAutoloadPlugin;
export const server = Code2LoRAAutoloadPlugin;
