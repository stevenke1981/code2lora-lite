param(
    [string]$RepoPath = ".",
    [string]$ContextDir = "",
    [switch]$SkipConfigCheck,
    [switch]$NoRefresh
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ResolvedRepo = Resolve-Path $RepoPath
$HookPath = Join-Path $ResolvedRepo.Path "hooks/code2lora-autoload.mjs"

if (-not (Test-Path $HookPath)) {
    throw "OpenCode autoload hook not found: $HookPath"
}

if ([string]::IsNullOrWhiteSpace($ContextDir)) {
    $runId = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $ContextDir = ".code2lora/opencode-autoload-smoke/$runId"
}

$StatusPath = Join-Path $ContextDir "autoload-status.json"

if (-not $SkipConfigCheck) {
    $configOutput = & opencode debug config 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "opencode debug config failed:`n$configOutput"
    }
    if ($configOutput -notmatch "code2lora-autoload\.mjs") {
        throw "OpenCode resolved config does not include hooks/code2lora-autoload.mjs"
    }
    Write-Host "OpenCode config includes code2lora autoload hook"
}

$nodeScript = Join-Path ([System.IO.Path]::GetTempPath()) ("code2lora-autoload-smoke-{0}.mjs" -f $PID)
$refresh = if ($NoRefresh) { "missing" } else { "always" }

$env:CODE2LORA_SMOKE_REPO = $ResolvedRepo.Path
$env:CODE2LORA_SMOKE_CONTEXT_DIR = $ContextDir
$env:CODE2LORA_SMOKE_STATUS_PATH = $StatusPath
$env:CODE2LORA_SMOKE_REFRESH = $refresh

@'
import { existsSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";

const repo = process.env.CODE2LORA_SMOKE_REPO;
const contextDir = process.env.CODE2LORA_SMOKE_CONTEXT_DIR;
const statusPath = process.env.CODE2LORA_SMOKE_STATUS_PATH;
const refresh = process.env.CODE2LORA_SMOKE_REFRESH;
const hookUrl = pathToFileURL(join(repo, "hooks", "code2lora-autoload.mjs"));
const module = await import(hookUrl.href);
const plugin = module.default ?? module.server;

if (typeof plugin !== "function") {
  throw new Error("autoload hook does not export a plugin function");
}

const hooks = await plugin(
  { directory: repo, worktree: repo },
  {
    contextDir,
    statusPath,
    refresh,
    strict: true,
    maxChars: 24000,
    maxFiles: 24,
    minReduction: 0.8,
    refreshTimeoutMs: 180000,
  },
);

const transform = hooks["experimental.chat.system.transform"];
if (typeof transform !== "function") {
  throw new Error("autoload hook did not register experimental.chat.system.transform");
}

const output = {};
await transform({ sessionID: "code2lora-autoload-smoke", model: { id: "smoke" } }, output);

if (!Array.isArray(output.system)) {
  throw new Error("hook did not initialize output.system");
}
if (!output.system.some((entry) => String(entry).includes("code2lora-lite-autoload"))) {
  throw new Error("hook did not inject the code2lora marker");
}
if (!existsSync(statusPath)) {
  throw new Error(`hook did not write status file: ${statusPath}`);
}

const status = JSON.parse(readFileSync(statusPath, "utf8"));
if (!status.injected) {
  throw new Error(`status reports injected=false: ${JSON.stringify(status)}`);
}
if (refresh === "always" && !status.lastRefresh?.ok) {
  throw new Error(`status reports refresh was not ok: ${JSON.stringify(status)}`);
}

console.log(JSON.stringify({
  ok: true,
  systemEntries: output.system.length,
  firstSystemChars: output.system[0].length,
  status,
}, null, 2));
'@ | Set-Content -Encoding UTF8 -Path $nodeScript

try {
    $result = node $nodeScript
    if ($LASTEXITCODE -ne 0) {
        throw "node smoke failed"
    }
    Write-Host "OpenCode autoload hook smoke passed"
    $result
}
finally {
    Remove-Item -LiteralPath $nodeScript -Force -ErrorAction SilentlyContinue
    Remove-Item Env:\CODE2LORA_SMOKE_REPO -ErrorAction SilentlyContinue
    Remove-Item Env:\CODE2LORA_SMOKE_CONTEXT_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\CODE2LORA_SMOKE_STATUS_PATH -ErrorAction SilentlyContinue
    Remove-Item Env:\CODE2LORA_SMOKE_REFRESH -ErrorAction SilentlyContinue
}
