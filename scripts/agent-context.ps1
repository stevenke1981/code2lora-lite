param(
    [string]$RepoPath = ".",
    [string]$OutputDir = ".code2lora/agent-context",
    [int]$MaxFiles = 24,
    [switch]$Release
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ResolvedRepo = Resolve-Path $RepoPath

Push-Location $ProjectRoot
try {
    $profileArg = if ($Release) { "--release" } else { $null }
    $cargoArgs = @()
    if ($profileArg) {
        $cargoArgs += $profileArg
    }
    $cargoArgs += @(
        "--no-default-features",
        "--",
        "agent-context",
        $ResolvedRepo.Path,
        "--output-dir",
        $OutputDir,
        "--max-files",
        [string]$MaxFiles
    )

    $json = cargo run @cargoArgs
    $jsonText = $json -join "`n"
    $report = $jsonText | ConvertFrom-Json

    Write-Host "Code2LoRA agent context ready"
    Write-Host "Context: $($report.context_path)"
    Write-Host "Metrics: $($report.metrics_path)"
    Write-Host "Codex prompt: $($report.codex_prompt_path)"
    Write-Host "OpenCode prompt: $($report.opencode_prompt_path)"
    Write-Host ("Estimated token reduction: {0:P1}" -f [double]$report.reduction_ratio)

    $jsonText
}
finally {
    Pop-Location
}
