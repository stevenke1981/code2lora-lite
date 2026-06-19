param(
    [string]$RepoPath = ".",
    [string]$OutputDir = ".code2lora/agent-context",
    [int]$MaxFiles = 24,
    [double]$MinReduction = 0.80,
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

    $passed = [double]$report.reduction_ratio -ge $MinReduction
    $auditPath = Join-Path (Split-Path -Parent $report.metrics_path) "audit.json"
    $audit = [ordered]@{
        passed = $passed
        min_reduction = $MinReduction
        reduction_ratio = [double]$report.reduction_ratio
        raw_token_estimate = [int]$report.raw_token_estimate
        context_token_estimate = [int]$report.context_token_estimate
        saved_token_estimate = [int]$report.saved_token_estimate
        symbols_included = [int]$report.symbols_included
        metrics_path = $report.metrics_path
        context_path = $report.context_path
    }
    $audit | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $auditPath
    Write-Host "Audit: $auditPath"

    if (-not $passed) {
        throw ("Token reduction gate failed: {0:P1} < required {1:P1}" -f [double]$report.reduction_ratio, $MinReduction)
    }

    $jsonText
}
finally {
    Pop-Location
}
