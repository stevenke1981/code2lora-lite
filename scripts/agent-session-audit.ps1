param(
    [string]$RepoPath = ".",
    [string]$ContextDir = ".code2lora/agent-context",
    [string[]]$OpenedFiles = @(),
    [string]$OpenedFilesPath = "",
    [double]$MinReduction = 0.70
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-TokenEstimate {
    param([string]$Text)
    return [math]::Ceiling($Text.Length / 4.0)
}

function Get-RelativeRepoPath {
    param(
        [string]$Root,
        [string]$Path
    )
    $RootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $PathFull = [System.IO.Path]::GetFullPath($Path)
    $RootUri = New-Object System.Uri($RootFull)
    $PathUri = New-Object System.Uri($PathFull)
    $RelativeUri = $RootUri.MakeRelativeUri($PathUri)
    return [System.Uri]::UnescapeDataString($RelativeUri.ToString()).Replace('/', [System.IO.Path]::DirectorySeparatorChar)
}

$ResolvedRepo = Resolve-Path $RepoPath
$ResolvedContextDir = if ([System.IO.Path]::IsPathRooted($ContextDir)) {
    Resolve-Path $ContextDir
} else {
    Resolve-Path (Join-Path $ResolvedRepo.Path $ContextDir)
}
$MetricsPath = Join-Path $ResolvedContextDir.Path "metrics.json"
if (-not (Test-Path $MetricsPath)) {
    throw "metrics.json not found at $MetricsPath. Run scripts/agent-context.ps1 first."
}

$Metrics = Get-Content $MetricsPath -Raw | ConvertFrom-Json
$Paths = New-Object System.Collections.Generic.List[string]

foreach ($Path in $OpenedFiles) {
    foreach ($Item in ($Path -split ",")) {
        $TrimmedItem = $Item.Trim()
        if (-not [string]::IsNullOrWhiteSpace($TrimmedItem)) {
            $Paths.Add($TrimmedItem)
        }
    }
}

if (-not [string]::IsNullOrWhiteSpace($OpenedFilesPath)) {
    $ResolvedOpenedFilesPath = if ([System.IO.Path]::IsPathRooted($OpenedFilesPath)) {
        Resolve-Path $OpenedFilesPath
    } else {
        Resolve-Path (Join-Path $ResolvedRepo.Path $OpenedFilesPath)
    }
    foreach ($Line in Get-Content $ResolvedOpenedFilesPath.Path) {
        $Trimmed = $Line.Trim()
        if ($Trimmed.Length -gt 0 -and -not $Trimmed.StartsWith("#")) {
            $Paths.Add($Trimmed)
        }
    }
}

$UniqueFiles = [ordered]@{}
foreach ($Path in $Paths) {
    $Candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $ResolvedRepo.Path $Path
    }
    if (-not (Test-Path $Candidate -PathType Leaf)) {
        throw "Opened file not found: $Path"
    }
    $Resolved = Resolve-Path $Candidate
    if (-not $Resolved.Path.StartsWith($ResolvedRepo.Path, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Opened file is outside repo: $($Resolved.Path)"
    }
    $Relative = Get-RelativeRepoPath -Root $ResolvedRepo.Path -Path $Resolved.Path
    $UniqueFiles[$Relative] = $Resolved.Path
}

$OpenedDetails = @()
$OpenedTokenEstimate = 0
foreach ($Entry in $UniqueFiles.GetEnumerator()) {
    $Text = Get-Content $Entry.Value -Raw
    $Tokens = Get-TokenEstimate -Text $Text
    $OpenedTokenEstimate += $Tokens
    $OpenedDetails += [ordered]@{
        path = $Entry.Key
        token_estimate = $Tokens
        chars = $Text.Length
    }
}

$RawTokenEstimate = [int]$Metrics.raw_token_estimate
$ContextTokenEstimate = [int]$Metrics.context_token_estimate
$SessionTokenEstimate = $ContextTokenEstimate + $OpenedTokenEstimate
$SavedTokenEstimate = [math]::Max(0, $RawTokenEstimate - $SessionTokenEstimate)
$ReductionRatio = if ($RawTokenEstimate -eq 0) { 0.0 } else { $SavedTokenEstimate / $RawTokenEstimate }
$Passed = $ReductionRatio -ge $MinReduction

$AuditPath = Join-Path $ResolvedContextDir.Path "session-audit.json"
$Audit = [ordered]@{
    passed = $Passed
    min_reduction = $MinReduction
    reduction_ratio = $ReductionRatio
    raw_token_estimate = $RawTokenEstimate
    context_token_estimate = $ContextTokenEstimate
    opened_file_token_estimate = $OpenedTokenEstimate
    session_token_estimate = $SessionTokenEstimate
    saved_token_estimate = $SavedTokenEstimate
    opened_file_count = $OpenedDetails.Count
    opened_files = $OpenedDetails
    metrics_path = $MetricsPath
}

$Audit | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path $AuditPath
Write-Host "Code2LoRA agent session audit ready"
Write-Host "Audit: $AuditPath"
Write-Host ("Estimated session reduction: {0:P1}" -f [double]$ReductionRatio)

if (-not $Passed) {
    throw ("Session token reduction gate failed: {0:P1} < required {1:P1}" -f [double]$ReductionRatio, $MinReduction)
}

$Audit | ConvertTo-Json -Depth 6
