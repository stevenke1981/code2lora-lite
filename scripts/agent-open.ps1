param(
    [string]$RepoPath = ".",
    [string]$ContextDir = ".code2lora/agent-context",
    [string[]]$Files = @(),
    [switch]$NoContent
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

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
$ContextPath = if ([System.IO.Path]::IsPathRooted($ContextDir)) {
    $ContextDir
} else {
    Join-Path $ResolvedRepo.Path $ContextDir
}
New-Item -ItemType Directory -Force -Path $ContextPath | Out-Null
$OpenedFilesPath = Join-Path $ContextPath "opened-files.txt"

$RequestedFiles = New-Object System.Collections.Generic.List[string]
foreach ($File in $Files) {
    foreach ($Item in ($File -split ",")) {
        $TrimmedItem = $Item.Trim()
        if (-not [string]::IsNullOrWhiteSpace($TrimmedItem)) {
            $RequestedFiles.Add($TrimmedItem)
        }
    }
}

if ($RequestedFiles.Count -eq 0) {
    throw "No files provided. Use -Files path1,path2 or -Files @('path1','path2')."
}

$Existing = [ordered]@{}
if (Test-Path $OpenedFilesPath) {
    foreach ($Line in Get-Content $OpenedFilesPath) {
        $Trimmed = $Line.Trim()
        if ($Trimmed.Length -gt 0 -and -not $Trimmed.StartsWith("#")) {
            $Existing[$Trimmed] = $true
        }
    }
}

$ResolvedFiles = @()
foreach ($File in $RequestedFiles) {
    $Candidate = if ([System.IO.Path]::IsPathRooted($File)) {
        $File
    } else {
        Join-Path $ResolvedRepo.Path $File
    }
    if (-not (Test-Path $Candidate -PathType Leaf)) {
        throw "File not found: $File"
    }
    $Resolved = Resolve-Path $Candidate
    if (-not $Resolved.Path.StartsWith($ResolvedRepo.Path, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "File is outside repo: $($Resolved.Path)"
    }
    $Relative = Get-RelativeRepoPath -Root $ResolvedRepo.Path -Path $Resolved.Path
    $Existing[$Relative] = $true
    $ResolvedFiles += [ordered]@{
        path = $Relative
        absolute_path = $Resolved.Path
    }
}

$Existing.Keys | Set-Content -Encoding UTF8 -Path $OpenedFilesPath

Write-Host "Recorded opened files:"
foreach ($File in $ResolvedFiles) {
    Write-Host "- $($File.path)"
}
Write-Host "Opened-files log: $OpenedFilesPath"

if (-not $NoContent) {
    foreach ($File in $ResolvedFiles) {
        Write-Output ""
        Write-Output "===== $($File.path) ====="
        Get-Content $File.absolute_path -Raw
    }
}
