param(
    [ValidateSet("All", "Codex", "OpenCode")]
    [string]$Target = "All",
    [string]$RepoPath = ".",
    [string]$CodexConfigPath = "$env:USERPROFILE\.codex\config.toml",
    [string]$OpenCodeConfigPath = "$env:USERPROFILE\.config\opencode\opencode.jsonc",
    [switch]$Apply,
    [switch]$SkipSmoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$ResolvedRepo = (Resolve-Path $RepoPath).Path
$McpScript = Join-Path $ProjectRoot "scripts\code2lora-mcp.ps1"
$SmokeScript = Join-Path $ProjectRoot "scripts\mcp-smoke.ps1"
$SummaryPath = Join-Path $ResolvedRepo ".code2lora\agent-context\mcp-install-summary.json"
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Read-TextUtf8 {
    param([string]$Path)
    return [System.IO.File]::ReadAllText($Path, $Utf8NoBom)
}

function Read-LinesUtf8 {
    param([string]$Path)
    return [System.IO.File]::ReadAllLines($Path, $Utf8NoBom)
}

function Write-TextUtf8 {
    param(
        [string]$Path,
        [string]$Content
    )
    [System.IO.File]::WriteAllText($Path, $Content, $Utf8NoBom)
}

function Write-LinesUtf8 {
    param(
        [string]$Path,
        [string[]]$Lines
    )
    [System.IO.File]::WriteAllLines($Path, $Lines, $Utf8NoBom)
}

function Escape-TomlString {
    param([string]$Value)
    return '"' + (($Value -replace "\\", "\\") -replace '"', '\"') + '"'
}

function Escape-JsonString {
    param([string]$Value)
    return ($Value -replace "\\", "\\") -replace '"', '\"'
}

function Backup-Config {
    param([string]$Path)
    if (-not (Test-Path $Path -PathType Leaf)) {
        return $null
    }
    $Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $BackupPath = "$Path.code2lora-$Stamp.bak"
    Copy-Item -LiteralPath $Path -Destination $BackupPath
    return $BackupPath
}

function Get-CodexBlock {
    $Args = @(
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $McpScript,
        "-RepoPath",
        $ResolvedRepo
    )
    $ArgText = ($Args | ForEach-Object { Escape-TomlString $_ }) -join ", "
    return @"
[mcp_servers."code2lora-lite"]
command = "powershell"
args = [$ArgText]
startup_timeout_sec = 120
"@
}

function Set-CodexConfig {
    param([string]$Path)
    $Block = (Get-CodexBlock).TrimEnd()
    $Pattern = '(?ms)^\[mcp_servers\.(?:"code2lora-lite"|code2lora-lite)\]\r?\n.*?(?=^\[|\z)'

    if (Test-Path $Path -PathType Leaf) {
        $Content = Read-TextUtf8 -Path $Path
    } else {
        $Parent = Split-Path -Parent $Path
        if (-not (Test-Path $Parent)) {
            New-Item -ItemType Directory -Force -Path $Parent | Out-Null
        }
        $Content = ""
    }

    if ($Content -match $Pattern) {
        $NewContent = [regex]::Replace($Content, $Pattern, $Block + "`r`n`r`n")
    } else {
        $Separator = if ($Content.Trim().Length -gt 0) { "`r`n`r`n" } else { "" }
        $NewContent = $Content.TrimEnd() + $Separator + $Block + "`r`n"
    }

    if ($Apply) {
        Write-TextUtf8 -Path $Path -Content $NewContent
    }

    return @{
        path = $Path
        configured = ($NewContent -match '\[mcp_servers\."code2lora-lite"\]')
        applied = [bool]$Apply
    }
}

function Count-JsonBraces {
    param([string]$Line)
    $Open = ([regex]::Matches($Line, "\{")).Count
    $Close = ([regex]::Matches($Line, "\}")).Count
    return $Open - $Close
}

function Get-OpenCodeEntryLines {
    $ScriptPath = Escape-JsonString $McpScript
    $Repo = Escape-JsonString $ResolvedRepo
    return @(
        '    "code2lora-lite": {',
        '      "type": "local",',
        '      "command": [',
        '        "powershell",',
        '        "-NoProfile",',
        '        "-ExecutionPolicy",',
        '        "Bypass",',
        '        "-File",',
        "        `"$ScriptPath`",",
        '        "-RepoPath",',
        "        `"$Repo`"",
        '      ],',
        '      "enabled": true,',
        '      "timeout": 120000',
        '    }'
    )
}

function Remove-OpenCodeEntry {
    param([string[]]$Lines)
    $Start = -1
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match '^\s*"code2lora-lite"\s*:\s*\{') {
            $Start = $i
            break
        }
    }
    if ($Start -lt 0) {
        return $Lines
    }

    $Depth = 0
    $End = $Start
    for ($i = $Start; $i -lt $Lines.Count; $i++) {
        $Depth += Count-JsonBraces -Line $Lines[$i]
        if ($Depth -eq 0) {
            $End = $i
            break
        }
    }

    if ($Lines[$End].TrimEnd().EndsWith(",")) {
        $Lines[$End] = $Lines[$End].TrimEnd().TrimEnd(",")
    } elseif ($End + 1 -lt $Lines.Count -and $Lines[$End + 1].TrimStart().StartsWith(",")) {
        $Lines[$End + 1] = $Lines[$End + 1].TrimStart().Substring(1)
    }

    $Out = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($i -lt $Start -or $i -gt $End) {
            $Out.Add($Lines[$i])
        }
    }
    return $Out.ToArray()
}

function Set-OpenCodeConfig {
    param([string]$Path)
    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "OpenCode config not found at $Path"
    }

    $Lines = @(Read-LinesUtf8 -Path $Path)
    $Lines = @(Remove-OpenCodeEntry -Lines $Lines)
    $McpStart = -1
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($Lines[$i] -match '^\s*"mcp"\s*:\s*\{') {
            $McpStart = $i
            break
        }
    }
    if ($McpStart -lt 0) {
        throw "OpenCode config has no top-level mcp object: $Path"
    }

    $Depth = 0
    $McpEnd = -1
    for ($i = $McpStart; $i -lt $Lines.Count; $i++) {
        $Depth += Count-JsonBraces -Line $Lines[$i]
        if ($i -gt $McpStart -and $Depth -eq 0) {
            $McpEnd = $i
            break
        }
    }
    if ($McpEnd -lt 0) {
        throw "Could not find end of OpenCode mcp object: $Path"
    }

    $Prev = $McpEnd - 1
    while ($Prev -gt $McpStart -and [string]::IsNullOrWhiteSpace($Lines[$Prev])) {
        $Prev--
    }
    if ($Prev -gt $McpStart -and -not $Lines[$Prev].TrimEnd().EndsWith(",") -and -not $Lines[$Prev].TrimEnd().EndsWith("{")) {
        $Lines[$Prev] = $Lines[$Prev].TrimEnd() + ","
    }

    $EntryLines = Get-OpenCodeEntryLines
    $Out = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        if ($i -eq $McpEnd) {
            foreach ($Line in $EntryLines) {
                $Out.Add($Line)
            }
        }
        $Out.Add($Lines[$i])
    }

    if ($Apply) {
        Write-LinesUtf8 -Path $Path -Lines $Out.ToArray()
    }

    return @{
        path = $Path
        configured = (($Out.ToArray() -join "`n") -match '"code2lora-lite"\s*:')
        applied = [bool]$Apply
    }
}

if (-not (Test-Path $McpScript -PathType Leaf)) {
    throw "MCP script not found: $McpScript"
}

$Summary = [ordered]@{
    repo_path = $ResolvedRepo
    mcp_script = $McpScript
    target = $Target
    applied = [bool]$Apply
    smoke = $null
    codex = $null
    opencode = $null
    backups = @()
}

if (-not $SkipSmoke) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $SmokeScript -RepoPath $ResolvedRepo | Write-Host
    if ($LASTEXITCODE -ne 0) {
        throw "MCP smoke failed"
    }
    $Summary.smoke = Get-Content (Join-Path $ResolvedRepo ".code2lora\agent-context\mcp-smoke.json") -Raw | ConvertFrom-Json
}

if ($Target -eq "All" -or $Target -eq "Codex") {
    if ($Apply) {
        $Backup = Backup-Config -Path $CodexConfigPath
        if ($null -ne $Backup) {
            $Summary.backups += $Backup
        }
    }
    $Summary.codex = Set-CodexConfig -Path $CodexConfigPath
}

if ($Target -eq "All" -or $Target -eq "OpenCode") {
    if ($Apply) {
        $Backup = Backup-Config -Path $OpenCodeConfigPath
        if ($null -ne $Backup) {
            $Summary.backups += $Backup
        }
    }
    $Summary.opencode = Set-OpenCodeConfig -Path $OpenCodeConfigPath
}

$SummaryDir = Split-Path -Parent $SummaryPath
if (-not (Test-Path $SummaryDir)) {
    New-Item -ItemType Directory -Force -Path $SummaryDir | Out-Null
}
Write-TextUtf8 -Path $SummaryPath -Content ($Summary | ConvertTo-Json -Depth 8)

if ($Apply) {
    Write-Host "Code2LoRA MCP config installed"
} else {
    Write-Host "Code2LoRA MCP config dry-run passed; rerun with -Apply to write config"
}
Write-Host "Summary: $SummaryPath"
$Summary | ConvertTo-Json -Depth 8
