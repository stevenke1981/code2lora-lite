param(
    [string]$RepoPath = "."
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$ResolvedRepo = Resolve-Path $RepoPath
$McpScript = Join-Path $ProjectRoot "scripts/code2lora-mcp.ps1"

function New-RequestJson {
    param(
        [int]$Id,
        [string]$Method,
        [object]$Params
    )
    return @{
        jsonrpc = "2.0"
        id = $Id
        method = $Method
        params = $Params
    } | ConvertTo-Json -Depth 12 -Compress
}

$Requests = @(
    (New-RequestJson -Id 1 -Method "initialize" -Params @{
        protocolVersion = "2024-11-05"
        capabilities = @{}
        clientInfo = @{ name = "code2lora-mcp-smoke"; version = "0" }
    }),
    (New-RequestJson -Id 2 -Method "tools/list" -Params @{}),
    (New-RequestJson -Id 3 -Method "tools/call" -Params @{
        name = "code2lora_agent_context"
        arguments = @{
            repoPath = $ResolvedRepo.Path
            minReduction = 0.80
            maxFiles = 24
        }
    }),
    (New-RequestJson -Id 4 -Method "tools/call" -Params @{
        name = "code2lora_read_context"
        arguments = @{
            repoPath = $ResolvedRepo.Path
        }
    }),
    (New-RequestJson -Id 5 -Method "tools/call" -Params @{
        name = "code2lora_agent_open"
        arguments = @{
            repoPath = $ResolvedRepo.Path
            files = @("AGENTS.md", "scripts/code2lora-mcp.ps1")
            noContent = $true
        }
    }),
    (New-RequestJson -Id 6 -Method "tools/call" -Params @{
        name = "code2lora_session_audit"
        arguments = @{
            repoPath = $ResolvedRepo.Path
            minReduction = 0.70
        }
    })
)

$Output = $Requests | powershell -NoProfile -ExecutionPolicy Bypass -File $McpScript -RepoPath $ResolvedRepo.Path
$Responses = @()
foreach ($Line in $Output) {
    if (-not [string]::IsNullOrWhiteSpace($Line)) {
        $Responses += ($Line | ConvertFrom-Json)
    }
}

if ($Responses.Count -ne 6) {
    throw "Expected 6 MCP responses, got $($Responses.Count)"
}

foreach ($Response in $Responses) {
    if ($Response.PSObject.Properties.Name -contains "error") {
        throw "MCP response $($Response.id) failed: $($Response.error.message)"
    }
}

$Tools = $Responses[1].result.tools
foreach ($ToolName in @("code2lora_agent_context", "code2lora_read_context", "code2lora_agent_open", "code2lora_session_audit")) {
    if (-not ($Tools | Where-Object { $_.name -eq $ToolName })) {
        throw "Missing MCP tool: $ToolName"
    }
}

$ContextText = [string]$Responses[3].result.content[0].text
if ($ContextText -notmatch "Code2LoRA Agent Context") {
    throw "code2lora_read_context returned unexpected content"
}

$AuditPath = Join-Path $ResolvedRepo.Path ".code2lora/agent-context/session-audit.json"
$Audit = Get-Content $AuditPath -Raw | ConvertFrom-Json
if (-not [bool]$Audit.passed) {
    throw "Session audit did not pass"
}

$Summary = [ordered]@{
    passed = $true
    response_count = $Responses.Count
    tools_count = $Tools.Count
    session_reduction_ratio = [double]$Audit.reduction_ratio
    session_token_estimate = [int]$Audit.session_token_estimate
    saved_token_estimate = [int]$Audit.saved_token_estimate
    audit_path = $AuditPath
}

$SummaryPath = Join-Path $ResolvedRepo.Path ".code2lora/agent-context/mcp-smoke.json"
$Summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 -Path $SummaryPath
Write-Host "Code2LoRA MCP smoke passed"
Write-Host "Summary: $SummaryPath"
$Summary | ConvertTo-Json -Depth 4
