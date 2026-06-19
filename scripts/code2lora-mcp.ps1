param(
    [string]$RepoPath = "."
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$DefaultRepo = Resolve-Path $RepoPath

function Write-McpResponse {
    param([object]$Response)
    $Json = $Response | ConvertTo-Json -Depth 20 -Compress
    [Console]::Out.WriteLine($Json)
    [Console]::Out.Flush()
}

function New-McpTextResult {
    param([string]$Text)
    return @{
        content = @(
            @{
                type = "text"
                text = $Text
            }
        )
    }
}

function Get-ArgValue {
    param(
        [object]$Arguments,
        [string]$Name,
        [object]$DefaultValue = $null
    )
    if ($null -eq $Arguments) {
        return $DefaultValue
    }
    if ($Arguments.PSObject.Properties.Name -contains $Name) {
        $Value = $Arguments.$Name
        if ($null -ne $Value) {
            return $Value
        }
    }
    return $DefaultValue
}

function Convert-ToStringArray {
    param([object]$Value)
    $Items = New-Object System.Collections.Generic.List[string]
    if ($null -eq $Value) {
        return @()
    }
    if ($Value -is [System.Array]) {
        foreach ($Item in $Value) {
            if ($null -ne $Item) {
                $Items.Add([string]$Item)
            }
        }
    } else {
        foreach ($Item in ([string]$Value -split ",")) {
            $Trimmed = $Item.Trim()
            if ($Trimmed.Length -gt 0) {
                $Items.Add($Trimmed)
            }
        }
    }
    return $Items.ToArray()
}

function Resolve-RepoArg {
    param([object]$Arguments)
    $Repo = [string](Get-ArgValue -Arguments $Arguments -Name "repoPath" -DefaultValue $DefaultRepo.Path)
    return (Resolve-Path $Repo).Path
}

function Quote-ProcessArg {
    param([string]$Arg)
    if ($Arg -match '^[A-Za-z0-9_./:=\\-]+$') {
        return $Arg
    }
    return '"' + ($Arg -replace '"', '\"') + '"'
}

function Invoke-PowerShellFile {
    param(
        [string]$Script,
        [string[]]$ScriptArgs
    )
    $PowerShellExe = (Get-Process -Id $PID).Path
    if ([string]::IsNullOrWhiteSpace($PowerShellExe)) {
        $PowerShellExe = if ($PSVersionTable.PSEdition -eq "Core") { "pwsh" } else { "powershell" }
    }
    $AllArgs = @("-NoProfile")
    if ($PSVersionTable.PSEdition -ne "Core") {
        $AllArgs += @("-ExecutionPolicy", "Bypass")
    }
    $AllArgs += @("-File", $Script) + $ScriptArgs
    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $PowerShellExe
    $StartInfo.Arguments = ($AllArgs | ForEach-Object { Quote-ProcessArg ([string]$_) }) -join " "
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true
    $StartInfo.UseShellExecute = $false
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $StartInfo
    [void]$Process.Start()
    $Stdout = $Process.StandardOutput.ReadToEnd()
    $Stderr = $Process.StandardError.ReadToEnd()
    $Process.WaitForExit()

    $Text = @($Stdout.TrimEnd(), $Stderr.TrimEnd()) |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { [string]$_ }
    $Combined = $Text -join "`n"
    if ($Process.ExitCode -ne 0) {
        throw $Combined
    }
    return $Combined
}

function Invoke-AgentContextTool {
    param([object]$Arguments)
    $Repo = Resolve-RepoArg -Arguments $Arguments
    $OutputDir = [string](Get-ArgValue -Arguments $Arguments -Name "outputDir" -DefaultValue ".code2lora/agent-context")
    $MaxFiles = [int](Get-ArgValue -Arguments $Arguments -Name "maxFiles" -DefaultValue 24)
    $MinReduction = [double](Get-ArgValue -Arguments $Arguments -Name "minReduction" -DefaultValue 0.80)
    $Release = [bool](Get-ArgValue -Arguments $Arguments -Name "release" -DefaultValue $false)

    $Script = Join-Path $ProjectRoot "scripts/agent-context.ps1"
    $Args = @("-RepoPath", $Repo, "-OutputDir", $OutputDir, "-MaxFiles", [string]$MaxFiles, "-MinReduction", [string]$MinReduction)
    if ($Release) {
        $Args += "-Release"
    }
    $Text = Invoke-PowerShellFile -Script $Script -ScriptArgs $Args
    return New-McpTextResult -Text $Text
}

function Invoke-ReadContextTool {
    param([object]$Arguments)
    $Repo = Resolve-RepoArg -Arguments $Arguments
    $ContextDir = [string](Get-ArgValue -Arguments $Arguments -Name "contextDir" -DefaultValue ".code2lora/agent-context")
    $ContextPath = if ([System.IO.Path]::IsPathRooted($ContextDir)) {
        Join-Path $ContextDir "context.md"
    } else {
        Join-Path (Join-Path $Repo $ContextDir) "context.md"
    }
    if (-not (Test-Path $ContextPath -PathType Leaf)) {
        throw "Context file not found at $ContextPath. Call code2lora_agent_context first."
    }
    return New-McpTextResult -Text (Get-Content $ContextPath -Raw)
}

function Invoke-AgentOpenTool {
    param([object]$Arguments)
    $Repo = Resolve-RepoArg -Arguments $Arguments
    $ContextDir = [string](Get-ArgValue -Arguments $Arguments -Name "contextDir" -DefaultValue ".code2lora/agent-context")
    $Files = @(Convert-ToStringArray (Get-ArgValue -Arguments $Arguments -Name "files" -DefaultValue @()))
    $NoContent = [bool](Get-ArgValue -Arguments $Arguments -Name "noContent" -DefaultValue $false)
    if ($Files.Count -eq 0) {
        throw "files is required"
    }
    $Script = Join-Path $ProjectRoot "scripts/agent-open.ps1"
    $Args = @("-RepoPath", $Repo, "-ContextDir", $ContextDir, "-Files", ($Files -join ","))
    if ($NoContent) {
        $Args += "-NoContent"
    }
    $Text = Invoke-PowerShellFile -Script $Script -ScriptArgs $Args
    return New-McpTextResult -Text $Text
}

function Invoke-SessionAuditTool {
    param([object]$Arguments)
    $Repo = Resolve-RepoArg -Arguments $Arguments
    $ContextDir = [string](Get-ArgValue -Arguments $Arguments -Name "contextDir" -DefaultValue ".code2lora/agent-context")
    $OpenedFilesPath = [string](Get-ArgValue -Arguments $Arguments -Name "openedFilesPath" -DefaultValue ".code2lora/agent-context/opened-files.txt")
    $OpenedFiles = @(Convert-ToStringArray (Get-ArgValue -Arguments $Arguments -Name "openedFiles" -DefaultValue @()))
    $MinReduction = [double](Get-ArgValue -Arguments $Arguments -Name "minReduction" -DefaultValue 0.70)
    $Script = Join-Path $ProjectRoot "scripts/agent-session-audit.ps1"
    $Args = @("-RepoPath", $Repo, "-ContextDir", $ContextDir, "-MinReduction", [string]$MinReduction)
    if ($OpenedFilesPath.Length -gt 0) {
        $Args += @("-OpenedFilesPath", $OpenedFilesPath)
    }
    if ($OpenedFiles.Count -gt 0) {
        $Args += @("-OpenedFiles", ($OpenedFiles -join ","))
    }
    $Text = Invoke-PowerShellFile -Script $Script -ScriptArgs $Args
    return New-McpTextResult -Text $Text
}

function Get-ToolsList {
    return @(
        @{
            name = "code2lora_agent_context"
            description = "Generate compact Code2LoRA context pack and token-reduction gate for a repository."
            inputSchema = @{
                type = "object"
                properties = @{
                    repoPath = @{ type = "string"; description = "Repository path"; default = $DefaultRepo.Path }
                    outputDir = @{ type = "string"; default = ".code2lora/agent-context" }
                    maxFiles = @{ type = "integer"; default = 24 }
                    minReduction = @{ type = "number"; default = 0.80 }
                    release = @{ type = "boolean"; default = $false }
                }
            }
        },
        @{
            name = "code2lora_read_context"
            description = "Read the generated compact context.md for Codex/OpenCode before opening raw source files."
            inputSchema = @{
                type = "object"
                properties = @{
                    repoPath = @{ type = "string"; default = $DefaultRepo.Path }
                    contextDir = @{ type = "string"; default = ".code2lora/agent-context" }
                }
            }
        },
        @{
            name = "code2lora_agent_open"
            description = "Open raw repository files and record them for session token audit."
            inputSchema = @{
                type = "object"
                required = @("files")
                properties = @{
                    repoPath = @{ type = "string"; default = $DefaultRepo.Path }
                    contextDir = @{ type = "string"; default = ".code2lora/agent-context" }
                    files = @{ type = "array"; items = @{ type = "string" } }
                    noContent = @{ type = "boolean"; default = $false }
                }
            }
        },
        @{
            name = "code2lora_session_audit"
            description = "Audit token savings for compact context plus actual files opened by the agent."
            inputSchema = @{
                type = "object"
                properties = @{
                    repoPath = @{ type = "string"; default = $DefaultRepo.Path }
                    contextDir = @{ type = "string"; default = ".code2lora/agent-context" }
                    openedFilesPath = @{ type = "string"; default = ".code2lora/agent-context/opened-files.txt" }
                    openedFiles = @{ type = "array"; items = @{ type = "string" } }
                    minReduction = @{ type = "number"; default = 0.70 }
                }
            }
        }
    )
}

function Invoke-ToolByName {
    param(
        [string]$Name,
        [object]$Arguments
    )
    switch ($Name) {
        "code2lora_agent_context" { return Invoke-AgentContextTool -Arguments $Arguments }
        "code2lora_read_context" { return Invoke-ReadContextTool -Arguments $Arguments }
        "code2lora_agent_open" { return Invoke-AgentOpenTool -Arguments $Arguments }
        "code2lora_session_audit" { return Invoke-SessionAuditTool -Arguments $Arguments }
        default { throw "Unknown tool: $Name" }
    }
}

while ($null -ne ($Line = [Console]::In.ReadLine())) {
    if ([string]::IsNullOrWhiteSpace($Line)) {
        continue
    }
    try {
        $Request = $Line | ConvertFrom-Json
        $Method = [string]$Request.method
        $Id = $Request.id
        if ($Method -eq "notifications/initialized") {
            continue
        }

        switch ($Method) {
            "initialize" {
                Write-McpResponse @{
                    jsonrpc = "2.0"
                    id = $Id
                    result = @{
                        protocolVersion = "2024-11-05"
                        capabilities = @{
                            tools = @{}
                        }
                        serverInfo = @{
                            name = "code2lora-lite"
                            version = "0.1.0"
                        }
                    }
                }
            }
            "tools/list" {
                Write-McpResponse @{
                    jsonrpc = "2.0"
                    id = $Id
                    result = @{
                        tools = Get-ToolsList
                    }
                }
            }
            "tools/call" {
                $ToolName = [string]$Request.params.name
                $Arguments = $Request.params.arguments
                $Result = Invoke-ToolByName -Name $ToolName -Arguments $Arguments
                Write-McpResponse @{
                    jsonrpc = "2.0"
                    id = $Id
                    result = $Result
                }
            }
            default {
                Write-McpResponse @{
                    jsonrpc = "2.0"
                    id = $Id
                    error = @{
                        code = -32601
                        message = "Method not found: $Method"
                    }
                }
            }
        }
    } catch {
        $ResponseId = $null
        try {
            if ($null -ne $Request -and ($Request.PSObject.Properties.Name -contains "id")) {
                $ResponseId = $Request.id
            }
        } catch {
            $ResponseId = $null
        }
        Write-McpResponse @{
            jsonrpc = "2.0"
            id = $ResponseId
            error = @{
                code = -32000
                message = $_.Exception.Message
            }
        }
    }
}
