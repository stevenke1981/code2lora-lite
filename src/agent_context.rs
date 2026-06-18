use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use walkdir::{DirEntry, WalkDir};

const MAX_FILE_BYTES: u64 = 1_000_000;

#[derive(Debug, Clone, Serialize)]
pub struct AgentContextReport {
    pub repo_path: String,
    pub output_dir: String,
    pub context_path: String,
    pub metrics_path: String,
    pub codex_prompt_path: String,
    pub opencode_prompt_path: String,
    pub files_scanned: usize,
    pub files_included: usize,
    pub raw_chars: usize,
    pub raw_token_estimate: usize,
    pub context_chars: usize,
    pub context_token_estimate: usize,
    pub saved_token_estimate: usize,
    pub reduction_ratio: f64,
    pub generated_at_unix: u64,
}

#[derive(Debug, Clone, Serialize)]
struct FileSignal {
    path: String,
    language: String,
    bytes: u64,
    lines: usize,
    score: i64,
    reason: String,
}

pub fn write_agent_context(
    repo_path: &Path,
    output_dir: &Path,
    max_files: usize,
) -> Result<AgentContextReport> {
    let repo_path = repo_path
        .canonicalize()
        .with_context(|| format!("Failed to resolve repo path {}", repo_path.display()))?;
    let output_dir = if output_dir.is_absolute() {
        output_dir.to_path_buf()
    } else {
        repo_path.join(output_dir)
    };
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("Failed to create {}", output_dir.display()))?;

    let scan = scan_repo(&repo_path, max_files)?;
    let context_path = output_dir.join("context.md");
    let metrics_path = output_dir.join("metrics.json");
    let codex_prompt_path = output_dir.join("codex-prompt.md");
    let opencode_prompt_path = output_dir.join("opencode-prompt.md");

    let mut context = render_context_markdown(&repo_path, &scan.signals, scan.raw_chars);
    let context_chars = context.chars().count();
    let raw_token_estimate = estimate_tokens(scan.raw_chars);
    let context_token_estimate = estimate_tokens(context_chars);
    let saved_token_estimate = raw_token_estimate.saturating_sub(context_token_estimate);
    let reduction_ratio = if raw_token_estimate == 0 {
        0.0
    } else {
        saved_token_estimate as f64 / raw_token_estimate as f64
    };

    context.push_str(&format!(
        "\n## Token Savings Estimate\n\n- Raw repo text estimate: {raw_token_estimate} tokens\n- Context pack estimate: {context_token_estimate} tokens\n- Estimated saved tokens: {saved_token_estimate}\n- Reduction ratio: {:.1}%\n",
        reduction_ratio * 100.0
    ));

    fs::write(&context_path, &context)
        .with_context(|| format!("Failed to write {}", context_path.display()))?;

    let report = AgentContextReport {
        repo_path: display_path(&repo_path),
        output_dir: display_path(&output_dir),
        context_path: display_path(&context_path),
        metrics_path: display_path(&metrics_path),
        codex_prompt_path: display_path(&codex_prompt_path),
        opencode_prompt_path: display_path(&opencode_prompt_path),
        files_scanned: scan.files_scanned,
        files_included: scan.signals.len(),
        raw_chars: scan.raw_chars,
        raw_token_estimate,
        context_chars: context.chars().count(),
        context_token_estimate,
        saved_token_estimate,
        reduction_ratio,
        generated_at_unix: unix_time(),
    };

    fs::write(&metrics_path, serde_json::to_string_pretty(&report)?)
        .with_context(|| format!("Failed to write {}", metrics_path.display()))?;
    fs::write(
        &codex_prompt_path,
        render_agent_prompt("Codex", &context_path, &report)?,
    )
    .with_context(|| format!("Failed to write {}", codex_prompt_path.display()))?;
    fs::write(
        &opencode_prompt_path,
        render_agent_prompt("OpenCode", &context_path, &report)?,
    )
    .with_context(|| format!("Failed to write {}", opencode_prompt_path.display()))?;

    Ok(report)
}

fn render_context_markdown(repo_path: &Path, signals: &[FileSignal], raw_chars: usize) -> String {
    let mut by_language: BTreeMap<&str, usize> = BTreeMap::new();
    for signal in signals {
        *by_language.entry(&signal.language).or_default() += 1;
    }

    let mut out = String::new();
    out.push_str("# Code2LoRA Agent Context Pack\n\n");
    out.push_str("Use this compact pack before opening broad source files. It is designed for Codex/OpenCode runs where token budget matters.\n\n");
    out.push_str(&format!("- Repository: `{}`\n", display_path(repo_path)));
    out.push_str(&format!("- Raw scanned characters: `{raw_chars}`\n"));
    out.push_str("- Adapter artifact path, when available: pass `checkpoints/final.safetensors` to `code2lora-lite adapt`.\n");
    out.push_str("- Reduced-context rule: inspect the files below first, then open additional files only when the task requires exact code.\n\n");

    out.push_str("## Agent Workflow\n\n");
    out.push_str(
        "1. Read this context pack instead of dumping the whole repository into the prompt.\n",
    );
    out.push_str("2. For code generation, create or reuse an adapter with `code2lora-lite adapt <repo> -m <hypernetwork> -o <adapter>`.\n");
    out.push_str("3. For assertion completion, call `code2lora-lite complete <repo> <adapter> --prefix <code>`.\n");
    out.push_str("4. If this pack lacks evidence for a change, open only the listed target files plus direct dependencies.\n\n");

    out.push_str("## Language Mix\n\n");
    for (language, count) in by_language {
        out.push_str(&format!("- {language}: {count} selected files\n"));
    }

    out.push_str("\n## High-Signal Files\n\n");
    out.push_str("| Path | Lang | Lines | Bytes | Why |\n");
    out.push_str("|------|------|-------|-------|-----|\n");
    for signal in signals {
        out.push_str(&format!(
            "| `{}` | {} | {} | {} | {} |\n",
            signal.path,
            signal.language,
            signal.lines,
            signal.bytes,
            signal.reason.replace('|', "/")
        ));
    }
    out
}

fn render_agent_prompt(
    agent: &str,
    context_path: &Path,
    report: &AgentContextReport,
) -> Result<String> {
    Ok(format!(
        "# {agent} Prompt\n\nRead `{}` first. Treat it as the compact repository context for this task.\n\nToken evidence:\n- Raw token estimate: {}\n- Context token estimate: {}\n- Estimated reduction: {:.1}%\n\nOnly open raw source files when the compact context does not contain enough evidence.\n",
        display_path(context_path),
        report.raw_token_estimate,
        report.context_token_estimate,
        report.reduction_ratio * 100.0
    ))
}

struct ScanResult {
    files_scanned: usize,
    raw_chars: usize,
    signals: Vec<FileSignal>,
}

fn scan_repo(repo_path: &Path, max_files: usize) -> Result<ScanResult> {
    let mut files_scanned = 0;
    let mut raw_chars = 0;
    let mut signals = Vec::new();

    for entry in WalkDir::new(repo_path)
        .into_iter()
        .filter_entry(should_descend)
        .filter_map(|entry| entry.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if !is_text_candidate(path) {
            continue;
        }
        let metadata = match entry.metadata() {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if metadata.len() > MAX_FILE_BYTES {
            continue;
        }
        let content = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(_) => continue,
        };
        files_scanned += 1;
        raw_chars += content.chars().count();

        let relative = relative_path(repo_path, path);
        let language = language_for(path);
        let lines = content.lines().count();
        let (score, reason) = score_file(&relative, path, lines, metadata.len());
        signals.push(FileSignal {
            path: relative,
            language,
            bytes: metadata.len(),
            lines,
            score,
            reason,
        });
    }

    signals.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
    signals.truncate(max_files.max(1));

    Ok(ScanResult {
        files_scanned,
        raw_chars,
        signals,
    })
}

fn should_descend(entry: &DirEntry) -> bool {
    let name = entry.file_name().to_string_lossy();
    !matches!(
        name.as_ref(),
        ".git"
            | "target"
            | ".cache"
            | ".code2lora"
            | ".codebase-memory"
            | ".opencode"
            | "node_modules"
            | "data"
            | "checkpoints"
            | "p7_checkpoints"
            | "${PROJECT_ROOT}"
    )
}

fn is_text_candidate(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    if matches!(name, "Cargo.lock") {
        return false;
    }
    matches!(
        name,
        "README.md" | "README.zh-TW.md" | "AGENTS.md" | "SPEC.md" | "PLAN.md" | "todos.md"
    ) || matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default(),
        "rs" | "toml" | "md" | "ps1" | "py" | "json" | "jsonc" | "yaml" | "yml" | "txt"
    )
}

fn score_file(relative: &str, path: &Path, lines: usize, bytes: u64) -> (i64, String) {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    let ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    let mut score = 100;
    let mut reasons = Vec::new();

    if matches!(name, "README.md" | "README.zh-TW.md") {
        score += 180;
        reasons.push("project overview");
    }
    if matches!(name, "Cargo.toml" | "AGENTS.md" | "todos.md") {
        score += 160;
        reasons.push("agent/build control");
    }
    if relative.starts_with("src/") {
        score += 120;
        reasons.push("runtime code");
    }
    if relative.contains("infer")
        || relative.contains("trainer")
        || relative.contains("repo_encoder")
    {
        score += 90;
        reasons.push("core Code2LoRA path");
    }
    if matches!(ext, "rs" | "toml" | "ps1") {
        score += 40;
        reasons.push("executable/config source");
    }
    if lines > 300 {
        score += 20;
        reasons.push("large surface");
    }
    if bytes > 200_000 {
        score -= 40;
        reasons.push("large file");
    }
    if reasons.is_empty() {
        reasons.push("supporting context");
    }
    (score, reasons.join(", "))
}

fn language_for(path: &Path) -> String {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
    {
        "rs" => "rust",
        "toml" => "toml",
        "md" => "markdown",
        "ps1" => "powershell",
        "py" => "python",
        "json" | "jsonc" => "json",
        "yaml" | "yml" => "yaml",
        other if other.is_empty() => "text",
        other => other,
    }
    .to_string()
}

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn display_path(path: &Path) -> String {
    let text = path.display().to_string();
    text.strip_prefix("\\\\?\\").unwrap_or(&text).to_string()
}

fn estimate_tokens(chars: usize) -> usize {
    chars.div_ceil(4)
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_context_reports_token_reduction() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "code2lora-agent-context-test-{}",
            std::process::id()
        ));
        let src = root.join("src");
        fs::create_dir_all(&src)?;
        fs::write(root.join("Cargo.toml"), "[package]\nname = \"demo\"\n")?;
        fs::write(
            root.join("README.md"),
            "# Demo\n\nThis repository demonstrates a small codebase.\n",
        )?;
        fs::write(
            src.join("lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n".repeat(200),
        )?;

        let report = write_agent_context(&root, Path::new(".code2lora/agent-context"), 8)?;
        assert!(std::path::PathBuf::from(&report.context_path).exists());
        assert!(std::path::PathBuf::from(&report.metrics_path).exists());
        assert!(std::path::PathBuf::from(&report.codex_prompt_path).exists());
        assert!(std::path::PathBuf::from(&report.opencode_prompt_path).exists());
        assert!(report.raw_token_estimate > report.context_token_estimate);
        assert!(report.reduction_ratio > 0.5);

        fs::remove_dir_all(&root).ok();
        Ok(())
    }
}
