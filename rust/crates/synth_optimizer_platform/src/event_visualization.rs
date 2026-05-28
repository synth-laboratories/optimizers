use std::io::{self, Write};
use std::sync::{Mutex, OnceLock};
use std::{env, fmt::Write as _};

use serde_json::Value;

pub(crate) fn terminal_events_enabled() -> bool {
    env::var("SYNTH_OPTIMIZERS_TERMINAL")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn terminal_detail_enabled(topic: &str) -> bool {
    env::var("SYNTH_OPTIMIZERS_TERMINAL_DETAIL")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.trim().to_ascii_lowercase())
                .any(|item| matches!(item.as_str(), "1" | "true" | "debug") || item == topic)
        })
        .unwrap_or(false)
}

pub(crate) fn render_terminal_event(event_type: &str, message: &str, fields: &Value) {
    let lines = terminal_lines_for_event(event_type, message, fields);
    if lines.is_empty() {
        return;
    }
    let mut stdout = io::stdout().lock();
    for line in lines {
        let _ = writeln!(stdout, "{line}");
    }
    let _ = stdout.flush();
}

fn terminal_lines_for_event(event_type: &str, message: &str, fields: &Value) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(summary) = maybe_rollout_section_summary_before_event(event_type, fields) {
        lines.push(summary);
    }
    if let Some(line) = terminal_line_for_event(event_type, message, fields) {
        lines.push(line);
    }
    if let Some(summary) = maybe_rollout_section_summary_after_event(event_type) {
        lines.push(summary);
    }
    lines
}

pub(crate) fn terminal_line_for_event(
    event_type: &str,
    message: &str,
    fields: &Value,
) -> Option<String> {
    match event_type {
        "gepa.run.started" => Some(format!(
            "{} {}",
            bold("GEPA run"),
            field_str(fields, "run_id").unwrap_or("unknown")
        )),
        "container.program.loaded" => {
            let modules = field_array_strings(fields, "mutable_fields").join(", ");
            Some(format!(
                "  program: {}  mutable={}",
                field_str(fields, "program_id").unwrap_or("unknown"),
                if modules.is_empty() {
                    "-".to_string()
                } else {
                    modules
                }
            ))
        }
        "dataset.rows.loaded" => Some(format!(
            "  dataset: train={} heldout={}",
            field_usize(fields, "train_rows").unwrap_or(0),
            field_usize(fields, "heldout_rows").unwrap_or(0)
        )),
        "candidate.evaluated" => Some(format!(
            "  seed {} train={}",
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown")),
            fmt_score(field_f64(fields, "train_reward"))
        )),
        "optimizer.state.transitioned" => terminal_state_transition_line(message, fields),
        "proposer.started" => Some(terminal_proposer_started_line(fields)),
        "proposer.completed" => Some(format!(
            "  generation {} proposer finished backend={} wall={} candidates={} warnings={}",
            field_usize(fields, "generation").unwrap_or(0),
            field_str(fields, "backend").unwrap_or("unknown"),
            field_f64(fields, "wall_seconds")
                .map(fmt_seconds)
                .unwrap_or_else(|| "-".to_string()),
            field_usize(fields, "proposal_count").unwrap_or(0),
            field_usize(fields, "warning_count").unwrap_or(0)
        )),
        "rollout.chunk.started" => Some(terminal_rollout_progress_line(fields, false)),
        "rollout.chunk.finished" => Some(terminal_rollout_progress_line(fields, true)),
        "candidate.duplicate_skipped" => Some(format!(
            "  generation {} duplicate skipped {}",
            field_usize(fields, "generation").unwrap_or(0),
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown"))
        )),
        "candidate.minibatch_evaluated" => Some(format!(
            "  candidate {} minibatch={} parent={}",
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown")),
            fmt_score(field_f64(fields, "minibatch_reward")),
            fmt_score(field_f64(fields, "parent_minibatch_reward"))
        )),
        "candidate.full_train_evaluated" => Some(format!(
            "  candidate {} train={} best={}",
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown")),
            fmt_score(field_f64(fields, "train_reward")),
            fmt_score(field_f64(fields, "best_train_reward"))
        )),
        "candidate.accepted" => Some(format!(
            "  {} {} {}",
            green("accepted"),
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown")),
            terminal_reason(fields)
        )),
        "candidate.rejected" => Some(format!(
            "  {} {} {}",
            red("rejected"),
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown")),
            terminal_reason(fields)
        )),
        "candidate.deferred" => Some(format!(
            "  deferred {} {}",
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown")),
            terminal_reason(fields)
        )),
        "frontier.updated" => Some(terminal_frontier_update_line(fields)),
        "frontier.snapshot" => Some(terminal_frontier_snapshot_line(fields)),
        "heldout.completed" => Some(format!(
            "  heldout {} train={} heldout={}",
            short_id(field_str(fields, "candidate_id").unwrap_or("unknown")),
            fmt_score(field_f64(fields, "train_reward")),
            fmt_score(field_f64(fields, "heldout_reward"))
        )),
        "heldout.skipped" => Some(format!(
            "  heldout skipped: required={} available={} best={}",
            field_usize(fields, "required_rollouts").unwrap_or(0),
            field_usize(fields, "available_rollouts").unwrap_or(0),
            short_id(field_str(fields, "best_candidate_id").unwrap_or("unknown"))
        )),
        "runtime.job.completed" => Some(terminal_runtime_job_completed_line(fields)),
        "runtime.throughput.warning" => Some(terminal_runtime_throughput_warning_line(fields)),
        "rollout.concurrency.adjusted" => Some(format!(
            "  adaptive rollout concurrency {} old={} new={} completed={} reason={}",
            field_str(fields, "direction").unwrap_or("-"),
            field_usize(fields, "old_limit").unwrap_or(0),
            field_usize(fields, "new_limit").unwrap_or(0),
            field_usize(fields, "completed_rollouts").unwrap_or(0),
            field_str(fields, "reason").unwrap_or("-")
        )),
        "score_chart.written" => Some(terminal_score_chart_line(fields)),
        "workspace.persisted" => None,
        "gepa.run.finished" => Some(terminal_finished_line(fields)),
        "gepa.stop" => Some(format!("  stop: {message}")),
        _ => None,
    }
}

fn terminal_score_chart_line(fields: &Value) -> String {
    let rows = fields
        .get("candidates")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if rows.is_empty() {
        return format!(
            "\n    {}\n      chart: {}",
            bold("GEPA visual summary"),
            field_str(fields, "chart_path").unwrap_or("-")
        );
    }

    let train_values = rows
        .iter()
        .filter_map(|row| field_f64(row, "train_reward"))
        .collect::<Vec<_>>();
    let heldout_values = rows
        .iter()
        .filter_map(|row| field_f64(row, "heldout_reward"))
        .collect::<Vec<_>>();
    let seed_id = field_str(fields, "seed_candidate_id").unwrap_or("-");
    let best_id = field_str(fields, "best_candidate_id").unwrap_or("-");
    let seed_heldout = rows
        .iter()
        .find(|row| field_bool(row, "is_seed").unwrap_or(false))
        .and_then(|row| field_f64(row, "heldout_reward"))
        .unwrap_or(0.0);
    let max_heldout = heldout_values
        .iter()
        .copied()
        .fold(1.0_f64, f64::max)
        .max(1.0);
    let heldout_is_tied = values_are_tied(&heldout_values);

    let mut out = String::new();
    let _ = write!(
        out,
        "\n    {}\n      train    {:<8} {} -> {}\n      heldout  {:<8} {} -> {}\n",
        bold("GEPA score summary"),
        ascii_trajectory(&train_values),
        fmt_score(train_values.first().copied()),
        fmt_score(max_f64(&train_values)),
        ascii_trajectory(&heldout_values),
        fmt_score(heldout_values.first().copied()),
        fmt_score(max_f64(&heldout_values)),
    );
    if heldout_is_tied {
        let _ = writeln!(
            out,
            "      heldout tie: {} candidates at {}",
            heldout_values.len(),
            fmt_score(heldout_values.first().copied())
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "      {:<9} {:<17} {:>7} {:>7} {:>8}  bar",
        "role", "candidate", "train", "heldout", "lift"
    );
    let mut ranked = rows.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        field_f64(right, "heldout_reward")
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&field_f64(left, "heldout_reward").unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| field_bool(right, "is_best").cmp(&field_bool(left, "is_best")))
            .then_with(|| field_bool(right, "is_seed").cmp(&field_bool(left, "is_seed")))
            .then_with(|| {
                field_f64(right, "train_reward")
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&field_f64(left, "train_reward").unwrap_or(f64::NEG_INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let max_rows = 10usize;
    for (rank, row) in ranked.iter().take(max_rows).enumerate() {
        let candidate_id = field_str(row, "candidate_id").unwrap_or("unknown");
        let heldout = field_f64(row, "heldout_reward").unwrap_or(0.0);
        let lift = heldout - seed_heldout;
        let is_seed = candidate_id == seed_id;
        let is_best = candidate_id == best_id;
        let role = match (is_seed, is_best) {
            (true, true) => "best+seed".to_string(),
            (true, false) => "seed".to_string(),
            (false, true) => "best".to_string(),
            (false, false) if heldout_is_tied => "tied".to_string(),
            (false, false) => format!("#{}", rank + 1),
        };
        let _ = writeln!(
            out,
            "      {:<9} {:<17} {:>7} {:>7} {:>8}  [{}]",
            role,
            short_id(candidate_id),
            fmt_score(field_f64(row, "train_reward")),
            fmt_score(field_f64(row, "heldout_reward")),
            format!("{lift:+.3}"),
            ascii_score_bar(heldout, 14, max_heldout)
        );
    }
    if ranked.len() > max_rows {
        let _ = writeln!(out, "      ... {} more candidates", ranked.len() - max_rows);
    }
    if !heldout_is_tied {
        append_score_scatter(&mut out, rows, seed_id, best_id);
    }
    append_baseline_to_best_diff(&mut out, fields);
    append_candidate_prompt_diffs(&mut out, fields);
    out
}

fn append_baseline_to_best_diff(out: &mut String, fields: &Value) {
    let diffs = fields
        .get("baseline_to_best_diff")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let _ = writeln!(out, "\n      baseline -> best diff");
    if diffs.is_empty() {
        let seed_id = field_str(fields, "seed_candidate_id").unwrap_or("-");
        let best_id = field_str(fields, "best_candidate_id").unwrap_or("-");
        let reason = if seed_id == best_id {
            "best is still the seed"
        } else {
            "best prompt payload matches the seed"
        };
        let _ = writeln!(out, "      none ({reason})");
        return;
    }
    for diff in diffs {
        let module = field_str(diff, "module").unwrap_or("module");
        let before = field_str(diff, "before").unwrap_or("");
        let after = field_str(diff, "after").unwrap_or("");
        let _ = writeln!(out, "      {module}");
        let _ = writeln!(out, "        - {}", truncate_inline(before, 180));
        let _ = writeln!(out, "        + {}", truncate_inline(after, 180));
    }
}

fn append_candidate_prompt_diffs(out: &mut String, fields: &Value) {
    let diffs = fields
        .get("candidate_prompt_diffs")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let _ = writeln!(out, "\n      candidate prompt diffs");
    if diffs.is_empty() {
        let _ = writeln!(out, "      none");
        return;
    }
    let mut ranked = diffs.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        field_f64(right, "train_reward")
            .unwrap_or(f64::NEG_INFINITY)
            .partial_cmp(&field_f64(left, "train_reward").unwrap_or(f64::NEG_INFINITY))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                field_f64(right, "heldout_reward")
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&field_f64(left, "heldout_reward").unwrap_or(f64::NEG_INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let max_candidates = 6usize;
    let max_modules = 3usize;
    for candidate in ranked.iter().take(max_candidates) {
        let candidate_id = field_str(candidate, "candidate_id").unwrap_or("unknown");
        let _ = writeln!(
            out,
            "      {} train={} heldout={} status={}",
            short_id(candidate_id),
            fmt_score(field_f64(candidate, "train_reward")),
            fmt_score(field_f64(candidate, "heldout_reward")),
            field_str(candidate, "status").unwrap_or("-")
        );
        let module_diffs = candidate
            .get("diff")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for diff in module_diffs.iter().take(max_modules) {
            let module = field_str(diff, "module").unwrap_or("module");
            let before = field_str(diff, "before").unwrap_or("");
            let after = field_str(diff, "after").unwrap_or("");
            let _ = writeln!(out, "        {module}");
            let _ = writeln!(out, "          - {}", truncate_inline(before, 180));
            let _ = writeln!(out, "          + {}", truncate_inline(after, 180));
        }
        if module_diffs.len() > max_modules {
            let _ = writeln!(
                out,
                "        ... {} more modules",
                module_diffs.len() - max_modules
            );
        }
    }
    if ranked.len() > max_candidates {
        let _ = writeln!(
            out,
            "      ... {} more candidates",
            ranked.len() - max_candidates
        );
    }
}

fn terminal_frontier_update_line(fields: &Value) -> String {
    let summary = frontier_summary(fields);
    let generation = field_usize(fields, "generation")
        .map(|generation| format!("gen={generation} "))
        .unwrap_or_default();
    let changed = field_str(fields, "changed_candidate_id")
        .map(short_id)
        .unwrap_or_else(|| "unknown".to_string());
    let delta = frontier_churn(fields);
    let mut out = format!(
        "  frontier {}+{} size={}{} best={} train={} coverage=train {} frontier_seeds={} best_seeds={} {}",
        generation,
        changed,
        summary.frontier_size,
        delta,
        summary.best_candidate_id,
        summary.best_train_reward,
        summary.coverage,
        summary.frontier_seed_percent,
        summary.best_seed_percent,
        summary.seed_list
    );
    append_frontier_detail(&mut out, fields, &summary);
    out
}

fn frontier_churn(fields: &Value) -> String {
    let delta = field_i64(fields, "frontier_size_delta");
    let added = field_usize(fields, "frontier_added_count");
    let removed = field_usize(fields, "frontier_removed_count");
    match (added, removed, delta) {
        (Some(added), Some(removed), Some(delta)) if added > 0 || removed > 0 => {
            format!(" (+{added}/-{removed} net {delta:+})")
        }
        (_, _, Some(delta)) => format_signed(delta),
        _ => String::new(),
    }
}

fn terminal_frontier_snapshot_line(fields: &Value) -> String {
    let summary = frontier_summary(fields);
    let generation = field_usize(fields, "generation")
        .map(|generation| format!("generation {generation} "))
        .unwrap_or_default();
    let mut out = format!(
        "  {}frontier summary: best={} train={} size={} coverage=train {} frontier_seeds={} best_seeds={} {}",
        generation,
        summary.best_candidate_id,
        summary.best_train_reward,
        summary.frontier_size,
        summary.coverage,
        summary.frontier_seed_percent,
        summary.best_seed_percent,
        summary.seed_list
    );
    append_frontier_detail(&mut out, fields, &summary);
    out
}

struct FrontierSummary {
    best_candidate_id: String,
    best_train_reward: String,
    frontier_size: usize,
    seed_count: usize,
    row_count: usize,
    coverage: String,
    frontier_seed_percent: String,
    best_seed_percent: String,
    seed_list: String,
}

fn frontier_summary(fields: &Value) -> FrontierSummary {
    let coverage = fields.get("coverage").unwrap_or(&Value::Null);
    let best_candidate_id = field_str(fields, "best_candidate_id").unwrap_or("unknown");
    let seed_count = field_usize(coverage, "train_seed_count")
        .or_else(|| field_usize(fields, "train_seed_count"))
        .unwrap_or(0);
    let covered_seed_count = field_usize(coverage, "covered_train_seed_count")
        .or_else(|| field_usize(fields, "covered_train_seed_count"))
        .unwrap_or(0);
    let row_count = field_usize(coverage, "train_row_count")
        .or_else(|| field_usize(fields, "train_row_count"))
        .unwrap_or(0);
    let covered_row_count = field_usize(coverage, "covered_train_example_count")
        .or_else(|| field_usize(fields, "covered_train_example_count"))
        .unwrap_or(0);
    let best_covered_seed_count = fields
        .get("members")
        .and_then(Value::as_array)
        .and_then(|members| {
            members.iter().find(|member| {
                field_bool(member, "is_best").unwrap_or(false)
                    || field_str(member, "candidate_id") == Some(best_candidate_id)
            })
        })
        .and_then(|member| field_usize(member, "covered_seed_count"))
        .unwrap_or(0);
    FrontierSummary {
        best_candidate_id: short_id(best_candidate_id),
        best_train_reward: fmt_score(field_f64(fields, "best_train_reward")),
        frontier_size: field_usize(fields, "frontier_size").unwrap_or(0),
        seed_count,
        row_count,
        coverage: format!(
            "{covered_row_count}/{row_count} rows, {covered_seed_count}/{seed_count} seeds"
        ),
        frontier_seed_percent: fmt_percent(covered_seed_count, seed_count),
        best_seed_percent: fmt_percent(best_covered_seed_count, seed_count),
        seed_list: compact_i64_list(&field_array_i64(fields, "covered_train_seeds")),
    }
}

fn append_frontier_detail(out: &mut String, fields: &Value, summary: &FrontierSummary) {
    if !terminal_detail_enabled("frontier") {
        return;
    }

    let members = fields
        .get("members")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let max_members = 6usize;
    for member in members.iter().take(max_members) {
        let marker = match (
            field_bool(member, "is_best").unwrap_or(false),
            field_bool(member, "is_changed").unwrap_or(false),
        ) {
            (true, true) => "*+",
            (true, false) => "* ",
            (false, true) => "+ ",
            (false, false) => "  ",
        };
        let _ = write!(
            out,
            "\n    {} {} train={} seeds={}/{} rows={}/{} W/L/T={}/{}/{} source={}",
            marker,
            short_id(field_str(member, "candidate_id").unwrap_or("unknown")),
            fmt_score(field_f64(member, "train_reward")),
            field_usize(member, "covered_seed_count").unwrap_or(0),
            summary.seed_count,
            field_usize(member, "covered_example_count").unwrap_or(0),
            summary.row_count,
            field_usize(member, "wins_vs_best").unwrap_or(0),
            field_usize(member, "losses_vs_best").unwrap_or(0),
            field_usize(member, "ties_vs_best").unwrap_or(0),
            field_str(member, "source").unwrap_or("-")
        );
    }
    if members.len() > max_members {
        let _ = write!(out, "\n      ... {} more", members.len() - max_members);
    }
}

fn terminal_state_transition_line(message: &str, fields: &Value) -> Option<String> {
    let details = fields.get("details").unwrap_or(&Value::Null);
    let rollouts_started = field_str(fields, "trigger") == Some("rollouts_started");
    match message {
        "Container, program, and dataset ready" => Some("  container ready".to_string()),
        "Proposer started" | "Async proposer started" => {
            Some(terminal_proposer_started_line(details))
        }
        "Seed candidate rollouts started" if rollouts_started => Some(
            terminal_candidate_rollouts_started_line(details, "seed-full-train"),
        ),
        "Candidate minibatch rollouts started" if rollouts_started => Some(
            terminal_candidate_rollouts_started_line(details, "minibatch"),
        ),
        "Parent minibatch reference rollouts started" if rollouts_started => Some(
            terminal_candidate_rollouts_started_line(details, "parent-minibatch-reference"),
        ),
        "Candidate full-train rollouts queued" | "Candidate full-train rollouts started"
            if rollouts_started =>
        {
            Some(terminal_candidate_rollouts_started_line(
                details,
                "full-train",
            ))
        }
        "Heldout rollouts queued" | "Heldout rollouts started" if rollouts_started => {
            Some(format!(
                "\n  heldout rollouts candidates={} rows={} n={}",
                field_usize(details, "candidate_count").unwrap_or(0),
                field_usize(details, "row_count").unwrap_or(0),
                field_usize(details, "rollout_count").unwrap_or(0)
            ))
        }
        _ => None,
    }
}

fn terminal_proposer_started_line(fields: &Value) -> String {
    let mut line = format!(
        "\n  generation {} proposer started",
        field_usize(fields, "generation").unwrap_or(0)
    );
    if let Some(backend) = field_str(fields, "backend") {
        let _ = write!(line, " backend={backend}");
    }
    if let Some(model) = field_str(fields, "model") {
        let _ = write!(line, " model={model}");
    }
    if let Some(proposal_count) = field_usize(fields, "proposal_count") {
        let _ = write!(line, " target={proposal_count}");
    }
    if let Some(parent_id) = field_str(fields, "parent_candidate_id") {
        let _ = write!(line, " parent={}", short_id(parent_id));
    }
    if let Some(frontier_size) = field_usize(fields, "frontier_size") {
        let _ = write!(line, " frontier={frontier_size}");
    }
    if let Some(candidate_count) = field_usize(fields, "candidate_count") {
        let _ = write!(line, " candidates={candidate_count}");
    }
    if let Some(rollout_count) = field_usize(fields, "rollout_row_count") {
        let _ = write!(line, " rollouts={rollout_count}");
    }
    if let Some(loss_count) = field_usize(fields, "loss_count") {
        let _ = write!(line, " losses={loss_count}");
    }
    if let Some(win_count) = field_usize(fields, "win_count") {
        let _ = write!(line, " wins={win_count}");
    }
    if let Some(workspace) = field_str(fields, "workspace") {
        let _ = write!(line, " workspace={workspace}");
    }
    line
}

fn terminal_rollout_chunk_started_line(fields: &Value) -> String {
    format!(
        "  rollout chunk started stage={} rows={} active={}/{} chunk={}",
        field_str(fields, "stage").unwrap_or("-"),
        field_usize(fields, "rows").unwrap_or(0),
        field_usize(fields, "active_rollout_workers").unwrap_or(0),
        field_usize(fields, "configured_rollout_workers").unwrap_or(0),
        short_id(field_str(fields, "chunk_id").unwrap_or("-"))
    )
}

fn terminal_rollout_chunk_finished_line(fields: &Value) -> String {
    format!(
        "  rollout chunk finished stage={} rows={} active={} wall={} rows/s={} chunk={}",
        field_str(fields, "stage").unwrap_or("-"),
        field_usize(fields, "rows").unwrap_or(0),
        field_usize(fields, "active_rollout_workers").unwrap_or(0),
        fmt_seconds(field_f64(fields, "wall_seconds").unwrap_or(0.0)),
        fmt_rate(field_f64(fields, "rows_per_second").unwrap_or(0.0)),
        short_id(field_str(fields, "chunk_id").unwrap_or("-"))
    )
}

fn terminal_rollout_progress_line(fields: &Value, finished: bool) -> String {
    if terminal_detail_enabled("rollout") {
        if finished {
            return terminal_rollout_chunk_finished_line(fields);
        }
        return terminal_rollout_chunk_started_line(fields);
    }
    let total = field_usize(fields, "total_rows").unwrap_or(0);
    let done = field_usize(fields, "completed_rows")
        .unwrap_or(0)
        .min(total);
    let width = 20usize;
    let filled = if total > 0 {
        (done.saturating_mul(width) + total / 2) / total
    } else {
        0
    }
    .min(width);
    let bar = format!("{}{}", "#".repeat(filled), ".".repeat(width - filled));
    let percent = if total > 0 {
        100.0 * done as f64 / total as f64
    } else {
        0.0
    };
    let rate = field_f64(fields, "rows_per_second").unwrap_or(0.0);
    let eta = if finished && rate > 0.0 && total > done {
        fmt_seconds((total - done) as f64 / rate)
    } else {
        "-".to_string()
    };
    let state = if finished { "done" } else { "run" };
    format!(
        "  rollout {:<20} [{}] {:>3}/{:<3} {:>5.1}% active={} rate={}/s eta={} {}",
        field_str(fields, "stage").unwrap_or("-"),
        bar,
        done,
        total,
        percent,
        field_usize(fields, "active_rollout_workers").unwrap_or(0),
        if rate > 0.0 {
            fmt_rate(rate)
        } else {
            "-".to_string()
        },
        eta,
        state,
    )
}

fn terminal_candidate_rollouts_started_line(details: &Value, label: &str) -> String {
    if let Some(candidate_id) = field_str(details, "candidate_id") {
        return format!(
            "  candidate {} {} rollouts n={}",
            short_id(candidate_id),
            label,
            field_usize(details, "row_count")
                .or_else(|| field_usize(details, "rollout_count"))
                .unwrap_or(0)
        );
    }
    format!(
        "  generation {} {} rollouts candidates={} n={}",
        field_usize(details, "generation").unwrap_or(0),
        label,
        field_usize(details, "candidate_count").unwrap_or(0),
        field_usize(details, "rollout_count")
            .or_else(|| field_usize(details, "row_count"))
            .unwrap_or(0)
    )
}

fn terminal_runtime_job_completed_line(fields: &Value) -> String {
    match field_str(fields, "runtime_kind").unwrap_or("runtime") {
        "proposer" => {
            let generation = field_usize(fields, "generation")
                .map(|generation| format!("generation {generation} "))
                .unwrap_or_default();
            format!(
                "  {}proposer runtime wall={} cache={} proposals={} tokens={}",
                generation,
                fmt_seconds(field_f64(fields, "wall_seconds").unwrap_or(0.0)),
                fmt_cache_bool(field_bool(fields, "cache_hit").unwrap_or(false)),
                field_usize(fields, "proposal_count").unwrap_or(0),
                fmt_tokens_millions(field_u64(fields, "total_tokens").unwrap_or(0))
            )
        }
        "rollout" | "rollout_batch" => {
            let rollout_count = field_usize(fields, "rollout_count").unwrap_or(0);
            let cache_hits = field_usize(fields, "cache_hits").unwrap_or(0);
            let cache_misses = field_usize(fields, "cache_misses").unwrap_or(0);
            if terminal_detail_enabled("rollout") {
                let avg = field_f64(fields, "avg_wall_seconds_per_rollout")
                    .map(fmt_seconds)
                    .unwrap_or_else(|| "-".to_string());
                let diagnostics = runtime_rollout_diagnostics_suffix(fields);
                format!(
                    "  rollout runtime stage={} mode={} workers={} candidates={} rollouts={} cache={}/{} wall={} avg={} tokens={}{}",
                    field_str(fields, "stage").unwrap_or("mixed"),
                    field_str(fields, "rollout_submission_mode").unwrap_or("-"),
                    field_usize(fields, "configured_rollout_workers").unwrap_or(0),
                    field_usize(fields, "candidate_count").unwrap_or(1),
                    rollout_count,
                    cache_hits,
                    cache_hits + cache_misses,
                    fmt_seconds(field_f64(fields, "wall_seconds").unwrap_or(0.0)),
                    avg,
                    fmt_tokens_millions(field_u64(fields, "total_tokens").unwrap_or(0)),
                    diagnostics,
                )
            } else {
                let diagnostics = field_f64(fields, "estimated_effective_concurrency")
                    .map(|value| format!(" eff={value:.1}x"))
                    .unwrap_or_default();
                format!(
                    "  rollout {} rows={} candidates={} cache={}/{} wall={} tokens={}{}",
                    field_str(fields, "stage").unwrap_or("mixed"),
                    rollout_count,
                    field_usize(fields, "candidate_count").unwrap_or(1),
                    cache_hits,
                    cache_hits + cache_misses,
                    fmt_seconds(field_f64(fields, "wall_seconds").unwrap_or(0.0)),
                    fmt_tokens_millions(field_u64(fields, "total_tokens").unwrap_or(0)),
                    diagnostics,
                )
            }
        }
        _ => format!(
            "  runtime job completed kind={} wall={}",
            field_str(fields, "runtime_kind").unwrap_or("unknown"),
            fmt_seconds(field_f64(fields, "wall_seconds").unwrap_or(0.0))
        ),
    }
}

fn terminal_runtime_throughput_warning_line(fields: &Value) -> String {
    format!(
        "  warning: rollout throughput low stage={} mode={} workers={} uncached={} wall={} throughput={}/s expected>={}/s eff_conc={}",
        field_str(fields, "stage").unwrap_or("mixed"),
        field_str(fields, "rollout_submission_mode").unwrap_or("-"),
        field_usize(fields, "configured_rollout_workers").unwrap_or(0),
        field_usize(fields, "cache_misses").unwrap_or(0),
        fmt_seconds(field_f64(fields, "wall_seconds").unwrap_or(0.0)),
        fmt_rate(field_f64(fields, "observed_uncached_rollouts_per_second").unwrap_or(0.0)),
        fmt_rate(field_f64(fields, "expected_min_uncached_rollouts_per_second").unwrap_or(0.0)),
        field_f64(fields, "estimated_effective_concurrency")
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "-".to_string())
    )
}

#[derive(Clone, Debug, Default)]
struct RolloutSectionSummary {
    stage: String,
    rollouts: u64,
    cache_hits: u64,
    cache_misses: u64,
    wall_seconds: f64,
    total_tokens: u64,
    jobs: u64,
}

impl RolloutSectionSummary {
    fn add(&mut self, fields: &Value) {
        self.rollouts = self
            .rollouts
            .saturating_add(field_u64(fields, "rollout_count").unwrap_or(0));
        self.cache_hits = self
            .cache_hits
            .saturating_add(field_u64(fields, "cache_hits").unwrap_or(0));
        self.cache_misses = self
            .cache_misses
            .saturating_add(field_u64(fields, "cache_misses").unwrap_or(0));
        self.wall_seconds += field_f64(fields, "wall_seconds").unwrap_or(0.0);
        self.total_tokens = self
            .total_tokens
            .saturating_add(field_u64(fields, "total_tokens").unwrap_or(0));
        self.jobs = self.jobs.saturating_add(1);
    }

    fn line(&self) -> String {
        let throughput = if self.wall_seconds > 0.0 {
            self.rollouts as f64 / self.wall_seconds
        } else {
            0.0
        };
        format!(
            "  rollout section done stage={} rollouts={} wall={} throughput={}/s cache={}/{} tokens={} jobs={}",
            self.stage,
            self.rollouts,
            fmt_seconds(self.wall_seconds),
            fmt_rate(throughput),
            self.cache_hits,
            self.cache_hits.saturating_add(self.cache_misses),
            fmt_tokens_millions(self.total_tokens),
            self.jobs,
        )
    }
}

static ROLLOUT_SECTION_SUMMARY: OnceLock<Mutex<Option<RolloutSectionSummary>>> = OnceLock::new();

fn rollout_section_summary_state() -> &'static Mutex<Option<RolloutSectionSummary>> {
    ROLLOUT_SECTION_SUMMARY.get_or_init(|| Mutex::new(None))
}

fn maybe_rollout_section_summary_before_event(event_type: &str, fields: &Value) -> Option<String> {
    let mut guard = rollout_section_summary_state().lock().ok()?;
    if event_type == "runtime.job.completed" && rollout_runtime_event(fields) {
        let stage = field_str(fields, "stage").unwrap_or("mixed").to_string();
        if guard
            .as_ref()
            .map(|summary| summary.stage.as_str() != stage.as_str())
            .unwrap_or(false)
        {
            let line = guard.take().map(|summary| summary.line());
            *guard = Some(RolloutSectionSummary {
                stage,
                ..Default::default()
            });
            if let Some(summary) = guard.as_mut() {
                summary.add(fields);
            }
            return line;
        }
        if guard.is_none() {
            *guard = Some(RolloutSectionSummary {
                stage,
                ..Default::default()
            });
        }
        if let Some(summary) = guard.as_mut() {
            summary.add(fields);
        }
        return None;
    }
    if rollout_section_boundary_event(event_type) {
        return guard.take().map(|summary| summary.line());
    }
    None
}

fn maybe_rollout_section_summary_after_event(event_type: &str) -> Option<String> {
    if event_type == "gepa.run.finished" {
        return rollout_section_summary_state()
            .lock()
            .ok()?
            .take()
            .map(|summary| summary.line());
    }
    None
}

fn rollout_runtime_event(fields: &Value) -> bool {
    matches!(
        field_str(fields, "runtime_kind"),
        Some("rollout" | "rollout_batch")
    )
}

fn rollout_section_boundary_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "candidate.evaluated"
            | "candidate.minibatch_evaluated"
            | "candidate.full_train_evaluated"
            | "candidate.accepted"
            | "candidate.rejected"
            | "candidate.deferred"
            | "frontier.updated"
            | "frontier.snapshot"
            | "heldout.completed"
            | "heldout.skipped"
            | "proposer.started"
            | "proposer.completed"
            | "score_chart.written"
            | "gepa.run.finished"
            | "gepa.stop"
    )
}

fn runtime_rollout_diagnostics_suffix(fields: &Value) -> String {
    let Some(effective_concurrency) = field_f64(fields, "estimated_effective_concurrency") else {
        return String::new();
    };
    let p50 = field_f64(fields, "uncached_latency_p50_seconds")
        .map(fmt_seconds)
        .unwrap_or_else(|| "-".to_string());
    let p95 = field_f64(fields, "uncached_latency_p95_seconds")
        .map(fmt_seconds)
        .unwrap_or_else(|| "-".to_string());
    let max = field_f64(fields, "uncached_latency_max_seconds")
        .map(fmt_seconds)
        .unwrap_or_else(|| "-".to_string());
    format!(" eff_conc={effective_concurrency:.1} lat[p50/p95/max]={p50}/{p95}/{max}")
}

fn terminal_finished_line(fields: &Value) -> String {
    let usage = fields.get("usage").unwrap_or(&Value::Null);
    let heldout = if field_bool(fields, "heldout_skipped").unwrap_or(false) {
        " heldout=skipped".to_string()
    } else {
        field_f64(fields, "heldout_reward")
            .map(|score| format!(" heldout={}", fmt_score(Some(score))))
            .unwrap_or_default()
    };
    let mut line = format!(
        "{} best={} rollouts={}{} cost=${:.4} tokens={}",
        bold("done"),
        short_id(field_str(fields, "best_candidate_id").unwrap_or("unknown")),
        field_usize(fields, "rollout_count").unwrap_or(0),
        heldout,
        field_f64(fields, "cost_usd").unwrap_or(0.0),
        fmt_tokens_millions(field_u64(usage, "total_tokens").unwrap_or(0))
    );
    if let Some(summary) = fields.get("runtime_summary") {
        append_runtime_summary_line(&mut line, "policy", summary.get("policy"));
        append_runtime_summary_line(&mut line, "proposer", summary.get("proposer"));
        append_candidate_runtime_summary_lines(&mut line, summary.get("candidates"));
    }
    line
}

fn append_runtime_summary_line(out: &mut String, label: &str, bucket: Option<&Value>) {
    let Some(bucket) = bucket else {
        return;
    };
    let tokens = field_u64(bucket, "total_tokens").unwrap_or(0);
    let calls = field_u64(bucket, "calls").unwrap_or(0);
    let jobs = field_u64(bucket, "jobs").unwrap_or(0);
    let cost = field_f64(bucket, "cost_usd").unwrap_or(0.0);
    let wall = field_f64(bucket, "wall_seconds").unwrap_or(0.0);
    let model = field_str(bucket, "model")
        .map(|model| format!(" model={model}"))
        .unwrap_or_default();
    let throughput = if wall > 0.0 && calls > 0 {
        format!(" throughput={:.2}/s", calls as f64 / wall)
    } else {
        String::new()
    };
    let _ = write!(
        out,
        "\n  usage {label}:{model} tokens={} cost=${cost:.4} time={} calls={calls} jobs={jobs}{throughput}",
        fmt_tokens_millions(tokens),
        fmt_seconds(wall)
    );
}

fn append_candidate_runtime_summary_lines(out: &mut String, candidates: Option<&Value>) {
    let Some(candidates) = candidates.and_then(Value::as_object) else {
        return;
    };
    if candidates.is_empty() {
        return;
    }
    let mut rows = candidates.iter().collect::<Vec<_>>();
    rows.sort_by(|(left_id, left), (right_id, right)| {
        field_u64(right, "calls")
            .unwrap_or(0)
            .cmp(&field_u64(left, "calls").unwrap_or(0))
            .then_with(|| {
                field_u64(right, "total_tokens")
                    .unwrap_or(0)
                    .cmp(&field_u64(left, "total_tokens").unwrap_or(0))
            })
            .then_with(|| left_id.cmp(right_id))
    });
    let max_rows = 12usize;
    let _ = write!(out, "\n  usage candidates:");
    for (candidate_id, bucket) in rows.iter().take(max_rows) {
        let tokens = field_u64(bucket, "total_tokens").unwrap_or(0);
        let calls = field_u64(bucket, "calls").unwrap_or(0);
        let cost = field_f64(bucket, "cost_usd").unwrap_or(0.0);
        let wall = field_f64(bucket, "wall_seconds").unwrap_or(0.0);
        let _ = write!(
            out,
            "\n    {} tokens={} cost=${cost:.4} time={} rollouts={calls}",
            short_id(candidate_id),
            fmt_tokens_millions(tokens),
            fmt_seconds(wall)
        );
    }
    if rows.len() > max_rows {
        let _ = write!(out, "\n    ... {} more candidates", rows.len() - max_rows);
    }
}

fn fmt_tokens_millions(tokens: u64) -> String {
    format!("{:.3}M", tokens as f64 / 1_000_000.0)
}

fn fmt_seconds(seconds: f64) -> String {
    if seconds.is_finite() {
        format!("{seconds:.2}s")
    } else {
        "-".to_string()
    }
}

fn fmt_cache_bool(cache_hit: bool) -> &'static str {
    if cache_hit {
        "hit"
    } else {
        "miss"
    }
}

fn fmt_rate(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.2}")
    } else {
        "-".to_string()
    }
}

fn append_score_scatter(out: &mut String, rows: &[Value], seed_id: &str, best_id: &str) {
    let points = rows
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            let candidate_id = field_str(row, "candidate_id")?;
            let heldout = field_f64(row, "heldout_reward")?;
            Some((idx, candidate_id, heldout))
        })
        .collect::<Vec<_>>();
    if points.is_empty() {
        return;
    }
    let width = 48usize;
    let height = 6usize;
    let min_score = points
        .iter()
        .map(|(_, _, score)| *score)
        .fold(f64::INFINITY, f64::min)
        .min(0.0);
    let max_score = points
        .iter()
        .map(|(_, _, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1.0);
    let score_span = (max_score - min_score).max(0.001);
    let mut grid = vec![vec![' '; width]; height];
    for (idx, candidate_id, score) in points {
        let col = if rows.len() <= 1 {
            width / 2
        } else {
            ((idx as f64 / (rows.len() - 1) as f64) * (width - 1) as f64).round() as usize
        }
        .min(width - 1);
        let row = (height - 1)
            .saturating_sub(
                (((score - min_score) / score_span) * (height - 1) as f64).round() as usize,
            )
            .min(height - 1);
        grid[row][col] = if candidate_id == best_id {
            '*'
        } else if candidate_id == seed_id {
            'o'
        } else {
            '.'
        };
    }
    let _ = writeln!(out, "\n      heldout score vs candidate order");
    for (row_idx, cells) in grid.iter().enumerate() {
        let score = max_score - (row_idx as f64 / (height - 1) as f64) * score_span;
        let line = cells.iter().collect::<String>();
        let _ = writeln!(out, "      {score:0.3} |{line}");
    }
    let _ = writeln!(out, "            +{}", "-".repeat(width));
}

fn terminal_reason(fields: &Value) -> String {
    field_str(fields, "reason")
        .map(|reason| truncate(reason, 96))
        .unwrap_or_default()
}

fn field_str<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn field_f64(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn field_usize(value: &Value, key: &str) -> Option<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn field_u64(value: &Value, key: &str) -> Option<u64> {
    value.get(key).and_then(Value::as_u64)
}

fn field_i64(value: &Value, key: &str) -> Option<i64> {
    value.get(key).and_then(Value::as_i64)
}

fn field_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn field_array_strings(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn field_array_i64(value: &Value, key: &str) -> Vec<i64> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default()
}

fn compact_i64_list(values: &[i64]) -> String {
    match values.len() {
        0 => "[]".to_string(),
        1..=8 => format!(
            "[{}]",
            values
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
        len => {
            let head = values
                .iter()
                .take(4)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let tail = values
                .iter()
                .rev()
                .take(2)
                .copied()
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("[{head},...,{tail}] n={len}")
        }
    }
}

fn format_signed(value: i64) -> String {
    match value.cmp(&0) {
        std::cmp::Ordering::Greater => format!(" (+{value})"),
        std::cmp::Ordering::Less => format!(" ({value})"),
        std::cmp::Ordering::Equal => " (+0)".to_string(),
    }
}

fn max_f64(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn min_f64(values: &[f64]) -> Option<f64> {
    values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
}

fn values_are_tied(values: &[f64]) -> bool {
    if values.len() <= 1 {
        return true;
    }
    let Some(low) = min_f64(values) else {
        return true;
    };
    let Some(high) = max_f64(values) else {
        return true;
    };
    (high - low).abs() < 1e-9
}

fn ascii_trajectory(values: &[f64]) -> String {
    if values_are_tied(values) {
        "flat".to_string()
    } else {
        ascii_sparkline(values)
    }
}

fn ascii_sparkline(values: &[f64]) -> String {
    if values.is_empty() {
        return "-".to_string();
    }
    let ticks = [' ', '.', ':', '-', '=', '+', '*', '#'];
    let finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return "-".repeat(values.len());
    }
    let low = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let high = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if (high - low).abs() < 1e-9 {
        return "-".repeat(values.len());
    }
    values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                return '?';
            }
            let idx = (((value - low) / (high - low)) * (ticks.len() - 1) as f64).round();
            ticks[idx.clamp(0.0, (ticks.len() - 1) as f64) as usize]
        })
        .collect()
}

fn ascii_score_bar(value: f64, width: usize, max_value: f64) -> String {
    let scale = max_value.max(1e-9);
    let filled = ((value / scale) * width as f64)
        .round()
        .clamp(0.0, width as f64) as usize;
    format!("{}{}", "#".repeat(filled), ".".repeat(width - filled))
}

fn fmt_score(value: Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.3}"),
        _ => "-".to_string(),
    }
}

fn fmt_percent(numerator: usize, denominator: usize) -> String {
    if denominator == 0 {
        "-".to_string()
    } else {
        format!("{:.1}%", numerator as f64 * 100.0 / denominator as f64)
    }
}

fn short_id(value: &str) -> String {
    if value.chars().count() <= 17 {
        value.to_string()
    } else {
        let mut out = value.chars().take(17).collect::<String>();
        out.push_str("...");
        out
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut out = value
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect::<String>();
    out.push_str("...");
    out
}

fn truncate_inline(value: &str, max_chars: usize) -> String {
    truncate(
        &value.split_whitespace().collect::<Vec<_>>().join(" "),
        max_chars,
    )
}

fn bold(value: &str) -> String {
    ansi("1", value)
}

fn green(value: &str) -> String {
    ansi("32", value)
}

fn red(value: &str) -> String {
    ansi("31", value)
}

fn ansi(code: &str, value: &str) -> String {
    if env::var_os("NO_COLOR").is_some() {
        return value.to_string();
    }
    let mut out = String::new();
    let _ = write!(&mut out, "\x1b[{code}m{value}\x1b[0m");
    out
}
