//! P0-9 lock — Rust half.
//!
//! A 2,000-line cap on every `.rs` file in the workspace, with an explicit
//! allowlist of the files that are already over it. The allowlist records each
//! offender's line count as a ceiling, so an offender may only shrink: adding
//! lines to one of these files fails here, and a new file crossing the cap
//! fails without any allowlist to hide behind.
//!
//! Run: `cargo test file_size_cap`

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Lines. Chosen in the v0.7 structure review (decision D-X-2) alongside the
/// 600-line renderer cap in Workshop.
const MAX_LINES: usize = 2_000;

/// Files already over the cap, with the count they may not exceed.
///
/// Paths are relative to the repo root. Entries leave this list one of two
/// ways: the file drops under the cap (then the entry must be deleted, which
/// this test enforces), or the file is split. Nothing may be added to this
/// list without a review that says why the split is not being done now.
///
/// `lib.rs` and `workspace.rs` are the two authorities the structure review
/// named as the least reviewable files in the repo (OPT-R-10); P2-2 splits
/// them.
const ALLOWLIST: &[(&str, usize)] = &[
    ("rust/crates/synth_gepa/src/codex_app_server.rs", 3_434),
    ("rust/crates/synth_gepa/src/lib.rs", 22_143),
    ("rust/crates/synth_gepa/src/service.rs", 6_273),
    ("rust/crates/synth_optimizer_platform/src/config.rs", 3_141),
    (
        "rust/crates/synth_optimizer_platform/src/workspace.rs",
        10_174,
    ),
];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .expect("repo root above rust/crates/<crate>")
        .to_path_buf()
}

/// Every `.rs` file in the workspace, keyed by its repo-relative path.
fn rust_line_counts() -> BTreeMap<String, usize> {
    fn walk(dir: &Path, root: &Path, out: &mut BTreeMap<String, usize>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().is_some_and(|name| name == "target") {
                    continue;
                }
                walk(&path, root, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = std::fs::read_to_string(&path).unwrap_or_default();
                let relative = path
                    .strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string();
                out.insert(relative, text.lines().count());
            }
        }
    }
    let root = repo_root();
    let mut out = BTreeMap::new();
    walk(&root.join("rust"), &root, &mut out);
    out
}

fn allowlist() -> BTreeMap<&'static str, usize> {
    ALLOWLIST.iter().copied().collect()
}

#[test]
fn file_size_cap_allowlist_is_sorted_and_unique() {
    let names: Vec<&str> = ALLOWLIST.iter().map(|(path, _)| *path).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(names, sorted, "ALLOWLIST must be sorted and unique");
}

#[test]
fn file_size_cap_has_no_unlisted_offenders() {
    let allowlist = allowlist();
    let offenders: Vec<String> = rust_line_counts()
        .into_iter()
        .filter(|(path, lines)| *lines > MAX_LINES && !allowlist.contains_key(path.as_str()))
        .map(|(path, lines)| format!("{path} ({lines} lines)"))
        .collect();
    assert!(
        offenders.is_empty(),
        "these files are over the {MAX_LINES}-line cap and are not allowlisted. \
         Split them; do not add them to the list without a review: {offenders:#?}"
    );
}

#[test]
fn file_size_cap_allowlist_only_shrinks() {
    let counts = rust_line_counts();
    let mut grown = Vec::new();
    for (path, ceiling) in ALLOWLIST {
        let Some(lines) = counts.get(*path) else {
            continue; // reported by the stale-entry test
        };
        if *lines > *ceiling {
            grown.push(format!("{path}: {lines} lines, ceiling {ceiling}"));
        }
    }
    assert!(
        grown.is_empty(),
        "an allowlisted file grew. These files may only shrink — move the new code \
         into a new module instead of raising the ceiling: {grown:#?}"
    );
}

#[test]
fn file_size_cap_allowlist_has_no_stale_entries() {
    let counts = rust_line_counts();
    let mut stale = Vec::new();
    for (path, _) in ALLOWLIST {
        match counts.get(*path) {
            None => stale.push(format!("{path}: no such file")),
            Some(lines) if *lines <= MAX_LINES => {
                stale.push(format!("{path}: {lines} lines, now under the cap"))
            }
            Some(_) => {}
        }
    }
    assert!(
        stale.is_empty(),
        "remove these from ALLOWLIST — the list only shrinks: {stale:#?}"
    );
}
