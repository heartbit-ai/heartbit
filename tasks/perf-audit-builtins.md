# Perf audit: builtin tools

## P-TOOL-1 [High]: Regex compiled per-call in fallback grep
- **File**: crates/heartbit-core/src/tool/builtins/grep.rs:216-217
- **Observation**: `regex::Regex::new(&re_pattern)` compiles the user's pattern on every `fallback_grep()` call. No LazyLock.
- **Hypothesized cost**: ~50-500μs per call (depends on pattern complexity). Compounds on repeated searches for same pattern.
- **Frequency**: hot-path when ripgrep unavailable (fallback path)
- **Fix sketch**: Cache the compiled `Regex` in a `LazyLock<HashMap<String, Regex>>` or use regex crate's built-in pattern cache if available. For now, LazyLock per-session pool keyed by pattern string.
- **Security delta**: N/A
- **Validation**: needs-bench (profile with hyperfine on common patterns like "fn ", "TODO", etc.)

## P-TOOL-2 [High]: HTML sanitization regex compiled per-call
- **File**: crates/heartbit-core/src/tool/builtins/webfetch.rs:232-249
- **Observation**: Three `Regex::new(p)` calls inside `sanitize_html_for_agent()`, executed on every webfetch call regardless of format. Patterns are identical across calls.
- **Hypothesized cost**: ~100-200μs per call (3× Regex::new). Adds latency to every HTML response.
- **Frequency**: hot-path for html/markdown format responses
- **Fix sketch**: Move the three regex patterns to `LazyLock<[Regex; 3]>` at module level. Compile once at first use.
- **Security delta**: N/A (F-NET-7 sanitization still applied, just cached)
- **Validation**: needs-bench

## P-TOOL-3 [Medium]: glob::Pattern compiled per-call in list.rs
- **File**: crates/heartbit-core/src/tool/builtins/list.rs:135
- **Observation**: `glob::Pattern::new(pat)` inside the loop for every ignore pattern on each list call. DEFAULT_IGNORES is constant.
- **Hypothesized cost**: ~10-50μs per ignore pattern (total ~200-500μs for 8 defaults + user patterns). Per call.
- **Frequency**: every list tool invocation
- **Fix sketch**: Pre-compile DEFAULT_IGNORES patterns into a `LazyLock<Vec<glob::Pattern>>` at module init. Compile user patterns once per call (unavoidable, but reduces default overhead).
- **Security delta**: N/A
- **Validation**: static-only (pattern compilation is deterministic)

## P-TOOL-4 [High]: glob::Pattern compiled per-call in grep.rs fallback
- **File**: crates/heartbit-core/src/tool/builtins/grep.rs:220-222
- **Observation**: `glob::Pattern::new(include)` inside fallback_grep. If include pattern is provided, it's compiled every call without caching.
- **Hypothesized cost**: ~20-100μs per grep fallback call with include filter
- **Frequency**: warm-path when ripgrep unavailable + include filter used
- **Fix sketch**: Cache include pattern same as regex (LazyLock<HashMap>). User patterns are dynamic so can't fully pre-cache, but caching last-used is viable.
- **Security delta**: N/A
- **Validation**: static-only

## P-TOOL-5 [Critical]: patch.rs 4-pass fuzzy matching on every context/remove line
- **File**: crates/heartbit-core/src/tool/builtins/patch.rs:319-323 (fuzzy_lines_match)
- **Observation**: For each hunk, each context/remove line is matched against all 4 passes (exact → trim-end → trim-both → unicode-normalize). No short-circuit on first pass. Worst case: N hunk lines × 4 passes × normalization cost.
- **Hypothesized cost**: 4× string operations per line match. For a 100-line hunk, potentially 400 comparisons. normalize_unicode() allocates a new String per call (line 330-342).
- **Frequency**: hot-path during patch application
- **Fix sketch**: 
  - Short-circuit: return on first pass match (exact match is most common).
  - Pre-normalize: if fuzzy mode needed, normalize both strings once before the loop, not per-line.
  - Avoid allocating in normalize_unicode for ASCII-only strings.
- **Security delta**: N/A (fuzzy matching is intentional for LLM drift tolerance)
- **Validation**: needs-bench (measure actual vs normalized paths in real patches)

## P-TOOL-6 [Medium]: normalize_unicode allocates on every character iteration
- **File**: crates/heartbit-core/src/tool/builtins/patch.rs:330-342
- **Observation**: `normalize_unicode()` calls `.chars().map().collect::<String>()` unconditionally. Even if no unicode substitutions are needed, allocates a new String and calls `.trim().to_string()` again.
- **Hypothesized cost**: 2× allocations + trim iteration for every call, even on ASCII-only input
- **Frequency**: up to 4× per fuzzy line match (in worst case 100-line hunk)
- **Fix sketch**: 
  - Check if any unicode chars exist before allocating: `if !s.contains(['\u{2018}'..'\u{201F}', ...])` short-circuit to `s.to_string()`.
  - Or use an iterator-based approach that only allocates if substitution detected.
- **Security delta**: N/A
- **Validation**: needs-bench on real patches

## P-TOOL-7 [High]: floor_char_boundary O(N) scan per truncated line
- **File**: crates/heartbit-core/src/tool/builtins/mod.rs:137-143
- **Observation**: Called on every line that exceeds MAX_LINE_LENGTH (read.rs) or MAX_OUTPUT_CHARS (bash.rs, webfetch.rs). Scans backward from target byte to find char boundary.
- **Hypothesized cost**: O(N) scan; for a 2000-char line truncated to 2000 chars, up to 4 bytes scanned (UTF-8 worst case). Acceptable for single lines, but compounds if many lines truncated.
- **Frequency**: warm-path (truncated files/output)
- **Fix sketch**: Not a bottleneck for single-digit truncations; acceptable as-is. Flag as "acceptable asymmetry" — only relevant if output is very large + heavily multi-byte.
- **Security delta**: N/A
- **Validation**: static-only

## P-TOOL-8 [Medium]: FileTracker mtime check via stat() on every write/edit
- **File**: crates/heartbit-core/src/tool/builtins/file_tracker.rs:68
- **Observation**: `check_unmodified()` calls `std::fs::metadata(path)` and `.modified()` to check mtime. Called on every write/edit/patch after checking read-before-write guard. If batch operations read once then write multiple times, each write incurs a stat syscall.
- **Hypothesized cost**: ~1-10ms per stat (depends on filesystem and cache). Necessary for security (TOCTOU guard), but not cached across related writes.
- **Frequency**: hot-path on every write/edit/patch call
- **Fix sketch**: 
  - Necessary for security; do not remove.
  - Consider: after first write, could cache the mtime within FileTracker session for brief windows (e.g., <100ms), but this re-introduces race condition risk. Not recommended.
- **Security delta**: Removing this re-opens TOCTOU (file modified externally between read and write). Do not optimize away.
- **Validation**: N/A (security-critical, acceptable cost)

## P-TOOL-9 [Medium]: is_protected linear scan per path check
- **File**: crates/heartbit-core/src/tool/builtins/mod.rs:35-50
- **Observation**: Linear iteration over all protected_paths (typically ~15-20 paths: *.env, *.pem, /etc/shadow, ~/.ssh, etc.). Called on every read/write/edit/patch/glob/grep/list operation.
- **Hypothesized cost**: ~5-20μs per call (15 paths × string comparison). Negligible for single calls, but compounds if agent does many file ops in sequence.
- **Frequency**: hot-path on every filesystem tool call
- **Fix sketch**: 
  - Convert protected_paths to a `HashSet<PathBuf>` for O(1) lookup (but pattern matching like `*.env` requires special handling).
  - Alternatively: split into two structures: exact-match HashSet + pattern Vec, check exact first.
- **Security delta**: N/A (same security semantics, faster)
- **Validation**: static-only

## P-TOOL-10 [High]: bash.rs UUID generation per spawn
- **File**: crates/heartbit-core/src/tool/builtins/bash.rs:184
- **Observation**: `uuid::Uuid::new_v4()` called on every bash command to generate the nonce-bearing cwd marker. UUID generation has non-trivial cost (RNG + formatting).
- **Hypothesized cost**: ~5-50μs per call (depends on RNG source; v4 is random, not fast path).
- **Frequency**: hot-path on every bash tool call
- **Fix sketch**: 
  - Use a counter + hash instead of full UUID: `static COUNTER: AtomicU64; let nonce = COUNTER.fetch_add(1, Ordering::Relaxed); format!("__HEARTBIT_CWD_{:x}__", nonce);` Much faster, cryptographically sufficient for the purpose (forge resistance).
  - Or use a cheaper RNG like `fastrand` or SIMD-based UUID.
- **Security delta**: UUID v4 is not required; a sequential counter is equally secure against forgery (attacker doesn't know next nonce). Improvement.
- **Validation**: needs-bench (profile UUID::new_v4 vs counter)

## P-TOOL-11 [Medium]: default_protected_paths() rebuilt on every config creation
- **File**: crates/heartbit-core/src/tool/builtins/mod.rs:197-222
- **Observation**: `default_protected_paths()` is called in `BuiltinToolsConfig::default()` (line 234), which may be called multiple times per session if configs are created/dropped. Allocates Vec, pushes ~15-20 PathBuf entries.
- **Hypothesized cost**: ~50-200μs per call (Vec allocation + PathBuf creation + env var lookup).
- **Frequency**: cold-path (typically called once per session, but can be hot in test-heavy scenarios)
- **Fix sketch**: Wrap in `LazyLock<Vec<PathBuf>>` so HOME lookup and Vec allocation happen once.
- **Security delta**: N/A
- **Validation**: static-only (deterministic, idempotent)

## P-TOOL-12 [Low]: read.rs entire file into memory for line-range queries
- **File**: crates/heartbit-core/src/tool/builtins/read.rs:151-169
- **Observation**: `tokio::fs::read()` slurps entire file up to 256KB into memory, then iterates `.lines()` to reach the target range. For a 256KB file with offset=1000, limit=10, still reads all 256KB.
- **Hypothesized cost**: Memory usage is capped (256KB) so not a leak; latency is OK for the limit. However, for large files, reading beyond requested range is wasteful.
- **Frequency**: warm-path (read is common, but file size is bounded)
- **Fix sketch**: 
  - For files >100KB, consider seeking to approximate line offset before reading (complex, risky for edge cases).
  - Alternatively: keep as-is (simple, bounded by 256KB limit, acceptable for most use cases).
  - Recommendation: Flag as "acceptable asymmetry" — not worth the complexity.
- **Security delta**: N/A
- **Validation**: static-only

## P-TOOL-13 [Medium]: grep fallback_grep reads entire file into memory
- **File**: crates/heartbit-core/src/tool/builtins/grep.rs:264-267
- **Observation**: `std::fs::read_to_string()` on every file during grep fallback walk. No streaming; files are fully buffered. If grep matches in a 10MB file after reading it entirely, CPU/memory cost is high.
- **Hypothesized cost**: Unbounded file reads (no size cap). A grep on a directory with large files can consume significant RAM.
- **Frequency**: warm-path (grep fallback used when ripgrep unavailable)
- **Fix sketch**: 
  - Add a MAX_FILE_SIZE for grep fallback (e.g., 10MB) to match read.rs heuristics.
  - Stream-read and match line-by-line instead of buffering entire file.
- **Security delta**: N/A
- **Validation**: needs-bench (measure memory usage on large codebases)

## P-TOOL-14 [Critical]: patch.rs fuzzy matching scans entire file for every hunk
- **File**: crates/heartbit-core/src/tool/builtins/patch.rs:152-284
- **Observation**: For each hunk, fuzzy_lines_match is called on every line in the hunk. For a 1119-line patch file with 10 hunks of 50 lines each, that's 500 fuzzy line matches. Each match can trigger up to 4 string normalizations + allocations. Worst case: O(hunks × lines_per_hunk × 4) string operations.
- **Hypothesized cost**: Compound cost from P-TOOL-5 and P-TOOL-6: for a 10-hunk patch, potentially 2000+ allocations + string operations.
- **Frequency**: hot-path during patch application (core use case)
- **Fix sketch**: 
  - Short-circuit fuzzy matching: try exact match first, skip subsequent passes if exact succeeds.
  - Pre-normalize: single pass to normalize both strings once, then compare.
  - Cache normalized lines if fuzzy mode is active.
- **Security delta**: N/A
- **Validation**: needs-bench (profile against real patches from agent sessions)

## P-TOOL-15 [Low]: glob symlink canonicalize on every result
- **File**: crates/heartbit-core/src/tool/builtins/glob.rs:129-135, 150-157
- **Observation**: For each glob result, `symlink_metadata()` is called, and if it's a symlink, `path.canonicalize()` is called twice (once at line 131, again at line 152). Canonicalize is O(N) filesystem traversal.
- **Hypothesized cost**: ~1-10ms per symlink (depends on depth). If glob returns 100 paths with many symlinks, potential 100-1000ms overhead.
- **Frequency**: warm-path (glob is common, but symlink filtering is secondary)
- **Fix sketch**: 
  - Cache the canonicalize result: call once, reuse for both policy checks.
  - Or optimize the double-check logic.
- **Security delta**: N/A (symlink filtering is intentional, prevents escapes)
- **Validation**: needs-bench (measure on repos with many symlinks)

## Summary Table

| Finding | Severity | Component | Quick Win |
|---------|----------|-----------|-----------|
| P-TOOL-1 | High | grep fallback | LazyLock regex cache |
| P-TOOL-2 | High | webfetch sanitize | LazyLock regex cache |
| P-TOOL-3 | Medium | list.rs ignore | LazyLock for defaults |
| P-TOOL-4 | High | grep include filter | LazyLock pattern cache |
| P-TOOL-5 | Critical | patch fuzzy match | Short-circuit + pre-normalize |
| P-TOOL-6 | Medium | patch normalize_unicode | Avoid alloc on ASCII |
| P-TOOL-7 | High | floor_char_boundary | Acceptable (bounded) |
| P-TOOL-8 | Medium | FileTracker mtime | Do not remove (security) |
| P-TOOL-9 | Medium | is_protected linear scan | HashSet + pattern split |
| P-TOOL-10 | High | bash UUID nonce | Counter + hash |
| P-TOOL-11 | Medium | default_protected_paths | LazyLock |
| P-TOOL-12 | Low | read slurp | Acceptable (bounded) |
| P-TOOL-13 | Medium | grep file reads | Add size cap + stream |
| P-TOOL-14 | Critical | patch multi-pass | Combine with P-TOOL-5 |
| P-TOOL-15 | Low | glob symlink double-check | Cache canonicalize |

## Top 3 Optimization Wins (per-call impact)

1. **P-TOOL-5 + P-TOOL-14 (patch fuzzy matching)**: 
   - Short-circuit on exact match + pre-normalize avoids 3/4 of string operations per line.
   - Real-world estimate: 100-line hunk goes from 400 allocations to ~50. Saves ~50-200μs per patch.

2. **P-TOOL-1 + P-TOOL-2 (regex caching)**:
   - LazyLock for grep fallback regex + webfetch sanitization regex.
   - Estimate: 50-500μs saved per grep call + 100-200μs per webfetch call.

3. **P-TOOL-10 (bash UUID to counter)**:
   - Replace `Uuid::new_v4()` with atomic counter.
   - Estimate: 5-50μs per bash call (small per-call, but compounds over many bash invocations).

## Cross-Cutting Recommendations

- **All regex patterns**: Migrate to `LazyLock<Regex>` or `lazy_static` for patterns used per-call.
- **Allocations in loops**: Audit `normalize_unicode`, `sanitize_html_for_agent`, and similar hot-path functions for unnecessary allocations.
- **Symlink handling**: Cache `canonicalize()` results to avoid repeated syscalls.
- **Protected paths**: Consider HashSet for exact matches, keep Vec for patterns.

## REJECTED Optimization Suggestions

- **Lifting MAX_WALK_DEPTH (skill.rs:127)**: Would re-open F-FS-7 symlink-fork DoS. Do not lift.
- **Removing nonce-bearing cwd marker (bash.rs:184)**: Would re-open F-FS-8 user-data spoofing. Keep UUID or upgrade to counter, do not remove.
- **Removing default_protected_paths**: Would re-open F-FS-9 secret exposure. Keep and optimize via LazyLock instead.
- **Removing is_protected normalization**: Would re-open F-FS-11 path bypass (trivial variants). Keep and optimize via HashSet.
- **Removing absolute-path defense-in-depth in patch.rs**: Would re-open F-FS-12 path traversal. Keep as-is.
- **Removing FileTracker mtime check**: Would re-open TOCTOU race. Keep as-is; acceptable cost.
- **Removing O_NOFOLLOW open**: Would re-open F-FS symlink TOCTOU. Keep as-is.
- **Removing HTML sanitization in webfetch**: Would re-open F-NET-7 script injection. Keep and optimize via LazyLock.
- **Removing webfetch UA anonymization**: Would re-open F-NET-5 fingerprinting. Keep as-is.


---

## AUDIT SUMMARY (5-line brief)

**Total Findings**: 15 (2 Critical, 6 High, 5 Medium, 2 Low)

**Breakdown by Severity**:
- Critical (2): P-TOOL-5, P-TOOL-14 — patch fuzzy matching 4-pass with no short-circuit + allocations
- High (6): P-TOOL-1, P-TOOL-2, P-TOOL-4, P-TOOL-7, P-TOOL-10 — regex/pattern compilation per-call, UUID overhead
- Medium (5): P-TOOL-3, P-TOOL-6, P-TOOL-8, P-TOOL-9, P-TOOL-11, P-TOOL-13
- Low (2): P-TOOL-12, P-TOOL-15

**Top 3 Wins**:
1. Patch fuzzy matching short-circuit + pre-normalize (saves ~50-200μs per patch, hot-path)
2. LazyLock regex caching for grep fallback + webfetch sanitize (saves ~50-500μs per call)
3. Bash UUID → counter (saves ~5-50μs per bash invocation, compounds across many calls)

**REJECTED Suggestions**: 9 optimizations flagged as "DO NOT IMPLEMENT" due to security implications (symlink DoS, TOCTOU, path traversal, script injection, fingerprinting). All are documented with F-* finding references.

