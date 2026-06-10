# Benchmarking heartbit on Terminal-Bench 2.0 (via Harbor)

This directory contains a [Harbor](https://www.harborframework.com) adapter that
runs the **heartbit** runtime headless against **Terminal-Bench 2.0** (89 tasks),
and the scripts to build the binary for the task containers.

> Harbor (Laude Institute) is the official harness for TB **2.0**. The old `tb`
> CLI corresponds to 0.x/1.0 — don't use it for 2.0.

## Architecture: `BaseInstalledAgent` (Option B), not the external `BaseAgent`

heartbit's `bash` builtin executes **in-process** (`tokio::process::Command::new("bash")`),
as an *unrestricted* shell (no Landlock/seccomp/allowlist by default). So the
clean integration is to run the `heartbit` binary **inside the task container**:
its shell and file tools then mutate that container natively — which is exactly
what TB2 grades (each task's tests run in the container *after* the agent
finishes; Harbor's `AgentContext` only carries token/cost accounting, not the
score).

The external `BaseAgent` (Option A) would force every heartbit shell command to
be re-routed back into the container through `environment.exec` — but heartbit's
bash is a Rust in-process tool with no pluggable exec backend, and its file tools
would hit the *host* filesystem. Option A is strictly more work and more fragile
for a compiled agent. **We use Option B.** (Note: Option B builds the binary
*once*, then `install()` puts it into each fresh container — it does **not**
rebuild per task.)

## What the adapter relies on (heartbit-cli benchmark hooks)

The adapter drives the no-config **ENV path** of `heartbit run` and uses three
env/flag-gated hooks added to heartbit-cli (default behaviour unchanged):

| Hook | Effect |
|------|--------|
| `--trace-file <path>` | After the run, serialise `AgentOutput` to JSON (final answer, `tokens_used`, `tool_call_results`, `estimated_cost_usd`, `model_name`, `goal_met`) for `populate_context_post_run`. |
| `HEARTBIT_WORKSPACE=<dir>` | Repoints the file-tool jail **and** the bash cwd at the task directory (the env path otherwise hardcodes `~/.heartbit/workspaces`). |
| `HEARTBIT_NONINTERACTIVE=1` | Drops the `question` tool so the agent can never block on stdin. |

(They are added in the same change-set as this adapter. Without them, fall back to
`prebuilt`/older binaries works only partially — see *Token accounting* below.)

## 1. Prerequisites

- Docker installed and running (`docker info`).
- `uv` installed.
- Install Harbor as an isolated tool:

```bash
uv tool install harbor
harbor --help
```

## 2. Validate the environment with the oracle solutions

Before wiring heartbit in, confirm the harness + your Docker work by running the
reference (oracle) solutions:

```bash
harbor run -d terminal-bench/terminal-bench-2 -a oracle
```

If the oracle tasks pass, your Docker environment is correct.

## 3. Build the heartbit binary for the containers

Two install modes, selected at run time by `HEARTBIT_INSTALL_MODE` (default `build`):

### `build` (robust, slow) — compile inside each container

Package an offline source bundle once (committed tree + vendored crates):

```bash
benchmarks/terminal-bench-2/scripts/package_source.sh
# -> benchmarks/terminal-bench-2/dist/heartbit-src.tar.gz
```

`install()` then uploads it, installs a Rust toolchain + `cmake libssl-dev
libcurl4-openssl-dev pkg-config build-essential`, and runs
`cargo build --release --offline -p heartbit-cli`. Immune to glibc/OpenSSL ABI
mismatch; **but** it needs apt + network (for rustup) in each task container and
adds many minutes per container. Fine for the local smoke; see *Scaling* for the
caveat.

### `prebuilt` (fast, recommended for scale) — upload a host-built binary

`prebuilt` mode auto-detects the binary: it prefers the **slim static musl**
build (portable to *any* container) and falls back to the glibc one.

**Portable static binary (recommended) — `build_musl.sh`:**

```bash
benchmarks/terminal-bench-2/scripts/build_musl.sh
# -> target/x86_64-unknown-linux-musl/release/heartbit   (fully static)
export HEARTBIT_INSTALL_MODE=prebuilt
```

This builds the **slim** feature set (run/chat env-path only — no
teloxide/kafka/restate/postgres/openssl, pure rustls+ring) for
`x86_64-unknown-linux-musl` inside `rust:alpine`. The result is **statically
linked** with **compiled-in CA roots (webpki-roots)**, so it runs in
scratch/distroless/alpine/glibc containers alike — the adapter installs **no**
runtime libs and needs **no** `ca-certificates`. This is the path for Daytona
`-n 32`.

**Glibc fallback — `build_prebuilt.sh`** (only if you can't run the musl build):

```bash
benchmarks/terminal-bench-2/scripts/build_prebuilt.sh rust:1-bookworm
# -> target/release/heartbit   (glibc-dynamic)
```

The glibc binary links OpenSSL 3 / libcurl / zlib; the adapter apt-installs
`ca-certificates libssl3 libcurl4 zlib1g` (best-effort). Only fits
Debian/Ubuntu-derived task images whose glibc/OpenSSL major matches your build.

## 4. Run heartbit on the benchmark

The adapter class is `heartbit_tb2.agent:HeartbitAgent`. Make it importable
(either run Harbor from this directory, or `uv pip install -e .` here).

**Smoke (3–4 easy tasks, local, 1 trial):**

```bash
cd benchmarks/terminal-bench-2
export ANTHROPIC_API_KEY="..."
harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent-import-path heartbit_tb2.agent:HeartbitAgent \
  -m anthropic/claude-haiku-4-5 \
  -n 1
# narrow scope while validating, e.g. add Harbor's task/category filter flags.
```

**Scale on a cloud sandbox (Daytona):** local runs are slow (many turns × many
commands). Harbor recommends a cloud sandbox to parallelise beyond CPU cores:

```bash
export DAYTONA_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export HEARTBIT_INSTALL_MODE=prebuilt   # strongly recommended at scale (see below)
harbor run \
  -d terminal-bench/terminal-bench-2 \
  --agent-import-path heartbit_tb2.agent:HeartbitAgent \
  -m anthropic/claude-haiku-4-5 \
  --env daytona \
  -n 32
```

The model id maps automatically: `anthropic/claude-haiku-4-5` →
`HEARTBIT_PROVIDER=anthropic` + `HEARTBIT_MODEL=claude-haiku-4-5`, and the
matching API key is forwarded from your shell env into the container.

### Tunable env vars

| Var | Default | Meaning |
|-----|---------|---------|
| `HEARTBIT_INSTALL_MODE` | `build` | `build` (compile in-container) or `prebuilt` (upload binary). |
| `HEARTBIT_BIN` | `target/release/heartbit` | Prebuilt binary path on the host (`prebuilt` mode). |
| `HEARTBIT_SRC_BUNDLE` | `dist/heartbit-src.tar.gz` | Source bundle path (`build` mode). |
| `HEARTBIT_MAX_TURNS` | `60` | heartbit ReAct turn cap per task. |
| `HEARTBIT_TOOL_TIMEOUT` | `600` | Per-shell-command timeout (seconds; heartbit caps at 600). |
| `HEARTBIT_RUN_TIMEOUT_SEC` | `1800` | Wall-clock cap the adapter puts on the whole run. |
| `HEARTBIT_TB_WORKDIR` | `$(pwd)` in container | Task working dir (workspace root). |

## Token accounting

`populate_context_post_run` reads `/logs/agent/heartbit-trace.json` (the
`--trace-file` artefact) and fills `n_input_tokens` / `n_output_tokens` /
`n_cache_tokens` / `cost_usd`. If the binary predates `--trace-file`, it falls
back to scraping the `Tokens used: X in / Y out` footer from stderr. Scoring does
**not** depend on either — it's the container state — so token accounting is
best-effort reporting.

## Honest scope & known limits (measure, don't paper over)

- **TB2 is not "pure coding".** Of the 89 tasks, many are sysadmin, security,
  scientific computing, build/compilation, low-level debugging (compile CompCert,
  run Windows 3.11 under QEMU, recover a SQLite WAL…). heartbit *can* attempt
  these — bash is an unrestricted shell — but expect whole categories to be hard.
  Filter by category/difficulty for a realistic first pass.
- **Binary portability.** The binary is glibc-dynamic (OpenSSL/libcurl/zlib). It
  is **not** portable to Alpine/musl/distroless-static without slimming the crate
  (make `teloxide` optional, drop `full`/kafka → potentially musl-static). Until
  then, `prebuilt` only fits glibc bases; `build` fits any base with apt + a Rust
  toolchain + network.
- **Bash frictions.** The bash cwd resets to the workspace each call and its env
  is stripped to ~10 vars; instruct via absolute paths / `export`. Single-command
  wall time caps at 600 s — long builds may need chunking.
- **No built-in finish/self-verify gate** on the `run` path. The agent stops when
  it decides it's done or hits `max_turns`. (A goal/replan self-check exists in
  the runtime but isn't wired into `heartbit run` today.)

## Scaling caveat (important)

`build` mode compiles heartbit (full features incl. the librdkafka C build) in
**every** task container — minutes per container × n trials × 89 tasks, and it
needs network for rustup. That's fine for the local smoke but untenable for
Daytona `-n 32`. **For scale, use the slim static musl binary** (`build_musl.sh`
→ `HEARTBIT_INSTALL_MODE=prebuilt`): built once, uploaded as a single static
file into each container, portable to any base, no per-task compile, no runtime
libs. Report which mode you used with your numbers.

### How the slim binary is built (`--features slim`)

heartbit-cli is feature-gated: `default = ["full"]` (the normal binary, with
serve/daemon/persona/restate), and `slim = ["sandbox", "heartbit/a2a"]` keeps
only the `run`/`chat` env-path (AgentRunner + builtins + Anthropic/MCP/A2A over
rustls). The heavy deps (teloxide/rdkafka/restate-sdk/sqlx/axum/prometheus/
opentelemetry) are optional and their subcommands `#[cfg]`-gated, so the slim
graph drops openssl-sys/native-tls/librdkafka entirely — which is what makes the
static musl link possible. The full build is unchanged.

## Methodology

For a comparable measurement: fix the number of trials per task (`-n`), report the
**variance** across trials (not a single pass@1), and keep the trajectories
(`/logs/agent/heartbit-trace.json` per run). Leaderboard logs live in the
HuggingFace repo `alexgshaw/terminal-bench-2-leaderboard`; submit by PR per its
README.

## Tests

The Harbor-free helpers (`heartbit_io.py`) are unit-tested:

```bash
cd benchmarks/terminal-bench-2
uv run --with pytest python -m pytest tests/ -q
```
