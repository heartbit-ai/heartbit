"""Harbor adapter that runs heartbit headless inside a Terminal-Bench 2.0 container.

Architecture: ``BaseInstalledAgent`` (Option B). heartbit's ``bash`` builtin
executes in-process, so running the ``heartbit`` binary *inside* the task
container makes its shell + file tools mutate that container natively — which is
exactly what Terminal-Bench 2.0 grades (the task's tests run in the container
after the agent finishes; ``AgentContext`` only carries token/cost accounting).

Two install modes (``HEARTBIT_INSTALL_MODE``):
  * ``build``    (default, robust): compile heartbit from a vendored source
                 bundle inside the container against its own glibc. Slow, needs a
                 Rust toolchain + apt; immune to glibc/OpenSSL ABI mismatch.
  * ``prebuilt`` (fast): upload a host-built ``heartbit`` binary and apt-install
                 only the runtime libs. Use once you've confirmed the TB2 base
                 image's glibc/OpenSSL matches your build host.

The adapter drives the no-config ENV path of ``heartbit run`` and relies on three
benchmark hooks added to heartbit-cli (all env/flag-gated, default behaviour
unchanged): ``--trace-file`` (serialised AgentOutput JSON), ``HEARTBIT_WORKSPACE``
(repoints the file-tool jail + bash cwd at the task dir), and
``HEARTBIT_NONINTERACTIVE`` (drops the blocking ``question`` tool).
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .heartbit_io import (
    build_heartbit_env,
    parse_stderr_tokens,
    parse_trace_tokens,
)

# Paths inside the container.
CONTAINER_BIN = "/usr/local/bin/heartbit"
CONTAINER_SRC = "/opt/heartbit-src"
# Harbor mounts the container's /logs/agent at self.logs_dir on the host, so
# anything heartbit writes here is readable post-run without an explicit download.
LOG_DIR = "/logs/agent"
TRACE_PATH = f"{LOG_DIR}/heartbit-trace.json"
STDOUT_PATH = f"{LOG_DIR}/heartbit.stdout"
STDERR_PATH = f"{LOG_DIR}/heartbit.stderr"

# Build-time and run-time package sets (Debian/Ubuntu).
BUILD_APT = "cmake libssl-dev libcurl4-openssl-dev pkg-config build-essential curl ca-certificates"
RUNTIME_APT = "ca-certificates libssl3 libcurl4 zlib1g"

# Host-side artefacts, resolved relative to the repo root (…/benchmarks/terminal-bench-2/heartbit_tb2/agent.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PREBUILT_BIN = _REPO_ROOT / "target" / "release" / "heartbit"
_DEFAULT_SRC_BUNDLE = _REPO_ROOT / "benchmarks" / "terminal-bench-2" / "dist" / "heartbit-src.tar.gz"


class HeartbitAgent(BaseInstalledAgent):
    """Run the heartbit multi-agent runtime headless against a TB2 task."""

    @staticmethod
    def name() -> str:
        return "heartbit"

    def version(self) -> str | None:
        return self._version or os.environ.get("HEARTBIT_VERSION")

    # ---- install -------------------------------------------------------------

    async def install(self, environment: BaseEnvironment) -> None:
        mode = os.environ.get("HEARTBIT_INSTALL_MODE", "build").strip().lower()
        if mode == "prebuilt":
            await self._install_prebuilt(environment)
        else:
            await self._install_build(environment)

    async def _install_prebuilt(self, environment: BaseEnvironment) -> None:
        bin_path = Path(os.environ.get("HEARTBIT_BIN", str(_DEFAULT_PREBUILT_BIN)))
        if not bin_path.is_file():
            raise FileNotFoundError(
                f"prebuilt heartbit binary not found at {bin_path}. Build it first "
                f"(benchmarks/terminal-bench-2/scripts/build_prebuilt.sh) or set HEARTBIT_BIN."
            )
        # Runtime libs first (best-effort: a non-apt base must ship them already).
        await self.exec_as_root(
            environment,
            command=f"apt-get update && apt-get install -y {RUNTIME_APT} || true",
            timeout_sec=600,
        )
        await environment.upload_file(source_path=bin_path, target_path=CONTAINER_BIN)
        await self.exec_as_root(environment, command=f"chmod +x {CONTAINER_BIN}")

    async def _install_build(self, environment: BaseEnvironment) -> None:
        bundle = Path(os.environ.get("HEARTBIT_SRC_BUNDLE", str(_DEFAULT_SRC_BUNDLE)))
        if not bundle.is_file():
            raise FileNotFoundError(
                f"source bundle not found at {bundle}. Create it with "
                f"benchmarks/terminal-bench-2/scripts/package_source.sh or set HEARTBIT_SRC_BUNDLE."
            )
        # 1) Build toolchain + system deps (root). librdkafka/openssl-sys need a C
        #    toolchain + cmake + the -dev headers; ca-certificates for any fetch.
        await self.exec_as_root(
            environment,
            command=f"apt-get update && apt-get install -y {BUILD_APT}",
            timeout_sec=1800,
        )
        # 2) Rust toolchain (edition 2024 needs >=1.85). Reuse an existing cargo if
        #    the base image already ships one; else install rustup (needs network).
        await self.exec_as_root(
            environment,
            command=(
                "if ! command -v cargo >/dev/null 2>&1; then "
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal; fi"
            ),
            timeout_sec=900,
        )
        # 3) Upload the vendored source bundle, then unpack it (offline build).
        target = self._upload_target(bundle)
        await environment.upload_file(source_path=bundle, target_path=target)
        await self.exec_as_root(
            environment,
            command=(
                f"rm -rf {CONTAINER_SRC} && mkdir -p {CONTAINER_SRC} && "
                f"tar -xzf {shlex.quote(target)} -C {CONTAINER_SRC}"
            ),
            timeout_sec=600,
        )
        # 4) Compile (offline against the vendored crates) and install the binary.
        await self.exec_as_root(
            environment,
            command=(
                'export PATH="$HOME/.cargo/bin:$PATH"; '
                f"cd {CONTAINER_SRC} && cargo build --release --offline -p heartbit-cli && "
                f"install -m 0755 target/release/heartbit {CONTAINER_BIN}"
            ),
            timeout_sec=3600,
        )

    @staticmethod
    def _upload_target(bundle: Path) -> str:
        return f"/tmp/{bundle.name}"

    # ---- run -----------------------------------------------------------------

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # Operate in the container's natural working directory unless overridden.
        workdir = os.environ.get("HEARTBIT_TB_WORKDIR")
        if not workdir:
            res = await environment.exec("pwd")
            workdir = (res.stdout or "/root").strip() or "/root"

        env = build_heartbit_env(
            model_name=self.model_name,
            base_env=dict(os.environ),
            workspace=workdir,
            max_turns=os.environ.get("HEARTBIT_MAX_TURNS", "60"),
            tool_timeout=os.environ.get("HEARTBIT_TOOL_TIMEOUT", "600"),
        )

        # ENV path of `heartbit run`: --trace-file gives a clean AgentOutput JSON.
        # The task is a clap `trailing_var_arg`, so --trace-file MUST precede the
        # instruction and NO `--` separator is used (a literal `--` would be
        # joined into the task text). </dev/null neutralises any stdin-blocking
        # tool; the instruction is a single shell-quoted arg; tee keeps raw logs.
        cmd = (
            f"{CONTAINER_BIN} run --trace-file {shlex.quote(TRACE_PATH)} "
            f"{shlex.quote(instruction)} "
            f"</dev/null >{shlex.quote(STDOUT_PATH)} 2>{shlex.quote(STDERR_PATH)}"
        )
        run_timeout = int(os.environ.get("HEARTBIT_RUN_TIMEOUT_SEC", "1800"))
        await self.exec_as_agent(
            environment, command=cmd, env=env, cwd=workdir, timeout_sec=run_timeout
        )

    # ---- post-run accounting (token/cost only; scoring is container state) ----

    def populate_context_post_run(self, context: AgentContext) -> None:
        trace = Path(self.logs_dir) / "heartbit-trace.json"
        tokens = parse_trace_tokens(trace.read_text()) if trace.is_file() else None
        if tokens is None:
            stderr = Path(self.logs_dir) / "heartbit.stderr"
            tokens = parse_stderr_tokens(stderr.read_text()) if stderr.is_file() else None
        if tokens is None:
            return
        context.n_input_tokens = tokens.get("n_input_tokens")
        context.n_output_tokens = tokens.get("n_output_tokens")
        context.n_cache_tokens = tokens.get("n_cache_tokens")
        context.cost_usd = tokens.get("cost_usd")
        context.metadata = tokens.get("metadata")
