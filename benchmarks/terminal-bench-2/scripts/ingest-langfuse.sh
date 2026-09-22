#!/usr/bin/env bash
# Ingest Harbor TB2 job traces into Langfuse (host-side, post-run).
#
# Usage:
#   benchmarks/terminal-bench-2/scripts/ingest-langfuse.sh --dry-run jobs/tb2-qwen-orch-max8192
#   LANGFUSE_PUBLIC_KEY=… LANGFUSE_SECRET_KEY=… \
#     benchmarks/terminal-bench-2/scripts/ingest-langfuse.sh jobs/tb2-qwen-orch-max8192
#
# Env:
#   LANGFUSE_HOST          default http://localhost:3000
#   LANGFUSE_PUBLIC_KEY    required unless --dry-run
#   LANGFUSE_SECRET_KEY    required unless --dry-run
#   LANGFUSE_IO_CHARS      truncate tool I/O (default 4000)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
TB2="$ROOT/benchmarks/terminal-bench-2"
cd "$TB2"
export PYTHONPATH="$TB2${PYTHONPATH:+:$PYTHONPATH}"

DRY=0
JOB=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    -h|--help)
      sed -n '2,16p' "$0"
      exit 0
      ;;
    *)
      JOB="$1"
      shift
      ;;
  esac
done

if [[ -z "$JOB" ]]; then
  echo "usage: $0 [--dry-run] <job-dir>" >&2
  exit 2
fi
# Accept relative paths from TB2 or absolute.
if [[ ! -d "$JOB" ]]; then
  if [[ -d "$TB2/$JOB" ]]; then
    JOB="$TB2/$JOB"
  else
    echo "job dir not found: $JOB" >&2
    exit 1
  fi
fi

HOST="${LANGFUSE_HOST:-http://localhost:3000}"
IO_CHARS="${LANGFUSE_IO_CHARS:-4000}"

python3 - "$JOB" "$DRY" "$HOST" "$IO_CHARS" <<'PY'
import json, os, sys
from pathlib import Path
from heartbit_tb2.langfuse_ingest import build_job_batch, post_batch

job = Path(sys.argv[1])
dry = sys.argv[2] == "1"
host = sys.argv[3]
io_chars = int(sys.argv[4])
batch = build_job_batch(job, io_chars=io_chars)
print(f"batch events={len(batch)} job={job}", file=sys.stderr)
if dry:
    print(json.dumps({"batch": batch}, indent=2, ensure_ascii=False))
    raise SystemExit(0)
pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
if not pk or not sk:
    print("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (or pass --dry-run)", file=sys.stderr)
    raise SystemExit(2)
resp = post_batch(batch, host=host, public_key=pk, secret_key=sk)
print(json.dumps(resp, indent=2, ensure_ascii=False))
PY
