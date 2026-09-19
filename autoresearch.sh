#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v bun >/dev/null 2>&1; then
  echo "GATE_FAIL bun not on PATH" >&2
  exit 1
fi

bun test tests
bun run bench/identity_bench.ts
