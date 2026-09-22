#!/usr/bin/env sh
# Bootstrap the environment and run the FORGE evaluation reproduction end-to-end.
#
# From a bare clone (Linux/macOS):
#     ./reproduce.sh                         # all four external cohorts
#     ./reproduce.sh --datasets stanford     # one external cohort
# Any flags are passed straight through to reproduce.py (see `./reproduce.sh --help`).
#
# Requires `uv` (https://docs.astral.sh/uv/). `uv sync --frozen` provisions the pinned
# Python 3.11 and all dependencies, so no pre-existing virtualenv is needed.
# Windows: run `uv sync --frozen && uv run python reproduce.py` (this wrapper is POSIX sh).
set -eu

# Run from the repo root regardless of the caller's working directory.
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: 'uv' is not installed." >&2
    echo "       install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "[reproduce.sh] Provisioning environment (uv sync --frozen)…"
uv sync --frozen

echo "[reproduce.sh] Running reproduction…"
exec uv run python reproduce.py "$@"
