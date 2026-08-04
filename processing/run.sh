#!/usr/bin/env bash
# Run a py5 sketch with the right Java + venv. Usage: ./run.sh flow.py
set -euo pipefail
cd "$(dirname "$0")"
export JAVA_HOME="/opt/homebrew/opt/openjdk@17"
exec .venv/bin/python "${1:-flow.py}"
