#!/usr/bin/env bash
# Local dev entry point.
# Usage: ./run_api.sh              -> reload mode on localhost:8000
#        ./run_api.sh prod         -> 2 workers, no reload
set -euo pipefail

MODE="${1:-dev}"

if [[ "$MODE" == "prod" ]]; then
  exec uvicorn api.main:app \
    --host 0.0.0.0 --port "${PORT:-8000}" \
    --timeout-keep-alive 300 \
    --workers 2
else
  exec uvicorn api.main:app \
    --host 0.0.0.0 --port "${PORT:-8000}" \
    --timeout-keep-alive 300 \
    --reload
fi
