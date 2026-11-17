#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python scripts/check-production.py

api_pid=""
ui_pid=""

cleanup() {
    status=$?
    trap - EXIT INT TERM

    for pid in "$api_pid" "$ui_pid"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    for pid in "$api_pid" "$ui_pid"; do
        if [[ -n "$pid" ]]; then
            wait "$pid" 2>/dev/null || true
        fi
    done

    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

gunicorn pondlite.serve:app \
    --bind 0.0.0.0:8420 \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 1 \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - &
api_pid=$!

gunicorn pondlite.web:app \
    --bind 0.0.0.0:8421 \
    --worker-class sync \
    --workers 2 \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - &
ui_pid=$!

# If either server exits, the EXIT trap stops the other one. Avoid wait -n
# so the script also works with the older Bash shipped by macOS.
while kill -0 "$api_pid" 2>/dev/null && kill -0 "$ui_pid" 2>/dev/null; do
    sleep 1
done

status=0
if kill -0 "$api_pid" 2>/dev/null; then
    set +e
    wait "$ui_pid"
    status=$?
    set -e
else
    set +e
    wait "$api_pid"
    status=$?
    set -e
fi
exit "$status"
