#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"

PYTHON=""
for p in python3 python; do
    if command -v "$p" >/dev/null 2>&1; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python not found." >&2
    exit 1
fi

exec "$PYTHON" -m nuclear_kinematics "$@"
