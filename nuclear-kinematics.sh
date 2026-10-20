#!/bin/sh
DIR="$(cd "$(dirname "$0")" && pwd)"

PYTHON=""
if [ -x "$DIR/.venv/bin/python" ]; then
    PYTHON="$DIR/.venv/bin/python"
fi
[ -z "$PYTHON" ] && for p in python3 python; do
    if command -v "$p" >/dev/null 2>&1; then
        PYTHON="$p"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python not found." >&2
    exit 1
fi

cd "$DIR" && exec "$PYTHON" -m nuclear_kinematics "$@"
