#!/usr/bin/env bash
# ─── Push to Talk — Double-Click Launcher ─────────────────────────────────────
# Double-click this file in Finder to launch Push to Talk.
# On first run it will set up a virtual environment and install dependencies.
# ──────────────────────────────────────────────────────────────────────────────

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="$APP_DIR/.venv"

# Set up venv on first run
if [ ! -d "$VENV_DIR" ]; then
    echo "First run — setting up Python environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
    echo "Setup complete!"
    echo ""
fi

# Run the app
cd "$APP_DIR"
exec "$VENV_DIR/bin/python" pushtotalk.py
