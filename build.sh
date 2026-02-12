#!/usr/bin/env bash
# ─── Push to Talk — Build Standalone App ────────────────────────────────────────
#
# Builds a single-file executable using PyInstaller. The result is a portable
# binary that includes Python, all dependencies, and the web UI templates.
# Recipients don't need Python installed — just download and run.
#
# Usage:
#   ./build.sh          # Build for the current platform
#
# Output:
#   macOS:  dist/PushToTalk.app  (double-clickable application bundle)
#   Linux:  dist/pushtotalk      (single executable binary)
#
# Prerequisites (installed automatically by this script):
#   pip install pyinstaller
# ────────────────────────────────────────────────────────────────────────────────

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
VENV_DIR="$APP_DIR/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${BLUE}[BUILD]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║     Push to Talk — Build Script       ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

# ─── 1. Set up venv if needed ───────────────────────────────────────────────────

if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

info "Installing dependencies..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r "$APP_DIR/requirements.txt"
"$PIP" install --quiet pyinstaller
ok "Dependencies ready."

# ─── 2. Build ───────────────────────────────────────────────────────────────────

cd "$APP_DIR"

OS="$(uname)"
if [ "$OS" = "Darwin" ]; then
    info "Building macOS .app bundle..."
    "$PY" -m PyInstaller \
        --name "PushToTalk" \
        --onedir \
        --windowed \
        --noconfirm \
        --clean \
        --add-data "templates:templates" \
        --hidden-import "pystray._darwin" \
        --hidden-import "faster_whisper" \
        --hidden-import "ctranslate2" \
        --osx-bundle-identifier "com.pushtotalk.app" \
        pushtotalk.py

    ok "Built: dist/PushToTalk.app"
    echo ""
    info "To distribute: zip the .app bundle"
    echo -e "  ${BOLD}cd dist && zip -r PushToTalk-macOS.zip PushToTalk.app${NC}"
    echo ""
    info "To install on a Mac:"
    echo "  1. Unzip PushToTalk-macOS.zip"
    echo "  2. Drag PushToTalk.app to /Applications"
    echo "  3. Double-click to launch"
    echo "  4. Grant Accessibility permission when prompted"

else
    info "Building Linux binary..."
    "$PY" -m PyInstaller \
        --name "pushtotalk" \
        --onedir \
        --noconfirm \
        --clean \
        --add-data "templates:templates" \
        --hidden-import "pystray._xorg" \
        --hidden-import "pystray._appindicator" \
        --hidden-import "faster_whisper" \
        --hidden-import "ctranslate2" \
        pushtotalk.py

    ok "Built: dist/pushtotalk/"
    echo ""
    info "To distribute: tar the output directory"
    echo -e "  ${BOLD}cd dist && tar czf pushtotalk-linux.tar.gz pushtotalk/${NC}"
    echo ""
    info "To install on Linux:"
    echo "  1. Extract: tar xzf pushtotalk-linux.tar.gz"
    echo "  2. Run:     ./pushtotalk/pushtotalk"
    echo "  3. (Optional) Copy to /opt and symlink to /usr/local/bin"
fi

# ─── 3. Summary ─────────────────────────────────────────────────────────────────

echo ""
SIZE=$(du -sh "$APP_DIR/dist" 2>/dev/null | cut -f1)
echo "  ─────────────────────────────────────────"
echo "  Build complete!  (${SIZE:-?})"
echo ""
echo "  Output in: $APP_DIR/dist/"
echo ""
echo "  Config:  ~/.pushtotalk/.env (API keys)"
echo "  Web UI:  http://localhost:8528"
echo "  ─────────────────────────────────────────"
echo ""
