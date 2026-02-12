#!/usr/bin/env bash
# ─── Push to Talk Installer ────────────────────────────────────────────────────
# Sets up Push to Talk as a background service that starts automatically.
# Works on macOS (launchd) and Linux (systemd).
#
# Usage:
#   ./install.sh           # Install and start
#   ./install.sh uninstall # Remove the service
# ────────────────────────────────────────────────────────────────────────────────

set -e

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_DIR="$HOME/.pushtotalk"
ENV_FILE="$CONFIG_DIR/.env"
PYTHON="${PYTHON:-python3}"
VENV_DIR="$APP_DIR/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ─── Uninstall ──────────────────────────────────────────────────────────────────

if [ "${1:-}" = "uninstall" ]; then
    echo ""
    info "Uninstalling Push to Talk service..."
    if [ "$(uname)" = "Darwin" ]; then
        PLIST="$HOME/Library/LaunchAgents/com.pushtotalk.app.plist"
        launchctl unload "$PLIST" 2>/dev/null || true
        rm -f "$PLIST"
        ok "macOS LaunchAgent removed."
    else
        systemctl --user stop pushtotalk.service 2>/dev/null || true
        systemctl --user disable pushtotalk.service 2>/dev/null || true
        rm -f "$HOME/.config/systemd/user/pushtotalk.service"
        systemctl --user daemon-reload 2>/dev/null || true
        ok "systemd service removed."
    fi
    echo ""
    info "Config files kept at $CONFIG_DIR (delete manually if desired)"
    exit 0
fi

# ─── Install ────────────────────────────────────────────────────────────────────

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║       Push to Talk — Installer        ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""

# 1. Create config directory
mkdir -p "$CONFIG_DIR"

# 2. Set up Python virtual environment
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created at $VENV_DIR"
else
    ok "Virtual environment already exists."
fi

info "Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
ok "Dependencies installed."

# 3. Set up .env file for API keys
if [ ! -f "$ENV_FILE" ]; then
    info "Creating config file at $ENV_FILE"
    cat > "$ENV_FILE" << 'ENVEOF'
# Push to Talk Configuration
# Uncomment and fill in to enable cloud features.

# Groq API key — enables fast cloud transcription (free at console.groq.com)
# GROQ_API_KEY=gsk_your_key_here

# Anthropic API key — enables LLM cleanup of transcriptions
# ANTHROPIC_API_KEY=sk-ant-your_key_here
ENVEOF
    ok "Config created. Edit $ENV_FILE to add your API keys."
else
    ok "Config file already exists at $ENV_FILE"
fi

PYTHON_BIN="$VENV_DIR/bin/python"

# 4. Platform-specific service setup
if [ "$(uname)" = "Darwin" ]; then
    # ─── macOS: LaunchAgent ─────────────────────────────────────────────
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST="$PLIST_DIR/com.pushtotalk.app.plist"
    LOG_DIR="$CONFIG_DIR/logs"
    mkdir -p "$PLIST_DIR" "$LOG_DIR"

    cat > "$PLIST" << PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pushtotalk.app</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>$APP_DIR/pushtotalk.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$APP_DIR</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_DIR/pushtotalk.log</string>
    <key>StandardErrorPath</key>
    <string>$LOG_DIR/pushtotalk.err</string>
</dict>
</plist>
PLISTEOF

    launchctl unload "$PLIST" 2>/dev/null || true
    launchctl load "$PLIST"
    ok "macOS LaunchAgent installed and started."
    echo ""
    info "Push to Talk will now start automatically on login."
    info "Logs: $LOG_DIR/pushtotalk.log"

else
    # ─── Linux: systemd user service ────────────────────────────────────
    SERVICE_DIR="$HOME/.config/systemd/user"
    SERVICE="$SERVICE_DIR/pushtotalk.service"
    LOG_DIR="$CONFIG_DIR/logs"
    mkdir -p "$SERVICE_DIR" "$LOG_DIR"

    cat > "$SERVICE" << SVCEOF
[Unit]
Description=Push to Talk — Voice Dictation
After=graphical-session.target sound.target

[Service]
Type=simple
WorkingDirectory=$APP_DIR
ExecStart=$PYTHON_BIN $APP_DIR/pushtotalk.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
SVCEOF

    systemctl --user daemon-reload
    systemctl --user enable pushtotalk.service
    systemctl --user restart pushtotalk.service
    ok "systemd user service installed and started."
    echo ""
    info "Push to Talk will now start automatically on login."
    info "Logs: journalctl --user -u pushtotalk -f"
fi

# 5. Summary
echo ""
echo "  ─────────────────────────────────────────"
echo "  Setup complete!"
echo ""
echo "  Web UI:    http://localhost:8528"
echo "  Config:    $ENV_FILE"
echo "  Hotkey:    Hold Option/Alt to dictate"
echo ""
echo "  To enable LLM cleanup, edit $ENV_FILE"
echo "  and add your ANTHROPIC_API_KEY."
echo ""
echo "  Commands:"
if [ "$(uname)" = "Darwin" ]; then
echo "    Stop:      launchctl unload ~/Library/LaunchAgents/com.pushtotalk.app.plist"
echo "    Start:     launchctl load ~/Library/LaunchAgents/com.pushtotalk.app.plist"
else
echo "    Stop:      systemctl --user stop pushtotalk"
echo "    Start:     systemctl --user start pushtotalk"
echo "    Logs:      journalctl --user -u pushtotalk -f"
fi
echo "    Uninstall: ./install.sh uninstall"
echo "  ─────────────────────────────────────────"
echo ""
