#!/usr/bin/env bash
set -euo pipefail

# Installs QRouter FastAPI as a launchd agent on macOS.
# This keeps the server running in the background, even after closing the terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLIST_SRC="${REPO_DIR}/deploy/launchd/com.qrouter.api.plist"
PLIST_DST="${HOME}/Library/LaunchAgents/com.qrouter.api.plist"
SERVICE_NAME="com.qrouter.api"

echo "📦 Installing QRouter FastAPI as a launchd service..."

# Check if venv exists
if [[ ! -d "${REPO_DIR}/venv" ]]; then
    echo "❌ Error: venv not found at ${REPO_DIR}/venv"
    echo "   Create it with: python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check if uvicorn is installed
if [[ ! -f "${REPO_DIR}/venv/bin/uvicorn" ]]; then
    echo "❌ Error: uvicorn not found in venv"
    echo "   Install dependencies: ./venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Create logs directory
mkdir -p "${REPO_DIR}/logs"

# Update plist with actual repo path (replace placeholder)
sed "s|/Users/dhrupadrajrai/qrouter|${REPO_DIR}|g" "${PLIST_SRC}" > "${PLIST_DST}"

# Unload if already running
if launchctl list | grep -q "${SERVICE_NAME}"; then
    echo "🔄 Unloading existing service..."
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
fi

# Load the service
echo "🚀 Loading service..."
launchctl load "${PLIST_DST}"

echo "✅ QRouter FastAPI is now running as a background service!"
echo ""
echo "Check status:"
echo "  launchctl list | grep ${SERVICE_NAME}"
echo ""
echo "View logs:"
echo "  tail -f ${REPO_DIR}/logs/qrouter.out.log"
echo ""
echo "Stop service:"
echo "  launchctl unload ${PLIST_DST}"
echo ""
echo "Or run: ./scripts/uninstall_launchd.sh"
