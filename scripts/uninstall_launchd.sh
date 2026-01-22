#!/usr/bin/env bash
set -euo pipefail

# Uninstalls the QRouter FastAPI launchd service.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLIST_DST="${HOME}/Library/LaunchAgents/com.qrouter.api.plist"
SERVICE_NAME="com.qrouter.api"

echo "🛑 Uninstalling QRouter FastAPI launchd service..."

# Unload if running
if launchctl list | grep -q "${SERVICE_NAME}"; then
    echo "Unloading service..."
    launchctl unload "${PLIST_DST}" 2>/dev/null || true
fi

# Remove plist
if [[ -f "${PLIST_DST}" ]]; then
    rm -f "${PLIST_DST}"
    echo "✅ Removed ${PLIST_DST}"
fi

echo "✅ Uninstalled!"

