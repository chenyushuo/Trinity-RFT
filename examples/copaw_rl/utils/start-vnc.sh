#!/usr/bin/env bash
set -euo pipefail

DISPLAY_NUM=":99"
VNC_PORT="5900"
NOVNC_PORT="6080"
VNC_START_CMD="${VNC_START_CMD:-}"

Xvfb "${DISPLAY_NUM}" -screen 0 1920x1080x24 -ac +extension RANDR &
export DISPLAY="${DISPLAY_NUM}"

# Give Xvfb a moment to initialize before starting desktop components.
sleep 0.5

if command -v openbox >/dev/null 2>&1; then
	openbox &
else
	echo "[WARN] openbox not found, continuing without window manager"
fi

# Set a visible root background so the session is not perceived as a dead black screen.
if command -v xsetroot >/dev/null 2>&1; then
	xsetroot -solid "#1f2937" || true
fi

# Optional: launch one visible app for quick validation, e.g. VNC_START_CMD='chromium --no-sandbox'
if [ -n "${VNC_START_CMD}" ]; then
	echo "[INFO] launching VNC_START_CMD: ${VNC_START_CMD}"
	sh -c "${VNC_START_CMD}" &
fi

x11vnc -display "${DISPLAY_NUM}" -rfbport "${VNC_PORT}" -forever -shared -nopw -noxdamage &

if [ -x /opt/noVNC/utils/novnc_proxy ]; then
	exec /opt/noVNC/utils/novnc_proxy --vnc "localhost:${VNC_PORT}" --listen "${NOVNC_PORT}"
fi

if command -v novnc_proxy >/dev/null 2>&1; then
	exec novnc_proxy --vnc "localhost:${VNC_PORT}" --listen "${NOVNC_PORT}"
fi

if command -v websockify >/dev/null 2>&1; then
	for web_dir in /usr/share/novnc /usr/local/share/novnc /opt/noVNC; do
		if [ -d "${web_dir}"; then
			echo "[INFO] starting websockify with web root: ${web_dir}"
			exec websockify --web="${web_dir}" "${NOVNC_PORT}" "localhost:${VNC_PORT}"
		fi
	done
fi

echo "[ERROR] noVNC startup failed: novnc_proxy/websockify or noVNC web files not found"
echo "[HINT] install one of the following inside the container:"
echo "  1) apt-get install -y novnc websockify"
echo "  2) git clone https://github.com/novnc/noVNC.git /opt/noVNC"
echo "     git clone https://github.com/novnc/websockify.git /opt/noVNC/utils/websockify"
exit 1
