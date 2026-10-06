#!/usr/bin/env bash
# Create bench/.venv with the Python dependencies of gmxbench.
# Uses `uv` when available (downloading a private copy into bench/.tools if needed),
# otherwise python3 -m venv + pip.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
py="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
    uv=uv
elif [ -x "$here/.tools/uv" ]; then
    uv="$here/.tools/uv"
elif "$py" -m pip --version >/dev/null 2>&1; then
    uv=""
else
    echo "installing a private copy of uv into $here/.tools" >&2
    mkdir -p "$here/.tools"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$here/.tools" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
    uv="$here/.tools/uv"
fi
if [ -n "$uv" ]; then
    "$uv" venv -q --python "$py" "$here/.venv"
    "$uv" pip install -q -p "$here/.venv/bin/python" -r "$here/requirements.txt"
else
    "$py" -m venv "$here/.venv"
    "$here/.venv/bin/python" -m pip install -q -r "$here/requirements.txt"
fi
echo "ok: $here/.venv"
