#!/usr/bin/env bash
# Run with the configured PPTI environment or the scratch project's venv.
set -euo pipefail

STK_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$STK_ROOT"

STK_ENV=${PPTI_ENV_PATH:-}
if [[ -z "$STK_ENV" && -f .ppti-env ]]; then
    IFS= read -r STK_ENV < .ppti-env
fi
if [[ -f .ppti-proxy ]]; then
    IFS= read -r STK_PROXY < .ppti-proxy
    export HTTPS_PROXY="$STK_PROXY" HTTP_PROXY="$STK_PROXY"
fi

if [[ -n "$STK_ENV" ]]; then
    STK_PYTHON="$STK_ENV/bin/python"
    [[ -x "$STK_PYTHON" ]] || STK_PYTHON="$STK_ENV/bin/python3"
elif [[ -x "$STK_ROOT/.venv/bin/python" ]]; then
    STK_PYTHON="$STK_ROOT/.venv/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    STK_PYTHON="$VIRTUAL_ENV/bin/python"
elif command -v pystk2 >/dev/null 2>&1; then
    STK_PYTHON="$(dirname -- "$(command -v pystk2)")/python"
else
    STK_PYTHON=$(command -v python3 || true)
fi

if [[ ! -x "$STK_PYTHON" ]]; then
    echo "Set PPTI_ENV_PATH to the preinstalled RL environment directory." >&2
    exit 1
fi

export PATH="$(dirname -- "$STK_PYTHON"):$PATH"
export PYTHONPATH="$STK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME=${PPTI_CACHE_DIR:-"$STK_ROOT/.cache"}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-"$STK_ROOT/.cache/config"}
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "$XDG_RUNTIME_DIR" || ! -w "$XDG_RUNTIME_DIR" ]]; then
    export XDG_RUNTIME_DIR="$STK_ROOT/.cache/runtime"
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-3}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-3}
mkdir -p "$XDG_CACHE_HOME"

if [[ $# -eq 0 ]]; then
    set -- make test
fi
STK_COMMAND=$1
shift
case "$STK_COMMAND" in
    python) exec "$STK_PYTHON" "$@" ;;
    pystk2) exec "$(dirname -- "$STK_PYTHON")/pystk2" "$@" ;;
    make)
        for STK_TARGET in "$@"; do
            if [[ "$STK_TARGET" == setup && ! -f .ppti-env ]]; then
                echo "Configure the PPTI runtime before running make setup." >&2
                exit 1
            fi
        done
        exec make "PYTHON=$STK_PYTHON" "PYSTK2=$(dirname -- "$STK_PYTHON")/pystk2" "$@"
        ;;
    *) exec "$STK_COMMAND" "$@" ;;
esac
