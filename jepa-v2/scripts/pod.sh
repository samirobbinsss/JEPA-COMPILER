#!/usr/bin/env bash
# Helpers to talk to the shared RunPod B200 pod (see ../../RUNPOD_CONNECT.md).
#
# Why this exists: the RunPod SSH proxy is special.
#   * `ssh host 'cmd'` is IGNORED — commands must be piped via stdin, ending `exit`.
#   * `scp`/SFTP do NOT work (no subsystem on the proxy).
#   * `printf '...%...'` mangles any '%' before it reaches the remote (breaks
#     Python %-formatting). So we transfer files as base64 over stdin.
#
# All our work lives under /workspace/jepa-v2 with an isolated venv — the pod is
# SHARED, so we never touch the system Python and never throttle the GPU.
set -euo pipefail

KEY="${JEPA_POD_KEY:-$HOME/.ssh/jepa_asml_remote}"
HOST="${JEPA_POD_HOST:-xmtmw5izfpool0-64411f3c@ssh.runpod.io}"
REMOTE_ROOT="/workspace/jepa-v2"
VENV_ACTIVATE=". $REMOTE_ROOT/.venv/bin/activate"

_clean() {
  grep -ivE "post-quantum|store now|upgraded|openssh.com/pq|Warning: (Permanently|Identity)|Enjoy your|docs.runpod|blog.runpod|For detailed|RUNPOD" \
    | tr -d '\r' | sed -E 's/\x1b\[[0-9;?]*[a-zA-Z]//g' \
    | grep -vE '_____|^\| |\|_|/ _|\| \|'
}

# Run a remote command (string) inside the project venv. Piped via stdin.
pod_run() {
  { printf 'cd %s 2>/dev/null || mkdir -p %s && cd %s\n' "$REMOTE_ROOT" "$REMOTE_ROOT" "$REMOTE_ROOT"
    printf '%s\n' "$VENV_ACTIVATE 2>/dev/null || true"
    printf '%s\n' "$1"
    printf 'exit\n'
  } | ssh -tt -o StrictHostKeyChecking=accept-new -i "$KEY" "$HOST" 2>&1 | _clean
}

# Upload a local file to REMOTE_ROOT/<dest> (base64 over stdin; scp doesn't work).
pod_put() {
  local src="$1" dest="${2:-$(basename "$1")}"
  local b64; b64=$(base64 < "$src" | tr -d '\n')
  { printf 'mkdir -p %s && mkdir -p "$(dirname %s/%s)"\n' "$REMOTE_ROOT" "$REMOTE_ROOT" "$dest"
    printf 'echo %s | base64 -d > %s/%s\n' "$b64" "$REMOTE_ROOT" "$dest"
    printf 'echo PUT_OK %s\n' "$dest"
    printf 'exit\n'
  } | ssh -tt -o StrictHostKeyChecking=accept-new -i "$KEY" "$HOST" 2>&1 | _clean | grep -E "PUT_OK|No such|error" || true
}

case "${1:-}" in
  run) shift; pod_run "$*";;
  put) shift; pod_put "$@";;
  shell) ssh -tt -o StrictHostKeyChecking=accept-new -i "$KEY" "$HOST";;
  *) echo "usage: pod.sh {run <cmd> | put <local> [remote] | shell}"; exit 1;;
esac
