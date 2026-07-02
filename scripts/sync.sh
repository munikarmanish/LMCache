#!/usr/bin/env bash
set -euo pipefail

HOST="c2"
USER="manish"
TARGET="/home/manish/code/LMCache"

usage() {
    cat <<EOF
Usage: $0 [--host HOST] [--user USER] [--target TARGET]

Rsync the local LMCache directory to a remote node.

Options:
    --host HOST       Remote host (default: c2)
    --user USER       Remote user (default: manish)
    --target TARGET   Remote target directory (default: /home/manish/code/LMCache)
    -h, --help        Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --user) USER="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing ${SOURCE_DIR}/ -> ${USER}@${HOST}:${TARGET}/"

ssh "${USER}@${HOST}" "mkdir -p ${TARGET}"

rsync -avz --delete \
    --exclude='__pycache__/' \
    --exclude='*.py[cod]' \
    --exclude='*.so' \
    --exclude='*.o' \
    --exclude='*.swp' \
    --exclude='*.swo' \
    --exclude='*~' \
    --exclude='.DS_Store' \
    --exclude='.mypy_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.tox/' \
    --exclude='.venv/' \
    --exclude='venv/' \
    --exclude='env/' \
    --exclude='*.egg-info/' \
    --exclude='build/' \
    --exclude='dist/' \
    --exclude='.coverage' \
    --exclude='htmlcov/' \
    --exclude='.idea/' \
    --exclude='.vscode/' \
    --exclude='node_modules/' \
    --exclude='*.log' \
    --exclude='*.csv' \
    --exclude='*.tmp' \
    "${SOURCE_DIR}/" "${USER}@${HOST}:${TARGET}/"

echo "Done."
