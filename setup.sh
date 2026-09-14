#!/usr/bin/env bash
# AuK Voice setup — macOS (Apple Silicon). Idempotent: safe to re-run.
set -euo pipefail

BASE="$HOME/.config/auk-voice"
mkdir -p "$BASE/voices" "$BASE/ckpts"
cp -n "$(dirname "$0")/config.example.toml" "$BASE/config.example.toml" 2>/dev/null || true
cp -n "$(dirname "$0")/server.py" "$BASE/server.py" 2>/dev/null || true
cp -n "$(dirname "$0")/auk_engine.py" "$BASE/auk_engine.py" 2>/dev/null || true

# venv with MPS torch
cd "$BASE"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install torch torchaudio  # MPS wheels are the default on arm64 macs
./venv/bin/pip install soundfile huggingface_hub[cli]

# AuK source
[ -d AuK ] || git clone https://github.com/Tencent-Hunyuan/AuK.git
./venv/bin/pip install -e ./AuK

# weights (~6-9 GB, skipped when present)
./venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
base = __import__("os").path.expanduser("~/.config/auk-voice/ckpts")
snapshot_download("tencent/AuK-Flash", local_dir=f"{base}/AuK-Flash")
snapshot_download("Qwen/Qwen2.5-Omni-3B", local_dir=f"{base}/Qwen2.5-Omni-3B")
print("weights ready")
PY

echo
echo "Done. Next:"
echo "  1. cp a clean 3-15 s voice sample to $BASE/voices/default.wav"
echo "  2. follow README.md to add the MCP server to Claude Desktop / Codex / Cursor"
