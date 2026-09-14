#!/usr/bin/env python3
"""auk-voice: local MCP server giving Claude Desktop / Codex / Cursor a cloned TTS voice.

Stdio MCP (JSON-RPC 2.0, no deps beyond stdlib + auk_engine). Tools:
  speak(text, voice?, instruction_override?)  -> renders locally with AuK and plays
  set_voice_instruction(text)                 -> agents adjust the base instruction at runtime
  get_voice_config()                          -> current config summary
  list_voices()                               -> available reference voices
Config: ~/.config/auk-voice/config.toml  (see config.example.toml)
"""
import base64, json, os, re, sys, subprocess, tempfile, threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auk_engine import AukEngine  # noqa: E402

CONFIG_DIR = os.path.expanduser("~/.config/auk-voice")
CONFIG_PATH = os.path.join(CONFIG_DIR, "config.toml")
VOICES_DIR = os.path.join(CONFIG_DIR, "voices")
DEFAULTS = {
    "voice_ref": "default.wav",
    "voice_instruction": 'Say the following with the same voice: "{text}".',
    "auto_play": True,
}

_lock = threading.Lock()
_state = dict(DEFAULTS)
_engine = None


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return
    try:
        import tomllib
        with open(CONFIG_PATH, "rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:
        log(f"config parse failed, using defaults: {e}")
        return
    voice = cfg.get("voice", {})
    play = cfg.get("playback", {})
    _state["voice_ref"] = voice.get("ref", DEFAULTS["voice_ref"])
    _state["voice_instruction"] = voice.get("instruction", DEFAULTS["voice_instruction"])
    _state["auto_play"] = play.get("auto_play", DEFAULTS["auto_play"])
    server = cfg.get("server", {})
    _state["ckpt_dir"] = server.get("ckpt_dir", os.path.join(CONFIG_DIR, "ckpts"))
    _state["repo_dir"] = server.get("repo_dir", os.path.join(CONFIG_DIR, "AuK"))


def save_instruction():
    """Persist runtime instruction/ref changes back into the [voice] section.

    ponytail: section-aware regex rewrite so comments elsewhere survive; full
    tomllib round-trip if config ever grows beyond these two keys.
    """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    inst = _state["voice_instruction"].replace('"', r'\"')
    ref = _state["voice_ref"].replace('"', r'\"')
    block = f'[voice]\nref = "{ref}"\ninstruction = "{inst}"\n'
    text = open(CONFIG_PATH, encoding="utf-8").read() if os.path.exists(CONFIG_PATH) else ""
    if re.search(r'^\[voice\]\s*$', text, re.M):
        text = re.sub(r'^\[voice\]\s*$(?:\n(?!\[).*)*', block, text, flags=re.M | re.S)
    else:
        text = text.rstrip("\n") + ("\n\n" if text.strip() else "") + block
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(text)


def log(msg):
    print(msg, file=sys.stderr, flush=True)  # stderr only — stdout is the MCP protocol


def get_engine():
    global _engine
    if _engine is None:
        ckpt = _state.get("ckpt_dir", os.path.join(CONFIG_DIR, "ckpts"))
        repo = _state.get("repo_dir", os.path.join(CONFIG_DIR, "AuK"))
        _engine = AukEngine(ckpt, repo)
    return _engine


def play_audio(path):
    if not _state["auto_play"]:
        return {"played": False, "file": path}
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["afplay", path])
        else:
            subprocess.Popen(["aplay", path])
        return {"played": True, "file": path}
    except Exception as e:
        return {"played": False, "error": str(e), "file": path}


# ---------- tools ----------

def tool_speak(args):
    text = (args.get("text") or "").strip()
    if not text:
        return {"error": "text is required"}
    ref = args.get("voice") or _state["voice_ref"]
    ref_path = ref if os.path.isabs(ref) else os.path.join(VOICES_DIR, os.path.basename(ref))
    instruction = args.get("instruction_override") or _state["voice_instruction"]
    if "{text}" not in instruction:
        instruction = instruction.rstrip(". ") + ': "{text}".'
    if ref != "none" and not os.path.exists(ref_path) and not os.path.isabs(ref):
        return {"error": f"voice '{ref}' not found in {VOICES_DIR}", "available": os.listdir(VOICES_DIR) if os.path.isdir(VOICES_DIR) else []}
    os.makedirs("/tmp/auk-voice", exist_ok=True)
    out = tempfile.mktemp(prefix="reply-", suffix=".flac", dir="/tmp/auk-voice")
    eng = get_engine()
    with _lock:
        try:
            secs = eng.render(text, instruction, None if ref == "none" else ref_path, out)
        except Exception as e:
            return {"error": f"render failed: {e}"}
    result = {"seconds": secs, "file": out, "voice": os.path.basename(ref), **play_audio(out)}
    return result


def tool_set_instruction(args):
    t = (args.get("instruction") or "").strip()
    if not t:
        return {"error": "instruction is required", "current": _state["voice_instruction"]}
    _state["voice_instruction"] = t
    save_instruction()
    return {"ok": True, "voice_instruction": t}


def tool_get_config(args):
    return {k: _state.get(k) for k in ("voice_ref", "voice_instruction", "auto_play")}


def tool_list_voices(args):
    if not os.path.isdir(VOICES_DIR):
        return {"voices": [], "hint": f"drop .wav reference clips into {VOICES_DIR}"}
    return {"voices": sorted(f for f in os.listdir(VOICES_DIR) if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a")))}


TOOLS = {
    "speak": (tool_speak, "Speak text aloud with the configured cloned voice. Args: text (required), voice (optional filename from list_voices), instruction_override (optional, {text} marks where the spoken text goes).", {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "voice": {"type": "string"},
            "instruction_override": {"type": "string"},
        },
        "required": ["text"],
    }),
    "set_voice_instruction": (tool_set_instruction, "Change the base voice instruction used by speak() for all future replies. Pass a template containing {text}.", {
        "type": "object",
        "properties": {"instruction": {"type": "string"}},
        "required": ["instruction"],
    }),
    "get_voice_config": (tool_get_config, "Return current voice reference and instruction template.", {"type": "object", "properties": {}}),
    "list_voices": (tool_list_voices, "List available voice reference files.", {"type": "object", "properties": {}}),
}


# ---------- MCP stdio protocol ----------

def recv_msg():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def send_msg(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    load_config()
    os.makedirs(VOICES_DIR, exist_ok=True)
    while True:
        try:
            msg = recv_msg()
        except Exception as e:
            log(f"bad message: {e}")
            continue
        if msg is None:
            break
        method = msg.get("method", "")
        mid = msg.get("id")
        if method == "initialize":
            send_msg({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "auk-voice", "version": "0.1.0"},
            }})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            send_msg({"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": n, "description": d, "inputSchema": s} for n, (fn, d, s) in TOOLS.items()
            ]}})
        elif method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})
            fn = TOOLS.get(name, (None, None, None))[0]
            if fn is None:
                send_msg({"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool {name}"}})
                continue
            try:
                result = fn(args)
                send_msg({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}], "isError": bool(result.get("error")),
                }})
            except Exception as e:
                send_msg({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": f"tool crashed: {e}"}], "isError": True,
                }})
        elif method == "ping":
            send_msg({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None:
            send_msg({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}})


if __name__ == "__main__":
    main()
