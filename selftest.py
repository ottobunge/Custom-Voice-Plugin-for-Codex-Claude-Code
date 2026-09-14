#!/usr/bin/env python3
"""Self-test: drives server.py over real stdio with a stubbed engine.
Verifies MCP handshake, tools/list, speak (render+config path), set_voice_instruction persistence.
Run from the plugin dir: python3 selftest.py   (exit 0 = pass)"""
import json, os, subprocess, sys, tempfile, threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import server  # noqa: E402

# --- stub the engine so no GPU/model needed ---
class StubEngine:
    def __init__(self):
        self.calls = []
    def render(self, text, instruction, ref_path, out_path):
        self.calls.append((text, instruction, ref_path, out_path))
        with open(out_path, "wb") as f:
            f.write(b"RIFFstub")
        return round(len(text.split()) / 2.5 + 1.0, 1)

server._engine = StubEngine()

# create fake voices dir + config
tmp = tempfile.mkdtemp(prefix="auk-selftest-")
server.VOICES_DIR = os.path.join(tmp, "voices")
server.CONFIG_DIR = tmp
server.CONFIG_PATH = os.path.join(tmp, "config.toml")
os.makedirs(server.VOICES_DIR)
with open(os.path.join(server.VOICES_DIR, "default.wav"), "wb") as f:
    f.write(b"RIFF")

# monkeypatch play to no-op
server.play_audio = lambda p: {"played": True, "file": p}

fails = []
def check(name, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(name)

# 1. speak with default voice
r = server.tool_speak({"text": "hello world, this is a test"})
check("speak renders", "error" not in r, r.get("error", ""))
check("speak returns seconds", isinstance(r.get("seconds"), float))
check("engine got template + text separately",
      len(server._engine.calls) == 1
      and server._engine.calls[0][0] == "hello world, this is a test"
      and "{text}" in server._engine.calls[0][1])
check("engine got ref path", server._engine.calls[0][2].endswith("default.wav"))

# 1b. engine-side {text} substitution + duration estimate
from auk_engine import estimate_seconds
check("engine duration estimate", estimate_seconds("one two three four five") == 3.0,
      str(estimate_seconds("one two three four five")))

# 2. speak with missing voice -> clean error
r = server.tool_speak({"text": "hi", "voice": "ghost.wav"})
check("missing voice errors cleanly", "error" in r and "available" in r)

# 3. instruction without {text} gets it appended (template check, engine substitutes)
r = server.tool_speak({"text": "hi there", "instruction_override": "Whisper excitedly"})
check("override gets {text} appended", 'Whisper excitedly: "{text}".' in server._engine.calls[-1][1],
      server._engine.calls[-1][1])

# 4. set_voice_instruction persists
r = server.tool_set_instruction({"instruction": 'Say like a pirate: "{text}".'})
check("set instruction ok", r.get("ok") is True)
check("persisted to config.toml", os.path.exists(server.CONFIG_PATH)
      and "pirate" in open(server.CONFIG_PATH).read())
# 4b. round-trip: what save_instruction writes into [voice] is what load_config reads back
server._state["voice_ref"] = "default.wav"
saved_inst = server._state["voice_instruction"]
server._state["voice_instruction"] = "cleared"
server.load_config()
check("config round-trip reload", server._state["voice_instruction"] == saved_inst,
      repr(server._state["voice_instruction"]))
server.load_config = lambda: None  # ensure reload path doesn't clobber state check
check("state holds new instruction", server._state["voice_instruction"].startswith("Say like a pirate"))

# 5. MCP protocol loop (real stdio): initialize -> tools/list -> tools/call(get_voice_config)
proc = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py")],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        env={**os.environ, "HOME": tmp})  # isolated HOME -> config defaults

def rpc(obj):
    proc.stdin.write((json.dumps(obj) + "\n").encode()); proc.stdin.flush()
    return json.loads(proc.stdout.readline())

r = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
check("MCP initialize", r.get("result", {}).get("serverInfo", {}).get("name") == "auk-voice")
r = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
names = {t["name"] for t in r["result"]["tools"]}
check("tools/list has all 4 tools", {"speak", "set_voice_instruction", "get_voice_config", "list_voices"} <= names, names)
r = rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "list_voices", "arguments": {}}})
check("tools/call list_voices over stdio", "voices" in r["result"]["content"][0]["text"])
proc.stdin.close(); proc.wait(timeout=10)

print()
print("ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
