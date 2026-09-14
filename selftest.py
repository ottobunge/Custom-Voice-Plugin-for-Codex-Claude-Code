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

# 1c. long text is chunked at sentence boundaries, each under the garble limit
from auk_engine import split_text, MAX_CHUNK_WORDS
long_text = (" ".join(f"Sentence number {i} says a few words here." for i in range(1, 12)))
chunks = split_text(long_text)
check("long text chunked", len(chunks) >= 2 and all(len(c.split()) <= MAX_CHUNK_WORDS for c in chunks),
      f"{[len(c.split()) for c in chunks]}")
check("chunking preserves words", " ".join(chunks).split() == long_text.split())
check("short text unchunked", split_text("hello world, this is a test") == ["hello world, this is a test"])
mono = "word " * 95  # no sentence boundaries at all -> hard word split
mono_chunks = split_text(mono)
check("monster sentence hard-split", all(len(c.split()) <= MAX_CHUNK_WORDS for c in mono_chunks),
      f"{[len(c.split()) for c in mono_chunks]}")

# 1d. real render loop: chunk fan-out through a fake generate + fake save module
import types, threading
gen_texts = []
class FakeGen:
    def generate(self, messages, *, gen_seconds):
        gen_texts.append(messages[0]["content"][0]["text"])
        class T:  # minimal tensor stand-in: render() only reads .shape[-1]
            def __init__(self, n): self.shape = [1, n]
        return T(int(gen_seconds * 24000)), 24000
fake_mod = types.ModuleType("auk.infer.infer_auk")
def fake_save(audio, sr, path):
    open(path, "wb").write(b"RIFFfake")
    return round(audio.shape[-1] / sr, 1)
fake_mod.save_audio = fake_save
sys.modules["auk.infer.infer_auk"] = fake_mod
eng = server.AukEngine.__new__(server.AukEngine)  # skip __init__/model load
eng._engine = FakeGen()
eng._asr = lambda audio, sr: ""  # numpy absent on build host; gate just retries to ceiling here
eng._lock = threading.Lock()
pieces, sr2, _worst = eng._render_all(split_text(long_text), 'Say: "{text}".', None)
check("render fans out per chunk", len(pieces) == len(chunks) and sr2 == 24000,
      f"{len(pieces)} pieces")
single_out = os.path.join(tmp, "single.wav")
secs = eng._write([pieces[0]], sr2, single_out)
check("write single piece seconds", secs == pieces[0].shape[-1] / sr2 and os.path.exists(single_out),
      f"{secs}")

# 1e. quality gate: render→ASR→similarity, retry-until-floor, keep best
from auk_engine import QUALITY_FLOOR, MAX_ATTEMPTS
check("similarity perfect match", eng._similarity("hello world there", "hello world there") == 1.0)
check("similarity garble lower", eng._similarity("helo wrld thre", "hello world there") < 0.9)

class FakeWhisper:
    """First ASR call per generate() call returns garble; the retry is heard perfectly."""
    def __init__(self, gen): self.gen, self.calls = gen, 0
    def transcribe(self, wave, fp16=False):
        self.calls += 1
        if self.calls % 2 == 1:
            return {"text": "garbled noise words"}
        import re as _re
        return {"text": _re.search(r'"(.*)"', self.gen.last_chunk).group(1)}

class GatedGen(FakeGen):
    def __init__(self):
        self.n, self.last_chunk = 0, ""
    def generate(self, messages, *, gen_seconds):
        self.n += 1
        self.last_chunk = messages[0]["content"][0]["text"]
        return super().generate(messages, gen_seconds=gen_seconds)[0], 24000

gen2 = GatedGen()
eng2 = server.AukEngine.__new__(server.AukEngine)
eng2._engine, eng2._whisper, eng2._lock = gen2, FakeWhisper(gen2), threading.Lock()
eng2.last_quality = None
eng2._asr = lambda audio, sr: eng2._get_whisper().transcribe(None, fp16=False)["text"]
piece, score, _ = eng2._render_gated("say this nicely please.", 'Say: "{text}".', None)
check("gate retries past garble", gen2.n == 2 and score == 1.0, f"renders={gen2.n} score={score}")
always_garble = server.AukEngine.__new__(server.AukEngine)
always_garble._engine = gen2
always_garble._whisper = None
always_garble._asr = lambda audio, sr: "unrelated gibberish text"
always_garble._lock = threading.Lock()
always_garble.last_quality = None
_, score2, _ = always_garble._render_gated("persist test chunk.", 'Say: "{text}".', None)
check("gate ceiling: keeps best after max attempts", gen2.n == 2 + MAX_ATTEMPTS and score2 < QUALITY_FLOOR,
      f"renders={gen2.n} score={score2}")
full = "Short first sentence here. And a second one that follows it along."
eng2.render(full, 'Say: "{text}".', None, os.path.join(tmp, "gated.wav"))
check("render exposes quality", eng2.last_quality == 1.0, str(eng2.last_quality))

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
