# AuK Voice — local TTS voice for Claude Desktop / Codex / Cursor (macOS, Apple Silicon)

One-file MCP server + local AuK-Flash engine. Every reply your agent gives can be spoken
aloud in a cloned voice. No cloud, no API keys; renders on your machine.

## Setup (one time)

```bash
bash setup.sh          # clones AuK, makes venv, downloads AuK-Flash + Qwen2.5-Omni-3B (~6-9 GB)
                       # needs Python >= 3.10 (checked; brew hint on failure). MPS is picked automatically.
cp ~/.config/auk-voice/config.example.toml ~/.config/auk-voice/config.toml   # if not already created
cp your-voice-sample.wav ~/.config/auk-voice/voices/default.wav   # 3-15 s clean speech
python3 server.py      # leave running? No — the GUI launches it on demand (below)
```

## Wire into Claude Desktop

Settings → Developer → Edit Config → `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "auk-voice": {
      "command": "/Users/YOU/.config/auk-voice/venv/bin/python",
      "args": ["/Users/YOU/.config/auk-voice/server.py"]
    }
  }
}
```

Restart Claude Desktop. Ask it: "list your available voices" / "say hello in my voice".

## Wire into Codex (desktop app + IDE + CLI share this)

`~/.codex/config.toml`:

```toml
[mcp_servers.auk-voice]
command = "/Users/YOU/.config/auk-voice/venv/bin/python"
args = ["/Users/YOU/.config/auk-voice/server.py"]
```

## Wire into Cursor

Cursor Settings → MCP → Add server: same command/args as above.

## Config

`~/.config/auk-voice/config.toml`:

```toml
[voice]
ref = "default.wav"                      # file in voices/
instruction = 'Say the following with the same voice: "{text}".'

[playback]
auto_play = true                          # false = just render, no sound
```

The agent can change the instruction itself via the `set_voice_instruction` tool
(e.g. "make your voice deep and villainous from now on"). It persists to config.toml.

Long replies (>40 words) are auto-chunked at sentence boundaries and stitched into
one audio file — AuK starts garbling past ~50 words in a single render.

Every chunk is quality-gated: it's transcribed with Whisper-small (CPU, ~461 MB
download on first render) and compared against the source text; renders below 0.94
word similarity are retried (up to 6 fresh takes each) and the best take is kept.
The `speak` result reports the weakest chunk's score as `quality`.

## Voice instruction cookbook

The `instruction` is natural language; `{text}` is replaced with what the agent wants spoken.

- Clone: `Say the following with the same voice: "{text}".`
- Villain: `Say the following in a deep, slow, menacing villain voice: "{text}".`
- Whisper: `Whisper the following excitedly: "{text}".`
- No reference file needed (pure description): set `ref = "none"` and use
  `Based on the following description: "a cheerful British game-show host", generate speech content "{text}".`

## Files

- `setup.sh` — install (clone AuK, venv, download weights)
- `server.py` — the MCP server (copy to ~/.config/auk-voice/ or reference in place)
- `auk_engine.py` — engine wrapper
- `config.example.toml`
