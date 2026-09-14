#!/usr/bin/env python3
"""AuK engine wrapper: lazy-loads AuK-Flash once, renders one clip at a time.

Contract (used by server.py and the setup self-test):
  render(text, instruction, ref_path, out_path) -> writes out_path (wav), returns seconds
Instruction templates use {text} as the spoken-text placeholder.
Long text is chunked at sentence boundaries (MAX_CHUNK_WORDS) and the chunk
audio is concatenated — AuK garbles beyond ~50 words.
"""
import os, re, sys, threading, time

WORDS_PER_SEC = 2.5  # ponytail: flat rate estimate like our production rigs; +1 s margin
MAX_CHUNK_WORDS = 40  # AuK garbles past ~50; chunk below it with sentence-boundary headroom


def estimate_seconds(text: str) -> float:
    return round(len(re.findall(r"\S+", text)) / WORDS_PER_SEC + 1.0, 1)


def split_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """Sentence-boundary chunks under max_words; oversized sentences split on words."""
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, buf = [], ""
    for s in sentences:
        cand = f"{buf} {s}".strip()
        if len(cand.split()) <= max_words:
            buf = cand
            continue
        if buf:
            chunks.append(buf)
        while len(s.split()) > max_words:  # monster sentence: hard word split
            words = s.split()
            chunks.append(" ".join(words[:max_words]))
            s = " ".join(words[max_words:])
        buf = s
    if buf:
        chunks.append(buf)
    return chunks


class AukEngine:
    def __init__(self, ckpt_dir: str, repo_dir: str):
        self.ckpt_dir = ckpt_dir
        self.repo_dir = repo_dir
        self._engine = None
        self._lock = threading.Lock()  # ponytail: one render at a time; GPU is single-tenant here

    def _load(self):
        if self._engine is not None:
            return
        src = os.path.join(self.repo_dir, "src")
        if os.path.isdir(src) and src not in sys.path:
            sys.path.insert(0, src)
        from auk.infer.infer_auk import AukInfer  # noqa: deferred — heavy
        import torch  # already loaded by infer_auk
        cfg = os.path.join(self.ckpt_dir, "AuK-Flash", "config.yaml")
        ckpt = os.path.join(self.ckpt_dir, "AuK-Flash", "auk_flash.safetensors")
        for p in (cfg, ckpt):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing {p} — run setup.sh first")
        qwen = os.path.join(self.ckpt_dir, "Qwen2.5-Omni-3B")
        # upstream default is cuda-or-cpu; Apple Silicon needs MPS passed explicitly
        device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else None
        self._engine = AukInfer(cfg, ckpt, device=device, qwen_path=qwen)

    def _render_chunk(self, text: str, instruction: str, ref_path: str | None):
        content = [{"type": "text", "text": instruction.format(text=text)}]
        if ref_path:
            content.append({"type": "audio", "audio": os.path.abspath(ref_path)})
        messages = [{"role": "user", "content": content}]
        audio, sr = self._engine.generate(messages, gen_seconds=estimate_seconds(text))
        return audio, sr

    def _render_all(self, chunks: list[str], instruction: str, ref_path: str):
        pieces, sr = [], None
        for chunk in chunks:
            audio, sr = self._render_chunk(chunk, instruction, ref_path)
            pieces.append(audio)
        return pieces, sr

    def _write(self, pieces: list, sr: int, out_path: str) -> float:
        from auk.infer.infer_auk import save_audio
        if len(pieces) == 1:
            audio = pieces[0]
        else:
            import torch
            audio = torch.cat(pieces, dim=-1)  # no crossfade: ponytail ceiling, add if joints audible
        save_audio(audio, sr, out_path)
        return round(audio.shape[-1] / sr, 1)  # actual rendered length, not the estimate

    def render(self, text: str, instruction: str, ref_path: str | None, out_path: str) -> float:
        with self._lock:
            self._load()
            pieces, sr = self._render_all(split_text(text), instruction, ref_path)
            return self._write(pieces, sr, out_path)
