#!/usr/bin/env python3
"""AuK engine wrapper: lazy-loads AuK-Flash once, renders one clip at a time.

Contract (used by server.py and the setup self-test):
  render(text, instruction, ref_path, out_path) -> writes out_path (wav), returns seconds
Instruction templates use {text} as the spoken-text placeholder.
"""
import os, re, sys, threading, time

WORDS_PER_SEC = 2.5  # ponytail: flat rate estimate like our production rigs; +1 s margin


def estimate_seconds(text: str) -> float:
    return round(len(re.findall(r"\S+", text)) / WORDS_PER_SEC + 1.0, 1)


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
        cfg = os.path.join(self.ckpt_dir, "AuK-Flash", "config.yaml")
        ckpt = os.path.join(self.ckpt_dir, "AuK-Flash", "auk_flash.safetensors")
        for p in (cfg, ckpt):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing {p} — run setup.sh first")
        self._engine = AukInfer(cfg, ckpt)

    def render(self, text: str, instruction: str, ref_path: str | None, out_path: str) -> float:
        with self._lock:
            self._load()
            content = [{"type": "text", "text": instruction.format(text=text)}]
            if ref_path:
                content.append({"type": "audio", "audio": os.path.abspath(ref_path)})
            messages = [{"role": "user", "content": content}]
            gen_seconds = estimate_seconds(text)
            audio, sr = self._engine.generate(messages, gen_seconds=gen_seconds)
            from auk.infer.infer_auk import save_audio
            save_audio(audio, sr, out_path)
            return gen_seconds
