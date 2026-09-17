"""Backend registry. Spec strings:

    hf:<hf model id>            transformers on MPS/CUDA/CPU, exact
    llamacpp[:<url>]            running llama-server, exact
    ollama:<model tag>          running Ollama, approximate (top-k logprobs)
    <preset name>               see PRESETS
"""

from __future__ import annotations

from .base import DEFAULT_SYSTEM, Backend, BranchSpec, ScoreOutput

PRESETS: dict[str, str] = {
    # small, run in-process on a 24 GB Mac
    "qwen2.5-1.5b": "hf:Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3.5-0.8b": "hf:Qwen/Qwen3.5-0.8B",
    "qwen3.5-2b": "hf:Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "hf:Qwen/Qwen3.5-4B",
    "gemma3-1b": "hf:google/gemma-3-1b-it",
    "gemma4-e2b": "hf:google/gemma-4-E2B-it",
    "lfm2.5-1.2b": "hf:LiquidAI/LFM2.5-1.2B-Instruct",
    # bigger models through servers
    "llamacpp": "llamacpp:http://localhost:8091",
    "ollama-gemma4-e4b": "ollama:gemma4:e4b",
    "ollama-gemma4-26b": "ollama:gemma4:26b",
}


def load_backend(spec: str, **kwargs) -> Backend:
    spec = PRESETS.get(spec, spec)
    kind, _, rest = spec.partition(":")
    if kind == "hf":
        from .hf import HFBackend

        return HFBackend(rest, **kwargs)
    if kind == "llamacpp":
        from .llamacpp import LlamaCppBackend

        return LlamaCppBackend(rest or "http://localhost:8091", **kwargs)
    if kind == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend(rest, **kwargs)
    raise ValueError(f"unknown backend spec {spec!r}; presets: {', '.join(PRESETS)}")


__all__ = ["Backend", "BranchSpec", "ScoreOutput", "DEFAULT_SYSTEM", "PRESETS", "load_backend"]
