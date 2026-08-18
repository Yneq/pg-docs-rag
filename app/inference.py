"""Selectable local text-generation backends.

Ollama remains the default.  The Transformers dependencies and model are loaded
only when the ``transformers`` backend is selected, so existing Ollama users do
not pay their installation or startup cost.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Protocol


class InferenceBackend(Protocol):
    """Small interface used by the RAG scripts."""

    def generate(self, prompt: str) -> str:
        """Generate text from a complete prompt."""


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class InferenceConfig:
    backend: str = "ollama"
    ollama_model: str = "llama3.2:latest"
    hf_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_load_in_4bit: bool = True
    hf_max_new_tokens: int = 512
    hf_temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        return cls(
            backend=os.getenv("LLM_BACKEND", "ollama").strip().lower(),
            ollama_model=os.getenv("OLLAMA_LLM_MODEL", "llama3.2:latest"),
            hf_model_id=os.getenv(
                "HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"
            ),
            hf_load_in_4bit=_env_bool("HF_LOAD_IN_4BIT", True),
            hf_max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "512")),
            hf_temperature=float(os.getenv("HF_TEMPERATURE", "0.0")),
        )

    def summary(self) -> str:
        if self.backend == "ollama":
            return f"backend=ollama, model={self.ollama_model}"
        if self.backend == "transformers":
            return (
                f"backend=transformers, model={self.hf_model_id}, "
                f"4-bit={self.hf_load_in_4bit}, "
                f"max_new_tokens={self.hf_max_new_tokens}"
            )
        return f"backend={self.backend} (unsupported)"


class OllamaBackend:
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        import ollama

        response = ollama.generate(model=self.model, prompt=prompt)
        return response["response"].strip()


class TransformersBackend:
    """Lazy-loading Hugging Face Transformers backend for local CUDA inference."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Transformers backend dependencies are missing. Install them with "
                "`pip install -r requirements-transformers.txt`."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Transformers backend requires a CUDA-capable PyTorch installation, "
                "but CUDA is not available."
            )

        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
        if self.config.hf_load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.hf_model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model_id,
            **model_kwargs,
        )

    def generate(self, prompt: str) -> str:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        generation_kwargs = {
            "max_new_tokens": self.config.hf_max_new_tokens,
            "do_sample": self.config.hf_temperature > 0,
        }
        if self.config.hf_temperature > 0:
            generation_kwargs["temperature"] = self.config.hf_temperature

        output = self._model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[-1]
        return self._tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        ).strip()


def create_inference_backend(
    config: InferenceConfig | None = None,
) -> InferenceBackend:
    config = config or InferenceConfig.from_env()
    if config.backend == "ollama":
        return OllamaBackend(config.ollama_model)
    if config.backend == "transformers":
        return TransformersBackend(config)
    raise ValueError(
        f"Unsupported LLM_BACKEND={config.backend!r}; use 'ollama' or 'transformers'."
    )
