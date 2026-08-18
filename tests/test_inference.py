import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.inference import (
    GenerationResult,
    InferenceConfig,
    OllamaBackend,
    TransformersBackend,
    create_inference_backend,
)


class InferenceConfigTests(unittest.TestCase):
    def test_ollama_is_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = InferenceConfig.from_env()

        self.assertEqual(config.backend, "ollama")
        self.assertEqual(config.ollama_model, "llama3.2:latest")
        self.assertIsInstance(create_inference_backend(config), OllamaBackend)

    def test_transformers_configuration_does_not_load_heavy_dependencies(self):
        environment = {
            "LLM_BACKEND": "transformers",
            "HF_MODEL_ID": "example/model",
            "HF_LOAD_IN_4BIT": "false",
            "HF_MAX_NEW_TOKENS": "128",
        }
        with patch.dict(os.environ, environment, clear=True):
            backend = create_inference_backend()

        self.assertIsInstance(backend, TransformersBackend)
        self.assertEqual(backend.config.hf_model_id, "example/model")
        self.assertFalse(backend.config.hf_load_in_4bit)
        self.assertEqual(backend.config.hf_max_new_tokens, 128)
        self.assertIsNone(backend._model)
        self.assertIsNone(backend._tokenizer)

    def test_unknown_backend_is_rejected(self):
        config = InferenceConfig(backend="unknown")
        with self.assertRaisesRegex(ValueError, "Unsupported LLM_BACKEND"):
            create_inference_backend(config)

    def test_ollama_returns_runtime_token_counts(self):
        fake_ollama = SimpleNamespace(
            generate=lambda **kwargs: {
                "response": " generated answer ",
                "eval_count": 42,
                "prompt_eval_count": 17,
            }
        )
        with patch.dict("sys.modules", {"ollama": fake_ollama}):
            result = OllamaBackend("example").generate_with_metrics("prompt")

        self.assertEqual(
            result,
            GenerationResult(
                text="generated answer",
                generated_tokens=42,
                prompt_tokens=17,
            ),
        )


if __name__ == "__main__":
    unittest.main()
