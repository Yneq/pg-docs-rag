"""Validate inference backend configuration without downloading a model."""

from pathlib import Path
import sys

# Keep this script runnable directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference import InferenceConfig, create_inference_backend


def main() -> None:
    config = InferenceConfig.from_env()
    create_inference_backend(config)
    print(f"Inference configuration OK: {config.summary()}")


if __name__ == "__main__":
    main()
