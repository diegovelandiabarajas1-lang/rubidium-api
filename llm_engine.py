"""
LLM Engine - Unified interface for local LLM inference.
Supports llama.cpp (via llama-cpp-python) and Ollama.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Iterator, Dict, Any, List
import json
import subprocess
import sys


class EngineType(Enum):
    LLAMA_CPP = "llama.cpp"
    OLLAMA = "ollama"
    AUTO = "auto"


@dataclass
class ModelConfig:
    name: str
    model_id: str
    huggingface_repo: str
    filename: str
    quantization: str
    size_gb: float
    context_window: int
    license: str
    architecture: str
    parameters: str
    generation_defaults: Dict[str, Any]
    engine_compatibility: List[str]
    default_engine: str

    @classmethod
    def from_json(cls, path: Path) -> "ModelConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def get_model_path(self, models_dir: Path) -> Path:
        return models_dir / self.filename


class BaseLLMEngine(ABC):
    """Abstract base class for LLM inference engines."""

    def __init__(self, config: ModelConfig, model_path: Path):
        self.config = config
        self.model_path = model_path
        self._model = None

    @abstractmethod
    def load(self) -> bool:
        """Load the model. Returns True on success."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Generate text from prompt."""
        pass

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[str]:
        """Stream tokens from prompt."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine is available (installed and model loaded)."""
        pass

    @abstractmethod
    def unload(self):
        """Unload model from memory."""
        pass

    @property
    @abstractmethod
    def engine_name(self) -> str:
        pass


class LlamaCppEngine(BaseLLMEngine):
    """llama.cpp engine via llama-cpp-python."""

    def __init__(self, config: ModelConfig, model_path: Path, n_ctx: int = 4096, n_gpu_layers: int = -1):
        super().__init__(config, model_path)
        self.n_ctx = min(n_ctx, config.context_window)
        self.n_gpu_layers = n_gpu_layers
        self._llama = None

    def load(self) -> bool:
        try:
            from llama_cpp import Llama
        except ImportError:
            print("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            return False

        if not self.model_path.exists():
            print(f"Model not found: {self.model_path}")
            return False

        try:
            self._llama = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            return True
        except Exception as e:
            print(f"Failed to load model with llama.cpp: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        output = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=False,
        )
        return output["choices"][0]["text"]

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[str]:
        if not self.is_available():
            raise RuntimeError("Model not loaded")

        stream = self._llama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            stream=True,
        )
        for chunk in stream:
            if chunk["choices"][0]["text"]:
                yield chunk["choices"][0]["text"]

    def is_available(self) -> bool:
        return self._llama is not None

    def unload(self):
        self._llama = None

    @property
    def engine_name(self) -> str:
        return "llama.cpp"


class OllamaEngine(BaseLLMEngine):
    """Ollama engine via REST API."""

    def __init__(self, config: ModelConfig, model_path: Path, host: str = "http://localhost:11434"):
        super().__init__(config, model_path)
        self.host = host
        self.model_name = config.model_id
        self._available = False

    def load(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                self._available = any(m["name"].startswith(self.model_name) for m in models)
                if not self._available:
                    print(f"Model {self.model_name} not found in Ollama. Run: ollama pull {self.model_name}")
            return self._available
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        if not self.is_available():
            raise RuntimeError("Ollama model not available")

        import requests
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "stop": stop or [],
            }
        }
        response = requests.post(f"{self.host}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "")

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[str]:
        if not self.is_available():
            raise RuntimeError("Ollama model not available")

        import requests
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "stop": stop or [],
            }
        }
        response = requests.post(f"{self.host}/api/generate", json=payload, stream=True, timeout=120)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done"):
                    break

    def is_available(self) -> bool:
        return self._available

    def unload(self):
        self._available = False

    @property
    def engine_name(self) -> str:
        return "ollama"


class LLMEngine:
    """Unified LLM engine interface with auto-detection."""

    def __init__(self, engine: BaseLLMEngine):
        self.engine = engine

    @classmethod
    def from_config(
        cls,
        config_path: Path,
        engine_type: EngineType = EngineType.AUTO,
        models_dir: Path = None,
        **engine_kwargs
    ) -> "LLMEngine":
        config = ModelConfig.from_json(config_path)
        models_dir = models_dir or config_path.parent
        model_path = config.get_model_path(models_dir)

        if engine_type == EngineType.AUTO:
            engine = cls._auto_detect_engine(config, model_path, **engine_kwargs)
        elif engine_type == EngineType.LLAMA_CPP:
            engine = LlamaCppEngine(config, model_path, **engine_kwargs)
        elif engine_type == EngineType.OLLAMA:
            engine = OllamaEngine(config, model_path, **engine_kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        if not engine.load():
            raise RuntimeError(f"Failed to load model with {engine.engine_name}")

        return cls(engine)

    @staticmethod
    def _auto_detect_engine(config: ModelConfig, model_path: Path, **kwargs) -> BaseLLMEngine:
        if "llama.cpp" in config.engine_compatibility:
            try:
                import llama_cpp
                return LlamaCppEngine(config, model_path, **kwargs)
            except ImportError:
                pass

        if "ollama" in config.engine_compatibility:
            try:
                import requests
                return OllamaEngine(config, model_path)
            except ImportError:
                pass

        raise RuntimeError(
            "No compatible engine found. Install llama-cpp-python or Ollama."
        )

    def generate(self, prompt: str, **kwargs) -> str:
        defaults = self.engine.config.generation_defaults
        params = {**defaults, **kwargs}
        return self.engine.generate(prompt, **params)

    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        defaults = self.engine.config.generation_defaults
        params = {**defaults, **kwargs}
        return self.engine.stream_generate(prompt, **params)

    def is_available(self) -> bool:
        return self.engine.is_available()

    def unload(self):
        self.engine.unload()

    @property
    def engine_name(self) -> str:
        return self.engine.engine_name

    @property
    def model_config(self) -> ModelConfig:
        return self.engine.config


def create_engine(
    model_id: str = "qwen2.5-1.5b-instruct-q4",
    engine_type: EngineType = EngineType.AUTO,
    models_dir: str = "models",
    **kwargs
) -> LLMEngine:
    """Convenience function to create engine from model ID."""
    models_dir = Path(models_dir)
    config_path = models_dir / f"{model_id}.json"
    return LLMEngine.from_config(config_path, engine_type, models_dir, **kwargs)


def list_available_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """List all available model configurations."""
    models_dir = Path(models_dir)
    models = []
    for config_file in models_dir.glob("*.json"):
        try:
            config = ModelConfig.from_json(config_file)
            model_path = config.get_model_path(models_dir)
            models.append({
                "model_id": config.model_id,
                "name": config.name,
                "size_gb": config.size_gb,
                "quantization": config.quantization,
                "context_window": config.context_window,
                "license": config.license,
                "downloaded": model_path.exists(),
                "path": str(model_path) if model_path.exists() else None,
            })
        except Exception:
            pass
    return models


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM Engine")
    parser.add_argument("--model", default="qwen2.5-1.5b-instruct-q4")
    parser.add_argument("--engine", choices=["auto", "llama.cpp", "ollama"], default="auto")
    parser.add_argument("--prompt", default="Hola, ¿cómo estás?")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    engine_type = EngineType(args.engine.replace(".", "_").upper())
    engine = create_engine(args.model, engine_type)

    print(f"Engine: {engine.engine_name}")
    print(f"Model: {engine.model_config.name}")

    if args.stream:
        print("Response: ", end="", flush=True)
        for token in engine.stream_generate(args.prompt):
            print(token, end="", flush=True)
        print()
    else:
        response = engine.generate(args.prompt)
        print(f"Response: {response}")