# Rubidium API - Local Models

This directory contains configuration files and tools for running local LLMs with Rubidium API
`qwen2.5-1.5b-instruct-q4.json` - Qwen2.5-1.5B-Instruct Q4_K_M (1.1 GB, Apache 2.0)

## Quick Start

```bash
# Download the recommended model (Qwen2.5-1.5B-Instruct Q4_K_M, 1.1 GB)
python models/download_model.py

# Or with custom model config
python models/download_model.py --model qwen2.5-1.5b-instruct-q4
```

## Available Models

| Model | Size | Quantization | Context | License | Best For |
|-------|------|--------------|---------|---------|----------|
| Qwen2.5-1.5B-Instruct-Q4_K_M | 1.1 GB | Q4_K_M | 128K | Apache 2.0 | Spanish, chat, code, reasoning |

## Downloading Models

```bash
# List available model configs
python models/download_model.py --list

# Download default model
python models/download_model.py

# Force re-download
python models/download_model.py --force

# Custom output directory
python models/download_model.py --output-dir ./my_models
```

## Model Configuration

Each model has a JSON config file with:
- `name`: Human-readable name
- `model_id`: Internal identifier
- `huggingface_repo`: HuggingFace repository
- `filename`: GGUF filename
- `quantization`: Quantization method
- `size_gb`: Approximate size in GB
- `context_window`: Context window in tokens
- `license`: License type
- `generation_defaults`: Default generation parameters
- `engine_compatibility`: Supported inference engines

## Inference Engines

### llama.cpp (Recommended)
```bash
# Install
pip install llama-cpp-python

# Or with GPU support (CUDA)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Ollama
```bash
# Install Ollama
# https://ollama.ai

# Pull model
ollama pull qwen2.5:1.5b-instruct-q4_k_m
```

## Integration

The `llm_engine.py` module provides a unified interface:

```python
from llm_engine import LLMEngine, EngineType

# Auto-detect best available engine
engine = LLMEngine.from_config("models/qwen2.5-1.5b-instruct-q4.json")

# Or specify engine
engine = LLMEngine.from_config(
    "models/qwen2.5-1.5b-instruct-q4.json",
    engine_type=EngineType.LLAMA_CPP
)

# Generate
response = engine.generate("Hola, ¿cómo estás?", max_tokens=200)
print(response)

# Streaming
for token in engine.stream_generate("Escribe un poema"):
    print(token, end="", flush=True)
```

## Model Details: Qwen2.5-1.5B-Instruct-Q4_K_M

- **Architecture**: Qwen2 (transformer)
- **Parameters**: 1.5B
- **Quantization**: Q4_K_M (4-bit, balanced quality/size)
- **Context Window**: 131,072 tokens (128K)
- **Size**: ~1.1 GB
- **License**: Apache 2.0
- **Languages**: Excellent Spanish, English, Chinese, coding
- **HuggingFace**: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- **File**: `qwen2.5-1.5b-instruct-q4_k_m.gguf`

## Hardware Requirements

| Engine | CPU RAM | GPU VRAM | Notes |
|--------|---------|----------|-------|
| llama.cpp (CPU) | 2 GB | - | Runs on any modern CPU |
| llama.cpp (CUDA) | 1 GB | 2 GB | Requires CUDA 11.8+ |
| llama.cpp (Metal) | 1 GB | 2 GB | Apple Silicon |
| Ollama | 2 GB | 2 GB | Includes overhead |

## Usage in Rubidium

### FastAPI (main.py)
```python
# Dual engine: local transformer + local LLM
from llm_engine import LLMEngine

# Initialize both engines
local_transformer = NumpyTransformer()  # Existing
local_llm = LLMEngine.from_config("models/qwen2.5-1.5b-instruct-q4.json")

@app.post("/generate")
def generate(req: GenerateRequest):
    # Use local LLM for better quality
    if req.engine == "llm" and local_llm.is_available():
        return local_llm.generate(req.seed, req.max_chars)
    # Fall back to local transformer
    return local_transformer.generate(...)
```

### Gradio (app.py)
```python
from llm_engine import LLMEngine, EngineType

engine = LLMEngine.from_config("models/qwen2.5-1.5b-instruct-q4.json")

def chat(message, history):
    return engine.generate(message)
```

## Verification

After download, verify the model:

```bash
# Check file exists and size
ls -lh models/qwen2.5-1.5b-instruct-q4_k_m.gguf

# Test with llama.cpp
python -c "
from llama_cpp import Llama
llm = Llama(model_path='models/qwen2.5-1.5b-instruct-q4_k_m.gguf', n_ctx=4096)
print(llm('Hola, ¿qué tal?', max_tokens=50))
"
```

## License

All models use their original licenses. Qwen2.5 is Apache 2.0 - permissive for commercial use.

## Troubleshooting

**Download fails:**
- Check internet connection
- Try with `--force` flag
- Check HuggingFace rate limits

**Model not loading:**
- Ensure `llama-cpp-python` is installed
- Check GGUF file integrity (SHA256)
- Verify sufficient RAM/VRAM

**Out of memory:**
- Reduce `n_ctx` (context window)
- Use smaller quantization (Q3_K_M)
- Close other applications

## Adding New Models

1. Create JSON config in `models/`
2. Add to `download_model.py --list`
3. Test with `llm_engine.py`

Example config structure:
```json
{
  "name": "Model Name",
  "model_id": "model-id",
  "huggingface_repo": "org/repo",
  "filename": "model.gguf",
  "quantization": "Q4_K_M",
  "size_gb": 1.5,
  "context_window": 8192,
  "license": "MIT",
  "engine_compatibility": ["llama.cpp"],
  "generation_defaults": {...}
}
```