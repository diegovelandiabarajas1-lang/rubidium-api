import os
import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformer import NumpyTransformer

try:
    import rubidium_core
    HAS_RUST = True
    print("Rubidium Core (Rust) loaded - inference acceleration available")
except ImportError:
    HAS_RUST = False
    print("Rubidium Core not available - using Python inference")

MODEL_PATH = "model.pkl"

app = FastAPI(title="Rubidium API - Transformer Generator", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transformer: NumpyTransformer = None
rust_model = None  # rubidium_core.RubidiumModel for fast inference

# Response cache: key -> (text, timestamp)
_response_cache: dict = {}
_CACHE_MAX = 200


class GenerateRequest(BaseModel):
    seed: str = ""
    max_chars: int = 200
    temperature: float = 0.8
    top_k: int = 20


class GenerateResponse(BaseModel):
    text: str


class TrainRequest(BaseModel):
    corpus: str = ""
    block_size: int = 128
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 4
    d_ff: int = 512
    max_steps: int = 1000
    learning_rate: float = 3e-4
    use_resources: bool = True


class StateResponse(BaseModel):
    is_trained: bool
    vocab_size: int
    model_size: str


def load_corpus_from_resources() -> str:
    texts = []
    if os.path.isdir("resources"):
        for fname in sorted(os.listdir("resources")):
            if fname.endswith(".txt"):
                path = os.path.join("resources", fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                except Exception:
                    pass
    return "\n".join(texts)


@app.on_event("startup")
def startup():
    global transformer, rust_model
    if os.path.exists(MODEL_PATH):
        try:
            transformer = NumpyTransformer()
            transformer.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH} (vocab={transformer.vocab_size})")

            # Try loading into Rust for fast inference
            if HAS_RUST:
                try:
                    rust_model = rubidium_core.RubidiumModel()
                    rust_model.load_from_pickle(os.path.abspath(MODEL_PATH))
                    print("Rust inference engine loaded successfully")
                except Exception as e:
                    print(f"Rust model load failed, using Python: {e}")
                    rust_model = None
            return
        except Exception as e:
            print(f"Could not load model: {e}")

    corpus = load_corpus_from_resources()
    if corpus.strip():
        print("No saved model found. Auto-training from resources...")
        transformer = NumpyTransformer(
            block_size=128, d_model=128, n_head=4, n_layer=4, d_ff=512,
            max_steps=500, learning_rate=3e-4
        )
        lines = [l.strip() for l in corpus.split("\n") if l.strip()]
        for line in lines:
            transformer.train(line)
        transformer.fit()
        if transformer.is_trained:
            transformer.save(MODEL_PATH)
            print("Auto-training complete and model saved.")


@app.get("/")
def root():
    engine = "rust" if rust_model is not None else "numpy"
    return {"service": "Rubidium API", "version": "2.0", "engine": engine, "status": "running"}


@app.get("/state")
def get_state():
    global transformer, rust_model
    if transformer is None or not transformer.is_trained:
        return {"is_trained": False, "vocab_size": 0, "model_size": "none"}
    params = sum(p.data.size for p in transformer._all_params())
    return {
        "is_trained": True,
        "vocab_size": transformer.vocab_size,
        "model_size": f"{params/1000:.1f}K params",
        "engine": "rust" if rust_model is not None else "numpy",
        "cache_size": len(_response_cache)
    }


@app.post("/train")
def train(req: TrainRequest):
    global transformer
    transformer = NumpyTransformer(
        block_size=req.block_size,
        d_model=req.d_model,
        n_head=req.n_head,
        n_layer=req.n_layer,
        d_ff=req.d_ff,
        max_steps=req.max_steps,
        learning_rate=req.learning_rate,
    )

    corpus_text = req.corpus if not req.use_resources else load_corpus_from_resources()
    if not corpus_text.strip():
        corpus_text = req.corpus

    lines = [l.strip() for l in corpus_text.split("\n") if l.strip()]
    for line in lines:
        transformer.train(line)

    transformer.fit()

    if transformer.is_trained:
        transformer.save(MODEL_PATH)
        return {"status": "success", "message": "Model trained and saved"}

    return {"status": "error", "message": "Training failed"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    global transformer, rust_model
    if transformer is None or not transformer.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")

    # Check cache
    cache_key = hashlib.md5(f"{req.seed}|{req.max_chars}|{req.temperature}|{req.top_k}".encode()).hexdigest()
    if cache_key in _response_cache:
        return GenerateResponse(text=_response_cache[cache_key])

    # Use Rust inference if available (much faster)
    if rust_model is not None:
        try:
            text = rust_model.generate(
                seed=req.seed,
                max_chars=req.max_chars,
                temperature=req.temperature,
                top_k=req.top_k
            )
        except Exception as e:
            # Fall back to Python
            text = transformer.generate(
                seed=req.seed,
                max_chars=req.max_chars,
                temperature=req.temperature,
                top_k=req.top_k
            )
    else:
        text = transformer.generate(
            seed=req.seed,
            max_chars=req.max_chars,
            temperature=req.temperature,
            top_k=req.top_k
        )

    # Store in cache (evict oldest if full)
    if len(_response_cache) >= _CACHE_MAX:
        _response_cache.pop(next(iter(_response_cache)))
    _response_cache[cache_key] = text

    return GenerateResponse(text=text)


@app.post("/save")
def save():
    global transformer
    if transformer is None:
        raise HTTPException(status_code=400, detail="No model to save")
    transformer.save(MODEL_PATH)
    return {"status": "saved", "path": MODEL_PATH}


@app.post("/load")
def load():
    global transformer, rust_model
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="No saved model found")
    transformer = NumpyTransformer()
    transformer.load(MODEL_PATH)
    _response_cache.clear()

    # Try loading into Rust
    if HAS_RUST:
        try:
            rust_model = rubidium_core.RubidiumModel()
            rust_model.load_from_pickle(os.path.abspath(MODEL_PATH))
        except Exception:
            rust_model = None

    return {"status": "loaded", "vocab_size": transformer.vocab_size}


@app.post("/clear-cache")
def clear_cache():
    count = len(_response_cache)
    _response_cache.clear()
    return {"status": "cleared", "entries_removed": count}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
