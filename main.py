import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformer import NumpyTransformer

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
    global transformer
    if os.path.exists(MODEL_PATH):
        try:
            transformer = NumpyTransformer()
            transformer.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH} (vocab={transformer.vocab_size})")
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
    return {"service": "Rubidium API", "version": "2.0", "engine": "numpy", "status": "running"}


@app.get("/state", response_model=StateResponse)
def get_state():
    global transformer
    if transformer is None or not transformer.is_trained:
        return StateResponse(is_trained=False, vocab_size=0, model_size="none")
    params = sum(p.data.size for p in transformer._all_params())
    return StateResponse(
        is_trained=True,
        vocab_size=transformer.vocab_size,
        model_size=f"{params/1000:.1f}K params"
    )


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
    global transformer
    if transformer is None or not transformer.is_trained:
        raise HTTPException(status_code=400, detail="Model not trained")

    text = transformer.generate(
        seed=req.seed,
        max_chars=req.max_chars,
        temperature=req.temperature,
        top_k=req.top_k
    )
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
    global transformer
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="No saved model found")
    transformer = NumpyTransformer()
    transformer.load(MODEL_PATH)
    return {"status": "loaded", "vocab_size": transformer.vocab_size}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
