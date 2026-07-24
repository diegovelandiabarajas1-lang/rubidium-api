import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from transformer import PyTorchTransformer

MODEL_PATH = "model.pth"

app = FastAPI(title="Rubidium API - Transformer Generator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

transformer: PyTorchTransformer = None


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
    d_model: int = 512
    n_head: int = 8
    n_layer: int = 8
    d_ff: int = 2048
    max_steps: int = 6000
    learning_rate: float = 1e-4
    tokenizer_type: str = "BPE"
    tokenizer_vocab_size: int = 4096
    use_resources: bool = True


class StateResponse(BaseModel):
    is_trained: bool
    vocab_size: int
    device: str


def load_corpus_from_resources() -> str:
    texts = []
    if os.path.isdir("resources"):
        for fname in os.listdir("resources"):
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
            transformer = PyTorchTransformer()
            transformer.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH} on {transformer.device}")
            return
        except Exception as e:
            print(f"Could not load model: {e}")

    corpus = load_corpus_from_resources()
    if corpus.strip():
        print("No saved model found. Auto-training from resources...")
        from tokenizers import TokenizerFactory, TokenizerConfig, TokenizerType
        transformer = PyTorchTransformer(max_steps=6000)
        tokenizer = TokenizerFactory.create(TokenizerConfig(type=TokenizerType.BPE, vocab_size=4096, add_special_tokens=True))
        transformer.tokenizer = tokenizer
        lines = [l.strip() for l in corpus.split("\n") if l.strip()]
        for line in lines:
            transformer.train(line)
        tokenizer.train(lines, 4096)
        transformer.fit()
        if transformer.is_trained:
            transformer.save(MODEL_PATH)
            print("Auto-training complete and model saved.")


@app.get("/")
def root():
    return {"service": "Rubidium API", "status": "running"}


@app.get("/state", response_model=StateResponse)
def get_state():
    global transformer
    if transformer is None or not transformer.is_trained:
        return StateResponse(is_trained=False, vocab_size=0, device="cpu")
    return StateResponse(
        is_trained=True,
        vocab_size=transformer.vocab_size,
        device=str(transformer.device)
    )


@app.post("/train")
def train(req: TrainRequest):
    global transformer
    from tokenizers import TokenizerFactory, TokenizerConfig, TokenizerType

    tokenizer_type_map = {
        "BPE": TokenizerType.BPE,
        "Unigram": TokenizerType.Unigram,
        "WordPiece": TokenizerType.WordPiece,
        "SentencePieceBPE": TokenizerType.SentencePieceBPE,
        "SentencePieceUnigram": TokenizerType.SentencePieceUnigram,
        "DynamicVocabulary": TokenizerType.DynamicVocabulary,
    }

    transformer = PyTorchTransformer(
        block_size=req.block_size,
        d_model=req.d_model,
        n_head=req.n_head,
        n_layer=req.n_layer,
        d_ff=req.d_ff,
        max_steps=req.max_steps,
        learning_rate=req.learning_rate,
    )

    tokenizer_config = TokenizerConfig(
        type=tokenizer_type_map.get(req.tokenizer_type, TokenizerType.BPE),
        vocab_size=req.tokenizer_vocab_size,
        add_special_tokens=True,
    )
    from tokenizers import TokenizerFactory
    tokenizer = TokenizerFactory.create(tokenizer_config)
    transformer.tokenizer = tokenizer

    corpus_text = req.corpus if not req.use_resources else load_corpus_from_resources()
    if not corpus_text.strip():
        corpus_text = req.corpus

    lines = [l.strip() for l in corpus_text.split("\n") if l.strip()]
    for line in lines:
        transformer.train(line)

    tokenizer.train(lines, req.tokenizer_vocab_size)
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
    transformer = PyTorchTransformer()
    transformer.load(MODEL_PATH)
    return {"status": "loaded", "vocab_size": transformer.vocab_size}


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)