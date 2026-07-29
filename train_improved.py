import sys, os, time, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from transformer import NumpyTransformer

corpus_dir = os.path.join(os.path.dirname(__file__), "resources")
texts = []
for fname in sorted(os.listdir(corpus_dir)):
    if fname.endswith(".txt"):
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8", errors="replace") as f:
            texts.append(f.read())

full_text = "\n".join(texts)
corpus_hash = hashlib.md5(full_text.encode()).hexdigest()[:8]
lines = [l.strip() for l in full_text.split("\n") if l.strip()]
print(f"Corpus: {len(lines)} lines, {len(full_text)} chars, hash={corpus_hash}")

# Model config: ~400K params (balance quality/speed)
config = {
    "block_size": 96,      # Context window
    "d_model": 96,         # Hidden dimension
    "n_head": 4,           # Attention heads
    "n_layer": 3,          # Transformer layers
    "d_ff": 384,           # Feedforward dimension
    "max_steps": 4000,     # Training steps
    "learning_rate": 2e-4, # Slightly lower for stability
}

print(f"\nConfig: {config}")

model = NumpyTransformer(
    block_size=config["block_size"],
    d_model=config["d_model"],
    n_head=config["n_head"],
    n_layer=config["n_layer"],
    d_ff=config["d_ff"],
    max_steps=config["max_steps"],
    learning_rate=config["learning_rate"],
)

for line in lines:
    model.train(line)

print("\nTraining...")
t0 = time.time()
model.fit()
t1 = time.time()
print(f"Training took {t1-t0:.1f}s")

if model.is_trained:
    model.save("model.pkl")
    params = sum(p.data.size for p in model._all_params())
    print(f"\nModel saved: {params/1000:.1f}K params, vocab={model.vocab_size}")

    # Test quality
    print("\n--- Quality Tests ---")
    tests = [
        ("Hola", 100),
        ("Buenos dias", 100),
        ("Que puedes hacer", 100),
        ("Quien eres", 100),
        ("Como estas", 100),
    ]
    for seed, chars in tests:
        t2 = time.time()
        r = model.generate(seed, max_chars=chars, temperature=0.8, top_k=20)
        t3 = time.time()
        print(f"[{t3-t2:.2f}s] '{seed}' -> {repr(r[:80])}")
