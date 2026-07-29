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
lines = [l.strip() for l in full_text.split("\n") if l.strip()]
print(f"Corpus: {len(lines)} lines, {len(full_text)} chars")

# Bigger model, more steps
config = {
    "block_size": 128,
    "d_model": 128,
    "n_head": 4,
    "n_layer": 4,
    "d_ff": 512,
    "max_steps": 8000,
    "learning_rate": 3e-4,
}

print(f"Config: {config}")
print(f"Estimated params: ~840K")

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

print("\nTraining 8000 steps...")
t0 = time.time()
model.fit()
t1 = time.time()
print(f"Training took {t1-t0:.1f}s")

if model.is_trained:
    model.save("model.pkl")
    params = sum(p.data.size for p in model._all_params())
    print(f"Model saved: {params/1000:.1f}K params, vocab={model.vocab_size}")

    # Quick quality test
    tests = ['Hola', 'Buenos', 'Quien']
    results = []
    for seed in tests:
        t2 = time.time()
        r = model.generate(seed, max_chars=60, temperature=0.8, top_k=20)
        t3 = time.time()
        results.append(f"[{t3-t2:.2f}s] {seed} -> {r}")
    with open('quality_test.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))
    print("Quality tests written to quality_test.txt")
