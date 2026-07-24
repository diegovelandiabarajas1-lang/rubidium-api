import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from transformer import NumpyTransformer

corpus_dir = os.path.join(os.path.dirname(__file__), "resources")
texts = []
for fname in sorted(os.listdir(corpus_dir)):
    if fname.endswith(".txt"):
        with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
            texts.append(f.read())

lines = [l.strip() for l in "\n".join(texts).split("\n") if l.strip()]
print(f"Training {len(lines)} lines, 2000 steps...")

model = NumpyTransformer(
    block_size=128, d_model=128, n_head=4, n_layer=4, d_ff=512,
    max_steps=2000, learning_rate=3e-4
)
for line in lines:
    model.train(line)
model.fit()

if model.is_trained:
    model.save("model.pkl")
    print("Saved model.pkl")
    import time
    t0 = time.time()
    r = model.generate("Hola", max_chars=50, temperature=0.8, top_k=20)
    t1 = time.time()
    print(f"Generate 50 chars in {t1-t0:.1f}s: {r}")
