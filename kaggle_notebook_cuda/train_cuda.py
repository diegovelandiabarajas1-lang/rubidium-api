#!/usr/bin/env python3
# ============================================================
# RUBIDIUM 250M - CUDA/C++ Training Engine
# Kaggle GPU: Tesla P100 16GB
# Config: V=32000, T=512, D=1536, H=24, L=10, FF=6144
# ============================================================
import subprocess, os, sys, time

print("=" * 60)
print("RUBIDIUM 250M - CUDA/C++ Training Engine")
print("=" * 60)

# 1. GPU info
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=10
)
print(f"GPU: {result.stdout.strip()}")

# 2. CUDA toolkit
result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=10)
print(f"CUDA: {result.stdout.strip()[:100]}")

# 3. cuDNN
try:
    import torch
    print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
    del torch
except:
    print("cuDNN: checking manually...")
    result = subprocess.run(["find", "/usr", "-name", "libcudnn*"], capture_output=True, text=True, timeout=10)
    print(f"cuDNN libraries: {result.stdout[:200]}")

# 4. Clone rubidium-train
print("\n--- Cloning rubidium-train ---")
if os.path.exists("/kaggle/working/rubidium-train"):
    subprocess.run(["rm", "-rf", "/kaggle/working/rubidium-train"])
subprocess.run([
    "git", "clone", "--depth=1",
    "https://github.com/diegovelandiabarajas1-lang/rubidium-train.git",
    "/kaggle/working/rubidium-train"
], check=True, timeout=60)
print("Cloned!")

# 5. Build
print("\n--- Building CUDA training engine ---")
os.makedirs("/kaggle/working/rubidium-train/build", exist_ok=True)
subprocess.run(
    ["cmake", ".."],
    cwd="/kaggle/working/rubidium-train/build",
    check=True, timeout=120
)
subprocess.run(
    ["make", "-j4"],
    cwd="/kaggle/working/rubidium-train/build",
    check=True, timeout=600
)
print("Build complete!")

# 6. Generate expanded corpus
print("\n--- Generating expanded corpus ---")
corpus_gen = "/kaggle/working/rubidium-train/src/generate_corpus_expanded.py"
if os.path.exists(corpus_gen):
    result = subprocess.run(
        [sys.executable, corpus_gen],
        cwd="/kaggle/working/rubidium-train/src",
        capture_output=True, text=True, timeout=300
    )
    print(result.stdout[-500:] if result.stdout else "No output")
    if result.returncode != 0:
        print(f"Warning: corpus generation failed: {result.stderr[:200]}")

# 7. Copy corpus to data directory
print("\n--- Copying corpus ---")
corpus_dst = "/kaggle/working/rubidium-train/data"
os.makedirs(corpus_dst, exist_ok=True)

# Copy from Kaggle datasets if available
corpus_src = "/kaggle/input/datasets/diegovelandiabarajas/rubidium-corpus-train/resources"
if os.path.exists(corpus_src):
    for f in os.listdir(corpus_src):
        if f.endswith(".txt"):
            subprocess.run(["cp", f"{corpus_src}/{f}", f"{corpus_dst}/{f}"])
            print(f"  Copied dataset: {f}")

# Copy expanded corpus if generated
expanded_jsonl = "/kaggle/working/rubidium-train/src/corpus_expanded.jsonl"
expanded_txt = "/kaggle/working/rubidium-train/src/corpus_expanded.txt"
if os.path.exists(expanded_txt):
    subprocess.run(["cp", expanded_txt, f"{corpus_dst}/corpus_expanded.txt"])
    print(f"  Copied expanded corpus: {os.path.getsize(expanded_txt)/1024/1024:.1f} MB")
if os.path.exists(expanded_jsonl):
    subprocess.run(["cp", expanded_jsonl, f"{corpus_dst}/corpus_expanded.jsonl"])

# Count corpus
txt_files = [f for f in os.listdir(corpus_dst) if f.endswith(".txt")]
total_chars = sum(os.path.getsize(f"{corpus_dst}/{f}") for f in txt_files)
print(f"  Corpus: {len(txt_files)} files, {total_chars/1024/1024:.1f} MB total")

# 8. Run training
print("\n--- Starting 250M training ---")
print("Config: V=32000, T=512, D=1536, H=24, L=10, FF=6144")
print("Target: 200K steps, BS=2, GA=16, Eff=32")

result = subprocess.run(
    ["./rubidium-train", corpus_dst],
    cwd="/kaggle/working/rubidium-train/build",
    timeout=36000  # 10 hours max
)

# 9. Check output
print("\n--- Training complete ---")
if os.path.exists("/kaggle/working/rubidium-train/build/model_final.bin"):
    size_mb = os.path.getsize("/kaggle/working/rubidium-train/build/model_final.bin") / 1e6
    print(f"Model saved: model_final.bin ({size_mb:.1f} MB)")
else:
    print("Warning: model_final.bin not found")

# List checkpoints
ckpt_dir = "/kaggle/working/rubidium-train/build/checkpoints"
if os.path.exists(ckpt_dir):
    ckpts = sorted(os.listdir(ckpt_dir))
    print(f"Checkpoints: {len(ckpts)}")
    for c in ckpts[-5:]:
        size_mb = os.path.getsize(f"{ckpt_dir}/{c}") / 1e6
        print(f"  {c} ({size_mb:.1f} MB)")

print("\nDone! Download model_final.bin from Kaggle outputs.")
