#!/usr/bin/env python3
# ============================================================
# RUBIDIUM 500M - CUDA/C++ Training Engine
# Kaggle GPU: Tesla P100 16GB
# ============================================================
import subprocess, os, sys, time

print("=" * 60)
print("RUBIDIUM 500M - CUDA/C++ Training Engine")
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
    check=True, timeout=300
)
print("Build complete!")

# 6. Copy corpus
print("\n--- Copying corpus ---")
corpus_src = "/kaggle/input/datasets/diegovelandiabarajas/rubidium-corpus-train/resources"
corpus_dst = "/kaggle/working/rubidium-train/data"
os.makedirs(corpus_dst, exist_ok=True)
for f in os.listdir(corpus_src):
    if f.endswith(".txt"):
        subprocess.run(["cp", f"{corpus_src}/{f}", f"{corpus_dst}/{f}"])
        print(f"  Copied: {f}")

# 7. Run training
print("\n--- Starting 500M training ---")
print("Config: d=2048, h=32, l=10, ff=8192, ~500M params")
print("Target: 200K steps")

result = subprocess.run(
    ["./rubidium-train", corpus_dst],
    cwd="/kaggle/working/rubidium-train/build",
    timeout=36000  # 10 hours max
)

# 8. Check output
print("\n--- Training complete ---")
if os.path.exists("/kaggle/working/rubidium-train/build/model_final.bin"):
    size_mb = os.path.getsize("/kaggle/working/rubidium-train/build/model_final.bin") / 1e6
    print(f"Model saved: model_final.bin ({size_mb:.1f} MB)")
else:
    print("Warning: model_final.bin not found")

# 9. Convert to pickle for Rust inference
print("\n--- Converting to pickle ---")
subprocess.run([
    sys.executable, "/kaggle/working/rubidium-train/convert_to_pickle.py",
    "/kaggle/working/rubidium-train/build/model_final.bin",
    "/kaggle/working/model_500m.pkl"
])

if os.path.exists("/kaggle/working/model_500m.pkl"):
    size_mb = os.path.getsize("/kaggle/working/model_500m.pkl") / 1e6
    print(f"Pickle saved: model_500m.pkl ({size_mb:.1f} MB)")
else:
    print("Warning: model_500m.pkl not found - use model_final.bin")

print("\nDone! Download model_500m.pkl from Kaggle outputs.")
