""" Push train_pytorch.py (100K steps) to existing Kaggle notebook using api """
import os, json, subprocess, shutil, tempfile, time

NOTEBOOK_TITLE = "rubidium-pytorch-train"
KAGGLE_USER = "diegovelandiabarajas"
FULL_NAME = f"{KAGGLE_USER}/{NOTEBOOK_TITLE}"

# Verify train_pytorch.py is updated
with open("train_pytorch.py", "r") as f:
    src = f.read()
assert "max_steps = 100000" in src, "Not updated to 100K!"
assert "warmup = 4000" in src, "warmup not updated!"

# Create push folder
tmpdir = tempfile.mkdtemp()
push_dir = os.path.join(tmpdir, "push")
os.makedirs(push_dir)

# Write kernel-metadata.json
metadata = {
    "id": FULL_NAME,
    "title": NOTEBOOK_TITLE,
    "code_file": "train_pytorch.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": False,
    "dataset_sources": ["diegovelandiabarajas/rubidium-corpus-train"],
    "accelerator": "GPU",
}
with open(os.path.join(push_dir, "kernel-metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# Copy train_pytorch.py
shutil.copy2("train_pytorch.py", os.path.join(push_dir, "train_pytorch.py"))

print(f"Pushing to Kaggle: {FULL_NAME}")
result = subprocess.run(
    ["kaggle", "kernels", "push", "-p", push_dir, "-t", "3600"],
    capture_output=True, text=True, timeout=120
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr)

shutil.rmtree(tmpdir)

# Monitor
for i in range(60):
    result = subprocess.run(
        ["kaggle", "kernels", "status", FULL_NAME],
        capture_output=True, text=True, timeout=30
    )
    s = result.stdout.strip()
    print(f"[{i*10}s] Status: {s}")
    if "complete" in s.lower() or "error" in s.lower():
        break
    if "running" in s.lower():
        print("Training running... waiting for completion")
    time.sleep(10)

print(f"\nCheck: https://www.kaggle.com/code/{FULL_NAME}")
print("After completion, download model with:")
print(f"  kaggle kernels output {FULL_NAME} -w")
print(f"  move .\\model_10m_100k.pkl to rubidium-api\\")
