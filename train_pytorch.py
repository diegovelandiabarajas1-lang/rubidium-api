# ============================================================
# RUBIDIUM - CUDA/C++ con PyTorch (cuBLAS + cuDNN + CUTLASS)
# ============================================================
import subprocess, sys, os, json

# 1. Check GPU type
gpu_info = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
    capture_output=True, text=True, timeout=10
).stdout.strip()
print(f"GPU detectada: {gpu_info}", flush=True)

# 2. Install compatible PyTorch if needed
try:
    import torch
    torch.zeros(1, device='cuda')
    gpu_works = True
    print(f"GPU funciona: {torch.cuda.get_device_name(0)}")
except Exception as e:
    gpu_works = False
    print(f"GPU no compatible: {e}")

if not gpu_works:
    print("Instalando PyTorch 2.4.0+cu118 (soporta sm_60)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--force-reinstall", "-q",
        "torch==2.4.0+cu118",
        "--extra-index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True, timeout=600)
    print("PyTorch instalado!", flush=True)
    result = subprocess.run([sys.executable, "-u"] + sys.argv)
    sys.exit(result.returncode)

# ============================================================
# TRAINING - runs in child process with compatible PyTorch
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle, time, math, numpy as np

print(f"\nPyTorch {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"cuDNN: {torch.backends.cudnn.enabled}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


class CausalSelfAttention(nn.Module):
    def __init__(self, D, H, T):
        super().__init__()
        self.nh = H; self.hd = D // H; self.T = T
        self.wq = nn.Linear(D, D, bias=True)
        self.wk = nn.Linear(D, D, bias=True)
        self.wv = nn.Linear(D, D, bias=True)
        self.wo = nn.Linear(D, D, bias=True)

    def forward(self, x):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, L, self.nh, self.hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                              scale=1.0/math.sqrt(self.hd))
        return self.wo(out.transpose(1, 2).contiguous().view(B, L, D))


class Block(nn.Module):
    def __init__(self, D, H, T, FF):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(D, H, T)
        self.ln2 = nn.LayerNorm(D)
        self.w1 = nn.Linear(D, FF, bias=True)
        self.w2 = nn.Linear(FF, D, bias=True)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.w2(F.relu(self.w1(self.ln2(x))))
        return x


class Rubidium(nn.Module):
    def __init__(self, V, T, D, H, L, FF):
        super().__init__()
        self.T = T
        self.te = nn.Embedding(V, D)
        self.pe = nn.Embedding(T, D)
        self.layers = nn.ModuleList([Block(D, H, T, FF) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=True)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        B, L = idx.shape
        h = self.te(idx) + self.pe(torch.arange(L, device=idx.device))
        for layer in self.layers: h = layer(h)
        logits = self.head(self.lnf(h))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss
    def generate(self, idx, n, temp=0.7, topk=40):
        for _ in range(n):
            logits, _ = self(idx[:, -self.T:])
            p = F.softmax(logits[:, -1] / temp, -1)
            if topk > 0:
                v, _ = torch.topk(p, min(topk, p.size(-1)))
                p[p < v[:, [-1]]] = 0
            idx = torch.cat([idx, torch.multinomial(p, 1)], 1)
        return idx


def load_corpus():
    import glob
    base = '/kaggle/input/datasets/diegovelandiabarajas/rubidium-corpus-train/resources'
    texts = [open(f, 'r', encoding='utf-8', errors='replace').read()
             for f in sorted(glob.glob(f'{base}/*.txt'))]
    full = '\n'.join(texts)
    print(f'Corpus: {len(full)} chars')
    return full


def train():
    print("\n" + "=" * 60)
    print("RUBIDIUM 10M - CUDA/C++ Training")
    print("cuBLAS GEMM + cuDNN FlashAttention + CUTLASS + CUB")
    print("=" * 60)

    device = torch.device('cuda')
    V = 256; T = 256; D = 2048; H = 32; L = 10; FF = 8192

    full_text = load_corpus()
    chars = sorted(set(full_text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    V = len(chars)
    print(f"Vocab: {V}")

    data = torch.tensor([c2i.get(c, 0) for c in full_text], dtype=torch.long)
    n = len(data)
    print(f"Tokens: {n}")

    model = Rubidium(V, T, D, H, L, FF).to(device)
    tp = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {tp/1e6:.1f}M")

    # torch.compile for speed
    cc = torch.cuda.get_device_capability(0)
    if cc[0] >= 7:
        print("Compilando con torch.compile (CUDA graphs)...")
        model = torch.compile(model, mode='reduce-overhead')
    else:
        print(f"sm_{cc[0]}{cc[1]}: torch.compile no disponible, modo eager")

    BS = 2; GA = 16; max_steps = 200000
    lr = 3e-4; warmup = 4000; gc = 1.0
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.1)

    def get_lr(s):
        if s < warmup: return lr * s / warmup
        p = (s - warmup) / (max_steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    print(f"Training: {max_steps} steps, BS={BS}, GA={GA}, Eff={BS*GA}")
    print("-" * 60)

    model.train()
    sl = float('inf')
    t0 = time.time()

    for step in range(1, max_steps + 1):
        lr_t = get_lr(step // GA)
        for pg in optim.param_groups: pg['lr'] = lr_t

        ix = torch.randint(0, n - T - 1, (BS,))
        x = torch.stack([data[i:i+T] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)

        _, loss = model(x, y)
        (loss / GA).backward()

        if step % GA == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gc)
            optim.step()
            optim.zero_grad(set_to_none=True)

        lv = loss.item()
        sl = lv if step == 1 else 0.98 * sl + 0.02 * lv

        if step % 100 == 0 or step == max_steps:
            elapsed = time.time() - t0
            sps = step / elapsed
            eta = (max_steps - step) / sps / 60
            print(f"Step {step}/{max_steps} | loss: {sl:.4f} | lr: {lr_t:.2e} | {sps:.1f} steps/s | ETA: {eta:.0f}min")

    print(f"\nTraining: {time.time()-t0:.0f}s ({max_steps/(time.time()-t0):.1f} steps/s)")

    # Save
    model.eval()
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    state = {
        "vocab_size": V, "block_size": T, "d_model": D, "n_head": H,
        "n_layer": L, "d_ff": FF,
        "char_to_id": c2i, "id_to_char": i2c,
        "token_emb": m.te.weight.data.cpu().float().numpy(),
        "pos_emb": m.pe.weight.data.cpu().float().numpy().reshape(1, T, D),
        "ln_f_w": m.lnf.weight.data.cpu().float().numpy(),
        "ln_f_b": m.lnf.bias.data.cpu().float().numpy(),
        "lm_w": m.head.weight.data.cpu().float().numpy(),
        "lm_b": m.head.bias.data.cpu().float().numpy(),
        "layers": [],
    }
    for layer in m.layers:
        a = layer.attn
        state["layers"].append({
            "ln1_w": layer.ln1.weight.data.cpu().float().numpy(),
            "ln1_b": layer.ln1.bias.data.cpu().float().numpy(),
            "attn_wq_w": a.wq.weight.data.cpu().float().numpy(),
            "attn_wq_b": a.wq.bias.data.cpu().float().numpy(),
            "attn_wk_w": a.wk.weight.data.cpu().float().numpy(),
            "attn_wk_b": a.wk.bias.data.cpu().float().numpy(),
            "attn_wv_w": a.wv.weight.data.cpu().float().numpy(),
            "attn_wv_b": a.wv.bias.data.cpu().float().numpy(),
            "attn_wo_w": a.wo.weight.data.cpu().float().numpy(),
            "attn_wo_b": a.wo.bias.data.cpu().float().numpy(),
            "ln2_w": layer.ln2.weight.data.cpu().float().numpy(),
            "ln2_b": layer.ln2.bias.data.cpu().float().numpy(),
            "ff_w1_w": layer.w1.weight.data.cpu().float().numpy(),
            "ff_w1_b": layer.w1.bias.data.cpu().float().numpy(),
            "ff_w2_w": layer.w2.weight.data.cpu().float().numpy(),
            "ff_w2_b": layer.w2.bias.data.cpu().float().numpy(),
        })

    with open('/kaggle/working/model_10m_final.pkl', 'wb') as f:
        pickle.dump(state, f)
    print("Saved: /kaggle/working/model_10m_final.pkl")

    # Quick test
    print("\n--- Quick Test ---")
    seeds = ['Hola', 'Buenos dias', 'Quien eres', 'Que puedes hacer']
    for seed in seeds:
        ids = [c2i.get(c, 0) for c in seed]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        g = m.generate(x, 120, 0.7, 40)
        txt = ''.join(i2c.get(i.item(), '?') for i in g[0])
        print(f"{seed} -> {txt}")


if __name__ == '__main__':
    train()
