#!/usr/bin/env python3
# ============================================================
# RUBIDIUM 250M - CPU Training (PyTorch)
# Config: V=32000, T=512, D=1536, H=24, L=10, FF=6144
# Optimized for CPU: mixed precision not needed, larger BS possible
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle, time, math, os, sys, glob, json, random
from pathlib import Path

print("=" * 60)
print("RUBIDIUM 250M - CPU Training")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: cpu")

# ============================================================
# MODEL
# ============================================================
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


# ============================================================
# CORPUS
# ============================================================
def load_corpus():
    """Load all .txt files from resources/ and data/"""
    texts = []
    for base in ['resources', 'data', '../resources']:
        if os.path.exists(base):
            for f in sorted(glob.glob(f'{base}/*.txt')):
                texts.append(open(f, 'r', encoding='utf-8', errors='replace').read())
                print(f"  Loaded: {f}")
    if not texts:
        print("ERROR: No corpus found! Place .txt files in resources/")
        sys.exit(1)
    full = '\n'.join(texts)
    print(f'Corpus: {len(full):,} chars (~{len(full)//4:,} tokens)')
    return full


# ============================================================
# TRAINING
# ============================================================
def train():
    # Config: 250M params
    V = 32000; T = 512; D = 1536; H = 24; L = 10; FF = 6144
    BS = 4; GA = 8; max_steps = 200000
    lr = 3e-4; warmup = 6000; gc = 1.0
    save_every = 5000
    device = torch.device('cpu')

    print(f"\nConfig: V={V} T={T} D={D} H={H} L={L} FF={FF}")

    # Load corpus
    print("\n--- Loading corpus ---")
    full_text = load_corpus()

    # Build vocab
    chars = sorted(set(full_text))
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for i, c in enumerate(chars)}
    V = len(chars)
    print(f"Vocab: {V}")

    # Encode
    data = torch.tensor([c2i.get(c, 0) for c in full_text], dtype=torch.long)
    n = len(data)
    print(f"Tokens: {n:,}")

    # Init model
    print("\n--- Initializing model ---")
    model = Rubidium(V, T, D, H, L, FF).to(device)
    tp = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {tp/1e6:.1f}M")
    print(f"Model size: {tp * 4 / 1024/1024:.1f} MB (FP32)")

    # Optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.1)

    def get_lr(s):
        if s < warmup: return lr * s / warmup
        p = (s - warmup) / (max_steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    print(f"\nTraining: {max_steps} steps, BS={BS}, GA={GA}, Eff={BS*GA}")
    print(f"LR: {lr}, Warmup: {warmup}, GradClip: {gc}")
    print("-" * 60)

    # Checkpoint dir
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    model.train()
    sl = float('inf')
    t0 = time.time()

    for step in range(1, max_steps + 1):
        lr_t = get_lr(step // GA)
        for pg in optim.param_groups: pg['lr'] = lr_t

        # Sample batch
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
            print(f"Step {step}/{max_steps} | loss: {sl:.4f} | lr: {lr_t:.2e} | {sps:.2f} steps/s | ETA: {eta:.0f}min", flush=True)

        # Save checkpoint
        if step % save_every == 0:
            ckpt_path = f'checkpoints/model_step_{step}.pkl'
            save_model(model, c2i, i2c, V, T, D, H, L, FF, ckpt_path)

    # Final save
    total_time = time.time() - t0
    print(f"\nTraining complete: {total_time/60:.1f} min ({max_steps/total_time:.2f} steps/s)")
    save_model(model, c2i, i2c, V, T, D, H, L, FF, 'model_final.pkl')

    # Quick test
    print("\n--- Quick Test ---")
    model.eval()
    seeds = ['Hola', 'Buenos dias', 'Quien eres', 'Que puedes hacer']
    for seed in seeds:
        ids = [c2i.get(c, 0) for c in seed]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        g = model.generate(x, 120, 0.7, 40)
        txt = ''.join(i2c.get(i.item(), '?') for i in g[0])
        print(f"{seed} -> {txt}")


def save_model(model, c2i, i2c, V, T, D, H, L, FF, path):
    """Save model as pickle for Rust inference"""
    model.eval()
    state = {
        "vocab_size": V, "block_size": T, "d_model": D, "n_head": H,
        "n_layer": L, "d_ff": FF,
        "char_to_id": c2i, "id_to_char": i2c,
        "token_emb": model.te.weight.data.cpu().float().numpy(),
        "pos_emb": model.pe.weight.data.cpu().float().numpy().reshape(1, T, D),
        "ln_f_w": model.lnf.weight.data.cpu().float().numpy(),
        "ln_f_b": model.lnf.bias.data.cpu().float().numpy(),
        "lm_w": model.head.weight.data.cpu().float().numpy(),
        "lm_b": model.head.bias.data.cpu().float().numpy(),
        "layers": [],
    }
    for layer in model.layers:
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
    with open(path, 'wb') as f:
        pickle.dump(state, f)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"Saved: {path} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    train()
