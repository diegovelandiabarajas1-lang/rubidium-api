import os
import pickle
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# RUBIDIUM TRANSFORMER - PyTorch (from scratch)
# Built with: FlashAttention, torch.compile, modern CUDA kernels
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size

        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.wo = nn.Linear(d_model, d_model, bias=True)

        self.mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()

    def forward(self, x):
        B, L, D = x.shape
        q = self.wq(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # FlashAttention via PyTorch's scaled_dot_product_attention
        # Automatically uses FlashAttention-2 on CUDA, memory-efficient on CPU
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=True,
            scale=1.0 / math.sqrt(self.head_dim)
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class RubidiumTransformer(nn.Module):
    def __init__(self, vocab_size=256, block_size=192, d_model=384,
                 n_head=6, n_layer=8, d_ff=1536):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, block_size)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, L = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(L, device=idx.device))
        h = tok_emb + pos_emb

        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)
        logits = self.lm_head(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.7, top_k=40):
        for _ in range(max_new_tokens):
            idx_trimmed = idx[:, -self.block_size:]
            logits, _ = self(idx_trimmed)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ============================================================
# TRAINING
# ============================================================

def load_corpus():
    texts = []
    base = '/kaggle/input/datasets/diegovelandiabarajas/rubidium-corpus-train/resources'
    for fname in sorted(os.listdir(base)):
        if fname.endswith('.txt'):
            with open(os.path.join(base, fname), 'r', encoding='utf-8', errors='replace') as f:
                texts.append(f.read())
    full_text = '\n'.join(texts)
    print(f'Corpus: {len(full_text)} chars')
    return full_text


def train_model():
    print("=" * 60)
    print("RUBIDIUM 10M - PyTorch + FlashAttention + torch.compile")
    print("=" * 60)

    # Config
    vocab_size = 256
    block_size = 192
    d_model = 384
    n_head = 6
    n_layer = 8
    d_ff = 1536
    max_steps = 20000
    learning_rate = 3e-4
    warmup_steps = 2000
    grad_clip = 1.0
    batch_size = 8
    grad_accum = 4  # effective batch = 32

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load corpus
    full_text = load_corpus()

    # Build vocab
    chars = sorted(list(set(full_text)))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    print(f"Vocab: {vocab_size}")

    # Encode text
    data = torch.tensor([char_to_id.get(c, 0) for c in full_text], dtype=torch.long, device=device)
    print(f"Data tokens: {len(data)}")

    # Create model
    model = RubidiumTransformer(
        vocab_size=vocab_size, block_size=block_size, d_model=d_model,
        n_head=n_head, n_layer=n_layer, d_ff=d_ff
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.1f}M")

    # torch.compile for maximum performance
    print("Compiling model with torch.compile...")
    model = torch.compile(model, mode='reduce-overhead')
    print("Model compiled!")

    # Optimizer: AdamW with cosine schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.1)

    def get_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    # Training loop
    print(f"\nTraining: {max_steps} micro-steps, batch={batch_size}, grad_accum={grad_accum}")
    print(f"Effective batch: {batch_size * grad_accum}")
    print(f"LR: {learning_rate}, warmup: {warmup_steps}, grad_clip: {grad_clip}")
    print("-" * 60)

    model.train()
    smooth_loss = float('inf')
    t0 = time.time()
    n = len(data)

    for step in range(1, max_steps + 1):
        lr = get_lr(step // grad_accum)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Sample batch
        ix = torch.randint(0, n - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])

        # Forward + backward
        _, loss = model(x, y)
        loss = loss / grad_accum
        loss.backward()

        # Optimizer step
        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Logging
        lv = loss.item() * grad_accum
        smooth_loss = lv if step == 1 else 0.98 * smooth_loss + 0.02 * lv

        if step % 100 == 0 or step == max_steps:
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed
            eta = (max_steps - step) / steps_per_sec
            print(f"Step {step}/{max_steps} | loss: {smooth_loss:.4f} | lr: {lr:.2e} | {steps_per_sec:.1f} steps/s | ETA: {eta/60:.0f}min")

    total_time = time.time() - t0
    print(f"\nTraining complete: {total_time/60:.1f} minutes")
    print(f"Speed: {max_steps/total_time:.1f} steps/sec")

    # Save in same format as numpy version for Rust inference
    print("\nSaving model...")
    model.eval()

    state = {
        "vocab_size": vocab_size,
        "block_size": block_size,
        "d_model": d_model,
        "n_head": n_head,
        "n_layer": n_layer,
        "d_ff": d_ff,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "token_emb": model.token_emb.weight.data.cpu().float().numpy(),
        "pos_emb": model.pos_emb.weight.data.cpu().float().numpy().reshape(1, block_size, d_model),
        "ln_f_w": model.ln_f.weight.data.cpu().float().numpy(),
        "ln_f_b": model.ln_f.bias.data.cpu().float().numpy(),
        "lm_w": model.lm_head.weight.data.cpu().float().numpy(),
        "lm_b": model.lm_head.bias.data.cpu().float().numpy(),
        "layers": [],
    }

    for layer in model.layers:
        attn = layer.attn
        mlp = layer.mlp
        layer_state = {
            "ln1_w": layer.ln1.weight.data.cpu().float().numpy(),
            "ln1_b": layer.ln1.bias.data.cpu().float().numpy(),
            "attn_wq_w": attn.wq.weight.data.cpu().float().numpy(),
            "attn_wq_b": attn.wq.bias.data.cpu().float().numpy(),
            "attn_wk_w": attn.wk.weight.data.cpu().float().numpy(),
            "attn_wk_b": attn.wk.bias.data.cpu().float().numpy(),
            "attn_wv_w": attn.wv.weight.data.cpu().float().numpy(),
            "attn_wv_b": attn.wv.bias.data.cpu().float().numpy(),
            "attn_wo_w": attn.wo.weight.data.cpu().float().numpy(),
            "attn_wo_b": attn.wo.bias.data.cpu().float().numpy(),
            "ln2_w": layer.ln2.weight.data.cpu().float().numpy(),
            "ln2_b": layer.ln2.bias.data.cpu().float().numpy(),
            "ff_w1_w": mlp.w1.weight.data.cpu().float().numpy(),
            "ff_w1_b": mlp.w1.bias.data.cpu().float().numpy(),
            "ff_w2_w": mlp.w2.weight.data.cpu().float().numpy(),
            "ff_w2_b": mlp.w2.bias.data.cpu().float().numpy(),
        }
        state["layers"].append(layer_state)

    with open('/kaggle/working/model_10m_final.pkl', 'wb') as f:
        pickle.dump(state, f)
    print("Saved: /kaggle/working/model_10m_final.pkl")

    # Quick test
    print("\n--- Quick Test ---")
    model_for_gen = model._orig_mod if hasattr(model, '_orig_mod') else model
    model_for_gen.eval()
    seeds = ['Hola', 'Buenos dias', 'Quien eres', 'Que puedes hacer', 'Como estas']
    for seed in seeds:
        ids = [char_to_id.get(c, 0) for c in seed]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        generated = model_for_gen.generate(x, max_new_tokens=120, temperature=0.7, top_k=40)
        text = ''.join(id_to_char.get(i.item(), '?') for i in generated[0])
        print(f"{seed} -> {text}")


if __name__ == '__main__':
    train_model()
