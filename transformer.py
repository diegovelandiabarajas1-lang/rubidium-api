import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int,
                 sliding_window: int = 0, sparse_topk: int = 0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size
        self.sliding_window = sliding_window
        self.sparse_topk = sparse_topk

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        Q = self.wq(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if self.sliding_window > 0:
            mask = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
            window_mask = (torch.arange(L, device=x.device).unsqueeze(1) -
                           torch.arange(L, device=x.device).unsqueeze(0))
            window_mask = (window_mask >= 0) & (window_mask < self.sliding_window)
            mask = mask & window_mask
            att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if self.sparse_topk > 0 and self.sparse_topk < L:
            topk_vals, _ = torch.topk(att, self.sparse_topk, dim=-1)
            threshold = topk_vals[..., -1:]
            att = torch.where(att < threshold, float("-inf"), att)

        att = F.softmax(att, dim=-1)
        y = torch.matmul(att, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=True)
        self.w2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, block_size: int,
                 sliding_window: int = 0, sparse_topk: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, sliding_window, sparse_topk)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, block_size: int = 128, d_model: int = 512,
                 n_head: int = 8, n_layer: int = 8, d_ff: int = 2048,
                 sliding_window: int = 0, sparse_topk: int = 0):
        super().__init__()
        self.block_size = block_size
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, block_size, d_model) * 0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, block_size, sliding_window, sparse_topk)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                return_logits: bool = False):
        B, L = idx.shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L} > block_size {self.block_size}"

        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding[:, :L, :]
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_logits:
            return logits

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
            return logits, loss

        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int = 20) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond, return_logits=True)
            logits = logits[:, -1, :] / max(temperature, 0.05)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx


class PyTorchTransformer:
    def __init__(self, block_size: int = 128, d_model: int = 512, n_head: int = 8,
                 n_layer: int = 8, d_ff: int = 2048, max_steps: int = 6000,
                 learning_rate: float = 1e-4, neg_samples: int = 0,
                 warmup_steps: int = 200, sliding_window: int = 0,
                 sparse_topk: int = 0, device: str = "cpu"):
        self.block_size = block_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.sliding_window = sliding_window
        self.sparse_topk = sparse_topk
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        self.tokenizer = None
        self.vocab_size = 0
        self.model: Optional[TransformerModel] = None
        self.optimizer = None
        self.scheduler = None
        self.corpus: List[str] = []
        self.is_trained = False
        self._char_to_id: Dict[str, int] = {}
        self._id_to_char: List[str] = []

    def get_embeddings(self):
        from text_similarity import WordEmbeddings
        emb = WordEmbeddings()
        if self.model is not None:
            emb.dimension = self.d_model
            weights = self.model.token_embedding.weight.detach().cpu().numpy()
            for token_id in range(min(100, self.vocab_size)):
                token = self._id_to_char[token_id] if token_id < len(self._id_to_char) else f"<{token_id}>"
                emb._vectors[token.lower()] = weights[token_id]
        return emb

    def train(self, text: str):
        if text and text.strip():
            self.corpus.append(text.strip())

    def fit(self):
        if not self.corpus:
            return

        full_text = "\n".join(self.corpus)

        if self.tokenizer is not None and self.tokenizer.vocab_size > 0:
            self.vocab_size = self.tokenizer.vocab_size
            all_tokens = self.tokenizer.encode(full_text)
            data = torch.tensor(all_tokens, dtype=torch.long, device=self.device)
        else:
            chars = sorted(list(set(full_text)))
            self._char_to_id = {ch: i for i, ch in enumerate(chars)}
            self._id_to_char = chars
            self.vocab_size = len(chars)
            data = torch.tensor([self._char_to_id.get(c, 0) for c in full_text],
                                dtype=torch.long, device=self.device)

        if len(data) < self.block_size + 2:
            return

        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            d_model=self.d_model,
            n_head=self.n_head,
            n_layer=self.n_layer,
            d_ff=self.d_ff,
            sliding_window=self.sliding_window,
            sparse_topk=self.sparse_topk
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=self.learning_rate, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_steps, eta_min=self.learning_rate * 0.1
        )

        n = len(data)
        smooth_loss = float("inf")

        for step in range(1, self.max_steps + 1):
            start = torch.randint(0, n - self.block_size - 1, (1,)).item()
            x = data[start:start + self.block_size].unsqueeze(0)
            y = data[start + 1:start + self.block_size + 1].unsqueeze(0)

            self.optimizer.zero_grad()
            _, loss = self.model(x, targets=y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            if self.warmup_steps > 0 and step <= self.warmup_steps:
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.learning_rate * step / self.warmup_steps
            else:
                self.scheduler.step()

            if torch.isnan(loss) or torch.isinf(loss):
                if step == 1:
                    smooth_loss = loss.item()
                continue

            lv = loss.item()
            smooth_loss = lv if step == 1 else 0.98 * smooth_loss + 0.02 * lv

            if step % 100 == 0 or step == self.max_steps:
                print(f"Step {step}/{self.max_steps}, loss: {smooth_loss:.4f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")

        self.is_trained = True

    @torch.no_grad()
    def generate(self, seed: str, max_chars: int = 200,
                 temperature: float = 0.8, top_k: int = 20) -> str:
        if not self.is_trained or self.model is None:
            return ""

        try:
            if self.tokenizer is not None:
                ids = self.tokenizer.encode(seed)
                if not ids:
                    ids = [0]
            else:
                ids = [self._char_to_id.get(c, self._char_to_id.get(" ", 0)) for c in seed]
                if not ids:
                    ids = [self._char_to_id.get(" ", 0)]

            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            idx_out = self.model.generate(idx, max_chars, temperature, top_k)

            out_ids = idx_out[0].tolist()
            if self.tokenizer is not None:
                result = self.tokenizer.decode(out_ids)
            else:
                result = "".join(self._id_to_char[i] if i < len(self._id_to_char) else "?" for i in out_ids)

            return result.replace("\n", " ").strip()
        except Exception:
            return ""

    def save(self, path: str):
        if self.model is None:
            return
        torch.save({
            "model_state": self.model.state_dict(),
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "d_ff": self.d_ff,
            "char_to_id": self._char_to_id,
            "id_to_char": self._id_to_char,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.vocab_size = checkpoint["vocab_size"]
        self.block_size = checkpoint["block_size"]
        self.d_model = checkpoint["d_model"]
        self.n_head = checkpoint["n_head"]
        self.n_layer = checkpoint["n_layer"]
        self.d_ff = checkpoint["d_ff"]
        self._char_to_id = checkpoint["char_to_id"]
        self._id_to_char = checkpoint["id_to_char"]

        self.model = TransformerModel(
            vocab_size=self.vocab_size,
            block_size=self.block_size,
            d_model=self.d_model,
            n_head=self.n_head,
            n_layer=self.n_layer,
            d_ff=self.d_ff,
            sliding_window=self.sliding_window,
            sparse_topk=self.sparse_topk
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.is_trained = True