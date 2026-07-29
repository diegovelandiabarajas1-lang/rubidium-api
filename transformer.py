import numpy as np
import pickle
import os
import time
from typing import Optional, List, Dict, Tuple

try:
    import rubidium_core
    HAS_RUST = True
    print("Rubidium Core (Rust) loaded - available for large models")
except ImportError:
    HAS_RUST = False


class AutogradTensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data.astype(np.float32) if data.dtype != np.float32 else data
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward = None
        self._children = []

    def build_topo(self, v, topo, visited):
        if id(v) not in visited:
            visited.add(id(v))
            for child in v._children:
                self.build_topo(child, topo, visited)
            topo.append(v)

    def backward(self, grad: Optional[np.ndarray] = None):
        if grad is None:
            grad = np.ones_like(self.data)

        topo = []
        visited = set()
        self.build_topo(self, topo, visited)

        self.grad = grad
        for v in reversed(topo):
            if v._backward is not None:
                v._backward(v.grad)

    def __add__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                # Sum grad to match self.data shape if needed
                g = grad
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != self.data.shape[axis] and self.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = grad
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != other.data.shape[axis] and other.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = backward
        out._children = [self, other]
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                g = grad
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != self.data.shape[axis] and self.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = -grad
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != other.data.shape[axis] and other.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = backward
        out._children = [self, other]
        return out

    def __neg__(self):
        return self * (-1.0)

    def __mul__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                g = grad * other.data
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != self.data.shape[axis] and self.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = grad * self.data
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != other.data.shape[axis] and other.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = backward
        out._children = [self, other]
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                g = grad / other.data
                while g.ndim > self.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != self.data.shape[axis] and self.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = -grad * self.data / (other.data ** 2)
                while g.ndim > other.data.ndim:
                    g = g.sum(axis=0)
                for axis in range(g.ndim):
                    if g.shape[axis] != other.data.shape[axis] and other.data.shape[axis] == 1:
                        g = g.sum(axis=axis, keepdims=True)
                other.grad = g if other.grad is None else other.grad + g
        out._backward = backward
        out._children = [self, other]
        return out

    def __matmul__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                if self.data.ndim >= 3 and other.data.ndim == 2:
                    batch_shape = self.data.shape[:-2]
                    M, K = self.data.shape[-2:]
                    N = other.data.shape[1]
                    x_flat = self.data.reshape(-1, K)
                    grad_flat = grad.reshape(-1, N)
                    g = grad_flat @ other.data.T
                    g = g.reshape(*batch_shape, M, K)
                elif self.data.ndim == 4 and other.data.ndim == 4:
                    b, h = self.data.shape[0], self.data.shape[1]
                    x_flat = self.data.reshape(b*h, self.data.shape[2], self.data.shape[3])
                    o_flat = other.data.reshape(b*h, other.data.shape[2], other.data.shape[3])
                    grad_flat = grad.reshape(b*h, grad.shape[2], grad.shape[3])
                    g_flat = np.einsum('bmn,bnk->bmk', grad_flat, o_flat.swapaxes(-2, -1))
                    g = g_flat.reshape(b, h, g_flat.shape[1], g_flat.shape[2])
                elif self.data.ndim == 2 and other.data.ndim == 2:
                    g = grad @ other.data.T
                else:
                    g = grad @ other.data.T
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                if self.data.ndim == 2 and grad.ndim == 2:
                    g = self.data.T @ grad
                elif self.data.ndim >= 3 and grad.ndim >= 3 and other.data.ndim == 2:
                    K = self.data.shape[-1]
                    N = other.data.shape[1]
                    x_flat = self.data.reshape(-1, K)
                    grad_flat = grad.reshape(-1, N)
                    g = x_flat.T @ grad_flat
                elif self.data.ndim == 4 and other.data.ndim == 4:
                    b, h = self.data.shape[0], self.data.shape[1]
                    x_flat = self.data.reshape(b*h, self.data.shape[2], self.data.shape[3])
                    o_flat = other.data.reshape(b*h, other.data.shape[2], other.data.shape[3])
                    grad_flat = grad.reshape(b*h, grad.shape[2], grad.shape[3])
                    g_flat = np.einsum('bmk,bmn->bkn', x_flat, grad_flat)
                    g = g_flat.reshape(b, h, g_flat.shape[1], g_flat.shape[2]).sum(axis=(0, 1))
                else:
                    g = np.tensordot(self.data, grad, axes=([-2], [-2]))
                other.grad = g if other.grad is None else other.grad + g
        out._backward = backward
        out._children = [self, other]
        return out

    def sum(self, axis=None, keepdims=False):
        out = AutogradTensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                g = np.broadcast_to(grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = backward
        out._children = [self]
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / n

    def exp(self):
        out = AutogradTensor(np.exp(self.data), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                g = grad * out.data
                self.grad = g if self.grad is None else self.grad + g
        out._backward = backward
        out._children = [self]
        return out

    def log(self):
        out = AutogradTensor(np.log(self.data + 1e-8), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                g = grad / self.data
                self.grad = g if self.grad is None else self.grad + g
        out._backward = backward
        out._children = [self]
        return out

    def relu(self):
        out = AutogradTensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                g = grad * (self.data > 0).astype(np.float32)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = backward
        out._children = [self]
        return out

    def reshape(self, shape):
        out = AutogradTensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                g = grad.reshape(self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
        out._backward = backward
        out._children = [self]
        return out

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        out = AutogradTensor(self.data.transpose(*axes), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                if axes:
                    inv_axes = np.argsort(axes)
                    # Handle potential shape mismatch from broadcast/reshape
                    expected_shape = self.data.transpose(*axes).shape
                    if grad.shape != expected_shape:
                        # If grad was summed/reduced, we can't transpose back properly
                        # Just return zero grad for this path
                        g = np.zeros_like(self.data)
                    else:
                        g = grad.transpose(*inv_axes)
                else:
                    g = grad.T
                self.grad = g if self.grad is None else self.grad + g
        out._backward = backward
        out._children = [self]
        return out

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return f"AutogradTensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


def softmax(x: AutogradTensor, axis=-1):
    e = np.exp(x.data - x.data.max(axis=axis, keepdims=True))
    s = e.sum(axis=axis, keepdims=True)
    out = AutogradTensor(e / s, requires_grad=x.requires_grad)
    def backward(grad):
        if x.requires_grad:
            out_grad = grad * out.data
            g = out_grad - out_grad.sum(axis=axis, keepdims=True) * out.data
            x.grad = g if x.grad is None else x.grad + g
    out._backward = backward
    out._children = [x]
    return out


def layer_norm(x: AutogradTensor, weight: AutogradTensor, bias: AutogradTensor, eps=1e-5):
    mean = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_norm = (x.data - mean) * inv_std
    out = AutogradTensor(weight.data * x_norm + bias.data, requires_grad=True)
    def backward(grad):
        if weight.requires_grad:
            g = (grad * x_norm).sum(axis=(0, 1))
            weight.grad = g if weight.grad is None else weight.grad + g
        if bias.requires_grad:
            g = grad.sum(axis=(0, 1))
            bias.grad = g if bias.grad is None else bias.grad + g
        if x.requires_grad:
            N = x.data.shape[-1]
            dy_w = grad * weight.data
            mean_dy_w = dy_w.mean(axis=-1, keepdims=True)
            mean_dy_w_x = (dy_w * x_norm).mean(axis=-1, keepdims=True)
            g = (dy_w - mean_dy_w - x_norm * mean_dy_w_x) * inv_std
            x.grad = g if x.grad is None else x.grad + g
    out._backward = backward
    out._children = [x, weight, bias]
    return out


class Linear:
    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.weight = AutogradTensor(np.random.randn(in_features, out_features).astype(np.float32) * scale, requires_grad=True)
        self.bias = AutogradTensor(np.zeros(out_features, dtype=np.float32), requires_grad=True)

    def __call__(self, x: AutogradTensor) -> AutogradTensor:
        return x @ self.weight + self.bias

    def parameters(self) -> List[AutogradTensor]:
        return [self.weight, self.bias]


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = AutogradTensor(np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02, requires_grad=True)

    def __call__(self, x: np.ndarray) -> AutogradTensor:
        B, L = x.shape
        V, D = self.weight.data.shape
        one_hot = np.zeros((B, L, V), dtype=np.float32)
        for b in range(B):
            for l in range(L):
                idx = int(x[b, l])
                if 0 <= idx < V:
                    one_hot[b, l, idx] = 1.0
        one_hot_tensor = AutogradTensor(one_hot, requires_grad=False)
        return one_hot_tensor @ self.weight

    def parameters(self) -> List[AutogradTensor]:
        return [self.weight]


class MultiHeadAttention:
    def __init__(self, d_model: int, n_head: int, block_size: int):
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size
        self._scale = 1.0 / np.sqrt(self.head_dim)

        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

        mask = np.triu(np.ones((block_size, block_size), dtype=np.float32), 1) * -1e9
        self.mask = mask

    def __call__(self, x: AutogradTensor) -> AutogradTensor:
        B, L, D = x.data.shape
        q = self.wq(x).reshape((B, L, self.n_head, self.head_dim)).transpose((0, 2, 1, 3))
        k = self.wk(x).reshape((B, L, self.n_head, self.head_dim)).transpose((0, 2, 1, 3))
        v = self.wv(x).reshape((B, L, self.n_head, self.head_dim)).transpose((0, 2, 1, 3))

        scale = 1.0 / np.sqrt(self.head_dim)
        att = (q @ k.transpose((0, 1, 3, 2))) * scale

        mask = AutogradTensor(self.mask[:L, :L], requires_grad=False)
        att = att + mask

        att = softmax(att, axis=-1)
        out = att @ v
        out = out.transpose((0, 2, 1, 3)).reshape((B, L, D))
        return self.wo(out)

    def parameters(self) -> List[AutogradTensor]:
        return self.wq.parameters() + self.wk.parameters() + self.wv.parameters() + self.wo.parameters()

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        B, L, D = x.shape
        wq, wk, wv, wo = self.wq.weight.data, self.wk.weight.data, self.wv.weight.data, self.wo.weight.data
        bq, bk, bv, bo = self.wq.bias.data, self.wk.bias.data, self.wv.bias.data, self.wo.bias.data
        q = (x @ wq + bq).reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (x @ wk + bk).reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (x @ wv + bv).reshape(B, L, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        scale = self._scale
        att = (q @ k.transpose(0, 1, 3, 2)) * scale
        att = att + self.mask[:L, :L]
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att = att / (att.sum(axis=-1, keepdims=True) + 1e-8)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return out @ wo + bo


class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def __call__(self, x: AutogradTensor) -> AutogradTensor:
        return self.w2(self.w1(x).relu())

    def parameters(self) -> List[AutogradTensor]:
        return self.w1.parameters() + self.w2.parameters()

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x @ self.w1.weight.data + self.w1.bias.data) @ self.w2.weight.data + self.w2.bias.data


class TransformerBlock:
    def __init__(self, d_model: int, n_head: int, d_ff: int, block_size: int):
        self.ln1_w = AutogradTensor(np.ones(d_model, dtype=np.float32), requires_grad=True)
        self.ln1_b = AutogradTensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        self.attn = MultiHeadAttention(d_model, n_head, block_size)
        self.ln2_w = AutogradTensor(np.ones(d_model, dtype=np.float32), requires_grad=True)
        self.ln2_b = AutogradTensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        self.mlp = FeedForward(d_model, d_ff)

    def __call__(self, x: AutogradTensor) -> AutogradTensor:
        x = x + self.attn(layer_norm(x, self.ln1_w, self.ln1_b))
        x = x + self.mlp(layer_norm(x, self.ln2_w, self.ln2_b))
        return x

    def parameters(self) -> List[AutogradTensor]:
        return [self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b] + self.attn.parameters() + self.mlp.parameters()

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        x = x + self.attn.forward_numpy(self.ln1_w.data * x_norm + self.ln1_b.data)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + 1e-5)
        x = x + self.mlp.forward_numpy(self.ln2_w.data * x_norm + self.ln2_b.data)
        return x


class NumpyTransformer:
    def __init__(self, vocab_size: int = 256, block_size: int = 128, d_model: int = 128,
                 n_head: int = 4, n_layer: int = 4, d_ff: int = 512,
                 max_steps: int = 1000, learning_rate: float = 3e-4):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff
        self.max_steps = max_steps
        self.learning_rate = learning_rate

        self.token_embedding = Embedding(vocab_size, d_model)
        self.pos_embedding = AutogradTensor(np.random.randn(1, block_size, d_model).astype(np.float32) * 0.02, requires_grad=True)
        self.layers = [TransformerBlock(d_model, n_head, d_ff, block_size) for _ in range(n_layer)]
        self.ln_f_w = AutogradTensor(np.ones(d_model, dtype=np.float32), requires_grad=True)
        self.ln_f_b = AutogradTensor(np.zeros(d_model, dtype=np.float32), requires_grad=True)
        self.lm_head = Linear(d_model, vocab_size)

        self._char_to_id = {chr(i): i for i in range(256)}
        self._id_to_char = {i: chr(i) for i in range(256)}
        self.is_trained = False
        self.corpus: List[str] = []
        self._layers = self.layers
        self._inv_sqrt_var = 1.0 / np.sqrt(self.d_model)

    def _all_params(self) -> List[AutogradTensor]:
        params = [self.pos_embedding, self.ln_f_w, self.ln_f_b]
        params += self.token_embedding.parameters()
        params += self.lm_head.parameters()
        for layer in self.layers:
            params += layer.parameters()
        return params

    def forward(self, x: np.ndarray) -> AutogradTensor:
        B, L = x.shape
        tok_emb = self.token_embedding(x)
        pos_emb = AutogradTensor(self.pos_embedding.data[:, :L, :], requires_grad=False)
        h = tok_emb + pos_emb
        for layer in self.layers:
            h = layer(h)
        h = layer_norm(h, self.ln_f_w, self.ln_f_b)
        logits = self.lm_head(h)
        return logits

    def loss(self, logits: AutogradTensor, targets: np.ndarray, label_smoothing: float = 0.1) -> AutogradTensor:
        B, L, V = logits.data.shape
        logits_flat = logits.reshape((B * L, V))
        targets_flat = targets.reshape(B * L)
        probs = softmax(logits_flat, axis=-1)
        # Add epsilon to avoid log(0) = -inf
        log_probs = (probs + 1e-8).log()
        
        # Label smoothing
        one_hot = np.zeros((B * L, V), dtype=np.float32)
        one_hot[np.arange(B * L), targets_flat] = 1.0
        if label_smoothing > 0:
            one_hot = one_hot * (1.0 - label_smoothing) + label_smoothing / V
        
        selected = log_probs * AutogradTensor(one_hot, requires_grad=False)
        selected = selected.sum(axis=-1)
        loss_val = selected.mean()
        neg_one = AutogradTensor(np.array(-1.0), requires_grad=False)
        return loss_val * neg_one

    def train(self, text: str):
        if text and text.strip():
            self.corpus.append(text.strip())

    def fit(self, grad_accum_steps: int = 1, warmup_steps: int = None, label_smoothing: float = 0.1,
            use_fp16: bool = True, loss_scale: float = 1024.0, prefetch_batches: int = 4,
            checkpoint_every: int = 5000, checkpoint_dir: str = "checkpoints"):
        """
        Optimized training loop with:
        - Mixed precision (fp16 forward/backward, fp32 master weights + loss scaling)
        - Data prefetching (background queue)
        - Gradient accumulation
        - Cosine LR with warmup
        - Checkpointing
        """
        if not self.corpus:
            return

        import os
        import queue
        import threading
        import time
        
        if checkpoint_every and checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        full_text = "\n".join(self.corpus)
        chars = sorted(list(set(full_text)))
        self._char_to_id = {ch: i for i, ch in enumerate(chars)}
        self._id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size)

        data = np.array([self._char_to_id.get(c, 0) for c in full_text], dtype=np.int32)
        if len(data) < self.block_size + 2:
            return

        params = self._all_params()
        
        # Mixed precision: master weights in fp32, fp16 copies for forward
        if use_fp16:
            master_params = [p.data.astype(np.float32).copy() for p in params]
            # fp16 working copies (updated from master each optim step)
            fp16_params = [mp.astype(np.float16) for mp in master_params]
            # Point autograd tensors to fp16 data
            for p, fp16 in zip(params, fp16_params):
                p.data = fp16
        else:
            master_params = [p.data.astype(np.float32).copy() for p in params]
            for p, mp in zip(params, master_params):
                p.data = mp

        m = [np.zeros_like(mp) for mp in master_params]
        v = [np.zeros_like(mp) for mp in master_params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        n = len(data)
        smooth_loss = float("inf")
        
        if warmup_steps is None:
            warmup_steps = self.max_steps // 10

        def get_lr(step):
            base_lr = self.learning_rate
            if step < warmup_steps:
                return base_lr * step / warmup_steps
            progress = (step - warmup_steps) / max(1, self.max_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

        # ---- Data prefetcher (background thread) ----
        batch_queue = queue.Queue(maxsize=prefetch_batches * 2)
        stop_prefetch = threading.Event()
        
        def prefetch_worker():
            rng = np.random.default_rng()
            while not stop_prefetch.is_set():
                try:
                    start = rng.integers(0, n - self.block_size - 1)
                    x = data[start:start + self.block_size].reshape(1, -1).astype(np.int32)
                    y = data[start + 1:start + self.block_size + 1].reshape(1, -1).astype(np.int32)
                    batch_queue.put((x, y), timeout=1.0)
                except queue.Full:
                    time.sleep(0.001)
                except Exception:
                    break
        
        prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        micro_step = 0
        optim_step = 0
        accum_grads = [np.zeros_like(mp) for mp in master_params]
        current_loss_scale = loss_scale
        min_loss_scale = 1.0
        growth_interval = 2000

        try:
            for step in range(1, self.max_steps + 1):
                # Get batch from prefetch queue
                try:
                    x, y = batch_queue.get(timeout=5.0)
                except queue.Empty:
                    # Fallback: generate synchronously
                    start = np.random.randint(0, n - self.block_size - 1)
                    x = data[start:start + self.block_size].reshape(1, -1)
                    y = data[start + 1:start + self.block_size + 1].reshape(1, -1)

                # Forward + backward (fp16 if enabled)
                logits = self.forward(x)
                loss = self.loss(logits, y, label_smoothing)
                
                # Scale loss for fp16
                if use_fp16:
                    loss.data = loss.data * current_loss_scale

                for p in params:
                    p.grad = None
                loss.backward()

                # Accumulate gradients (unscale if fp16)
                for i, p in enumerate(params):
                    if p.grad is not None:
                        g = p.grad
                        if use_fp16:
                            g = g / current_loss_scale
                        accum_grads[i] += g.astype(np.float32)

                micro_step += 1
                if micro_step % grad_accum_steps != 0:
                    continue

                optim_step += 1
                lr = get_lr(optim_step)

                # Optimizer step on master weights (fp32)
                for i, mp in enumerate(master_params):
                    g = accum_grads[i] / grad_accum_steps
                    accum_grads[i].fill(0)
                    
                    # Clip
                    g = np.clip(g, -5.0, 5.0)
                    
                    # Adam update
                    m[i] = beta1 * m[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * g * g
                    m_hat = m[i] / (1 - beta1 ** optim_step)
                    v_hat = v[i] / (1 - beta2 ** optim_step)
                    mp -= lr * m_hat / (np.sqrt(v_hat) + eps)
                
                # Update fp16 working copies from master
                if use_fp16:
                    for p, mp in zip(params, master_params):
                        p.data = mp.astype(np.float16)
                
                # Dynamic loss scaling
                if use_fp16:
                    # Check for inf/nan grads
                    has_inf = any(np.any(np.isinf(g)) or np.any(np.isnan(g)) for g in accum_grads)
                    if has_inf:
                        current_loss_scale = max(current_loss_scale / 2, min_loss_scale)
                    elif optim_step % growth_interval == 0:
                        current_loss_scale = min(current_loss_scale * 2, 65536.0)

                # Loss for logging (unscaled)
                lv = loss.data.item() if loss.data.size == 1 else float(loss.data)
                if use_fp16:
                    lv = lv / current_loss_scale
                smooth_loss = lv if optim_step == 1 else 0.98 * smooth_loss + 0.02 * lv

                if optim_step % 100 == 0 or step == self.max_steps:
                    print(f"Step {optim_step}/{self.max_steps // grad_accum_steps}, loss: {smooth_loss:.4f}, lr: {lr:.2e}, scale: {current_loss_scale:.0f}")

                # Checkpoint
                if checkpoint_every and optim_step % checkpoint_every == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"model_step_{optim_step}.pkl")
                    self.save(ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")

        finally:
            stop_prefetch.set()
            prefetch_thread.join(timeout=1.0)

        self.is_trained = True
        self._layers = self.layers
        self._inv_sqrt_var = 1.0 / np.sqrt(self.d_model)

    def forward_numpy(self, x: np.ndarray) -> np.ndarray:
        B, L = x.shape
        tok_emb = self.token_embedding.weight.data[x]
        pos_emb = self.pos_embedding.data[:, :L, :]
        h = tok_emb + pos_emb
        _layers = self._layers
        for i in range(len(_layers)):
            h = _layers[i].forward_numpy(h)
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = (h - mean) * self._inv_sqrt_var
        h = self.ln_f_w.data * h + self.ln_f_b.data
        return h @ self.lm_head.weight.data + self.lm_head.bias.data

    def generate(self, seed: str, max_chars: int = 200, temperature: float = 0.8, top_k: int = 20) -> str:
        if not self.is_trained:
            return ""

        ids = [self._char_to_id.get(c, 0) for c in seed]
        if not ids:
            ids = [0]

        temp = max(temperature, 0.05)
        _id_to_char = self._id_to_char
        vocab = len(_id_to_char)
        _forward = self.forward_numpy

        for _ in range(max_chars):
            ids_trimmed = ids[-self.block_size:]
            x = np.array([ids_trimmed], dtype=np.int32)
            logits = _forward(x)[0, -1, :] / temp

            if top_k > 0 and top_k < vocab:
                topk_vals = np.partition(logits, -top_k)[-top_k:]
                logits[logits < topk_vals.min()] = -1e9

            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()
            next_id = int(np.random.choice(vocab, p=probs))
            ids.append(next_id)

        return "".join(_id_to_char.get(i, "?") for i in ids).replace("\n", " ").strip()

    def save(self, path: str):
        state = {
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "d_model": self.d_model,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "d_ff": self.d_ff,
            "char_to_id": self._char_to_id,
            "id_to_char": self._id_to_char,
            "token_emb": self.token_embedding.weight.data,
            "pos_emb": self.pos_embedding.data,
            "ln_f_w": self.ln_f_w.data,
            "ln_f_b": self.ln_f_b.data,
            "lm_w": self.lm_head.weight.data,
            "lm_b": self.lm_head.bias.data,
            "layers": [],
        }
        for layer in self.layers:
            layer_state = {
                "ln1_w": layer.ln1_w.data,
                "ln1_b": layer.ln1_b.data,
                "ln2_w": layer.ln2_w.data,
                "ln2_b": layer.ln2_b.data,
                "attn_wq_w": layer.attn.wq.weight.data,
                "attn_wq_b": layer.attn.wq.bias.data,
                "attn_wk_w": layer.attn.wk.weight.data,
                "attn_wk_b": layer.attn.wk.bias.data,
                "attn_wv_w": layer.attn.wv.weight.data,
                "attn_wv_b": layer.attn.wv.bias.data,
                "attn_wo_w": layer.attn.wo.weight.data,
                "attn_wo_b": layer.attn.wo.bias.data,
                "ff_w1_w": layer.mlp.w1.weight.data,
                "ff_w1_b": layer.mlp.w1.bias.data,
                "ff_w2_w": layer.mlp.w2.weight.data,
                "ff_w2_b": layer.mlp.w2.bias.data,
            }
            state["layers"].append(layer_state)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vocab_size = state["vocab_size"]
        self.block_size = state["block_size"]
        self.d_model = state["d_model"]
        self.n_head = state["n_head"]
        self.n_layer = state["n_layer"]
        self.d_ff = state["d_ff"]
        self._char_to_id = state["char_to_id"]
        self._id_to_char = state["id_to_char"]

        self.token_embedding = Embedding(self.vocab_size, self.d_model)
        self.token_embedding.weight.data = state["token_emb"]
        self.pos_embedding = AutogradTensor(state["pos_emb"], requires_grad=True)
        self.ln_f_w = AutogradTensor(state["ln_f_w"], requires_grad=True)
        self.ln_f_b = AutogradTensor(state["ln_f_b"], requires_grad=True)
        self.lm_head = Linear(self.d_model, self.vocab_size)
        self.lm_head.weight.data = state["lm_w"]
        self.lm_head.bias.data = state["lm_b"]

        self.layers = []
        for layer_state in state["layers"]:
            block = TransformerBlock(self.d_model, self.n_head, self.d_ff, self.block_size)
            block.ln1_w.data = layer_state["ln1_w"]
            block.ln1_b.data = layer_state["ln1_b"]
            block.ln2_w.data = layer_state["ln2_w"]
            block.ln2_b.data = layer_state["ln2_b"]
            block.attn.wq.weight.data = layer_state["attn_wq_w"]
            block.attn.wq.bias.data = layer_state["attn_wq_b"]
            block.attn.wk.weight.data = layer_state["attn_wk_w"]
            block.attn.wk.bias.data = layer_state["attn_wk_b"]
            block.attn.wv.weight.data = layer_state["attn_wv_w"]
            block.attn.wv.bias.data = layer_state["attn_wv_b"]
            block.attn.wo.weight.data = layer_state["attn_wo_w"]
            block.attn.wo.bias.data = layer_state["attn_wo_b"]
            block.mlp.w1.weight.data = layer_state["ff_w1_w"]
            block.mlp.w1.bias.data = layer_state["ff_w1_b"]
            block.mlp.w2.weight.data = layer_state["ff_w2_w"]
            block.mlp.w2.bias.data = layer_state["ff_w2_b"]
            self.layers.append(block)

        self.is_trained = True
        self._layers = self.layers
        self._inv_sqrt_var = 1.0 / np.sqrt(self.d_model)