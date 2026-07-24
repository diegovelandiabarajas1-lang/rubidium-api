import numpy as np
import pickle
import os
import time
from typing import Optional, List, Dict, Tuple


class AutogradTensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False):
        self.data = data.astype(np.float32) if data.dtype != np.float32 else data
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        self._backward = None
        self._children = []

    def backward(self, grad: Optional[np.ndarray] = None):
        if grad is None:
            grad = np.ones_like(self.data)
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        if self._backward is not None:
            self._backward(grad)

    def __add__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                self.backward(grad)
            if other.requires_grad:
                other.backward(grad)
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
                self.backward(grad)
            if other.requires_grad:
                other.backward(-grad)
        out._backward = backward
        out._children = [self, other]
        return out

    def __mul__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                self.backward(grad * other.data)
            if other.requires_grad:
                other.backward(grad * self.data)
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
                self.backward(grad / other.data)
            if other.requires_grad:
                other.backward(-grad * self.data / (other.data ** 2))
        out._backward = backward
        out._children = [self, other]
        return out

    def __matmul__(self, other):
        if not isinstance(other, AutogradTensor):
            other = AutogradTensor(np.array(other))
        out = AutogradTensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        def backward(grad):
            if self.requires_grad:
                self.backward(grad @ other.data.T)
            if other.requires_grad:
                other.backward(self.data.T @ grad)
        out._backward = backward
        out._children = [self, other]
        return out

    def sum(self, axis=None, keepdims=False):
        out = AutogradTensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                if axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                self.backward(np.broadcast_to(grad, self.data.shape))
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
                self.backward(grad * out.data)
        out._backward = backward
        out._children = [self]
        return out

    def log(self):
        out = AutogradTensor(np.log(self.data + 1e-8), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                self.backward(grad / self.data)
        out._backward = backward
        out._children = [self]
        return out

    def relu(self):
        out = AutogradTensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                self.backward(grad * (self.data > 0).astype(np.float32))
        out._backward = backward
        out._children = [self]
        return out

    def reshape(self, shape):
        out = AutogradTensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                self.backward(grad.reshape(self.data.shape))
        out._backward = backward
        out._children = [self]
        return out

    def transpose(self, *axes):
        out = AutogradTensor(self.data.transpose(*axes), requires_grad=self.requires_grad)
        def backward(grad):
            if self.requires_grad:
                if axes:
                    inv_axes = [0] * len(axes)
                    for i, a in enumerate(axes):
                        inv_axes[a] = i
                    self.backward(grad.transpose(*inv_axes))
                else:
                    self.backward(grad.T)
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
            x.backward(out_grad - out_grad.sum(axis=axis, keepdims=True) * out.data)
    out._backward = backward
    out._children = [x]
    return out


def layer_norm(x: AutogradTensor, weight: AutogradTensor, bias: AutogradTensor, eps=1e-5):
    mean = x.data.mean(axis=-1, keepdims=True)
    var = x.data.var(axis=-1, keepdims=True)
    x_norm = (x.data - mean) / np.sqrt(var + eps)
    out = AutogradTensor(weight.data * x_norm + bias.data, requires_grad=True)
    def backward(grad):
        if weight.requires_grad:
            weight.backward(grad * x_norm)
        if bias.requires_grad:
            bias.backward(grad)
        if x.requires_grad:
            x.backward(grad * weight.data / np.sqrt(var + eps))
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
        return AutogradTensor(self.weight.data[x], requires_grad=False)

    def parameters(self) -> List[AutogradTensor]:
        return [self.weight]


class MultiHeadAttention:
    def __init__(self, d_model: int, n_head: int, block_size: int):
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.block_size = block_size

        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

        mask = np.triu(np.ones((block_size, block_size), dtype=np.float32), diagonal=1) * -1e9
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


class FeedForward:
    def __init__(self, d_model: int, d_ff: int):
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def __call__(self, x: AutogradTensor) -> AutogradTensor:
        return self.w2(self.w1(x).relu())

    def parameters(self) -> List[AutogradTensor]:
        return self.w1.parameters() + self.w2.parameters()


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

    def loss(self, logits: AutogradTensor, targets: np.ndarray) -> AutogradTensor:
        B, L, V = logits.data.shape
        logits_flat = logits.reshape((B * L, V))
        targets_flat = targets.reshape(B * L)
        probs = softmax(logits_flat, axis=-1)
        log_probs = probs.log()
        indices = np.arange(B * L)
        loss_val = -log_probs.data[indices, targets_flat].mean()
        return AutogradTensor(np.array(loss_val), requires_grad=True)

    def train(self, text: str):
        if text and text.strip():
            self.corpus.append(text.strip())

    def fit(self):
        if not self.corpus:
            return

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
        m = [np.zeros_like(p.data) for p in params]
        v = [np.zeros_like(p.data) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        n = len(data)
        smooth_loss = float("inf")

        for step in range(1, self.max_steps + 1):
            start = np.random.randint(0, n - self.block_size - 1)
            x = data[start:start + self.block_size].reshape(1, -1)
            y = data[start + 1:start + self.block_size + 1].reshape(1, -1)

            logits = self.forward(x)
            loss = self.loss(logits, y)

            for p in params:
                if p.grad is not None:
                    p.grad = None
            loss.backward()

            lr = self.learning_rate
            if step < self.max_steps // 10:
                lr = self.learning_rate * step / (self.max_steps // 10)

            for i, p in enumerate(params):
                if p.grad is not None:
                    g = np.clip(p.grad, -5.0, 5.0)
                    m[i] = beta1 * m[i] + (1 - beta1) * g
                    v[i] = beta2 * v[i] + (1 - beta2) * g ** 2
                    m_hat = m[i] / (1 - beta1 ** step)
                    v_hat = v[i] / (1 - beta2 ** step)
                    p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)

            lv = loss.data.item() if loss.data.size == 1 else float(loss.data)
            smooth_loss = lv if step == 1 else 0.98 * smooth_loss + 0.02 * lv

            if step % 100 == 0 or step == self.max_steps:
                print(f"Step {step}/{self.max_steps}, loss: {smooth_loss:.4f}")

        self.is_trained = True

    def generate(self, seed: str, max_chars: int = 200, temperature: float = 0.8, top_k: int = 20) -> str:
        if not self.is_trained:
            return ""

        ids = [self._char_to_id.get(c, 0) for c in seed]
        if not ids:
            ids = [0]

        for _ in range(max_chars):
            ids_trimmed = ids[-self.block_size:]
            x = np.array([ids_trimmed], dtype=np.int32)
            logits = self.forward(x)
            logits = logits.data[0, -1, :] / max(temperature, 0.05)

            if top_k > 0:
                topk_vals = np.sort(logits)[-top_k:]
                logits[logits < topk_vals[0]] = -1e9

            exp_logits = np.exp(logits - logits.max())
            probs = exp_logits / exp_logits.sum()
            next_id = np.random.choice(len(probs), p=probs)
            ids.append(next_id)

        return "".join(self._id_to_char.get(i, "?") for i in ids).replace("\n", " ").strip()

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
