use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use ndarray::{Array1, Array2, Array3, s};
use std::collections::HashMap;

fn layer_norm_2d(x: &Array2<f32>, gamma: &Array1<f32>, beta: &Array1<f32>, eps: f32) -> Array2<f32> {
    let (rows, d) = x.dim();
    let mut out = Array2::<f32>::zeros((rows, d));
    for i in 0..rows {
        let mean = x.row(i).mean().unwrap();
        let var = x.row(i).iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            out[[i, j]] = gamma[j] * (x[[i, j]] - mean) * inv_std + beta[j];
        }
    }
    out
}

fn softmax_last_axis(x: &mut Array2<f32>) {
    let (rows, cols) = x.dim();
    for i in 0..rows {
        let mut max = f32::NEG_INFINITY;
        for j in 0..cols {
            if x[[i, j]] > max { max = x[[i, j]]; }
        }
        let mut sum = 0.0f32;
        for j in 0..cols {
            x[[i, j]] = (x[[i, j]] - max).exp();
            sum += x[[i, j]];
        }
        for j in 0..cols {
            x[[i, j]] /= sum;
        }
    }
}

fn embed_2d(weights: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let d = weights.ncols();
    let seq_len = indices.len();
    let mut out = Array2::<f32>::zeros((seq_len, d));
    for (i, &idx) in indices.iter().enumerate() {
        let clamped = idx.min(weights.nrows() - 1);
        out.row_mut(i).assign(&weights.row(clamped));
    }
    out
}

fn dict_get_f32(_py: Python, dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<f32> {
    let v = dict.get_item(key)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string())
    })?;
    v.extract()
}

fn dict_get_usize(_py: Python, dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<usize> {
    let v = dict.get_item(key)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string())
    })?;
    v.extract()
}

fn dict_get_array1(py: Python, dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Array1<f32>> {
    let np = py.import_bound("numpy")?;
    let obj = dict.get_item(key)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string())
    })?;
    let arr = np.getattr("asarray")?.call1((obj, np.getattr("float32")?))?;
    let arr: PyReadonlyArray1<f32> = arr.extract()?;
    Ok(arr.as_array().to_owned())
}

fn dict_get_array2(py: Python, dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Array2<f32>> {
    let np = py.import_bound("numpy")?;
    let obj = dict.get_item(key)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string())
    })?;
    let arr = np.getattr("asarray")?.call1((obj, np.getattr("float32")?))?;
    let arr: PyReadonlyArray2<f32> = arr.extract()?;
    Ok(arr.as_array().to_owned())
}

fn dict_get_array3(py: Python, dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Array3<f32>> {
    let np = py.import_bound("numpy")?;
    let obj = dict.get_item(key)?.ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyKeyError, _>(key.to_string())
    })?;
    let arr = np.getattr("asarray")?.call1((obj, np.getattr("float32")?))?;
    let arr: PyReadonlyArray3<f32> = arr.extract()?;
    Ok(arr.as_array().to_owned())
}

struct LayerWeights {
    ln1_w: Array1<f32>,
    ln1_b: Array1<f32>,
    wq_w: Array2<f32>,
    wq_b: Array1<f32>,
    wk_w: Array2<f32>,
    wk_b: Array1<f32>,
    wv_w: Array2<f32>,
    wv_b: Array1<f32>,
    wo_w: Array2<f32>,
    wo_b: Array1<f32>,
    ln2_w: Array1<f32>,
    ln2_b: Array1<f32>,
    w1_w: Array2<f32>,
    w1_b: Array1<f32>,
    w2_w: Array2<f32>,
    w2_b: Array1<f32>,
}

struct Weights {
    token_emb: Array2<f32>,
    pos_emb: Array3<f32>,
    ln_f_w: Array1<f32>,
    ln_f_b: Array1<f32>,
    lm_w: Array2<f32>,
    lm_b: Array1<f32>,
    layers: Vec<LayerWeights>,
}

struct Config {
    vocab_size: usize,
    block_size: usize,
    d_model: usize,
    n_head: usize,
    n_layer: usize,
    d_ff: usize,
    head_dim: usize,
    char_to_id: HashMap<char, usize>,
    id_to_char: HashMap<usize, char>,
}

struct Model {
    w: Weights,
    cfg: Config,
}

impl Model {
    fn forward_single(&self, tokens: &[usize]) -> Array1<f32> {
        let seq_len = tokens.len();
        let d = self.cfg.d_model;
        let n_head = self.cfg.n_head;
        let head_dim = self.cfg.head_dim;

        let tok_emb = embed_2d(&self.w.token_emb, tokens);
        let pos_emb = self.w.pos_emb.slice(s![0, ..seq_len, ..]).to_owned();
        let mut h = tok_emb + pos_emb;

        for layer in &self.w.layers {
            let h_norm = layer_norm_2d(&h, &layer.ln1_w, &layer.ln1_b, 1e-5);

            let q = &h_norm.dot(&layer.wq_w) + &layer.wq_b;
            let k = &h_norm.dot(&layer.wk_w) + &layer.wk_b;
            let v = &h_norm.dot(&layer.wv_w) + &layer.wv_b;

            let q = q.to_shape((seq_len, n_head, head_dim)).unwrap().into_owned();
            let k = k.to_shape((seq_len, n_head, head_dim)).unwrap().into_owned();
            let v = v.to_shape((seq_len, n_head, head_dim)).unwrap().into_owned();

            let q = q.permuted_axes([1, 0, 2]);
            let k = k.permuted_axes([1, 0, 2]);
            let v = v.permuted_axes([1, 0, 2]);

            let scale = 1.0 / (head_dim as f32).sqrt();
            let mut att = Array3::<f32>::zeros((n_head, seq_len, seq_len));
            for h_idx in 0..n_head {
                let q_h = q.slice(s![h_idx, .., ..]);
                let k_h = k.slice(s![h_idx, .., ..]);
                let mut scores = q_h.dot(&k_h.t()) * scale;

                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        scores[[i, j]] = f32::NEG_INFINITY;
                    }
                }
                softmax_last_axis(&mut scores);
                att.slice_mut(s![h_idx, .., ..]).assign(&scores);
            }

            let mut attn_out = Array3::<f32>::zeros((n_head, seq_len, head_dim));
            for h_idx in 0..n_head {
                let scores = att.slice(s![h_idx, .., ..]);
                let v_h = v.slice(s![h_idx, .., ..]);
                attn_out.slice_mut(s![h_idx, .., ..]).assign(&scores.dot(&v_h));
            }

            let attn_out = attn_out.permuted_axes([1, 0, 2]);
            let attn_out = attn_out.to_shape((seq_len, d)).unwrap().into_owned();

            let attn_out = attn_out.dot(&layer.wo_w) + &layer.wo_b;
            h = h + attn_out;

            let h_norm = layer_norm_2d(&h, &layer.ln2_w, &layer.ln2_b, 1e-5);
            let hidden = (&h_norm.dot(&layer.w1_w) + &layer.w1_b).mapv(|x| x.max(0.0));
            let mlp_out = hidden.dot(&layer.w2_w) + &layer.w2_b;
            h = h + mlp_out;
        }

        let h = layer_norm_2d(&h, &self.w.ln_f_w, &self.w.ln_f_b, 1e-5);
        let logits = h.dot(&self.w.lm_w) + &self.w.lm_b;
        logits.row(seq_len - 1).to_owned()
    }

    fn generate(&self, seed: &str, max_chars: usize, temperature: f32, top_k: usize) -> String {
        let mut tokens: Vec<usize> = seed
            .chars()
            .map(|c| *self.cfg.char_to_id.get(&c).unwrap_or(&0))
            .collect();

        if tokens.is_empty() {
            return String::new();
        }

        let temp = temperature.max(0.05);
        let vocab = self.cfg.vocab_size;

        for _ in 0..max_chars {
            let start = if tokens.len() > self.cfg.block_size {
                tokens.len() - self.cfg.block_size
            } else {
                0
            };
            let input = &tokens[start..];

            let logits = self.forward_single(input);
            let mut logits_scaled: Array1<f32> = logits.mapv(|x| x / temp);

            if top_k > 0 && top_k < vocab {
                let mut indexed: Vec<(usize, f32)> = logits_scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let threshold = indexed[top_k - 1].1;
                for v in logits_scaled.iter_mut() {
                    if *v < threshold { *v = f32::NEG_INFINITY; }
                }
            }

            let max_val = logits_scaled.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut probs: Array1<f32> = logits_scaled.mapv(|x| (x - max_val).exp());
            let sum = probs.sum();
            probs.mapv_inplace(|x| x / sum);

            let r: f32 = rand::random();
            let mut cumsum = 0.0f32;
            let mut next_id = 0usize;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    next_id = i;
                    break;
                }
            }

            tokens.push(next_id);
        }

        tokens[seed.len()..]
            .iter()
            .map(|&id| *self.cfg.id_to_char.get(&id).unwrap_or(&'?'))
            .collect()
    }
}

#[pyclass]
struct RubidiumModel {
    model: Option<Model>,
}

#[pymethods]
impl RubidiumModel {
    #[new]
    fn new() -> Self {
        Self { model: None }
    }

    fn load_from_pickle(&mut self, py: Python, path: &str) -> PyResult<()> {
        let pickle = py.import_bound("pickle")?;
        let builtins = py.import_bound("builtins")?;
        let file = builtins.getattr("open")?.call1((path, "rb"))?;
        let state = pickle.getattr("load")?.call1((file,))?;
        let state = state.downcast::<pyo3::types::PyDict>()?;

        let vocab_size = dict_get_usize(py, state, "vocab_size")?;
        let block_size = dict_get_usize(py, state, "block_size")?;
        let d_model = dict_get_usize(py, state, "d_model")?;
        let n_head = dict_get_usize(py, state, "n_head")?;
        let n_layer = dict_get_usize(py, state, "n_layer")?;
        let d_ff = dict_get_usize(py, state, "d_ff")?;
        let head_dim = d_model / n_head;

        let char_to_id_binding = state.get_item("char_to_id")?.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>("char_to_id".to_string())
        })?;
        let char_to_id_py = char_to_id_binding.downcast::<pyo3::types::PyDict>()?;
        let id_to_char_binding = state.get_item("id_to_char")?.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>("id_to_char".to_string())
        })?;
        let id_to_char_py = id_to_char_binding.downcast::<pyo3::types::PyDict>()?;

        let mut char_to_id = HashMap::new();
        for (k, v) in char_to_id_py.iter() {
            if let (Ok(ch), Ok(id)) = (k.extract::<char>(), v.extract::<usize>()) {
                char_to_id.insert(ch, id);
            }
        }

        let mut id_to_char = HashMap::new();
        for (k, v) in id_to_char_py.iter() {
            if let (Ok(id), Ok(ch)) = (k.extract::<usize>(), v.extract::<char>()) {
                id_to_char.insert(id, ch);
            }
        }

        let token_emb = dict_get_array2(py, state, "token_emb")?;
        let pos_emb = dict_get_array3(py, state, "pos_emb")?;
        let ln_f_w = dict_get_array1(py, state, "ln_f_w")?;
        let ln_f_b = dict_get_array1(py, state, "ln_f_b")?;
        let lm_w = dict_get_array2(py, state, "lm_w")?.reversed_axes();  // (V,D) -> (D,V)
        let lm_b = dict_get_array1(py, state, "lm_b")?;

        let layers_binding = state.get_item("layers")?.ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>("layers".to_string())
        })?;
        let layers_py = layers_binding.downcast::<pyo3::types::PyList>()?;

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..layers_py.len() {
            let l_binding = layers_py.get_item(i)?;
            let l = l_binding.downcast::<pyo3::types::PyDict>()?;
            // PyTorch saves weights as (out_features, in_features)
            // Rust ndarray dot: (M,N) @ (N,P) -> (M,P), need (in, out)
            // So we transpose: w1_w (FF,D) -> (D,FF), w2_w (D,FF) -> (FF,D)
            layers.push(LayerWeights {
                ln1_w: dict_get_array1(py, l, "ln1_w")?,
                ln1_b: dict_get_array1(py, l, "ln1_b")?,
                wq_w: dict_get_array2(py, l, "attn_wq_w")?,
                wq_b: dict_get_array1(py, l, "attn_wq_b")?,
                wk_w: dict_get_array2(py, l, "attn_wk_w")?,
                wk_b: dict_get_array1(py, l, "attn_wk_b")?,
                wv_w: dict_get_array2(py, l, "attn_wv_w")?,
                wv_b: dict_get_array1(py, l, "attn_wv_b")?,
                wo_w: dict_get_array2(py, l, "attn_wo_w")?,
                wo_b: dict_get_array1(py, l, "attn_wo_b")?,
                ln2_w: dict_get_array1(py, l, "ln2_w")?,
                ln2_b: dict_get_array1(py, l, "ln2_b")?,
                w1_w: dict_get_array2(py, l, "ff_w1_w")?.reversed_axes(),
                w1_b: dict_get_array1(py, l, "ff_w1_b")?,
                w2_w: dict_get_array2(py, l, "ff_w2_w")?.reversed_axes(),
                w2_b: dict_get_array1(py, l, "ff_w2_b")?,
            });
        }

        let total_params = token_emb.len() + pos_emb.len() + ln_f_w.len() + ln_f_b.len()
            + lm_w.len() + lm_b.len()
            + layers.iter().map(|l| {
                l.ln1_w.len() + l.ln1_b.len() + l.ln2_w.len() + l.ln2_b.len()
                    + l.wq_w.len() + l.wq_b.len() + l.wk_w.len() + l.wk_b.len()
                    + l.wv_w.len() + l.wv_b.len() + l.wo_w.len() + l.wo_b.len()
                    + l.w1_w.len() + l.w1_b.len() + l.w2_w.len() + l.w2_b.len()
            }).sum::<usize>();

        self.model = Some(Model {
            w: Weights { token_emb, pos_emb, ln_f_w, ln_f_b, lm_w, lm_b, layers },
            cfg: Config { vocab_size, block_size, d_model, n_head, n_layer, d_ff, head_dim, char_to_id, id_to_char },
        });

        println!("Loaded: {}M params, vocab={}, d={}, layers={}",
            total_params / 1_000_000, vocab_size, d_model, n_layer);

        Ok(())
    }

    fn generate(&self, seed: &str, max_chars: usize, temperature: f32, top_k: usize) -> PyResult<String> {
        let model = self.model.as_ref()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Model not loaded".to_string()))?;
        Ok(model.generate(seed, max_chars, temperature, top_k))
    }
}

#[pymodule]
fn rubidium_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RubidiumModel>()?;
    Ok(())
}
