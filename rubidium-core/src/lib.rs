use numpy::{PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use ndarray::{Array2, Array3};

#[pyfunction]
fn forward_numpy(
    py: Python<'_>,
    token_emb: PyReadonlyArray2<f32>,
    pos_emb: PyReadonlyArray3<f32>,
    ln_f_w: PyReadonlyArray2<f32>,
    ln_f_b: PyReadonlyArray2<f32>,
    lm_w: PyReadonlyArray2<f32>,
    lm_b: PyReadonlyArray2<f32>,
    x_arr: PyReadonlyArray2<i32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let tok = token_emb.as_array();
    let pos = pos_emb.as_array();
    let lfw = ln_f_w.as_array();
    let lfb = ln_f_b.as_array();
    let lmw = lm_w.as_array();
    let lmb = lm_b.as_array();
    let x = x_arr.as_array();
    let (b, l) = x.dim();
    let d_model = tok.shape()[1];
    let vocab = lmw.shape()[1];

    // Token + positional embedding
    let mut h: Array3<f32> = Array3::zeros((b, l, d_model));
    for i in 0..b {
        for j in 0..l {
            let idx = x[[i, j]] as usize;
            for k in 0..d_model {
                h[[i, j, k]] = tok[[idx, k]] + pos[[0, j, k]];
            }
        }
    }

    // Final layer norm
    let mut h_norm: Array3<f32> = Array3::zeros((b, l, d_model));
    for i in 0..b {
        for j in 0..l {
            let mut mean: f32 = 0.0;
            for k in 0..d_model {
                mean += h[[i, j, k]];
            }
            mean /= d_model as f32;
            let mut var: f32 = 0.0;
            for k in 0..d_model {
                let diff = h[[i, j, k]] - mean;
                var += diff * diff;
            }
            var /= d_model as f32;
            let inv_std = (var + 1e-5_f32).sqrt().recip();
            for k in 0..d_model {
                h_norm[[i, j, k]] = (h[[i, j, k]] - mean) * inv_std * lfw[[0, k]] + lfb[[0, k]];
            }
        }
    }

    // LM head: matmul + bias
    let mut out: Array3<f32> = Array3::zeros((b, l, vocab));
    for i in 0..b {
        for j in 0..l {
            for v in 0..vocab {
                let mut sum: f32 = lmb[[0, v]];
                for k in 0..d_model {
                    sum += h_norm[[i, j, k]] * lmw[[k, v]];
                }
                out[[i, j, v]] = sum;
            }
        }
    }

    let result = PyArray3::from_array_bound(py, &out);
    Ok(result.into())
}

#[pyfunction]
fn attention_forward(
    py: Python<'_>,
    wq_w: PyReadonlyArray2<f32>,
    wq_b: PyReadonlyArray2<f32>,
    wk_w: PyReadonlyArray2<f32>,
    wk_b: PyReadonlyArray2<f32>,
    wv_w: PyReadonlyArray2<f32>,
    wv_b: PyReadonlyArray2<f32>,
    wo_w: PyReadonlyArray2<f32>,
    wo_b: PyReadonlyArray2<f32>,
    mask: PyReadonlyArray2<f32>,
    x: PyReadonlyArray3<f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let x_arr = x.as_array();
    let wq = wq_w.as_array();
    let bq = wq_b.as_array();
    let wk = wk_w.as_array();
    let bk = wk_b.as_array();
    let wv = wv_w.as_array();
    let bv = wv_b.as_array();
    let wo = wo_w.as_array();
    let bo = wo_b.as_array();
    let m = mask.as_array();
    let (b, l, d) = x_arr.dim();
    let n_head = 4;
    let head_dim = d / n_head;
    let scale = (head_dim as f32).sqrt().recip();

    // Compute Q, K, V
    let mut q: Array3<f32> = Array3::zeros((b, l, d));
    let mut k: Array3<f32> = Array3::zeros((b, l, d));
    let mut v: Array3<f32> = Array3::zeros((b, l, d));
    for i in 0..b {
        for j in 0..l {
            for h in 0..d {
                let mut sq: f32 = bq[[0, h]];
                let mut sk: f32 = bk[[0, h]];
                let mut sv: f32 = bv[[0, h]];
                for c in 0..d {
                    sq += x_arr[[i, j, c]] * wq[[c, h]];
                    sk += x_arr[[i, j, c]] * wk[[c, h]];
                    sv += x_arr[[i, j, c]] * wv[[c, h]];
                }
                q[[i, j, h]] = sq;
                k[[i, j, h]] = sk;
                v[[i, j, h]] = sv;
            }
        }
    }

    // Multi-head attention
    let mut att_out: Array3<f32> = Array3::zeros((b, l, d));
    for i in 0..b {
        for head in 0..n_head {
            let hd = head * head_dim;
            let mut scores: Array2<f32> = Array2::zeros((l, l));
            for qi in 0..l {
                for ki in 0..l {
                    let mut s: f32 = 0.0;
                    for dd in 0..head_dim {
                        s += q[[i, qi, hd + dd]] * k[[i, ki, hd + dd]];
                    }
                    scores[[qi, ki]] = s * scale + m[[qi, ki]];
                }
            }
            for qi in 0..l {
                let mut max_val: f32 = f32::NEG_INFINITY;
                for ki in 0..l {
                    if scores[[qi, ki]] > max_val {
                        max_val = scores[[qi, ki]];
                    }
                }
                let mut sum_exp: f32 = 0.0;
                for ki in 0..l {
                    scores[[qi, ki]] = (scores[[qi, ki]] - max_val).exp();
                    sum_exp += scores[[qi, ki]];
                }
                for ki in 0..l {
                    scores[[qi, ki]] /= sum_exp + 1e-8;
                }
            }
            for oi in 0..l {
                for oh in 0..head_dim {
                    let mut s: f32 = 0.0;
                    for si in 0..l {
                        s += scores[[oi, si]] * v[[i, si, hd + oh]];
                    }
                    att_out[[i, oi, hd + oh]] = s;
                }
            }
        }
    }

    // Output projection
    let mut out: Array3<f32> = Array3::zeros((b, l, d));
    for i in 0..b {
        for j in 0..l {
            for h in 0..d {
                let mut s: f32 = bo[[0, h]];
                for c in 0..d {
                    s += att_out[[i, j, c]] * wo[[c, h]];
                }
                out[[i, j, h]] = s;
            }
        }
    }

    let result = PyArray3::from_array_bound(py, &out);
    Ok(result.into())
}

#[pyfunction]
fn layer_norm_forward(
    py: Python<'_>,
    x: PyReadonlyArray3<f32>,
    w: PyReadonlyArray2<f32>,
    b: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let x_arr = x.as_array();
    let w_arr = w.as_array();
    let b_arr = b.as_array();
    let (batch, seq, d) = x_arr.dim();

    let mut out: Array3<f32> = Array3::zeros((batch, seq, d));
    for i in 0..batch {
        for j in 0..seq {
            let mut mean: f32 = 0.0;
            for k in 0..d {
                mean += x_arr[[i, j, k]];
            }
            mean /= d as f32;
            let mut var: f32 = 0.0;
            for k in 0..d {
                let diff = x_arr[[i, j, k]] - mean;
                var += diff * diff;
            }
            var /= d as f32;
            let inv_std = (var + 1e-5_f32).sqrt().recip();
            for k in 0..d {
                out[[i, j, k]] = (x_arr[[i, j, k]] - mean) * inv_std * w_arr[[0, k]] + b_arr[[0, k]];
            }
        }
    }

    let result = PyArray3::from_array_bound(py, &out);
    Ok(result.into())
}

#[pyfunction]
fn softmax_forward(
    py: Python<'_>,
    x: PyReadonlyArray3<f32>,
) -> PyResult<Py<PyArray3<f32>>> {
    let x_arr = x.as_array();
    let (b, l, v) = x_arr.dim();

    let mut out: Array3<f32> = Array3::zeros((b, l, v));
    for i in 0..b {
        for j in 0..l {
            let mut max_val: f32 = f32::NEG_INFINITY;
            for k in 0..v {
                if x_arr[[i, j, k]] > max_val {
                    max_val = x_arr[[i, j, k]];
                }
            }
            let mut sum_exp: f32 = 0.0;
            for k in 0..v {
                out[[i, j, k]] = (x_arr[[i, j, k]] - max_val).exp();
                sum_exp += out[[i, j, k]];
            }
            for k in 0..v {
                out[[i, j, k]] /= sum_exp + 1e-8;
            }
        }
    }

    let result = PyArray3::from_array_bound(py, &out);
    Ok(result.into())
}

#[pymodule]
fn rubidium_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(forward_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(attention_forward, m)?)?;
    m.add_function(wrap_pyfunction!(layer_norm_forward, m)?)?;
    m.add_function(wrap_pyfunction!(softmax_forward, m)?)?;
    Ok(())
}
