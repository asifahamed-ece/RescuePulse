# Rescue Pulse — Python ML Pipeline (`scripts/`)

This directory documents the Python training and quantization scripts that produce the artifacts deployed to the ESP32-S3. The scripts live in `scripts/`; this README explains the end-to-end workflow and the critical implementation details you'll be asked about in a viva.

> **Path anchoring:** Every script derives the repository root from its own location (`ROOT = Path(__file__).resolve().parent.parent`), so all `datasets/`, `models/`, and `firmware/` paths resolve correctly regardless of the current working directory. You can run any script from anywhere, e.g. `python scripts/train_model.py`.

---

## 🔄 Workflow

```
process_raw_data.py  →  audit_dataset.py  →  extract_features.py
      │                      │                     │
      ▼                      ▼                     ▼
  16 kHz mono 3 s WAV   Clean manifest      MFCC (64,13) windows
  (siren/ + noise/)     (drop conflicts)    + SpecAugment
                                            + mfcc_stats.npz
                                                    │
                                                    ▼
                                            train_model.py
                                                    │
                                                    ▼
                                            siren_classifier.keras
                                                    │
                                                    ▼
                                            quantize_model.py
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
              model_data.cc                model_config.h            siren_classifier_quantized.tflite
              (int8 TFLite bytes)          (g_mfcc_mu/std,           (validated int8 model)
                                           scale/zero-point)
                                                    │
                                                    ▼
                                            gen_mfcc_test_vectors.py
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
                  mel_tables.h              test_vectors.h             (parity check)
                  (g_ham, g_mel_fb,         (siren/noise PCM,
                   g_dct)                   expected MFCC, int8 input)
```

### Stage-by-Stage

| Script | Stage | Output |
|--------|-------|--------|
| `process_raw_data.py` | Raw audio → standardized WAV | `datasets/processed/{siren,noise}/*.wav`, `datasets/manifest.csv` |
| `audit_dataset.py` | Label-contamination detection | `datasets/manifest_clean.csv` |
| `extract_features.py` | MFCC + augmentation + split | `datasets/features/X_{train,val}.npy`, `y_*.npy`, `clip_*.npy`, `mfcc_stats.npz`, `mfcc_config.json` |
| `train_model.py` | 1D CNN training + eval | `models/siren_classifier.keras`, `training_curves.png`, `confusion_matrix.png` |
| `quantize_model.py` | int8 PTQ + firmware export | `models/siren_classifier_quantized.tflite`, `models/model_data.cc`, `models/model_config.h` |
| `gen_mfcc_test_vectors.py` | Parity test vectors | `firmware/main/mel_tables.h`, `firmware/main/test_vectors.h` |
| `test_audio.py` | Single-file inference demo | Console classification of any audio file |

---

## 🧠 The DCT-II Matrix Generation (and the Off-by-One Bug)

The MFCC pipeline ends with a **Discrete Cosine Transform (DCT-II)** that decorrelates the 40 log-mel energies into 13 cepstral coefficients. Librosa uses `scipy.fftpack.dct(norm='ortho')` internally, so the firmware must reproduce that exact orthonormal DCT-II matrix.

`gen_mfcc_test_vectors.py` generates the matrix with `ortho_dct2()`:

```python
def ortho_dct2(n_filters, width):
    """Orthonormal DCT-II matrix [n_filters][width].

    Matches scipy.fftpack.dct(norm='ortho'), which is what
    librosa.feature.mfcc() uses internally:
        D[k, n] = sqrt(1/N)                 for k=0
        D[k, n] = sqrt(2/N) * cos(pi*k*(2n+1)/(2N))  for k=1..n_filters-1
    """
    n = np.arange(width)
    k = np.arange(n_filters)[:, None]          # k = 0, 1, ..., n_filters-1
    D = np.cos(np.pi * k * (2 * n[None, :] + 1) / (2 * width))
    D[0, :] *= np.sqrt(1.0 / width)            # row 0: constant sqrt(1/N)
    D[1:, :] *= np.sqrt(2.0 / width)           # rows 1..: sqrt(2/N)
    return D
```

### The Bug

The **original** implementation had an off-by-one error in the frequency index `k`:

```python
# BUGGY version:
k = np.arange(1, n_filters + 1)[:, None]   # k = 1, 2, ..., 13  (WRONG!)
D = np.cos(np.pi * k * (2 * n[None, :] + 1) / (2 * width))
D *= np.sqrt(2.0 / width)
D[0, :] *= 1.0 / np.sqrt(2.0)
```

Two problems:

1. **Off-by-one in `k`:** `k` started at 1 instead of 0, so the DCT basis rows were shifted by one frequency index. The DC row (k=0) was missing entirely, and the highest coefficient (k=13) was computed with the wrong frequency.
2. **Wrong DC scaling:** The DC row (k=0) should be scaled by `sqrt(1/N)`, but the buggy code applied `sqrt(2/N)` to *all* rows and then tried to "fix" row 0 by multiplying by `1/sqrt(2)` — which yields `sqrt(2/N) * 1/sqrt(2) = sqrt(1/N)`. While that arithmetic happens to give the right DC magnitude, the off-by-one in `k` meant the *entire matrix* was wrong, not just the DC row.

### The Fix (commit `1f84d19`)

```python
# FIXED version:
k = np.arange(n_filters)[:, None]          # k = 0, 1, ..., n_filters-1
D = np.cos(np.pi * k * (2 * n[None, :] + 1) / (2 * width))
D[0, :] *= np.sqrt(1.0 / width)            # row 0: constant sqrt(1/N)
D[1:, :] *= np.sqrt(2.0 / width)           # rows 1..: sqrt(2/N)
```

- `k` now correctly spans **0..12** (matching the 13 MFCC coefficients).
- Row 0 (DC) is scaled by `sqrt(1/N)`.
- Rows 1..12 are scaled by `sqrt(2/N)`.

This exactly reproduces `scipy.fftpack.dct(norm='ortho')`, guaranteeing the firmware's `g_dct[520]` table (13×40) matches what librosa used during training — a critical train/deploy parity requirement.

---

## 📊 `mfcc_stats.npz` → `g_mfcc_mu` / `g_mfcc_std`

### How the stats are computed

`extract_features.py` standardizes each MFCC coefficient **per-coefficient** (not globally) using statistics computed **only from the training set** (to avoid data leakage into validation):

```python
mu  = X_tr.reshape(-1, N_MFCC).mean(0)          # shape (13,)
std = X_tr.reshape(-1, N_MFCC).std(0) + 1e-6    # shape (13,), +1e-6 avoids div-by-zero
X_tr = (X_tr - mu) / std
X_va = (X_va - mu) / std                        # val uses TRAIN stats
```

The stats are saved to `datasets/features/mfcc_stats.npz`:

```python
np.savez(OUT / "mfcc_stats.npz", mu=mu, std=std)
```

### How they reach the ESP32

`quantize_model.py` loads `mfcc_stats.npz` and embeds the values directly into `model_config.h` as C arrays:

```c
static const float g_mfcc_mu[13]  = {-117.434525f, 4.487606f, -24.293510f, ...};
static const float g_mfcc_std[13] = {58.905327f, 23.372980f, 16.025259f, ...};
```

The firmware then applies the identical standardization before quantization:

```c
float z = (mfcc[t][c] - g_mfcc_mu[c]) / g_mfcc_std[c];
float qf = roundf(z / g_in_scale) + (float)g_in_zp;
```

**Why this matters:** If the firmware used different mu/std than training, the standardized features would be shifted/scaled differently, and the int8 quantized input would fall outside the calibrated range — silently degrading accuracy. Embedding the exact train-set stats guarantees the on-device features match the training distribution.

---

## 🔢 Quantization (`quantize_model.py`)

### Full int8 Post-Training Quantization

The model is converted to a **fully int8** TFLite graph (no float tensors) because TFLite Micro on the ESP32-S3 has no float kernel support:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

- **Representative dataset:** 200 training windows sampled *without replacement* with a fixed seed (42), fed one-at-a-time exactly like the deployed firmware. This calibrates the activation ranges for quantization.
- **Validation:** The int8 model is run on the *entire* validation set and compared against the float32 oracle to quantify the accuracy drop (target < 2%).

### Firmware Artifacts

| Artifact | Contents |
|----------|----------|
| `model_data.cc` | `g_model_data[]` — the raw int8 TFLite bytes (108,392 B ≈ 106 KB), 12 hex bytes per line, plus `g_model_data_len`. |
| `model_config.h` | Audio front-end constants (`RP_*`), `g_mfcc_mu[13]`/`g_mfcc_std[13]`, and the interpreter's per-tensor quantization params (`g_in_scale`, `g_in_zp`, `g_out_scale`, `g_out_zp`). |

The quantization parameters are read from the interpreter's `quantization_parameters` dict (with a fallback to the legacy top-level `scale`/`zero_point` for TF 2.10+ portability):

```python
in_scale, in_zp = _quant_params(in_d, "input")
out_scale, out_zp = _quant_params(out_d, "output")
```

---

## 🧪 Parity Testing (`gen_mfcc_test_vectors.py`)

This script bridges the Python and C worlds:

1. **Dumps exact librosa tables** into `mel_tables.h`:
   - `g_ham[512]` — periodic Hamming window (`librosa.filters.get_window("hamming", 512, fftbins=True)`).
   - `g_mel_fb[10280]` — 40×257 Slaney mel filterbank (`librosa.filters.mel(...)`).
   - `g_dct[520]` — 13×40 orthonormal DCT-II (`ortho_dct2(13, 40)`).

2. **Generates two end-to-end test vectors** into `test_vectors.h` (one siren, one noise):
   - Raw 16,640-sample PCM (`tv_siren_pcm`, `tv_noise_pcm`).
   - Expected MFCC output (`tv_siren_mfcc`, `tv_noise_mfcc`).
   - Quantized int8 input (`tv_siren_in8`, `tv_noise_in8`).
   - Expected class (`TV_SIREN_CLASS`, `TV_NOISE_CLASS`).

3. **Runs the int8 model** on each vector in an isolated subprocess to confirm the expected class.

The firmware's `RP_PARITY_TEST` harness then replays these vectors and asserts:
- MFCC relative L2 error < 1% vs. the Python-computed MFCC.
- Inference class matches the expected class.

> **Note:** TensorFlow and librosa cannot be imported in the same process (librosa's numba/LLVM corrupts TF's LLVM pass registry → segfault). `gen_mfcc_test_vectors.py` and `test_audio.py` therefore isolate TF inside subprocesses.