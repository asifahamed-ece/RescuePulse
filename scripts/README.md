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

---

## Phase 2: Traffic Control Integration

### Overview

Phase 2 adds hardware-driven traffic light management that responds dynamically to siren detections from Phase 1. The traffic control system is **entirely firmware-based** — no Python training or model quantization is required. The integration happens at the C/FreeRTOS level through:

1. **Detection message queue:** Inference task sends `detection_msg_t` via FreeRTOS queue
2. **State machine:** Traffic control task receives messages and drives GPIO outputs
3. **GPIO management:** 9 pins control 3 traffic lanes (LEFT/CENTER/RIGHT, each with RED/YELLOW/GREEN)

### Traffic Control Firmware Architecture

The traffic control module (`src/traffic_ctrl.c` / `src/traffic_ctrl.h`) implements a three-state finite state machine:

```
NORMAL (autonomous cycling)
  ↓ (siren on different lane or yellow phase)
CLEARANCE (2s all-red safety)
  ↓ (after 2s)
EMERGENCY (priority green on siren lane)
  ↓ (10s no siren)
NORMAL
```

### Message Flow: Phase 1 → Phase 2

**Phase 1 (Inference) generates:**
```c
detection_msg_t msg = {
    .siren_active = true,
    .direction = DOA_RIGHT,  // Mapped 1:1 from DoA
    .confidence = 0.98f      // Model confidence
};
xQueueSend(g_traffic_queue, &msg, 0);  // Non-blocking
```

**Phase 2 (Traffic Control) receives and acts:**
```c
if (xQueueReceive(g_traffic_queue, &msg, 100ms timeout) == pdTRUE) {
    if (msg->siren_active) {
        // Transition to EMERGENCY or CLEARANCE
    }
}
```

### GPIO Pin Mapping

| Lane | Component | GPIO | Signal |
|------|-----------|------|--------|
| **LEFT** | Traffic Light | 1 | RED |
| | | 2 | YELLOW |
| | | 3 | GREEN |
| **CENTER** | Traffic Light | 4 | RED |
| | | 5 | YELLOW |
| | | 6 | GREEN |
| **RIGHT** | Traffic Light | 13 | RED |
| | | 14 | YELLOW |
| | | 21 | GREEN |

All 9 pins are configured as digital GPIO outputs with no pull resistors. Drive strength is suitable for direct LED control (add 150Ω current-limiting resistors on hardware side).

### Timing Configuration

All timings are `#define` values in `src/traffic_ctrl.c`:

```c
#define NORMAL_GREEN_MS          8000    /* 8 seconds green per lane */
#define NORMAL_YELLOW_MS         2000    /* 2 seconds yellow per lane */
#define CLEARANCE_MS             2000    /* All-red safety interval */
#define EMERGENCY_TIMEOUT_MS     10000   /* Exit emergency if no siren for 10s */
#define QUEUE_TIMEOUT_MS         100     /* Poll detection queue every 100ms */
```

### State Machine Behaviors

**MODE_NORMAL** — Autonomous Cycling
- Lane sequence: LEFT → CENTER → RIGHT → LEFT (repeating)
- Timing: GREEN (8s) → YELLOW (2s) → RED
- Triggered on: System boot or emergency timeout
- Lights all lanes independently with no external input

**MODE_CLEARANCE** — 2-Second All-Red
- All traffic lights set to RED
- Ensures intersection clears before emergency vehicle passes
- Prevents T-bone collisions during mode transitions
- Triggered on: Siren detected on non-green lane or during yellow phase

**MODE_EMERGENCY** — Priority Green
- Siren lane: GREEN (continuous)
- All other lanes: RED
- Automatically exits after 10 seconds without siren
- Optimization: If siren on already-green lane during green phase, skip clearance

### Optimization: Smart Clearance Avoidance

If a siren is detected on a lane that is **already GREEN** in NORMAL mode and **not in YELLOW phase**, the system directly transitions to EMERGENCY without the 2-second clearance delay:

```c
if (s_state.mode == MODE_NORMAL && 
    s_state.current_lane == msg->direction && 
    !s_state.in_yellow) {
    /* Siren on active GREEN lane - extend without clearance */
    s_state.mode = MODE_EMERGENCY;
    s_state.emergency_lane = msg->direction;
    /* Keep lights as-is; no change needed */
}
```

This reduces emergency vehicle wait time by ~2 seconds when the vehicle is already on a green lane.

### Serial Output Examples

**Boot & Normal Operation:**
```
I (xxx) traffic_ctrl: Initializing Emergency Vehicle Priority Traffic Controller
I (xxx) traffic_ctrl: GPIO pins configured: 1-6, 13-14, 21
I (xxx) traffic_ctrl: Traffic control task launched (priority 3, stack 4096 bytes)
I (xxx) traffic_ctrl: Entering NORMAL mode: starting with LANE_LEFT GREEN
I (xxx) traffic_ctrl: Normal cycle: LANE_CENTER now GREEN
I (xxx) traffic_ctrl: Normal cycle: LANE_RIGHT now GREEN
```

**Emergency Detection:**
```
W (xxx) rescuepulse: 🚨 SIREN DETECTED [RIGHT] (Conf: 0.98) [5/5]
I (xxx) traffic_ctrl: 🚨 Siren detected: LANE_RIGHT (confidence: 0.98)
I (xxx) traffic_ctrl: Entering CLEARANCE mode: 2s all-red before emergency LANE_RIGHT
I (xxx) traffic_ctrl: Entering EMERGENCY mode: LANE_RIGHT GREEN (emergency vehicle)
```

**Emergency Timeout:**
```
I (xxx) traffic_ctrl: No siren for 10001 ms, exiting emergency mode
I (xxx) traffic_ctrl: Entering NORMAL mode: starting with LANE_LEFT GREEN
```

### Porting Traffic Control to Other Systems

#### Change GPIO Pins

Edit `src/traffic_ctrl.h`:
```c
#define TL_LEFT_RED     <new_pin>
#define TL_LEFT_YELLOW  <new_pin>
#define TL_LEFT_GREEN   <new_pin>
/* ... etc for CENTER and RIGHT ... */
```

#### Change Timing

Edit `src/traffic_ctrl.c`:
```c
#define NORMAL_GREEN_MS      10000    /* Increase for high-traffic areas */
#define EMERGENCY_TIMEOUT_MS 15000    /* Extend for larger vehicles */
```

#### Add More Lanes

1. Extend `lane_t` enum in `traffic_ctrl.h`
2. Add GPIO defines for new lane
3. Update `set_lane_lights()` switch statement
4. Update `lane_name()` function
5. Update lane cycling logic (modulo in `update_normal_mode()`)

#### Different Detection System

If your detection doesn't use DoA estimation, just send the same message structure:

```c
detection_msg_t msg = {
    .siren_active = true,
    .direction = LANE_CENTER,  /* Your system's lane assignment */
    .confidence = 0.95f
};
xQueueSend(g_traffic_queue, &msg, 0);
```

The traffic controller doesn't care where the messages come from — it only reads from the queue.

### Memory & Performance

| Metric | Value | Notes |
|--------|-------|-------|
| State struct | 32 B | Static allocation |
| Message queue | 160 B | 10 × 16-byte messages |
| Task stack | 4 KB | Fixed at link time |
| **Total overhead** | **4.3 KB** | No dynamic allocations |
| **Queue latency** | 5-20 ms | FreeRTOS IPC typical |
| **GPIO switching** | <1 µs | Hardware-level |
| **CPU usage** | <0.1% | 100 ms poll interval on Core 1 |

### Firmware Integration Points

**`app_main()` initialization:**
```c
if (traffic_ctrl_init() != ESP_OK) {
    ESP_LOGE(TAG, "Traffic Controller Init: FAIL");
    return;
}
ESP_LOGI(TAG, "Traffic Controller Init: OK");
```

**Inference task message sending:**
```c
detection_msg_t msg = {
    .siren_active = true,
    .direction = (lane_t)doa_dir,
    .confidence = confidence
};
xQueueSend(g_traffic_queue, &msg, 0);
```

**Task scheduling (both on Core 1):**
- Inference task: Priority 4 (higher)
- Traffic control task: Priority 3 (lower)

This ensures detections are always processed before traffic state updates.

### No Python Changes Required

Phase 2 is **100% firmware** — no changes to:
- Dataset processing scripts
- Model training pipeline
- Quantization artifacts
- MFCC test vector generation

The Python pipeline remains unchanged; Phase 2 consumes the trained model output and adds hardware-level traffic management on top.

---