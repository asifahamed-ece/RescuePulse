# RescuePulse — Comprehensive Project Execution & Technical Report

**Project Title:** Edge AI Emergency Siren Detection and Direction of Arrival (DoA)
**Platform:** ESP32-S3 (Edge AI, On-Device Inference)
**Document Purpose:** Internal reference for the documentation team preparing the final project report. This document explains the deep-down execution flow, tech stack at each layer, the exact engineering approach, and the current achievement status.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [Project Approach & Methodology](#3-project-approach--methodology)
4. [End-to-End Execution Flow](#4-end-to-end-execution-flow)
5. [Technology Stack by Layer](#5-technology-stack-by-layer)
6. [Layer 1 — Data Pipeline (Python / ML Training)](#6-layer-1--data-pipeline-python--ml-training)
7. [Layer 2 — Embedded Firmware (ESP32-S3 / C / C++)](#7-layer-2--embedded-firmware-esp32-s3--c--c)
8. [Signal Processing & Direction of Arrival](#8-signal-processing--direction-of-arrival)
9. [Neural Network Architecture & Quantization](#9-neural-network-architecture--quantization)
10. [Dual-Core FreeRTOS Execution Model](#10-dual-core-freertos-execution-model)
11. [Hardware Configuration](#11-hardware-configuration)
12. [Memory Architecture & Resource Budget](#12-memory-architecture--resource-budget)
13. [Validation & Parity Testing](#13-validation--parity-testing)
14. [Current Achievement Status](#14-current-achievement-status)
15. [Repository Structure](#15-repository-structure)
16. [Build & Deployment Procedure](#16-build--deployment-procedure)
17. [Limitations & Future Work](#17-limitations--future-work)
18. [License](#18-license)

---

## 1. Executive Summary

RescuePulse is a real-time, fully edge-deployed acoustic intelligence system running on the ESP32-S3 microcontroller. It performs two tightly-coupled tasks entirely on-chip with zero cloud dependency and sub-15 ms per-frame latency:

1. **Emergency Siren Classification** — A 1D Convolutional Neural Network (CNN), quantized to full INT8, distinguishes emergency vehicle sirens (ambulance, fire engine, police) from heavy urban noise (traffic, horns, engines, construction, speech).
2. **Direction of Arrival (DoA) Estimation** — A dual-microphone MEMS array combined with ESP-DSP cross-correlation localizes the siren source direction (LEFT / RIGHT / CENTER) before the vehicle is visually in range.

The entire signal chain — stereo I2S capture, MFCC feature extraction, TDOA correlation, and TFLite Micro inference — executes locally on the MCU using a deterministic, dual-core FreeRTOS pipeline with zero dynamic allocation in the hot path. The C-based MFCC implementation achieves <1% L2 relative error compared to 64-bit Python Librosa references, and the INT8 quantized model shows no accuracy loss relative to its FP32 baseline.

### Headline Metrics

| Metric | Value |
|---|---|
| Inference latency per frame | < 15 ms |
| Model flash footprint (INT8) | 108 KB |
| Total internal DRAM usage | 76.4% (250 KB / 328 KB) |
| TFLite tensor arena (external PSRAM) | 200 KB of 8 MB |
| MFCC parity vs Python (L2 error) | < 1% |
| Quantization impact vs FP32 | No accuracy loss |

---

## 2. Problem Statement & Motivation

Emergency vehicles (ambulances, fire engines, police cars) produce characteristic siren signals that must be detected and localized in noisy urban environments. Traditional approaches rely on cloud-connected systems that introduce unacceptable latency and fail without network connectivity.

**RescuePulse solves three simultaneous challenges:**

1. **Detection under noise** — Distinguishing siren acoustic signatures from dense, overlapping urban noise. A simple frequency-threshold detector fails here; siren spectra overlap heavily with traffic, engines, and human speech. A learned neural representation is required.
2. **Localization with minimal hardware** — Estimating the direction of arrival using only two microphones and the time-difference-of-arrival (TDOA) principle, rather than expensive phased arrays or beamforming hardware.
3. **Edge constraint satisfaction** — Running all of this on a resource-constrained MCU (limited RAM, fixed-point arithmetic, no FPU headroom for large floats) with deterministic, real-time performance and no cloud round-trip.

The system is purpose-built for edge deployment: every design decision — from the INT8 quantization to the static ping-pong buffers to the dual-core split — is driven by the constraint that inference must be real-time, on-chip, and reliable.

---

## 3. Project Approach & Methodology

The project was executed as a vertically-integrated edge-ML pipeline, where the **desktop ML training environment** and the **embedded inference firmware** are designed together to guarantee mathematical parity. The approach can be summarized in four guiding principles:

### Principle 1: Parity-First Design
The MFCC feature extraction, audio constants (sample rate, FFT size, hop, mel bins), and standardization statistics are defined **once** and shared exactly between the Python training pipeline and the C firmware. A dedicated parity test (`RP_PARITY_TEST`) compares the C MFCC output against pre-computed Python reference vectors and fails the build if the L2 relative error exceeds 1%.

### Principle 2: Quantize Early, Validate Late
The model is trained in FP32 with TensorFlow/Keras, then converted via **full integer post-training quantization** to a TFLite INT8 model. The quantized model is validated against the full validation set and compared to the float32 oracle to confirm no accuracy degradation before any firmware artifact is emitted.

### Principle 3: Deterministic Embedded Execution
The firmware uses **zero `malloc` in the inference loop**, static ping-pong buffers, Flash-mapped model data, and a fixed dual-core FreeRTOS task split. This guarantees bounded, predictable timing — essential for real-time audio.

### Principle 4: Robustness Through Gating & Voting
Single-window predictions are noisy. The system layers two defenses:
- **Noise/silence gating** via dynamic AC RMS thresholding to prevent false triggers during quiet or ambient intervals.
- **Debounced majority voting** over a 5-window rolling buffer with a ≥70% confidence gate, eliminating transient false positives.

---

## 4. End-to-End Execution Flow

This is the complete sequence of events from acoustic wave to final siren alert, executed continuously on the ESP32-S3:

```
[Acoustic Source]
        │
        ▼
[INMP441 MEMS Mic 1 (Left)  ── L/R tied to GND]
[INMP441 MEMS Mic 2 (Right) ── L/R tied to 3.3V]
        │  (I2S bus, shared SCK/WS/SD, 16 kHz stereo)
        ▼
┌──────────────────────────────────────────────────────────────┐
│  CORE 0 — I2S DMA Capture                                    │
│  • 32-bit interleaved stereo at 16,000 Hz                    │
│  • DMA writes into 2×2×16640 Int16 ping-pong buffers (133 KB)│
│  • De-interleave: even slot → Left, odd slot → Right (>>16)  │
└──────────────────────────────────────────────────────────────┘
        │ (full stereo frame ready)
        ▼
┌──────────────────────────────────────────────────────────────┐
│  CORE 1 — Signal Processing & Inference                      │
│                                                              │
│  Step A: RMS Metering (silence/noise gating)                 │
│    • Per-channel AC RMS computed from the stereo block.      │
│    • If RMS < threshold → classify as Background Noise,      │
│      skip inference (saves power & avoids false positives).  │
│                                                              │
│  Step B: TDOA Cross-Correlation (Direction of Arrival)       │
│    • dsps_corr_f32 (ESP-DSP) over Left/Right channels.       │
│    • Lag τ with max correlation → direction:                  │
│        τ < -thr  →  LEFT                                    │
│        τ > +thr  →  RIGHT                                   │
│        |τ| ≤ thr  →  CENTER                                 │
│    • Cross-correlation vectors averaged across windows for   │
│      stability in resonant environments.                    │
│                                                              │
│  Step C: MFCC Feature Extraction                            │
│    • Run on the channel with higher RMS volume.              │
│    • Pre-emphasis (α=0.97)                                   │
│    • Hamming window + 512-pt real FFT, 256 hop, 64 frames    │
│    • 40 Mel filterbanks (20 Hz – 8000 Hz)                    │
│    • Log power spectrum (-80 dB floor)                       │
│    • DCT-II → 13 MFCC coefficients                          │
│    • Output: (64, 13) MFCC spectrogram                       │
│                                                              │
│  Step D: Standardization                                     │
│    • Apply model_config.h affine constants (mu, std)         │
│    • Same standardization used during Python training.       │
│                                                              │
│  Step E: INT8 TFLite Micro Inference                        │
│    • Input: standardized (64, 13) MFCC → INT8 quantized      │
│    • 1D CNN classifier → 2 outputs [Noise, Siren]            │
│    • Softmax → confidence scores                            │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  5-Window Rolling Majority Vote                             │
│  • Maintains a rolling buffer of last 5 predictions.        │
│  • Requires ≥70% confidence to accept a siren classification.│
│  • Majority vote across the 5 windows → final decision.      │
│  • Direction (LEFT/RIGHT/CENTER) carried from TDOA.          │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│  Output & Display                                            │
│  • Serial log (ESP_LOG): "🚨 SIREN DETECTED [LEFT] (0.99)"  │
│  • ST7735S SPI TFT (128×128 RGB) renders live status,        │
│    confidence, DoA arrow, and RMS meters.                    │
└──────────────────────────────────────────────────────────────┘
```

### Timing Budget (per ~1.04 s inference block)

The inference block length is exactly `N_WIN_SAMPLES = (64-1) × 256 + 512 = 16,640 samples` ≈ 1.04 seconds of audio at 16 kHz. The processing completes well within that window (sub-15 ms inference), meaning the pipeline runs in real time — it never falls behind the audio capture rate.

---

## 5. Technology Stack by Layer

| Layer | Technology | Role |
|---|---|---|
| **Hardware (MCU)** | ESP32-S3-WROOM-1-N16R8 | Dual-core Xtensa LX7 @ 240 MHz, 16 MB Flash, 8 MB Octal PSRAM |
| **Microphones** | 2× INMP441 I2S MEMS | Stereo acoustic capture for DoA |
| **Display** | ST7735S SPI TFT (128×128) | Real-time visual output |
| **Embedded Framework** | ESP-IDF (via PlatformIO) | RTOS, drivers, build system |
| **Embedded RTOS** | FreeRTOS | Dual-core task scheduling |
| **Audio Capture** | I2S DMA driver | 16 kHz, 32-bit stereo lossless |
| **DSP Library** | ESP-DSP (`dsps_corr_f32`) | Hardware-optimized cross-correlation for TDOA |
| **Inference Engine** | TensorFlow Lite for Microcontrollers (esp-tflite-micro) | INT8 on-device inference |
| **Managed Components** | espressif__esp-dsp, espressif__esp-tflite-micro, espressif__esp-nn | PlatformIO-managed ESP AI/DSP libraries |
| **ML Training (Desktop)** | Python 3.11, TensorFlow/Keras | Model architecture, training, evaluation |
| **Feature Ref (Desktop)** | Librosa | 64-bit MFCC reference for parity validation |
| **Data Handling** | NumPy | Array manipulation, feature storage (.npy) |
| **Augmentation** | SpecAugment (time/freq masking) + jitter | Training robustness |
| **Quantization** | TensorFlow Lite Converter (full INT8 PTQ) | Model compression for edge |
| **Build System** | PlatformIO Core / VS Code PlatformIO IDE | Cross-compilation, flashing |
| **Target Board Profile** | `lolin_s3_pro` | Matches 16 MB Flash + 8 MB Octal PSRAM |

---

## 6. Layer 1 — Data Pipeline (Python / ML Training)

The desktop ML pipeline lives in `scripts/` and produces the trained, quantized model plus the firmware parity artifacts. It is a 5-phase pipeline:

### Phase 1: Dataset Audit (`scripts/audit_dataset.py`)
- Validates the raw audio dataset.
- Performs deduplication and integrity checks.
- Emits a clean manifest (`datasets/manifest_clean.csv`) mapping audio files to labels.

### Phase 2: Raw Data Processing (`scripts/process_raw_data.py`)
- Resamples all audio to the firmware sample rate (16 kHz).
- Normalizes audio levels.
- Organizes into `datasets/processed/siren/` and `datasets/processed/noise/`.

### Phase 3: Feature Extraction (`scripts/extract_features.py`)
- Computes MFCC features using **Librosa** with parameters **identical** to the firmware:
  - `SR = 16000`, `N_FFT = 512`, `HOP = 256`, `N_MFCC = 13`, `N_MELS = 40`
  - `FMIN = 20`, `FMAX = 8000`, `window = "hamming"`, `pre_emphasis = 0.97`
  - Window length: 64 frames ≈ 1.04 s (`N_WIN_SAMPLES = 16,640` — exactly one deployed inference block).
- Performs a **stratified, clip-level train/validation split with no data leakage**: clips are split, not windows, and each clip is assigned wholly to train or val.
- **Augmentation** (SpecAugment-style): random time-shifts, time-band masking, frequency-band masking, and Gaussian jitter (σ=0.05) with 90% augmentation probability per window.
- Standardization stats (mean μ, std σ) are computed and saved to `datasets/features/mfcc_stats.npz` — these exact constants are later baked into `model_config.h` for on-device parity.
- Outputs: `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `clip_tr.npy`, `clip_va.npy`, `mfcc_stats.npz`.

### Phase 4: Model Training (`scripts/train_model.py`)
- **Architecture:** 1D CNN over the time axis of the (64, 13) MFCC spectrogram.
- **Regularization:** L2 weight decay (1e-4) on all conv/dense layers, BatchNorm, and Dropout (0.4 / 0.2) to prevent overfitting.
- **Optimizer:** Adam, learning rate 5e-4.
- **Loss:** Sparse categorical cross-entropy.
- **Callbacks:** EarlyStopping (patience 7, restore best weights), ReduceLROnPlateau (patience 4, factor 0.5).
- **Batch size:** 32, **Epochs:** 50.
- **Evaluation is done at two levels:**
  1. **Window-level** — per 1.04 s window accuracy, precision, recall, F1.
  2. **Clip-level** — majority vote over 5 validation windows per clip (this is the real-world metric — one decision per clip, not per window).
- **Target gates** (checked after training):
  - val accuracy > 90%, siren precision > 90%, siren recall > 85%, siren F1 > 87%, train-val gap < 5%, INT8 estimate < 200 KB, clip-level acc > 90%.
- **Outputs:** `models/siren_classifier.keras`, `models/training_curves.png`, `models/confusion_matrix.png`.

### Phase 5: Quantization & Firmware Export (`scripts/quantize_model.py`)
- **Full integer post-training quantization** (inputs, outputs, and all ops quantized to INT8 — required because TFLite Micro on ESP32-S3 has no float kernel support).
- Uses a **200-sample representative dataset** for calibration (a tuned accuracy/speed tradeoff).
- **Validates the INT8 model against the full validation set** and compares to the FP32 oracle.
- Emits firmware-ready artifacts:
  - `models/model_data.cc` — the INT8 TFLite model binary as a C array, Flash-mapped in `.flash.rodata`.
  - `models/model_config.h` — standardization (μ, σ) and quantization constants.
- `scripts/gen_mfcc_test_vectors.py` generates C header test vectors (`test_vectors.h`) used by the on-device parity test.

### Auxiliary Scripts
- `scripts/decode_raw_dump.py` — decodes raw audio blocks dumped from the ESP32 (debugging, enabled via `RP_DUMP_RAW_BLOCK`).
- `scripts/test_audio.py` — standalone audio testing utility.

### Dataset Structure
```
datasets/
├── raw/                  # Original unprocessed audio (Sirens/, traffic/)
├── processed/            # 16 kHz normalized audio (siren/, noise/)
├── features/             # .npy feature arrays + stats for training
└── manifest.csv          # File → label mapping
```

---

## 7. Layer 2 — Embedded Firmware (ESP32-S3 / C / C++)

The firmware lives in `Rescue_Pulse_PIO/` and is built with PlatformIO + ESP-IDF. The source tree:

| File | Language | Role |
|---|---|---|
| `src/main.c` | C | Core orchestration: init, TDOA DoA, RMS metering, majority voting, display loop |
| `src/i2s_capture.c/.h` | C | 16 kHz, 32-bit stereo I2S DMA driver with ping-pong buffers |
| `src/mfcc.c/.h` | C | ESP-DSP accelerated MFCC extraction (mirrors Python/Librosa) |
| `src/inference.cpp/.h` | C++ | TFLite Micro bridge, tensor arena management, INT8 invocation |
| `src/model_data.cc` | C++ | Flash-mapped INT8 TFLite model binary (108 KB) |
| `src/model_config.h` | C | Standardization (μ, σ) and affine quantization constants |
| `src/mel_tables.h` | C | Precomputed Mel filterbank & DCT tables (206 KB, compile-time) |
| `src/test_vectors.h` | C | Pre-computed Python reference vectors for parity testing |
| `src/display_st7735s.c/.h` | C | ST7735S SPI TFT (128×128) display driver |
| `platformio.ini` | — | Build flags, board config, lib dependencies |
| `partitions_16MB.csv` | — | 16 MB flash partition table |

### Key Build Flags (`platformio.ini`)
- `-DBOARD_HAS_PSRAM` — enables 8 MB Octal PSRAM for the tensor arena.
- `-mfix-esp32-psram-cache-issue` — ESP32-S3 PSRAM cache workaround.
- `-DRP_PARITY_TEST` — enables the on-device MFCC-vs-Python parity test at boot.
- `-DRP_DUMP_RAW_BLOCK` (commented) — optional raw audio block dumping for debugging.

### Managed Components (PlatformIO)
- `espressif__esp-dsp` — DSP routines, notably `dsps_corr_f32` for TDOA cross-correlation.
- `espressif__esp-tflite-micro` — TensorFlow Lite for Microcontrollers runtime.
- `espressif__esp-nn` — ESP-NN optimized neural network kernels.

---

## 8. Signal Processing & Direction of Arrival

### 8.1 Dual-Channel Capture & De-interleaving
The I2S peripheral captures 32-bit interleaved stereo slots at 16,000 Hz into DMA scratch buffers. The capture driver unpacks each frame:
- **Left Channel (Mic 1):** even slot index, arithmetic right-shifted `>> 16`.
- **Right Channel (Mic 2):** odd slot index, arithmetic right-shifted `>> 16`.

This produces two synchronized mono Int16 streams.

### 8.2 Time Difference of Arrival (TDOA)
Sound reaching two microphones separated by distance *d* arrives at slightly different times. The lag τ that maximizes the normalized cross-correlation

```
R_LR(τ) = Σ_n x_L[n + τ] · x_R[n]
```

reveals the inter-microphone delay. Using the hardware-accelerated `dsps_corr_f32` from ESP-DSP:
- **τ < −threshold** → left microphone received the wavefront first → **source on LEFT**.
- **τ > +threshold** → right microphone received the wavefront first → **source on RIGHT**.
- **|τ| ≤ threshold** → frontal / equidistant arrival → **source in CENTER**.

Cross-correlation vectors are averaged across multiple windows to ensure stability in resonant environments.

### 8.3 MFCC Feature Extraction
Inference runs on the **channel with higher RMS volume** (the dominant signal). The MFCC pipeline mirrors the Python/Librosa reference exactly:
1. **Pre-emphasis:** `y[n] = x[n] − 0.97·x[n−1]` (resets per window).
2. **Hamming window & STFT:** 512-point real FFT, 256-sample hop, over 64 frames.
3. **Mel filterbank:** 40 triangular filters spanning 20 Hz–8000 Hz.
4. **Log power spectrum:** logarithmic scaling with −80 dB floor.
5. **DCT-II:** orthogonal projection to 13 MFCC coefficients.
6. **Output:** (64, 13) MFCC spectrogram.

The result is standardized with the training-set μ/σ before being fed to the classifier.

---

## 9. Neural Network Architecture & Quantization

### 9.1 Model Architecture (1D CNN)
```
Input: (64, 13) standardized MFCC spectrogram
  │
  ├─► Conv1D (64 filters, kernel=3, padding='same') + BatchNorm + ReLU + MaxPool(2)   → (32, 64)
  ├─► Conv1D (128 filters, kernel=3, padding='same') + BatchNorm + ReLU + MaxPool(2)  → (16, 128)
  ├─► Conv1D (128 filters, kernel=3, padding='same') + BatchNorm + ReLU + Dropout(0.3)
  ├─► GlobalAveragePooling1D                                                          → (128,)
  ├─► Dense (64 units) + ReLU + Dropout(0.4)
  └─► Dense (2 units, Softmax)                                                        → [Noise, Siren]
```

**Design rationale:** Conv1D over the time axis is a cheap, effective temporal pattern detector. GlobalAveragePooling keeps the parameter count tiny; BatchNorm stabilizes training; Dropout + L2 regularization fight overfitting. The architecture was intentionally kept small to satisfy the INT8 < 200 KB deployment target.

### 9.2 Quantization & Memory Budget
- **Model type:** TensorFlow Lite INT8 — **full integer** post-training quantization (all ops + I/O quantized, since TFLite Micro on ESP32-S3 has no float kernel support).
- **Calibration:** 200-sample representative dataset.
- **Model flash footprint:** 108,392 bytes (≈ 108 KB), placed in `.flash.rodata`.
- **Validation:** INT8 model checked against the FP32 oracle on the full validation set; no accuracy loss observed.

---

## 10. Dual-Core FreeRTOS Execution Model

The ESP32-S3 is a dual-core Xtensa LX7. The work is split to guarantee real-time audio capture without ever dropping samples:

| Core | Responsibility | Why here |
|---|---|---|
| **Core 0** | Lossless stereo I2S DMA capture into ping-pong buffers | Audio capture is I/O-bound and must never be starved by compute |
| **Core 1** | RMS metering, TDOA cross-correlation, MFCC extraction, TFLite inference, majority voting | Compute-bound DSP + NN workload, isolated from the capture path |

This separation means that even if inference momentarily spikes, the I2S DMA keeps filling buffers on Core 0 without data loss.

### Deterministic Memory Architecture
- **Zero `malloc`** inside FreeRTOS task execution loops.
- **Static ping-pong buffers** (2×2×16640 Int16 = 133 KB) for stereo capture.
- **Flash-mapped model data** — the 108 KB model lives in flash, not RAM.
- **TFLite tensor arena (200 KB)** allocated in external Octal PSRAM, keeping the 328 KB of internal DRAM free for audio buffers.

---

## 11. Hardware Configuration

### Bill of Materials

| Component | Specification | Function |
|---|---|---|
| **MCU** | ESP32-S3-WROOM-1-N16R8 (Edgehax N16R8 Pro) | Dual-core Xtensa LX7 @ 240 MHz, 16 MB Flash, 8 MB Octal PSRAM |
| **Microphone 1** | INMP441 I2S MEMS Microphone | Left acoustic sensor |
| **Microphone 2** | INMP441 I2S MEMS Microphone | Right acoustic sensor |
| **Display** | ST7735S SPI TFT (128×128 RGB) | Real-time status, DoA, confidence, RMS meters |
| **Power** | 3.3V regulated power rail | Low-noise analog/digital supply |

### Dual INMP441 Microphones (I2S Audio Bus)

| Signal | Mic 1 (Left) | Mic 2 (Right) | ESP32-S3 Pin | Function |
|---|---|---|---|---|
| VDD | 3.3V | 3.3V | 3.3V | Power supply |
| GND | GND | GND | GND | Common ground |
| SCK / BCLK | SCK | SCK | GPIO 15 | Bit Clock (shared) |
| WS / LRCLK | WS | WS | GPIO 16 | Word Select (shared) |
| SD / DOUT | SD | SD | GPIO 17 | Serial Data (shared single DIN) |
| L/R | Tied to GND | Tied to 3.3V | — | Hardware slot selection |

**How shared SD works:** In standard Philips I2S, Mic 1 drives the data bus during the Left slot (WS LOW) and tri-states during the Right slot (WS HIGH); Mic 2 drives during the Right slot and tri-states during the Left slot. Both microphones share a single data pin (GPIO 17).

### ST7735S SPI TFT Display (128×128 RGB)

| ST7735S Pin | ESP32-S3 GPIO | Function |
|---|---|---|
| SCL / SCLK | GPIO 12 | SPI Clock |
| SDA / MOSI | GPIO 11 | SPI Master Out / Data |
| DC / RS | GPIO 10 | Data / Command Select |
| CS | GPIO 9 | Chip Select (or GND if none) |
| RST / RES | GPIO 8 | Hardware Reset |
| BLK / LED | GPIO 7 | Backlight (3.3V) |
| VCC | 3.3V | Power supply |
| GND | GND | Ground |

---

## 12. Memory Architecture & Resource Budget

| Resource | Size | Usage |
|---|---|---|
| Model in Flash (`.flash.rodata`) | 108 KB | INT8 TFLite model binary |
| `s_audio_buf` (ping-pong stereo) | 133 KB | 2×2×16640 Int16 DMA buffers |
| TDOA / MFCC static buffers | ~18 KB | DSP working memory |
| Stacks (Core 0 + Core 1) | 20 KB total | FreeRTOS task stacks |
| **Total internal DRAM** | **250 KB / 328 KB (76.4%)** | >77 KB headroom remaining |
| TFLite tensor arena | 200 KB | Allocated in external Octal PSRAM (8 MB) |

The system leaves comfortable headroom for future features (e.g., OTA updates, WiFi connectivity) without breaching the real-time budget.

---

## 13. Validation & Parity Testing

### On-Device Mathematical Parity Test
The firmware includes an automated test harness (`RP_PARITY_TEST` build flag) that validates the C MFCC implementation against pre-computed Python Librosa reference vectors at boot.

**Boot serial output (passing):**
```
I (xxx) rescuepulse: Siren Test: L2 Error = 0.00341 (PASS) [3.4 ms]
I (xxx) rescuepulse: Siren Inference: Predicted 1 (Expected 1) - PASS [scores 0.0118 / 0.9882]
I (xxx) rescuepulse: Noise Test: L2 Error = 0.00412 (PASS) [3.3 ms]
I (xxx) rescuepulse: Noise Inference: Predicted 0 (Expected 0) - PASS [scores 0.9921 / 0.0079]
```

The C MFCC extraction achieves **<1% L2 relative error** versus the 64-bit Python reference — confirming that what the model learned on the desktop is exactly what the ESP32 computes.

### Desktop Validation
- Window-level accuracy, precision, recall, F1 are computed per 1.04 s window.
- **Clip-level majority vote** evaluation over 5 validation windows per clip reflects real-world performance (one decision per clip).
- Confusion matrix and training curves are saved to `models/`.

---

## 14. Current Achievement Status

### What Has Been Achieved
RescuePulse is a **fully functional, end-to-end edge AI system**. The following are complete and verified:

1. **Real-time acoustic classification** — The INT8 CNN reliably distinguishes emergency sirens from heavy urban noise on-device.
2. **Direction of arrival** — Dual-microphone TDOA via ESP-DSP cross-correlation localizes siren source to LEFT / RIGHT / CENTER.
3. **Mathematical parity** — C MFCC matches Python Librosa to <1% L2 error; verified via on-device test harness.
4. **Lossless INT8 quantization** — 108 KB quantized model with no accuracy loss vs the FP32 baseline.
5. **Dual-core real-time pipeline** — Core 0 capture + Core 1 inference; sub-15 ms per-frame latency; no dropped audio.
6. **Deterministic memory** — Zero malloc in the hot path; static ping-pong buffers; flash-mapped model; PSRAM arena.
7. **Robustness features** — RMS silence/noise gating + 5-window majority vote with ≥70% confidence gating.
8. **Live hardware validation** — Demonstrated on physical ESP32-S3 hardware with dual INMP441 microphones and ST7735S display (see serial logs and asset photos in the repo).
9. **Complete training pipeline** — Audited dataset, processed audio, feature extraction with augmentation, trained model, quantized model, and exported firmware artifacts.

### Live Hardware Execution Logs (Evidence)
Serial capture from an ESP32-S3 running live dual-microphone inference:
```
I (1410146) rescuepulse: 🔇 Background Noise [0/5] (Conf: 0.91) [RMS L:0.028 R:0.031]
W (1415336) rescuepulse: 🚨 SIREN DETECTED [LEFT] (Conf: 0.99) [3/5] [RMS L:0.087 R:0.043, Lag: -4, MaxPCM: 9100]
W (1420546) rescuepulse: 🚨 SIREN DETECTED [LEFT] (Conf: 0.88) [4/5] [RMS L:0.082 R:0.047, Lag: -4, MaxPCM: 8092]
W (1425746) rescuepulse: 🚨 SIREN DETECTED [RIGHT] (Conf: 1.00) [5/5] [RMS L:0.091 R:0.165, Lag: 5, MaxPCM: 15097]
W (1430936) rescuepulse: 🚨 SIREN DETECTED [RIGHT] (Conf: 0.98) [5/5] [RMS L:0.066 R:0.090, Lag: 5, MaxPCM: 9646]
W (1436146) rescuepulse: 🚨 SIREN DETECTED [CENTER] (Conf: 0.97) [5/5] [RMS L:0.083 R:0.060, Lag: -1, MaxPCM: 11096]
W (1441346) rescuepulse: 🚨 SIREN DETECTED [CENTER] (Conf: 0.98) [5/5] [RMS L:0.094 R:0.064, Lag: -1, MaxPCM: 10679]
W (1446536) rescuepulse: 🚨 SIREN DETECTED [CENTER] (Conf: 0.84) [5/5] [RMS L:0.041 R:0.042, Lag: 0, MaxPCM: 6456]
I (1451746) rescuepulse: 🔇 Background Noise [1/5] (Conf: 0.93) [RMS L:0.037 R:0.040]
```

These logs demonstrate the full system working in real time: background noise gating, siren detection with high confidence (0.84–1.00), correct DoA classification with consistent lags (−4 LEFT, +5 RIGHT, −1/0 CENTER), and the 5-window majority vote counter.

### Model Training Outcomes
Training curves and confusion matrix are saved as `models/training_curves.png` and `models/confusion_matrix.png` and demonstrate the model converges with minimal overfitting (train-val gap < 5%) and strong classification performance.

### Achievement Summary Table

| Capability | Status |
|---|---|
| Siren classification (edge, INT8) | ✅ Complete & verified |
| Direction of arrival (dual-mic TDOA) | ✅ Complete & verified |
| Python ↔ C MFCC parity (<1% L2) | ✅ Complete & verified |
| INT8 quantization (no accuracy loss) | ✅ Complete & verified |
| Dual-core FreeRTOS real-time pipeline | ✅ Complete & verified |
| Deterministic memory (no malloc) | ✅ Complete & verified |
| Noise/silence gating + majority voting | ✅ Complete & verified |
| Live hardware demonstration | ✅ Complete & verified |
| Training pipeline (data → quantized model) | ✅ Complete & verified |
| ST7735S TFT live display | ✅ Complete & verified |

---

## 15. Repository Structure

```
RescuePulse/
├── Rescue_Pulse_PIO/            # PlatformIO ESP32-S3 firmware project
│   ├── src/
│   │   ├── main.c               # Core orchestration, TDOA DoA, RMS metering, voting
│   │   ├── i2s_capture.c/.h      # 16 kHz 32-bit stereo I2S DMA driver
│   │   ├── mfcc.c/.h             # ESP-DSP accelerated MFCC extraction
│   │   ├── inference.cpp/.h      # TFLite Micro C++ bridge & tensor arena
│   │   ├── model_data.cc         # Flash-mapped INT8 TFLite model binary
│   │   ├── model_config.h        # Standardization & quantization constants
│   │   ├── mel_tables.h          # Precomputed filterbank & DCT tables
│   │   ├── test_vectors.h        # Pre-computed validation test vectors
│   │   ├── display_st7735s.c/.h  # ST7735S SPI TFT display driver
│   ├── platformio.ini           # Build flags, board config, dependencies
│   └── partitions_16MB.csv      # 16 MB flash partition configuration
│
├── datasets/                    # Audio datasets & metadata manifests
│   ├── raw/                     # Original audio (Sirens/, traffic/)
│   ├── processed/               # 16 kHz normalized (siren/, noise/)
│   ├── features/                # .npy feature arrays + stats
│   └── manifest.csv             # File → label mapping
│
├── models/                      # Trained Keras models & quantization artifacts
│   ├── siren_classifier.keras   # Trained FP32 Keras model
│   ├── siren_classifier_quantized.tflite  # INT8 quantized model
│   ├── model_data.cc            # Firmware-ready model binary (C array)
│   ├── model_config.h           # Standardization/quantization constants
│   ├── training_curves.png       # Accuracy/loss training plots
│   └── confusion_matrix.png     # Confusion matrix visualization
│
├── scripts/                     # Python ML training & validation pipeline
│   ├── audit_dataset.py         # Audio dataset validation & deduplication
│   ├── process_raw_data.py      # Resampling & audio normalization
│   ├── extract_features.py      # Batch MFCC extraction with SpecAugment
│   ├── train_model.py           # Keras 1D CNN training with clip-level voting
│   ├── quantize_model.py        # Post-training INT8 TFLite quantization
│   ├── gen_mfcc_test_vectors.py # C header generation for parity tests
│   ├── decode_raw_dump.py       # Decode raw audio blocks from ESP32
│   └── test_audio.py            # Standalone audio testing utility
│
├── assets/                      # Schematics, logs, and documentation media
└── README.md
```

---

## 16. Build & Deployment Procedure

### Prerequisites
- PlatformIO Core (CLI) or VS Code PlatformIO IDE.
- USB-C cable connected to the ESP32-S3 USB/JTAG port.
- ESP32-S3-WROOM-1-N16R8 board (or LOLIN S3 Pro equivalent).

### Compilation & Flashing
```bash
cd Rescue_Pulse_PIO
pio run -t clean && pio run       # clean and compile
pio run -t upload                 # flash to the ESP32-S3
pio device monitor -b 115200     # open serial monitor (115200 baud)
```

### Offline Mathematical Parity Verification
```bash
pio run -t upload -e edgehax_esp32s3_pro
```
On boot (with `RP_PARITY_TEST` enabled in `platformio.ini`), the serial console prints the MFCC L2 error and inference pass/fail against known vectors.

### Reproducing the ML Pipeline (Desktop)
```bash
python scripts/audit_dataset.py        # 1. Audit & dedupe dataset
python scripts/process_raw_data.py     # 2. Resample & normalize to 16 kHz
python scripts/extract_features.py     # 3. MFCC extraction + augmentation
python scripts/train_model.py          # 4. Train the 1D CNN
python scripts/quantize_model.py       # 5. INT8 quantize & export firmware artifacts
python scripts/gen_mfcc_test_vectors.py # 6. Generate on-device parity test vectors
```

---

## 17. Limitations & Future Work

### Current Limitations
- **DoA resolution** is coarse (3-zone: LEFT / RIGHT / CENTER) — sufficient for alerting but not for precise angular bearing. This is a hardware constraint of a 2-microphone baseline.
- **Detection is siren vs noise (binary).** Multi-class siren type (ambulance vs fire vs police) is not yet a model output.
- **No wireless connectivity** — the system currently logs over serial and displays on the local TFT; no cloud or smartphone alerting yet.

### Planned Future Enhancements
Per the project roadmap, the following are planned next steps:
- **WiFi connectivity** — push alerts to the cloud or a local dashboard.
- **Bluetooth (BLE) notifications** — alert a paired smartphone.
- **OTA firmware updates** — update the INT8 model over-the-air.
- **Finer DoA resolution** — exploring additional microphones or refined correlation methods.
- **Unit tests for firmware** — formal test coverage for ESP32 code modules.
- **GitHub Actions CI/CD** — automated builds and testing.
- **Pre-built release binaries** — v1.0.0 with pre-built firmware for non-developers.
- **Hardware schematics** — formal circuit diagrams and PCB designs.

---

## 18. License

RescuePulse is licensed under the **Apache 2.0 License**.

---

### Acknowledgements for the Documentation Team

When preparing the final project report from this document, the following are the key technical claims that should be emphasized and that are backed by evidence in the repository:

1. **"Zero cloud dependency, sub-15 ms per-frame latency"** — backed by the dual-core FreeRTOS split and the absence of any network code.
2. **"<1% L2 relative error vs Python Librosa"** — backed by the `RP_PARITY_TEST` harness and `test_vectors.h`.
3. **"No accuracy loss from INT8 quantization"** — backed by `quantize_model.py`'s oracle comparison and the 108 KB flash footprint.
4. **"Deterministic memory / zero malloc in the hot path"** — backed by the static ping-pong buffer design and PSRAM arena.
5. **"Clip-level majority voting"** — the real-world metric; single-window predictions are intentionally not the final output.

The training metrics plots (`models/training_curves.png`, `models/confusion_matrix.png`) and the live hardware photos in `assets/` should be embedded directly into the final report as visual evidence.
