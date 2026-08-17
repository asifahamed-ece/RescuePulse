# 🚑 Rescue Pulse

```
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║         🚧 WIP: Building & Training in Progress 🚧        ║
  ║                                                           ║
  ║            Edge AI Siren Detection System                 ║
  ║                                                           ║
  ║     [Mic] → ESP32-S3 → MFCC → CNN → [Siren/Noise]         ║
  ║                                                           ║
  ║          ⚡ Real-time • Low Power • On-device             ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
```

## Objective

**Rescue Pulse** is an edge-computing IoT system that detects emergency vehicle sirens in real-time using machine learning on a microcontroller. The goal is to alert traffic officers before emergency vehicles reach their location by classifying audio as either **"siren"** or **"noise"**.

### Core Idea

```
   Audio Input (16kHz)
            │
            ▼
  ┌─────────────────┐
  │  Pre-emphasis   │  y[n] = x[n] - 0.97·x[n-1]
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  STFT + Mel     │  512-pt FFT, 40 Mel bins
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  MFCC (13 coeffs)│  64-frame window (~1.04s)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  Quantized CNN  │  int8 TFLite Micro
  └────────┬────────┘
           │
           ▼
     [Siren / Noise]
```

---

## 🔧 Tech Stack

**ML Pipeline (Python)**
- TensorFlow 2.x / Keras
- Librosa (audio processing)
- NumPy, Matplotlib

**Firmware (C/C++)**
- ESP-IDF 5.1+ / PlatformIO
- FreeRTOS (dual-core tasks)
- TensorFlow Lite for Microcontrollers

**Hardware Target**
- ESP32-S3 N16R8 Pro (240MHz, 16MB Flash, 8MB PSRAM)
- INMP441 I2S MEMS Microphone

---

## 📁 Project Structure

```
rescue-pulse/
├── datasets/
│   ├── manifest.csv          # Dataset metadata
│   ├── processed/            # Cleaned 16kHz WAV files
│   └── features/             # Extracted MFCC (.npy)
│
├── models/
│   ├── siren_classifier.keras
│   └── model_config.h        # Quantization params
│
├── firmware/main/
│   ├── main.c                # Entry point, test harness
│   ├── mfcc.c/h              # MFCC feature extraction
│   ├── inference.cpp/h       # TFLite inference wrapper
│   ├── model_data.cc         # Quantized model binary
│   └── model_config.h        # Audio/quant config
│
├── scripts/                  # Python ML pipeline (paths anchored to repo root)
│   ├── audit_dataset.py      # Data cleaning
│   ├── process_raw_data.py   # Audio preprocessing
│   ├── extract_features.py   # MFCC extraction
│   ├── train_model.py        # CNN training
│   ├── quantize_model.py     # int8 quantization
│   └── gen_mfcc_test_vectors.py  # Firmware test data
```

---

## 🚧 Current Status: **IN PROGRESS**

### ✅ Completed

**ML Pipeline**
- [x] Data auditing and cleaning pipeline
- [x] Audio preprocessing (16kHz resample, normalization)
- [x] MFCC feature extraction with SpecAugment
- [x] 1D CNN architecture (Conv1D → BN → Pool → Dense)
- [x] Per-coefficient standardization
- [x] Clip-level majority vote evaluation
- [x] Post-training int8 quantization

**Model Architecture**
```
Input: (64, 13) MFCC window
  │
  ├─→ Conv1D(64, 3) + BN + MaxPool
  ├─→ Conv1D(128, 3) + BN + MaxPool  
  ├─→ Conv1D(128, 3) + BN + Dropout
  ├─→ GlobalAvgPool
  ├─→ Dense(64) + Dropout
  └─→ Dense(2) softmax [noise, siren]
```

**Target Metrics**
- Val Accuracy > 90%
- Siren Precision > 90%
- Siren Recall > 85%
- Siren F1 > 87%
- Model Size < 200KB (int8)

**Firmware**
- [x] MFCC extraction parity with Python (L2 error < 1%)
- [x] TFLite Micro integration
- [x] int8 quantization parameters exported
- [x] Test vector validation harness
- [x] Real-time I2S audio capture task (dual-core, ping-pong buffering)
- [x] Live streaming inference with 5-window majority vote
- [ ] OLED display integration
- [ ] LED indication system

---

## 🧪 Key Features

**Data Pipeline**
- Stratified train/val split (80/20) at clip level
- 5-window majority vote for validation clips
- Feature-space augmentation (time-shift, SpecAugment, jitter)
- Per-coefficient standardization (train stats only)

**Model Design**
- Lightweight Conv1D for temporal pattern detection
- L2 regularization + dropout for generalization
- Global average pooling to minimize parameters
- Clip-level evaluation matching real-world use case

**Edge Deployment**
- Manual pre-emphasis filter (matches librosa removal in v0.11+)
- Fixed-point MFCC computation for ESP32
- int8 quantization with per-tensor affine mapping
- Parity testing with Python-generated test vectors

---

## 📝 Workflow

```
Raw Audio → Audit → Process → Extract → Train → Quantize → Deploy
   │           │         │         │        │         │
   ▼           ▼         ▼         ▼        ▼         ▼
 .wav files  Clean     16kHz     MFCC     CNN      int8
             labels    WAV       (64,13)  Keras    TFLite
```

---

## 🏗️ System Architecture

The project follows a **train-on-PC → deploy-on-edge** TinyML workflow:

```
┌─────────────────────────── PYTHON (PC) ───────────────────────────┐
│                                                                   │
│  Raw Audio ──► process_raw_data.py ──► audit_dataset.py           │
│  (siren/noise)    16 kHz mono 3 s WAV    SHA-256 dedup            │
│                                          (drop label conflicts)   │
│                                                                   │
│  extract_features.py ──► train_model.py ──► quantize_model.py     │
│  MFCC (64,13) windows    1D CNN (Keras)    int8 PTQ (TFLite)      │
│  + SpecAugment           + standardization  + representative set  │
│  + per-coeff stats        + clip majority    (200 samples, seed)  │
│  → mfcc_stats.npz         vote eval                               │
│                                                                   │
│  gen_mfcc_test_vectors.py                                         │
│  Dumps librosa tables + 2 end-to-end test vectors                 │
│  → mel_tables.h, test_vectors.h                                   │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼  quantize_model.py exports:
                              │  model_data.cc  (int8 TFLite bytes)
                              │  model_config.h (g_mfcc_mu/std, scale/zp)
                              ▼
┌─────────────────────────── ESP-IDF (ESP32-S3) ────────────────────┐
│                                                                   │
│  INMP441 ──► I2S DMA ──► Ping-Pong ──► MFCC ──► int8 ──► TFLite   │
│  (16 kHz)    (Core 0)     buffers      (Core 1)  quant   Micro    │
│                                                                   │
│  5-window majority vote ──► [SIREN DETECTED / NOISE]              │
└───────────────────────────────────────────────────────────────────┘
```

**Key parity guarantees** between Python and firmware:
- Identical audio front-end constants (`RP_SR`, `RP_N_FFT`, `RP_HOP`, `RP_N_MFCC`, `RP_N_MELS`, `RP_FMIN`, `RP_FMAX`, `RP_N_WIN`, `RP_PRE_EMPH`).
- Precomputed librosa tables (`g_ham`, `g_mel_fb`, `g_dct`) embedded as C arrays.
- Per-coefficient standardization constants (`g_mfcc_mu`, `g_mfcc_std`) exported from `mfcc_stats.npz`.
- TFLite int8 quantization parameters (`g_in_scale`, `g_in_zp`, `g_out_scale`, `g_out_zp`) exported from the interpreter.

---

## 🔌 Hardware Requirements

| Component | Model / Spec | Notes |
|-----------|--------------|-------|
| **MCU** | Edgehax ESP32-S3 Pro (ESP32-S3-WROOM-1-N16R8) | 240 MHz dual-core Xtensa LX7, 16 MB Flash, 8 MB Octal PSRAM |
| **Microphone** | INMP441 I2S MEMS | 24-bit, 60 dB SNR, omnidirectional |
| **Power** | 3.3 V regulated | INMP441 VDD must be clean 3.3 V |
| **Host** | Any PC with PlatformIO Core | For build, flash, and serial monitor |

### GPIO Wiring Map (INMP441 → ESP32-S3)

| INMP441 Pin | ESP32-S3 GPIO | Notes |
|-------------|---------------|-------|
| **SCK** (BCLK) | **GPIO 15** | I2S bit clock (master output) |
| **WS** (LRCLK) | **GPIO 16** | I2S word select / frame sync |
| **SD** (DIN) | **GPIO 17** | I2S serial data (mic → MCU) |
| **L/R** | **GND** | Selects left channel (LOW = left) |
| **VDD** | **3.3 V** | Power supply |
| **GND** | **GND** | Common ground |

> **Note:** The INMP441 outputs 24-bit MSB-justified audio in 32-bit I2S slots. The firmware configures the ESP32-S3 I2S peripheral with 32-bit slot width and right-shifts each sample by 16 bits (`>> 16`) to extract the 16-bit PCM used by the MFCC front-end.

---

## 🛠️ Build & Flash Guide

The firmware lives in the `Rescue_Pulse_PIO/` PlatformIO project. All commands are run from that directory.

### 1. Build

```bash
cd Rescue_Pulse_PIO
pio run
```

### 2. Flash to the board

```bash
pio run -t upload
```

### 3. Monitor serial output (115200 baud)

```bash
pio device monitor
```

### Optional: Offline parity test build

Compile with the `RP_PARITY_TEST` macro to run the embedded siren/noise test vectors (validates MFCC L2 error < 1% and inference class match) before starting the live I2S pipeline:

```bash
pio run -t upload -e edgehax_esp32s3_pro -- -DRP_PARITY_TEST
```

> The `platformio.ini` environment is `edgehax_esp32s3_pro` (board `lolin_s3_pro`, framework `espidf`, 16 MB flash, 8 MB octal PSRAM, QIO flash mode).

---

## 📅 Project Phases

- **Phase 1 — Data Collection & Preprocessing:** Gathered raw siren/noise audio; `process_raw_data.py` standardized everything to 16 kHz mono 3-second 16-bit PCM WAV; `audit_dataset.py` performed SHA-256 content-hash dedup and dropped label-contaminated clips.
- **Phase 2 — Model Training:** `extract_features.py` produced (64, 13) MFCC windows with SpecAugment + per-coefficient standardization; `train_model.py` trained a lightweight 1D CNN (Conv1D → BN → Pool → Dense) with clip-level majority-vote evaluation.
- **Phase 3 — Quantization:** `quantize_model.py` performed full int8 post-training quantization (200-sample representative set, fixed seed 42) and validated the int8 model against the float32 oracle on the full validation set.
- **Phase 4 — C-Parity Testing:** `gen_mfcc_test_vectors.py` dumped exact librosa tables (`mel_tables.h`) and two end-to-end test vectors (`test_vectors.h`); the firmware's `RP_PARITY_TEST` harness confirmed MFCC L2 error < 1% and correct inference class.
- **Phase 5 — Live I2S Streaming:** Dual-core FreeRTOS pipeline with ping-pong double buffering, real-time MFCC + TFLite Micro inference on Core 1, and 5-window majority-vote debounce for stable siren detection.

---

*Last updated: Work in Progress*
