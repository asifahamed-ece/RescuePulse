# 🚑 Rescue Pulse

```
  ╔═══════════════════════════════════════════════════════════╗
  ║                                                           ║
  ║      ██████╗ ██╗   ██╗████████╗    ███████╗               ║
  ║     ██╔═══██╗██║   ██║╚══██╔══╝    ╚══███╔╝               ║
  ║     ██║   ██║██║   ██║   ██║         ███╔╝                ║
  ║     ██║▄▄ ██║██║   ██║   ██║        ███╔╝                 ║
  ║     ╚██████╔╝╚██████╔╝   ██║       ███████╗               ║
  ║      ╚══▀▀═╝  ╚═════╝    ╚═╝       ╚══════╝               ║
  ║                                                           ║
  ║            Edge AI Siren Detection System                 ║
  ║                                                           ║
  ║     [Mic] → ESP32-S3 → MFCC → CNN → [Siren/Noise]        ║
  ║                                                           ║
  ║              ⚡ Real-time • Low Power • On-device         ║
  ║                                                           ║
  ╚═══════════════════════════════════════════════════════════╝
```

## 🎯 Objective

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
├── audit_dataset.py          # Data cleaning
├── process_raw_data.py       # Audio preprocessing
├── extract_features.py       # MFCC extraction
├── train_model.py            # CNN training
├── quantize_model.py         # int8 quantization
└── gen_mfcc_test_vectors.py  # Firmware test data
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
- [ ] Real-time I2S audio capture task
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

*Last updated: Work in Progress*
