# 🚑 Rescue Pulse: Edge AI Siren Detection & Direction System

**Rescue Pulse** is an edge-computing IoT device designed to help traffic officers detect approaching emergency vehicles in high-noise urban environments. Using dual MEMS microphones and an ESP32-S3, the system classifies sirens in real-time and calculates the direction of approach (Left/Right) using GCC-PHAT TDOA, alerting officers *before* the vehicle reaches the intersection.

---

### 🧠 System Architecture
* **Dual-Core Parallelism (FreeRTOS):**
  * **Core 0:** Non-blocking I2S DMA audio capture (Zero-copy).
  * **Core 1:** DSP (MFCC extraction), CNN Inference (TFLite Micro), and TDOA calculation.
* **Dual ML Training Pipeline:**
  * **Track A (Custom):** Python (TensorFlow/Librosa) → int8 Quantization → ESP-NN.
  * **Track B (Rapid):** Edge Impulse Studio for rapid prototyping and validation.

### 🛠️ Hardware Stack
* **MCU:** ESP32-S3 N16R8 Pro (Dual-core Xtensa LX7 @ 240MHz, 8MB PSRAM)
* **Audio Inputs:** 2× INMP441 I2S MEMS Microphones (15cm spacing)
* **Outputs:** SSD1306 OLED (128x64) + Directional LED Array
* **Wireless:** Wi-Fi (MQTT Cloud Logging) + BLE 5.0 (Phone Notifications)

### 💻 Software Stack
* **Firmware:** ESP-IDF 5.1+ (C/C++)
* **ML Frameworks:** TensorFlow Lite for Microcontrollers, ESP-DSP
* **Python Pipeline:** Librosa, TensorFlow 2.x, Scikit-learn

---

### 📂 Repository Structure
```text
rescue-pulse/
├── firmware/          # ESP-IDF Project (Core 0 & Core 1 tasks)
├── ml_pipeline/       # Python scripts for Track A (Preprocess, Train, Quantize)
├── edge_impulse/      # Track B C++ Library exports
├── docs/              # Circuit diagrams, block diagrams, and reports
└── README.md
