# RescuePulse Firmware Architecture (`src/`)

This directory contains the ESP-IDF and FreeRTOS firmware running on the ESP32-S3 microcontroller. The firmware executes continuous dual-microphone audio sampling, Time Difference of Arrival (TDOA) Direction of Arrival estimation, real-time MFCC feature extraction, and INT8 TensorFlow Lite for Microcontrollers inference with zero runtime dynamic allocations.

---

## Technical Stack & Libraries

| Layer | Component | Details |
|---|---|---|
| **Build Framework** | PlatformIO with ESP-IDF v6 | Target: `edgehax_esp32s3_pro` (`lolin_s3_pro` board profile) |
| **DSP Acceleration** | Espressif ESP-DSP (`esp-dsp ^1.4.0`) | `dsps_fft2r_fc32` real FFT, `dsps_corr_f32` cross-correlation |
| **Neural Inference** | TensorFlow Lite for Microcontrollers | 108 KB quantized INT8 model, 200 KB PSRAM tensor arena |
| **Operating System** | FreeRTOS (Dual-Core SMP) | Core 0: I2S DMA Capture; Core 1: Feature Extraction & Inference |
| **Peripheral Driver** | ESP32-S3 `i2s_std` Driver | 16 kHz, 32-bit slot width, Stereo DMA RX |

---

## Dual-Core Pipeline Architecture

The firmware utilizes both Xtensa LX7 cores to ensure zero sample drops during neural inference:

```
┌────────────────────────────── Core 0 ──────────────────────────────┐
│  audio_capture_task  (Priority 5, 4 KB Stack)                      │
│                                                                    │
│  i2s_capture_read_stereo(chunk_l, chunk_r, 256)                    │
│      ├── Read 32-bit interleaved slots from DMA                    │
│      ├── Unpack Slot 0 (>>16) -> Left Channel (Mic 1)              │
│      └── Unpack Slot 1 (>>16) -> Right Channel (Mic 2)             │
│                                                                    │
│  When 16,640 samples accumulated:                                  │
│      s_rd = s_wr;  s_wr = 1 - s_wr;                                │
│      xSemaphoreGive(s_block_ready);                                │
└────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ (Binary Semaphore)
┌────────────────────────────── Core 1 ──────────────────────────────┐
│  inference_task  (Priority 4, 16 KB Stack)                         │
│                                                                    │
│  xSemaphoreTake(s_block_ready, portMAX_DELAY)                       │
│      ├── 1. Compute True AC RMS (DC-subtracted) on Left & Right    │
│      ├── 2. RMS Gating: Skip inference if max(RMS_L, RMS_R) < 0.02│
│      ├── 3. Estimate DoA: ESP-DSP Cross-Correlation (TDOA)        │
│      │      -> Peak lag gives LEFT, RIGHT, or CENTER               │
│      ├── 4. MFCC Feature Extraction on the louder channel (ESP-DSP)│
│      ├── 5. INT8 Affine Quantization                               │
│      ├── 6. TFLite Micro Inference (1D CNN)                        │
│      └── 7. 5-Window Debounced Majority Vote                       │
└────────────────────────────────────────────────────────────────────┘
```

---

## Real-World Serial Output

```text
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

---

## Key Algorithms and Implementation Details

### 1. Dual-Channel Stereo Ping-Pong Buffers
To ensure safe, lockless data exchange between Core 0 and Core 1 without memory allocation:

```c
/* Dimension 1: Ping-Pong (2), Dimension 2: Channel L/R (2), Dimension 3: Samples */
static int16_t s_audio_buf[2][2][N_SAMPLES]; /* 2 * 2 * 16640 * 2 = 133,120 bytes */
static volatile int s_wr = 0;
static volatile int s_rd = 0;
```

### 2. Time Difference of Arrival (TDOA) Direction Estimation
Using `dsps_corr_f32`, cross-correlation is averaged across multiple time segments within the active audio window:

```c
#define TDOA_MAX_LAG        32
#define TDOA_PAT_LEN        1024
#define TDOA_SIG_LEN        (TDOA_PAT_LEN + 2 * TDOA_MAX_LAG) /* 1088 */
#define TDOA_LAG_THRESHOLD  2

/* ESP-DSP Cross-Correlation: Signal (Left Channel), Pattern (Right Channel) */
esp_err_t err = dsps_corr_f32(s_tdoa_sig, TDOA_SIG_LEN, s_tdoa_pat, TDOA_PAT_LEN, s_tdoa_corr);
```
- A peak at `lag > +2` indicates the sound reached Mic 2 (Right) first $\rightarrow$ **DOA_RIGHT**.
- A peak at `lag < -2` indicates the sound reached Mic 1 (Left) first $\rightarrow$ **DOA_LEFT**.
- A peak within $[-2, +2]$ indicates approximately equal arrival time $\rightarrow$ **DOA_CENTER**.

### 3. Louder-Channel Selective Inference
To preserve classification accuracy even when the siren is strongly offset to one side:
1. True AC RMS is calculated independently for both `buf_l` and `buf_r`.
2. The channel with higher RMS volume is routed into `mfcc_extract_block()`.

### 4. Deterministic Memory Footprint
- Model weights (`g_model_data[]` ~108 KB) are mapped to **Flash RoData** (`.flash.rodata`).
- All internal buffers are statically allocated at link time:
  - SRAM Utilization: **76.4% (250 KB / 328 KB)**.
  - Headroom: **$>77\text{ KB}$ internal SRAM free**.
  - PSRAM: **8 MB Octal PSRAM** hosting the 200 KB TFLite tensor arena.

---

## File Reference

| File | Purpose |
|---|---|
| [`main.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/main.c) | System orchestration, FreeRTOS tasks, TDOA direction estimation, AC RMS volume metering, majority voting. |
| [`display_st7789.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/display_st7789.c) / [`display_st7789.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/display_st7789.h) | ST7789 SPI TFT display driver (240x240 RGB565) rendering live directional alerts, noise classification, and RMS bars. |
| [`i2s_capture.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/i2s_capture.c) / [`i2s_capture.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/i2s_capture.h) | 16 kHz 32-bit stereo I2S DMA driver, unpacking Left (Mic 1) and Right (Mic 2) PCM samples. |
| [`mfcc.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/mfcc.c) / [`mfcc.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/mfcc.h) | ESP-DSP accelerated MFCC pipeline (0.97 pre-emphasis, Hamming window, 512-pt FFT, 40 Mel filters, DCT-II). |
| [`inference.cpp`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/inference.cpp) / [`inference.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/inference.h) | C++ bridge for TensorFlow Lite for Microcontrollers, memory allocation, and interpreter invocation. |
| [`model_data.cc`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/model_data.cc) | Flash-mapped INT8 quantized TFLite model byte array. |
| [`model_config.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/model_config.h) | Audio constants, per-coefficient standardization parameters, and quantization affine scales. |
| [`mel_tables.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/mel_tables.h) | Precomputed Librosa-matching Hamming window, Mel filterbank, and DCT matrix tables. |
| [`test_vectors.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/test_vectors.h) | Pre-computed siren and noise test vectors for automated on-device mathematical parity testing. |