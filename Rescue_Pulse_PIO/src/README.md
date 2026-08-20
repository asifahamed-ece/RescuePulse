# Rescue Pulse — Firmware Source (`src/`)

This directory contains the ESP32-S3 firmware that runs the real-time siren-detection pipeline. It captures audio from an INMP441 I2S microphone, extracts MFCC features, and runs a quantized int8 CNN via TensorFlow Lite for Microcontrollers — all on-device, with no cloud dependency.

---

## 🧰 Tech Stack

| Component | Purpose |
|-----------|---------|
| **ESP-IDF v6** (via PlatformIO) | RTOS, I2S driver, build system |
| **PlatformIO** | Build/flash/monitor toolchain (`edgehax_esp32s3_pro` env) |
| **ESP-DSP** (`espressif/esp-dsp ^1.4.0`) | Optimized `dsps_fft2r_fc32` real FFT + bit-reversal |
| **TFLite Micro** (`espressif/esp-tflite-micro ^1.3.2`) | int8 inference runtime |
| **FreeRTOS** | Dual-core task scheduling, semaphores |

---

## 🏗️ Architecture & Logic

### Dual-Core FreeRTOS Design

The ESP32-S3 has two Xtensa LX7 cores. The pipeline is split across them to keep audio capture lossless while the ML inference runs:

```
┌─────────────────────────── Core 0 ───────────────────────────┐
│  audio_capture_task  (priority 5, 4 KB stack)                │
│                                                              │
│  i2s_capture_read(chunk, 256) ──►  De-interleave stereo slots│
│                               ──►  memcpy into buf[s_wr]     │
│                                                              │
│  When 16,640 samples collected:                              │
│      s_rd = s_wr;  s_wr = 1 - s_wr;                          │
│      xSemaphoreGive(s_block_ready);                          │
└──────────────────────────────────────────────────────────────┘
                              │  (binary semaphore)
                              ▼
┌─────────────────────────── Core 1 ───────────────────────────┐
│  inference_task  (priority 4, 16 KB stack)                   │
│                                                              │
│  xSemaphoreTake(s_block_ready)                               │
│      ──► True AC RMS volume meter (DC-subtracted)            │
│      ──► mfcc_extract_block()  (ESP-DSP FFT)                 │
│      ──► quantize_mfcc()       (int8 affine map)             │
│      ──► inference_run()       (TFLite Micro)                │
│      ──► 5-window majority vote                              │
└──────────────────────────────────────────────────────────────┘
```

- **Core 0** is dedicated to I2S DMA capture. It never blocks on ML work, so no audio samples are dropped.
- **Core 1** waits on the semaphore, then performs the full MFCC + inference chain.
- Task priorities (capture=5 > inference=4) ensure the producer always wins the CPU when both are ready.

### Ping-Pong Double Buffering

To prevent a race between the audio **producer** (Core 0) and the ML **consumer** (Core 1), two static buffers are used:

```c
static int16_t s_audio_buf[2][N_SAMPLES];   /* 2 × 16,640 int16 = 66,560 B */
static volatile int s_wr = 0;               /* buffer being filled */
static volatile int s_rd = 0;               /* buffer ready for inference */
```

**Protocol:**
1. Core 0 fills `s_audio_buf[s_wr]` in 256-sample chunks.
2. When full, Core 0 publishes it: `s_rd = s_wr; s_wr = 1 - s_wr;` then gives the semaphore.
3. Core 1 takes the semaphore, reads `s_audio_buf[s_rd]`, and processes it.
4. Meanwhile Core 0 is already filling the *other* buffer — hence "ping-pong".

Because the producer only ever writes `s_wr` and the consumer only ever reads `s_rd`, and the switch is a single atomic publish (`s_rd = s_wr`), no mutex is needed — the semaphore alone guarantees the hand-off.

### 5-Window Majority Vote & Confidence Threshold

A single 1.04-second window can produce a transient spurious classification. To debounce, the firmware accumulates predictions over **5 consecutive windows** requiring both high confidence ($\ge 70\%$) and majority agreement ($\ge 3/5$):

```c
#define VOTE_WINDOWS    5
#define VOTE_THRESH     3
#define CONF_THRESHOLD  0.70f

if (pred == 1 && confidence >= CONF_THRESHOLD) {
    vote_siren++;
}
vote_count++;

if (vote_count >= VOTE_WINDOWS) {
    bool siren = (vote_siren >= VOTE_THRESH);
    if (siren) {
        ESP_LOGW(TAG, "🚨 SIREN DETECTED [%d/%d] (conf: %.2f)", vote_siren, VOTE_WINDOWS, confidence);
    } else {
        ESP_LOGI(TAG, "🔇 Background Noise [%d/%d] (conf: %.2f)", vote_siren, VOTE_WINDOWS, confidence);
    }
    vote_siren = 0;
    vote_count = 0;
}
```

---

## ⚠️ Critical Implementation Notes (Viva & Architecture)

### 1. 32-bit I2S Slots & Stereo DMA De-interleaving

The INMP441 is a 24-bit MEMS microphone operating on standard Philips I2S. In Philips I2S mode, there are always **2 slots per audio frame (Left and Right)**. 
- With `L/R` tied to GND, audio data is transmitted during the **Left slot (Slot 0)**.
- The Right slot (Slot 1) is tristated/zero.

The ESP32-S3 I2S DMA controller captures 32-bit words for **both slots**. `i2s_capture_read()` reads $2 \times N$ 32-bit words and extracts only the even indices (Left channel):

```c
size_t frames_read = bytes_read / (2 * sizeof(int32_t));
for (size_t i = 0; i < frames_read; i++) {
    buf[i] = (int16_t)(s_scratch[2 * i] >> 16);
}
```

This prevents zero-padding between consecutive samples, preserving continuous 16 kHz sampling and spectral integrity.

### 2. ESP-DSP Packed Real-FFT Output Format

`dsps_fft2r_fc32()` computes a real FFT and stores the result in a **packed complex format**:

```
index:  0        1        2        3        ...  N-2      N-1
        Re[0]   Re[1]    Im[1]    Re[2]    ...  Re[N/2-1] Im[N/2-1]
        (DC)    └── bin 1 ──┘      └── bin 2 ──┘   ...   └── bin N/2-1 ──┘
```

`mfcc.c` unpacks this into the power spectrum:

```c
s_power[0] = s_fft[0] * s_fft[0];                       /* DC bin, Re[0] */
for (int k = 1; k < N_FFT / 2; k++) {
    float re = s_fft[2 * k];
    float im = s_fft[2 * k + 1];
    s_power[k] = re * re + im * im;
}
s_power[N_FFT / 2] = s_fft[N_FFT] * s_fft[N_FFT];       /* Nyquist bin, Re[N/2] */
```

### 3. Strict "No Heap Allocation in Tasks" Rule

All audio buffers, MFCC matrices, and inference arrays are **static**:

```c
static int16_t s_audio_buf[2][N_SAMPLES];   /* ping-pong audio */
static float   s_mfcc[N_WIN][N_MFCC];       /* MFCC output */
static int8_t  s_q_in[N_TOTAL];             /* quantized input */
static int8_t  s_q_out[2];                  /* quantized output */
```

Pre-allocating statically guarantees **deterministic memory usage and execution timing**, completely eliminating runtime heap fragmentation on the MCU.

---

## 📄 File Map

| File | Description |
|------|-------------|
| **`main.c`** | Application entry point. Initializes MFCC, TFLite, and I2S; manages dual-core tasks, AC RMS gating, quantization, and majority voting. |
| **`mfcc.c`** | MFCC feature extraction: pre-emphasis (0.97), Hamming windowing, 512-pt ESP-DSP FFT, 40-bin mel filterbank, `10·log10` dB conversion with `mmax-80` floor, and 13-coeff orthonormal DCT-II. |
| **`mfcc.h`** | Public API for `mfcc_init()` and `mfcc_extract_block()`. |
| **`inference.cpp`** | C++ bridge to TFLite Micro. Allocates the 200 KB tensor arena (PSRAM-first), registers ops in `MicroMutableOpResolver`, and invokes the model. |
| **`inference.h`** | Public API for `inference_init()` and `inference_run()`. |
| **`i2s_capture.c`** | I2S Standard Mode RX driver for INMP441: 16 kHz, 32-bit stereo slot de-interleaving, GPIO 4/5/6, DMA double-buffering. |
| **`i2s_capture.h`** | Public API for `i2s_capture_init()` and `i2s_capture_read()`. |
| **`mel_tables.h`** | Precomputed librosa-matching tables: `g_ham[512]`, `g_mel_fb[10280]`, and `g_dct[520]`. |
| **`model_config.h`** | Audio constants, standardization vectors (`g_mfcc_mu/std`), and TFLite int8 quantization scales. |
| **`model_data.cc`** | Quantized int8 TFLite model array (`g_model_data[]` ~106 KB). |
| **`test_vectors.h`** | Pre-computed PCM and MFCC test vectors for the `RP_PARITY_TEST` offline test suite. |