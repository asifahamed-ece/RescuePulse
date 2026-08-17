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
│  i2s_capture_read(chunk, 512)  ──►  memcpy into buf[s_wr]    │
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
│      ──► RMS volume meter                                    │
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
1. Core 0 fills `s_audio_buf[s_wr]` in 512-sample chunks.
2. When full, Core 0 publishes it: `s_rd = s_wr; s_wr = 1 - s_wr;` then gives the semaphore.
3. Core 1 takes the semaphore, reads `s_audio_buf[s_rd]`, and processes it.
4. Meanwhile Core 0 is already filling the *other* buffer — hence "ping-pong".

Because the producer only ever writes `s_wr` and the consumer only ever reads `s_rd`, and the switch is a single atomic publish (`s_rd = s_wr`), no mutex is needed — the semaphore alone guarantees the hand-off.

### 5-Window Majority Vote (Debounce)

A single 1.04-second window can produce a spurious classification. To debounce, the firmware accumulates predictions over **5 consecutive windows** and only raises an alarm when **≥ 3** vote "siren":

```c
#define VOTE_WINDOWS 5
#define VOTE_THRESH  3

if (pred == 1) vote_siren++;
vote_count++;

if (vote_count >= VOTE_WINDOWS) {
    bool siren = (vote_siren >= VOTE_THRESH);
    /* log SIREN DETECTED / NOISE */
    vote_siren = 0;
    vote_count = 0;
}
```

This mirrors the clip-level majority-vote evaluation used during training/validation, giving ~5.2 s of temporal smoothing against transient false positives.

---

## ⚠️ Critical Implementation Notes (Viva Prep)

### 1. 32-bit I2S Slots for the INMP441

The INMP441 is a 24-bit MEMS microphone that transmits **MSB-justified** data. Its 24-bit samples are left-aligned inside a 32-bit I2S frame (the low 8 bits are zero). The firmware therefore configures the ESP32-S3 I2S peripheral with **32-bit slot width**:

```c
i2s_std_config_t std_cfg = {
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                    I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
    ...
};
```

Each 32-bit word read from the DMA buffer is then right-shifted by 16 bits to extract the 16-bit signed PCM used by the MFCC front-end:

```c
int16_t sample = (int16_t)(raw32 >> 16);
```

**Why 32-bit slots?** If we configured 16-bit slots, the I2S peripheral would only capture the *low* 16 bits of each 32-bit frame — which for the INMP441 are the zero-padding bits, producing silence. The 32-bit slot + `>> 16` shift recovers the actual 24-bit microphone data (truncated to 16-bit for the MFCC pipeline).

### 2. ESP-DSP Packed Real-FFT Output Format

`dsps_fft2r_fc32()` computes a real FFT but stores the result in a **packed complex format** to save memory. The output layout for an N-point FFT is:

```
index:  0        1        2        3        ...  N-2      N-1
        Re[0]   Re[1]    Im[1]    Re[2]    ...  Re[N/2-1] Im[N/2-1]
        (DC)    └── bin 1 ──┘      └── bin 2 ──┘   ...   └── bin N/2-1 ──┘
```

- **Index 0** = DC component (real only).
- **Index N** = Nyquist component (real only).
- **Indices 1..N-1** = interleaved real/imaginary pairs for bins 1..N/2-1.

`mfcc.c` unpacks this into a power spectrum:

```c
s_power[0] = s_fft[0] * s_fft[0];                       /* DC bin, Re[0] */
for (int k = 1; k < N_FFT / 2; k++) {
    float re = s_fft[2 * k];
    float im = s_fft[2 * k + 1];
    s_power[k] = re * re + im * im;
}
s_power[N_FFT / 2] = s_fft[N_FFT] * s_fft[N_FFT];       /* Nyquist bin, Re[N/2] */
```

Note that `dsps_bit_rev_fc32()` must be called *after* the FFT to reorder the output into natural frequency order.

### 3. Strict "No Heap Allocation in Tasks" Rule

All audio, MFCC, and inference buffers are **static, allocated at compile time** — no `malloc`/`new` is ever called inside a task:

```c
static int16_t s_audio_buf[2][N_SAMPLES];   /* ping-pong audio */
static float   s_mfcc[N_WIN][N_MFCC];       /* MFCC output */
static int8_t  s_q_in[N_TOTAL];             /* quantized input */
static int8_t  s_q_out[2];                  /* quantized output */
static float   s_fft[2 * N_FFT];            /* FFT workspace */
static float   s_power[N_FFT / 2 + 1];      /* power spectrum */
static float   s_mel[N_MELS];               /* mel energies */
static float   s_db[N_WIN][N_MELS];         /* dB spectrogram */
static float   s_y[N_SAMPLES];              /* pre-emphasized audio */
```

**Why?** On a microcontroller, heap fragmentation and allocation latency are unpredictable. A `malloc` inside a real-time task can block for an unbounded time (or fail entirely), risking dropped audio or a watchdog reset. By pre-allocating everything statically, the pipeline has **deterministic memory usage and timing**.

The only heap allocation in the entire firmware happens once, during `inference_init()` — the 200 KB TFLite tensor arena, which is deliberately placed in **PSRAM** (8 MB on the S3-Pro) with a fallback to internal heap:

```c
s_arena = (uint8_t *)heap_caps_malloc(TENSOR_ARENA_SIZE,
                                      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
```

---

## 📄 File Map

| File | Description |
|------|-------------|
| **`main.c`** | Application entry point. Initializes MFCC, TFLite, and I2S; creates the two pinned FreeRTOS tasks; implements the ping-pong buffer hand-off, the int8 quantization of MFCC features, the 5-window majority vote, and the optional `RP_PARITY_TEST` offline harness. |
| **`mfcc.c`** | MFCC feature extraction: pre-emphasis (0.97), Hamming windowing, 512-pt ESP-DSP real FFT, power spectrum unpacking, 40-bin mel filterbank, `10·log10` dB conversion with global `mmax-80` floor, and 13-coefficient DCT-II via the precomputed `g_dct` table. |
| **`mfcc.h`** | Public API for `mfcc_init()` and `mfcc_extract_block()`. |
| **`inference.cpp`** | C++ bridge to TFLite Micro. Allocates the 200 KB tensor arena (PSRAM-first), registers the 9 required ops in a `MicroMutableOpResolver<10>`, loads the embedded model, and runs `Invoke()`. |
| **`inference.h`** | Public API for `inference_init()` and `inference_run()`. |
| **`i2s_capture.c`** | I2S Standard Mode RX driver for the INMP441: 16 kHz, 32-bit slots (mono), GPIO 15/16/17, 6 DMA descriptors × 240 frames, `auto_clear=true`. Provides blocking `i2s_capture_read()`. |
| **`i2s_capture.h`** | Public API for `i2s_capture_init()` and `i2s_capture_read()`. |
| **`mel_tables.h`** | Auto-generated by `gen_mfcc_test_vectors.py`. Contains the exact librosa tables: `g_ham[512]` (periodic Hamming), `g_mel_fb[10280]` (40×257 Slaney mel filterbank), `g_dct[520]` (13×40 orthonormal DCT-II). |
| **`model_config.h`** | Auto-generated by `quantize_model.py`. Audio front-end constants (`RP_*`), per-coefficient standardization (`g_mfcc_mu[13]`, `g_mfcc_std[13]`), and TFLite int8 quantization params (`g_in_scale`, `g_in_zp`, `g_out_scale`, `g_out_zp`). |
| **`model_data.cc`** | Auto-generated by `quantize_model.py`. Embeds the full int8 TFLite model bytes as `g_model_data[]` (108,392 bytes ≈ 106 KB) plus `g_model_data_len`. |
| **`test_vectors.h`** | Auto-generated by `gen_mfcc_test_vectors.py`. Contains one siren and one noise clip: raw PCM (`tv_siren_pcm`, `tv_noise_pcm`), expected MFCC (`tv_siren_mfcc`, `tv_noise_mfcc`), quantized int8 input (`tv_siren_in8`, `tv_noise_in8`), and expected classes (`TV_SIREN_CLASS`, `TV_NOISE_CLASS`). Used by the `RP_PARITY_TEST` harness. |
| **`CMakeLists.txt`** | ESP-IDF component registration: lists source files and `REQUIRES esp-dsp esp-tflite-micro esp_timer driver esp_driver_i2s`. |
| **`idf_component.yml`** | Component manager manifest: `espressif/esp-dsp ^1.4.0`, `espressif/esp-tflite-micro ^1.3.2`, `idf >= 5.1`. |