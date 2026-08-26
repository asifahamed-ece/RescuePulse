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
| [`main.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/main.c) | System orchestration, FreeRTOS tasks, TDOA direction estimation, AC RMS volume metering, majority voting, detection message generation. |
| [`display_st7789.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/display_st7789.c) / [`display_st7789.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/display_st7789.h) | ST7789 SPI TFT display driver (240x240 RGB565) rendering live directional alerts, noise classification, and RMS bars. |
| [`i2s_capture.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/i2s_capture.c) / [`i2s_capture.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/i2s_capture.h) | 16 kHz 32-bit stereo I2S DMA driver, unpacking Left (Mic 1) and Right (Mic 2) PCM samples. |
| [`mfcc.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/mfcc.c) / [`mfcc.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/mfcc.h) | ESP-DSP accelerated MFCC pipeline (0.97 pre-emphasis, Hamming window, 512-pt FFT, 40 Mel filters, DCT-II). |
| [`inference.cpp`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/inference.cpp) / [`inference.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/inference.h) | C++ bridge for TensorFlow Lite for Microcontrollers, memory allocation, and interpreter invocation. |
| [`model_data.cc`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/model_data.cc) | Flash-mapped INT8 quantized TFLite model byte array. |
| [`model_config.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/model_config.h) | Audio constants, per-coefficient standardization parameters, and quantization affine scales. |
| [`mel_tables.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/mel_tables.h) | Precomputed Librosa-matching Hamming window, Mel filterbank, and DCT matrix tables. |
| [`test_vectors.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/test_vectors.h) | Pre-computed siren and noise test vectors for automated on-device mathematical parity testing. |
| [`traffic_ctrl.c`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/traffic_ctrl.c) / [`traffic_ctrl.h`](file:///home/shadow/Projects/RescuePulse/Rescue_Pulse_PIO/src/traffic_ctrl.h) | **PHASE 2**: Emergency vehicle priority traffic light state machine, GPIO management, detection message handling, mode transitions (NORMAL/CLEARANCE/EMERGENCY). |

---

## Phase 2: Traffic Light Control System

### Architecture Overview

Phase 2 implements a responsive traffic light management system that receives acoustic siren detections from Phase 1 (inference_task) and dynamically controls 9 GPIO pins to manage three independent traffic lanes. The system operates as a finite state machine with three distinct modes designed to balance emergency vehicle priority with intersection safety.

### System Components

**Detection Pipeline → Traffic Control:**

```
┌──────────────────┐
│ inference_task   │  (Core 1, Priority 4)
│ (Phase 1)        │
│                  │
│ • Captures audio │
│ • Extracts MFCC  │
│ • Runs TFLite    │
│ • Generates      │
│   detection_msg  │
└────────┬─────────┘
         │ xQueueSend(g_traffic_queue, &msg, 0)
         │ Non-blocking, message loss acceptable
         ▼
    ┌─────────────────────┐
    │  FreeRTOS Queue     │
    │  (depth: 10)        │
    │  Message Type:      │
    │  detection_msg_t    │
    └─────────────────────┘
         │
         │ xQueueReceive(g_traffic_queue, &msg, 100ms timeout)
         ▼
┌──────────────────┐
│ traffic_ctrl_task│  (Core 1, Priority 3)
│ (Phase 2)        │
│                  │
│ • Processes      │
│   detections     │
│ • Manages state  │
│   machine        │
│ • Drives GPIOs   │
└──────────────────┘
```

### State Machine Design

**Three Operating Modes:**

1. **MODE_NORMAL** — Standard cyclic traffic control
   - Lane sequence cycles continuously: LEFT → CENTER → RIGHT → LEFT
   - Each lane: GREEN (8s) → YELLOW (2s) → RED
   - No external input required; fully autonomous
   - Triggered on: System boot or emergency timeout (10s no siren)

2. **MODE_CLEARANCE** — Safety transition (2 seconds)
   - All lanes forced to RED
   - Ensures intersection is empty before emergency vehicle passes
   - Prevents T-bone collisions during mode transitions
   - Triggered on: Siren detected on different lane or during yellow phase

3. **MODE_EMERGENCY** — Emergency vehicle priority
   - Siren-detected lane: GREEN (continuous)
   - All other lanes: RED
   - Green light held indefinitely while siren is detected
   - Triggered on: Siren on non-green lane (after clearance) OR siren on currently-green lane in normal mode
   - Exit condition: No siren detection for 10+ seconds

### Detection Message Structure

```c
typedef struct {
    bool  siren_active;   /* true if siren detected in this frame */
    lane_t direction;     /* LANE_CENTER (0), LANE_LEFT (1), LANE_RIGHT (2) */
    float confidence;     /* Model confidence score (0.0 - 1.0) */
} detection_msg_t;
```

**Message Frequency:** Approximately every 100-150 ms from the inference task, allowing real-time response to siren events while minimizing CPU overhead.

### GPIO Mapping & Control

Nine GPIO pins implement the three traffic lanes:

```c
// Lane LEFT: GPIOs 1, 2, 3
#define TL_LEFT_RED     1
#define TL_LEFT_YELLOW  2
#define TL_LEFT_GREEN   3

// Lane CENTER: GPIOs 4, 5, 6
#define TL_CENTER_RED     4
#define TL_CENTER_YELLOW  5
#define TL_CENTER_GREEN   6

// Lane RIGHT: GPIOs 13, 14, 21
#define TL_RIGHT_RED     13
#define TL_RIGHT_YELLOW  14
#define TL_RIGHT_GREEN   21
```

**Pin Selection Rationale:**
- Avoided USB pins (19, 20)
- Avoided Flash/PSRAM pins (26-37)
- Avoided I2S pins (15-17)
- Avoided Display SPI pins (7-12)
- All pins are GPIO-capable with sufficient drive strength for direct LED control

**Light Control Logic:**

```c
static void set_lane_lights(lane_t lane, bool red, bool yellow, bool green)
{
    /* Only one light is ON at any time */
    switch (lane) {
        case LANE_LEFT:
            gpio_set_level(TL_LEFT_RED, red ? 1 : 0);
            gpio_set_level(TL_LEFT_YELLOW, yellow ? 1 : 0);
            gpio_set_level(TL_LEFT_GREEN, green ? 1 : 0);
            break;
        /* ... similar for CENTER and RIGHT ... */
    }
}
```

### State Transition Logic

**Optimization: Smart Clearance Avoidance**

If a siren is detected on a lane that is **already GREEN** in NORMAL mode and **not in YELLOW phase**, the system skips the clearance phase and directly transitions to EMERGENCY mode:

```c
} else if (s_state.mode == MODE_NORMAL && 
           s_state.current_lane == msg->direction && 
           !s_state.in_yellow) {
    /* Siren on active GREEN lane - extend without clearance */
    ESP_LOGI(TAG, "Siren on active GREEN lane - extending without clearance");
    s_state.mode = MODE_EMERGENCY;
    s_state.emergency_lane = msg->direction;
    s_state.state_start_us = esp_timer_get_time();
    /* Keep current lane green, no need to change lights */
}
```

This reduces emergency vehicle waiting time by ~2 seconds when the vehicle is already on a green lane.

**Emergency Timeout Safety**

Emergency mode automatically expires after 10 seconds without detection:

```c
static void update_emergency_mode(void)
{
    int64_t now_us = esp_timer_get_time();
    int64_t since_last_siren_ms = (now_us - s_state.last_siren_us) / 1000;

    if (since_last_siren_ms >= EMERGENCY_TIMEOUT_MS) {
        /* Exit emergency, do clearance, then return to normal */
        enter_clearance_mode(LANE_CENTER);
        vTaskDelay(pdMS_TO_TICKS(CLEARANCE_MS));
        enter_normal_mode();
    }
}
```

### Timing Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `NORMAL_GREEN_MS` | 8000 ms | Standard green light duration per lane |
| `NORMAL_YELLOW_MS` | 2000 ms | Caution phase before red |
| `CLEARANCE_MS` | 2000 ms | All-red safety interval |
| `EMERGENCY_TIMEOUT_MS` | 10000 ms | Max time to hold emergency green |
| `QUEUE_TIMEOUT_MS` | 100 ms | Detection message poll interval |

### Queue-Based Decoupling

The FreeRTOS queue provides clean decoupling between the inference task (Phase 1) and traffic control (Phase 2):

- **Non-blocking send:** Detection messages are sent with `xQueueSend(..., 0)` — if the queue is full, the oldest message is discarded (acceptable, as the latest detection is most relevant)
- **Timeout receive:** Traffic control polls the queue every 100 ms, processing one message per iteration
- **Priority preservation:** Inference (priority 4) executes before traffic control (priority 3), ensuring detections are generated before traffic state updates
- **Memory efficiency:** Fixed 10-element queue avoids dynamic allocation

### Integration with Main System

**Initialization (app_main):**

```c
if (traffic_ctrl_init() != ESP_OK) {
    ESP_LOGE(TAG, "Traffic Controller Init: FAIL");
    return;
}
ESP_LOGI(TAG, "Traffic Controller Init: OK");
```

**Detection Message Sending (inference_task):**

```c
if (siren) {
    detection_msg_t msg = {
        .siren_active = true,
        .direction = (lane_t)doa_dir,
        .confidence = confidence
    };
    xQueueSend(g_traffic_queue, &msg, 0);
}
```

### Serial Output Examples

```
I (xxx) traffic_ctrl: Entering NORMAL mode: starting with LANE_LEFT GREEN
I (xxx) rescuepulse: 🚨 SIREN DETECTED [RIGHT] (Conf: 0.98)
I (xxx) traffic_ctrl: 🚨 Siren detected: LANE_RIGHT (confidence: 0.98)
I (xxx) traffic_ctrl: Entering CLEARANCE mode: 2s all-red before emergency LANE_RIGHT
I (xxx) traffic_ctrl: Entering EMERGENCY mode: LANE_RIGHT GREEN (emergency vehicle)
I (xxx) traffic_ctrl: No siren for 10001 ms, exiting emergency mode
```

### Memory & Performance

- **Static allocation:** All state variables are statically allocated at link time (no runtime heap)
- **Queue depth:** 10 messages (160 bytes total)
- **Task stack:** 4096 bytes
- **Task priority:** 3 (lower than inference, higher than idle)
- **CPU overhead:** ~1-2% per 100 ms poll interval (simple queue check and state update)

### Extensibility & Porting

The traffic control module is designed for easy porting to different systems:

1. **GPIO customization:** Edit `traffic_ctrl.h` pin definitions
2. **Timing tuning:** Modify `#define` constants in `traffic_ctrl.c`
3. **Lane count:** Extend `lane_t` enum and `set_lane_lights()` switch statement
4. **Detection integration:** Any system sending `detection_msg_t` via `g_traffic_queue` will work
5. **State machine logic:** Modify `update_*_mode()` functions for custom behavior

---