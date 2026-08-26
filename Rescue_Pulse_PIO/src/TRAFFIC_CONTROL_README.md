# Traffic Control Module (Phase 2) - Detailed Documentation

## Quick Start

The traffic control module manages a 3-lane traffic light system with adaptive emergency vehicle priority. It operates as a FreeRTOS task that receives siren detection messages from the acoustic inference engine and dynamically controls GPIO-driven traffic lights.

### Basic Initialization

```c
// In app_main():
if (traffic_ctrl_init() != ESP_OK) {
    ESP_LOGE(TAG, "Traffic Controller Init: FAIL");
    return;
}
```

This call:
- Creates a 10-element FreeRTOS message queue (`g_traffic_queue`)
- Configures 9 GPIO pins as digital outputs
- Launches the traffic control FreeRTOS task on Core 1 (priority 3)
- Initializes the state machine to NORMAL mode with LANE_LEFT active

### Sending Detections

```c
// In inference_task():
detection_msg_t msg = {
    .siren_active = true,
    .direction = DOA_LEFT,  // or DOA_CENTER, DOA_RIGHT
    .confidence = 0.98f
};
xQueueSend(g_traffic_queue, &msg, 0);  // Non-blocking send
```

The queue is checked every 100 ms. Message loss is acceptable (only the latest matters).

---

## Architecture & Design

### System Flow

```
Audio Input (Stereo INMP441 Microphones)
    ↓
I2S DMA Capture (Core 0)
    ↓
TDOA DoA Estimation + MFCC + TFLite Inference (Core 1, Priority 4)
    ↓
detection_msg_t → g_traffic_queue
    ↓
Traffic Control Task (Core 1, Priority 3)
    ↓
GPIO Updates (TL_*_RED/YELLOW/GREEN)
    ↓
Traffic Light LEDs
```

### State Machine (Finite State Machine)

Three states with explicit transitions:

```
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║  NORMAL_MODE                                                ║
║  ├─ Cycle: LEFT→CENTER→RIGHT→LEFT                         ║
║  ├─ Timing: GREEN(8s)→YELLOW(2s)→RED                      ║
║  ├─ Entry: System boot or emergency timeout               ║
║  ├─ Exit: Siren detected                                  ║
║  └─ Output: Autonomous lane sequencing                    ║
║                                                             ║
║           Siren on ≠ current lane                          ║
║           or during yellow                                 ║
║                    ↓                                        ║
║  CLEARANCE_MODE                                             ║
║  ├─ Duration: 2 seconds                                    ║
║  ├─ Output: All RED                                        ║
║  ├─ Purpose: Intersection safety                           ║
║  ├─ Entry: From NORMAL or EMERGENCY                        ║
║  └─ Exit: Auto after 2 seconds                             ║
║                    ↓                                        ║
║  EMERGENCY_MODE                                             ║
║  ├─ Output: Siren lane GREEN, all else RED                 ║
║  ├─ Duration: Until 10s no siren                           ║
║  ├─ Entry: From CLEARANCE or direct (if already green)     ║
║  └─ Exit: Timeout or no siren                              ║
║           (return via CLEARANCE)                           ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
```

### Type Definitions

**Lane Enumeration:**
```c
typedef enum {
    LANE_CENTER = 0,
    LANE_LEFT   = 1,
    LANE_RIGHT  = 2
} lane_t;
```

Maps 1:1 with acoustic DoA direction (DOA_CENTER=0, DOA_LEFT=1, DOA_RIGHT=2).

**Detection Message:**
```c
typedef struct {
    bool  siren_active;   /* true if siren detected */
    lane_t direction;     /* Which lane: LANE_LEFT/CENTER/RIGHT */
    float confidence;     /* Model confidence 0.0–1.0 */
} detection_msg_t;
```

**Internal State:**
```c
typedef struct {
    traffic_mode_t mode;
    lane_t current_lane;      /* Active lane (NORMAL or EMERGENCY) */
    int64_t state_start_us;   /* When mode started (microseconds) */
    int64_t last_siren_us;    /* Last siren detection timestamp */
    bool in_yellow;           /* NORMAL mode: in yellow phase? */
    lane_t emergency_lane;    /* Which lane in EMERGENCY mode */
} traffic_state_t;
```

---

## GPIO Configuration

### Pin Layout

| Functionality | Lane | GPIO | Signal |
|---|---|---|---|
| Traffic Light | LEFT | 1 | RED |
| | | 2 | YELLOW |
| | | 3 | GREEN |
| | CENTER | 4 | RED |
| | | 5 | YELLOW |
| | | 6 | GREEN |
| | RIGHT | 13 | RED |
| | | 14 | YELLOW |
| | | 21 | GREEN |

### Configuration Code

```c
gpio_config_t io_conf = {
    .pin_bit_mask = (1ULL << TL_LEFT_RED)   | (1ULL << TL_LEFT_YELLOW)   | (1ULL << TL_LEFT_GREEN) |
                    (1ULL << TL_CENTER_RED) | (1ULL << TL_CENTER_YELLOW) | (1ULL << TL_CENTER_GREEN) |
                    (1ULL << TL_RIGHT_RED)  | (1ULL << TL_RIGHT_YELLOW)  | (1ULL << TL_RIGHT_GREEN),
    .mode = GPIO_MODE_OUTPUT,
    .pull_up_en = GPIO_PULLUP_DISABLE,
    .pull_down_en = GPIO_PULLDOWN_DISABLE,
    .intr_type = GPIO_INTR_DISABLE,
};
gpio_config(&io_conf);
```

### Output Drive

Each GPIO is set HIGH (1) or LOW (0) directly:
```c
gpio_set_level(TL_LEFT_GREEN, 1);  // Turn ON
gpio_set_level(TL_LEFT_GREEN, 0);  // Turn OFF
```

For LED control, add current-limiting resistors (~150Ω for typical 20 mA LED).

---

## State Machine Behavior

### NORMAL Mode (Autonomous Cycling)

**Entry:**
- System startup
- Emergency timeout (10+ seconds without siren)

**Sequence:**
```
START: LANE_LEFT GREEN, CENTER RED, RIGHT RED
       │
       ├─ Wait 8 seconds (GREEN)
       ├─ Set LANE_LEFT YELLOW
       ├─ Wait 2 seconds (YELLOW)
       ├─ Set LANE_LEFT RED
       ├─ Advance: LANE_CENTER GREEN
       ├─ Wait 8 seconds (GREEN)
       ├─ Set LANE_CENTER YELLOW
       ├─ Wait 2 seconds (YELLOW)
       ├─ Set LANE_CENTER RED
       ├─ Advance: LANE_RIGHT GREEN
       ├─ Wait 8 seconds (GREEN)
       ├─ Set LANE_RIGHT YELLOW
       ├─ Wait 2 seconds (YELLOW)
       ├─ Set LANE_RIGHT RED
       ├─ Advance: LANE_LEFT GREEN (repeat)
       │
       └─ Exit: On siren detection
```

**Code:**
```c
static void update_normal_mode(void)
{
    int64_t now_us = esp_timer_get_time();
    int64_t elapsed_ms = (now_us - s_state.state_start_us) / 1000;

    if (!s_state.in_yellow) {
        if (elapsed_ms >= NORMAL_GREEN_MS) {
            s_state.in_yellow = true;
            s_state.state_start_us = now_us;
            set_lane_lights(s_state.current_lane, false, true, false);  /* YELLOW */
        }
    } else {
        if (elapsed_ms >= NORMAL_YELLOW_MS) {
            set_lane_lights(s_state.current_lane, true, false, false);  /* RED */
            s_state.current_lane = (lane_t)((s_state.current_lane + 1) % 3);
            s_state.in_yellow = false;
            s_state.state_start_us = now_us;
            set_lane_lights(s_state.current_lane, false, false, true);  /* GREEN */
            ESP_LOGI(TAG, "Normal cycle: LANE_%s now GREEN", lane_name(s_state.current_lane));
        }
    }
}
```

### CLEARANCE Mode (2-Second All-Red)

**Entry:**
- From NORMAL: Siren detected on different lane or during yellow
- From EMERGENCY: Timeout (10s no siren)

**Behavior:**
```
All Lanes → RED (1, 4, 13 HIGH)
     │
     └─ Wait 2 seconds for intersection to clear
     │
     └─ Exit to EMERGENCY (if siren detected) or NORMAL (if timeout)
```

**Rationale:**
- Prevents T-bone collisions
- Ensures all vehicles stop before direction changes
- 2 seconds is standard intersection clearance time

**Code:**
```c
static void enter_clearance_mode(lane_t target_emergency_lane)
{
    s_state.mode = MODE_CLEARANCE;
    s_state.emergency_lane = target_emergency_lane;
    s_state.state_start_us = esp_timer_get_time();
    all_red();  /* Set all 3 lanes to RED */
    ESP_LOGI(TAG, "Entering CLEARANCE mode: 2s all-red before emergency LANE_%s",
             lane_name(target_emergency_lane));
}

static void update_clearance_mode(void)
{
    int64_t now_us = esp_timer_get_time();
    int64_t elapsed_ms = (now_us - s_state.state_start_us) / 1000;

    if (elapsed_ms >= CLEARANCE_MS) {
        enter_emergency_mode();
    }
}
```

### EMERGENCY Mode (Priority Green)

**Entry:**
- From CLEARANCE (2s all-red complete)
- Direct from NORMAL (if siren on already-green lane during green phase)

**Behavior:**
```
Siren Lane → GREEN (continuous)
All Other Lanes → RED
     │
     ├─ Extend GREEN indefinitely while siren active
     ├─ Update timestamp on each siren message
     │
     └─ Exit: No siren for 10+ seconds
            (transition to CLEARANCE, then NORMAL)
```

**Optimization: Skip Clearance When Possible**

If the siren is detected on a lane that's already GREEN in NORMAL mode and not in YELLOW phase, transition directly to EMERGENCY without clearance:

```c
} else if (s_state.mode == MODE_NORMAL && 
           s_state.current_lane == msg->direction && 
           !s_state.in_yellow) {
    ESP_LOGI(TAG, "Siren on active GREEN lane - extending without clearance");
    s_state.mode = MODE_EMERGENCY;
    s_state.emergency_lane = msg->direction;
    s_state.state_start_us = esp_timer_get_time();
    /* Lights already correct; no change needed */
}
```

**Code:**
```c
static void enter_emergency_mode(void)
{
    s_state.mode = MODE_EMERGENCY;
    s_state.state_start_us = esp_timer_get_time();
    ESP_LOGI(TAG, "Entering EMERGENCY mode: LANE_%s GREEN", lane_name(s_state.emergency_lane));
    
    for (int i = 0; i < 3; i++) {
        lane_t lane = (lane_t)i;
        if (lane == s_state.emergency_lane) {
            set_lane_lights(lane, false, false, true);  /* GREEN */
        } else {
            set_lane_lights(lane, true, false, false);  /* RED */
        }
    }
}

static void update_emergency_mode(void)
{
    int64_t now_us = esp_timer_get_time();
    int64_t since_last_siren_ms = (now_us - s_state.last_siren_us) / 1000;

    if (since_last_siren_ms >= EMERGENCY_TIMEOUT_MS) {
        ESP_LOGI(TAG, "No siren for %lld ms, exiting emergency mode", since_last_siren_ms);
        s_state.mode = MODE_CLEARANCE;
        s_state.state_start_us = now_us;
        all_red();
        vTaskDelay(pdMS_TO_TICKS(CLEARANCE_MS));
        enter_normal_mode();
    }
}
```

---

## Message Handling

### Queue Management

- **Type:** FreeRTOS Queue
- **Depth:** 10 elements (can hold up to 10 `detection_msg_t` at once)
- **Send Mode:** Non-blocking (`xQueueSend(..., 0)`)
  - If queue full, oldest message is discarded
  - Acceptable because latest detection is most relevant
- **Receive Mode:** Blocking with timeout (`xQueueReceive(..., 100ms)`)
  - Polls at 100 ms intervals
  - Processes one message per iteration

### Message Processing

```c
void traffic_ctrl_task(void *arg)
{
    detection_msg_t msg;
    memset(&s_state, 0, sizeof(s_state));
    enter_normal_mode();

    while (1) {
        /* Check for detection messages (100ms timeout) */
        if (xQueueReceive(g_traffic_queue, &msg, pdMS_TO_TICKS(QUEUE_TIMEOUT_MS)) == pdTRUE) {
            handle_detection_msg(&msg);
        }

        /* Update state machine */
        switch (s_state.mode) {
            case MODE_NORMAL:
                update_normal_mode();
                break;
            case MODE_CLEARANCE:
                update_clearance_mode();
                break;
            case MODE_EMERGENCY:
                update_emergency_mode();
                break;
        }
    }
}
```

### Confidence Handling

The confidence score is logged but does not affect state transitions:
```c
ESP_LOGI(TAG, "🚨 Siren detected: LANE_%s (confidence: %.2f)",
         lane_name(msg->direction), msg->confidence);
```

For future enhancements, confidence could gate emergency activation:
```c
if (msg->siren_active && msg->confidence >= 0.75f) {
    /* Only trust detections above 75% confidence */
}
```

---

## Timing Configuration

Edit these `#define` values in `traffic_ctrl.c` to customize timing:

```c
#define NORMAL_GREEN_MS          8000    /* 8 seconds */
#define NORMAL_YELLOW_MS         2000    /* 2 seconds */
#define CLEARANCE_MS             2000    /* 2 seconds */
#define EMERGENCY_TIMEOUT_MS     10000   /* 10 seconds */
#define QUEUE_TIMEOUT_MS         100     /* Poll every 100 ms */
```

### Recommended Adjustments

- **High-traffic intersection:** Increase `NORMAL_GREEN_MS` to 10-15 seconds
- **Pedestrian-heavy:** Increase `NORMAL_YELLOW_MS` to 3 seconds
- **Safety-critical:** Increase `CLEARANCE_MS` to 3 seconds
- **Faster emergency response:** Decrease `EMERGENCY_TIMEOUT_MS` to 5 seconds

---

## Serial Output & Logging

### Log Examples

**System Boot:**
```
I (xxx) traffic_ctrl: Initializing Emergency Vehicle Priority Traffic Controller
I (xxx) traffic_ctrl: GPIO pins configured: 1-6, 13-14, 21
I (xxx) traffic_ctrl: Traffic control task launched (priority 3, stack 4096 bytes)
I (xxx) traffic_ctrl: Traffic control task started on core 1
I (xxx) traffic_ctrl: Entering NORMAL mode: starting with LANE_LEFT GREEN
```

**Normal Operation:**
```
I (xxx) traffic_ctrl: Normal cycle: LANE_CENTER now GREEN
I (xxx) traffic_ctrl: Normal cycle: LANE_RIGHT now GREEN
I (xxx) traffic_ctrl: Normal cycle: LANE_LEFT now GREEN
```

**Siren Detection:**
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

---

## Memory & Performance

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| `s_state` (traffic_state_t) | ~32 bytes | Static allocation |
| Message Queue | ~160 bytes | 10 × 16-byte messages |
| Task Stack | 4096 bytes | Configured in platformio.ini |
| **Total** | **~4.3 KB** | Fixed overhead |

### CPU Usage

- **Poll interval:** 100 ms
- **Processing per poll:** ~100 microseconds
- **CPU overhead:** <0.1% of Core 1
- **No dynamic allocation:** All memory static at link time

### Jitter & Latency

- **Queue-to-GPIO latency:** ~5-20 ms (typical FreeRTOS IPC)
- **State transition time:** <1 ms (simple arithmetic)
- **GPIO switching time:** <1 microsecond (hardware)

---

## Porting to Other Systems

### To a Different Board

1. **Edit GPIO pins** in `traffic_ctrl.h`:
   ```c
   #define TL_LEFT_RED     <new_pin>
   #define TL_LEFT_YELLOW  <new_pin>
   /* ... etc ... */
   ```

2. **Adjust timing** in `traffic_ctrl.c`:
   ```c
   #define NORMAL_GREEN_MS   <ms>
   #define EMERGENCY_TIMEOUT_MS <ms>
   /* ... etc ... */
   ```

3. **Recompile:**
   ```bash
   pio run -t clean && pio run -t upload
   ```

### To a Different Lane Count

1. **Extend `lane_t` enum:**
   ```c
   typedef enum {
       LANE_CENTER = 0,
       LANE_LEFT   = 1,
       LANE_RIGHT  = 2,
       LANE_EXTRA  = 3    /* New lane */
   } lane_t;
   ```

2. **Add GPIO defines:**
   ```c
   #define TL_EXTRA_RED     <pin>
   #define TL_EXTRA_YELLOW  <pin>
   #define TL_EXTRA_GREEN   <pin>
   ```

3. **Update `set_lane_lights()` switch statement**

4. **Update `lane_name()` function**

### To a Different Detection System

If your detection system doesn't use DoA estimation:

1. **Keep the same message structure:**
   ```c
   typedef struct {
       bool  siren_active;
       lane_t direction;
       float confidence;
   } detection_msg_t;
   ```

2. **Send messages to `g_traffic_queue`:**
   ```c
   detection_msg_t msg = { true, LANE_CENTER, 0.9f };
   xQueueSend(g_traffic_queue, &msg, 0);
   ```

The traffic controller doesn't care where the messages come from—it only reads from the queue.

---

## Troubleshooting

### LEDs Not Lighting

1. **Check GPIO configuration:** Verify pins in `traffic_ctrl.h` match your wiring
2. **Check polarity:** Confirm GPIO HIGH (1) drives your LED (verify current-limiting resistors)
3. **Test manually:**
   ```c
   gpio_set_level(TL_LEFT_RED, 1);
   vTaskDelay(pdMS_TO_TICKS(1000));
   gpio_set_level(TL_LEFT_RED, 0);
   ```

### State Machine Stuck

1. **Check queue:** Verify messages are being sent:
   ```
   I (xxx) traffic_ctrl: 🚨 Siren detected: LANE_RIGHT
   ```
2. **Check logs:** Look for mode transition messages
3. **Verify timing:** Ensure `esp_timer_get_time()` is advancing

### Erratic Lane Switching

1. **Increase confidence threshold:** Filter low-confidence detections
2. **Extend emergency timeout:** Increase `EMERGENCY_TIMEOUT_MS`
3. **Add hysteresis:** Require multiple consecutive messages before mode change

---

## Customization Examples

### Example 1: Always Clear Before Priority

Remove the optimization that skips clearance:

```c
// In handle_detection_msg():
// Comment out this block:
// } else if (s_state.mode == MODE_NORMAL && ...) {
//     /* Direct transition */
// }

// This forces all transitions through CLEARANCE mode
```

### Example 2: Extend Priority Duration for Large Vehicles

```c
#define EMERGENCY_TIMEOUT_MS  20000  /* 20 seconds instead of 10 */
```

### Example 3: Multiple Priority Levels

Add a `priority_t` field to detection_msg_t and handle different timeouts:

```c
typedef struct {
    bool  siren_active;
    lane_t direction;
    float confidence;
    uint8_t priority;  /* NEW: 1=low, 2=medium, 3=high */
} detection_msg_t;

// In update_emergency_mode():
int timeout_ms = (s_state.emergency_priority == 3) ? 15000 : 10000;
```

---

## API Reference

### Functions

#### `esp_err_t traffic_ctrl_init(void)`

Initializes the traffic control module:
- Creates FreeRTOS queue
- Configures GPIO pins
- Launches traffic control task

**Returns:** `ESP_OK` on success, `ESP_FAIL` on error

**Call once from:** `app_main()` after I2S initialization

#### `void traffic_ctrl_task(void *arg)`

Main traffic control task (launched automatically by `traffic_ctrl_init()`):
- Polls for detection messages
- Updates state machine
- Drives GPIO outputs

**Do not call directly** — launched by `traffic_ctrl_init()`

### Variables

#### `QueueHandle_t g_traffic_queue`

Global FreeRTOS queue handle for detection messages

**Access from:** Inference task to send `detection_msg_t`

```c
xQueueSend(g_traffic_queue, &msg, 0);
```

### Macros

All configurable via `#define` in `traffic_ctrl.c`:
- `QUEUE_DEPTH` — Message queue size
- `TASK_STACK_SIZE` — Task stack bytes
- `TASK_PRIORITY` — FreeRTOS priority
- `TASK_CORE` — Core (0 or 1)
- `NORMAL_GREEN_MS` — Green light duration
- `NORMAL_YELLOW_MS` — Yellow light duration
- `CLEARANCE_MS` — All-red clearance duration
- `EMERGENCY_TIMEOUT_MS` — Emergency exit timeout
- `QUEUE_TIMEOUT_MS` — Poll interval

---

## License

This module is part of the RescuePulse project, licensed under Apache 2.0.
