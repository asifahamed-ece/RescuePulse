# Traffic Control Module (Phase 2) - Implementation Guide

## Overview

The traffic control module implements a 3-lane traffic light system that responds to emergency vehicle siren detections. It's designed as an independent FreeRTOS task that receives detection messages from the acoustic inference engine and drives GPIO outputs for traffic light LEDs.

The module is self-contained in `traffic_ctrl.c` and `traffic_ctrl.h`, making it portable to different hardware configurations with minimal changes.

## Getting Started

### Initialization

Initialize the traffic control system once during startup:

```c
// In app_main():
if (traffic_ctrl_init() != ESP_OK) {
    ESP_LOGE(TAG, "Traffic Controller Init: FAIL");
    return;
}
```

This sets up:
- 10-element FreeRTOS message queue for detection messages
- 9 GPIO pins as digital outputs (3 lanes × 3 colors each)
- FreeRTOS task running on Core 1 (priority 3)
- State machine initialized to NORMAL mode

### Sending Detection Messages

From the inference task, send detection messages whenever siren analysis completes:

```c
detection_msg_t msg = {
    .siren_active = true,
    .direction = DOA_LEFT,  // or DOA_CENTER, DOA_RIGHT
    .confidence = 0.98f
};
xQueueSend(g_traffic_queue, &msg, 0);  // Non-blocking
```

The queue accepts messages every ~100-150ms from the inference task. Message loss is acceptable since only the most recent detection matters.

---

## System Architecture

### Data Flow

```
Microphone Input (Stereo)
    ↓
I2S DMA Capture (Core 0)
    ↓
Inference & DoA Estimation (Core 1, Priority 4)
    ↓
detection_msg_t → g_traffic_queue
    ↓
Traffic Control Task (Core 1, Priority 3)
    ↓
GPIO Output → Traffic Light LEDs
```

### Core Concepts

**Lane Mapping:** Three traffic lanes, each with three colored lights:
- LANE_LEFT (0): GPIOs 1 (RED), 2 (YELLOW), 3 (GREEN)
- LANE_CENTER (1): GPIOs 4 (RED), 5 (YELLOW), 6 (GREEN)
- LANE_RIGHT (2): GPIOs 13 (RED), 14 (YELLOW), 21 (GREEN)

**Detection Message Structure:**
```c
typedef struct {
    bool  siren_active;   // true if siren detected in this frame
    lane_t direction;     // LANE_CENTER, LANE_LEFT, or LANE_RIGHT
    float confidence;     // Model confidence (0.0 - 1.0)
} detection_msg_t;
```

---

## Operating Modes

The system runs a three-state finite state machine:

### NORMAL Mode (Autonomous Cycling)

In this mode, the system cycles through lanes independently without external input.

**Sequence:**
1. Start: LANE_LEFT green for 8 seconds
2. Transition: Yellow for 2 seconds
3. Switch: Turn red, advance to LANE_CENTER
4. Repeat for LANE_CENTER and LANE_RIGHT
5. Cycle back to LANE_LEFT

**When it runs:**
- System startup
- After emergency timeout (10+ seconds without siren detection)

**Code reference:** `enter_normal_mode()`, `update_normal_mode()`

### CLEARANCE Mode (Safety Transition)

All traffic lights set to red for 2 seconds to ensure the intersection is completely clear.

**Purpose:** Prevents collisions when switching from normal cycling to emergency priority, or when exiting emergency mode.

**Timing:** 2 seconds (defined by `CLEARANCE_MS`)

**When it runs:**
- Siren detected on a different lane than currently active
- Siren detected during yellow phase (to avoid mid-intersection conflicts)
- Exiting emergency mode after 10-second timeout

**Code reference:** `enter_clearance_mode()`, `update_clearance_mode()`

### EMERGENCY Mode (Priority Green)

When a siren is detected, the lane where it originated receives continuous green light while all other lanes remain red.

**Behavior:**
- Siren lane: GREEN (held until siren stops or 10-second timeout)
- All other lanes: RED

**Auto-exit:** After 10 seconds without siren detection, the system transitions back to NORMAL via CLEARANCE mode.

**Optimization:** If the siren is detected on a lane that's already green in NORMAL mode and not in the yellow phase, the system skips the clearance delay and goes directly to EMERGENCY mode, saving ~2 seconds.

**Code reference:** `enter_emergency_mode()`, `update_emergency_mode()`

---

## GPIO Configuration

### Pin Assignment

All 9 GPIO pins are configured as digital outputs during initialization:

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
```

**Pin selection rationale:**
- Avoided USB pins (19, 20) to prevent enumeration conflicts
- Avoided Flash/PSRAM pins (26-37) to prevent memory access conflicts
- Avoided I2S pins (15-17) already used for microphone input
- Avoided display SPI pins (7-12) used for ST7735S

### Modifying Pin Assignments

Edit pin definitions in `traffic_ctrl.h`:

```c
#define TL_LEFT_RED     1        // Change to your GPIO
#define TL_LEFT_YELLOW  2
#define TL_LEFT_GREEN   3
// ... repeat for CENTER and RIGHT lanes
```

Then update the `set_lane_lights()` function if your pin count changes.

---

## Timing Configuration

All timing values are configured as macros in `traffic_ctrl.c`. Adjust these for your intersection requirements:

```c
#define NORMAL_GREEN_MS          8000    /* Green light duration */
#define NORMAL_YELLOW_MS         2000    /* Yellow light duration */
#define CLEARANCE_MS             2000    /* All-red safety interval */
#define EMERGENCY_TIMEOUT_MS     10000   /* Max time to hold emergency green */
#define QUEUE_TIMEOUT_MS         100     /* Detection message poll interval */
```

### Recommended Values by Scenario

**High-traffic intersection:** Increase `NORMAL_GREEN_MS` to 12-15 seconds to allow more vehicles per cycle.

**Pedestrian-heavy area:** Increase `NORMAL_YELLOW_MS` to 3 seconds for safer crossing transitions.

**Safety-critical location:** Increase `CLEARANCE_MS` to 3 seconds to ensure complete intersection clearing.

**Faster emergency response:** Decrease `EMERGENCY_TIMEOUT_MS` to 5 seconds for shorter emergency vehicle dwell time.

---

## State Transitions

The system moves between states based on detection messages and timeouts:

```
NORMAL (Green → Yellow → Red cycle)
  ↓
  Siren detected on different lane
  or during yellow phase
  ↓
CLEARANCE (All red for 2s)
  ↓
EMERGENCY (Siren lane green)
  ↓
  10 seconds without siren
  ↓
CLEARANCE (All red for 2s)
  ↓
NORMAL (Resume cycling)
```

### Transition Logic

**From NORMAL to EMERGENCY:**

If the siren is detected on a lane already showing green and the yellow phase hasn't started, we skip clearance:

```c
if (s_state.mode == MODE_NORMAL && 
    s_state.current_lane == msg->direction && 
    !s_state.in_yellow) {
    // Lane already green - extend without clearance
    s_state.mode = MODE_EMERGENCY;
    s_state.emergency_lane = msg->direction;
}
```

This saves approximately 2 seconds of wait time when the emergency vehicle is already on a green lane.

**From EMERGENCY to NORMAL:**

After 10 seconds without detection, the system performs a safety clearance before resuming normal cycling:

```c
if (since_last_siren_ms >= EMERGENCY_TIMEOUT_MS) {
    // Exit emergency, clear intersection, then resume normal
    all_red();
    vTaskDelay(pdMS_TO_TICKS(CLEARANCE_MS));
    enter_normal_mode();
}
```

---

## Message Handling

### Queue Mechanics

- **Type:** FreeRTOS Queue
- **Capacity:** 10 messages
- **Send mode:** Non-blocking (`xQueueSend(..., 0)`)
  - If queue is full, the oldest message is discarded
  - Only the latest detection matters anyway
- **Receive mode:** Blocking with 100ms timeout
  - Task checks for messages every 100ms
  - Processes one message per cycle

### Processing Flow

```c
void traffic_ctrl_task(void *arg)
{
    detection_msg_t msg;
    enter_normal_mode();  // Start in NORMAL

    while (1) {
        // Check for new detection
        if (xQueueReceive(g_traffic_queue, &msg, pdMS_TO_TICKS(100)) == pdTRUE) {
            handle_detection_msg(&msg);
        }

        // Update current state
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

The confidence score is logged for debugging but doesn't affect state transitions:

```c
ESP_LOGI(TAG, "🚨 Siren detected: LANE_%s (confidence: %.2f)",
         lane_name(msg->direction), msg->confidence);
```

For future enhancements, you can add a confidence gate to ignore low-confidence detections:

```c
if (msg->siren_active && msg->confidence >= 0.75f) {
    // Only act on detections above 75% confidence
}
```

---

## Debugging & Monitoring

### Serial Output Examples

**System initialization:**
```
I (xxx) traffic_ctrl: Initializing Emergency Vehicle Priority Traffic Controller
I (xxx) traffic_ctrl: GPIO pins configured: 1-6, 13-14, 21
I (xxx) traffic_ctrl: Traffic control task launched (priority 3, stack 4096 bytes)
I (xxx) traffic_ctrl: Entering NORMAL mode: starting with LANE_LEFT GREEN
```

**Normal cycling:**
```
I (xxx) traffic_ctrl: Normal cycle: LANE_CENTER now GREEN
I (xxx) traffic_ctrl: Normal cycle: LANE_RIGHT now GREEN
```

**Emergency detection:**
```
W (xxx) rescuepulse: 🚨 SIREN DETECTED [RIGHT] (Conf: 0.98)
I (xxx) traffic_ctrl: 🚨 Siren detected: LANE_RIGHT (confidence: 0.98)
I (xxx) traffic_ctrl: Entering CLEARANCE mode: 2s all-red before emergency LANE_RIGHT
I (xxx) traffic_ctrl: Entering EMERGENCY mode: LANE_RIGHT GREEN (emergency vehicle)
```

**Emergency timeout:**
```
I (xxx) traffic_ctrl: No siren for 10001 ms, exiting emergency mode
I (xxx) traffic_ctrl: Entering NORMAL mode: starting with LANE_LEFT GREEN
```

### Troubleshooting

**LEDs not lighting up:**
1. Verify GPIO pins in `traffic_ctrl.h` match your wiring
2. Check GPIO polarity (HIGH should activate LED)
3. Verify current-limiting resistors on hardware
4. Test manually: `gpio_set_level(TL_LEFT_RED, 1);`

**State machine not transitioning:**
1. Check for detection messages in logs
2. Verify inference task is sending to `g_traffic_queue`
3. Check that `esp_timer_get_time()` is advancing

**Erratic lane switching:**
1. Add confidence gating to filter weak detections
2. Extend `EMERGENCY_TIMEOUT_MS` to prevent flapping
3. Verify microphone isn't picking up traffic light relay clicks

---

## Porting to Different Systems

### Different GPIO Pins

Edit `traffic_ctrl.h`:
```c
#define TL_LEFT_RED     <your_pin>
#define TL_LEFT_YELLOW  <your_pin>
#define TL_LEFT_GREEN   <your_pin>
// Repeat for CENTER and RIGHT
```

### Different Lane Count

1. Extend the `lane_t` enum in `traffic_ctrl.h`
2. Add GPIO defines for each new lane
3. Update `set_lane_lights()` switch statement
4. Update `lane_name()` function
5. Modify lane cycling: change `(lane % 3)` to `(lane % NUM_LANES)`

### Different Detection System

If your detection system doesn't use DoA estimation, just send the same message structure:

```c
detection_msg_t msg = {
    .siren_active = true,
    .direction = LANE_CENTER,  // Your system's decision
    .confidence = 0.95f
};
xQueueSend(g_traffic_queue, &msg, 0);
```

The traffic controller is agnostic to the detection source.

### Different MCU Platform

The code uses only standard ESP-IDF APIs:
- `gpio_config()` / `gpio_set_level()` — Standard GPIO
- `xQueueCreate()` / `xQueueReceive()` — FreeRTOS queue
- `xTaskCreatePinnedToCore()` — FreeRTOS task creation
- `esp_timer_get_time()` — System timer

Port by replacing these with your platform's equivalents.

---

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| State struct size | 32 bytes | Static at link time |
| Message queue size | 160 bytes | 10 × 16-byte messages |
| Task stack | 4096 bytes | Fixed allocation |
| Total memory | ~4.3 KB | All static, no heap |
| Queue latency | 5-20 ms | FreeRTOS IPC typical |
| GPIO switching | <1 µs | Hardware-level |
| CPU usage | <0.1% | 100ms poll on Core 1 |

---

## Integration with Acoustic Detection

The traffic control module expects detection messages from the inference task. Ensure the inference task calls:

```c
xQueueSend(g_traffic_queue, &msg, 0);
```

whenever it completes siren analysis (roughly every 100-150ms).

The traffic control task runs at lower priority (3 vs 4), ensuring inference always completes before traffic state updates. This prevents traffic decisions from blocking acoustic detection.

---

## API Reference

### Functions

**`esp_err_t traffic_ctrl_init(void)`**
- Initializes the traffic control module
- Creates queue, configures GPIO, launches task
- Call once from `app_main()` after I2S init
- Returns `ESP_OK` on success, `ESP_FAIL` on error

**`void traffic_ctrl_task(void *arg)`**
- Main task loop (launched by `traffic_ctrl_init()`)
- Do not call directly

### Global Variables

**`QueueHandle_t g_traffic_queue`**
- Global queue handle for detection messages
- Access from inference task to send `detection_msg_t`

### Configuration Macros

All in `traffic_ctrl.c`:
- `QUEUE_DEPTH` — Queue size
- `TASK_STACK_SIZE` — Stack bytes
- `TASK_PRIORITY` — FreeRTOS priority
- `TASK_CORE` — Core assignment (0 or 1)
- `NORMAL_GREEN_MS` — Green light duration
- `NORMAL_YELLOW_MS` — Yellow light duration
- `CLEARANCE_MS` — All-red duration
- `EMERGENCY_TIMEOUT_MS` — Emergency exit timeout
- `QUEUE_TIMEOUT_MS` — Poll interval

---

## Future Enhancements

**Confidence Gating:** Add a minimum confidence threshold before accepting detections.

**Multi-Vehicle Priority:** Handle simultaneous detections from different lanes (pick highest confidence).

**Adaptive Timing:** Adjust green light duration based on traffic flow or time of day.

**Vehicle Counting:** Track vehicle throughput per lane and optimize cycle times.

**Remote Control:** Accept timing adjustments via wireless commands.

**Monitoring Dashboard:** Log state transitions and traffic flow metrics for analysis.
