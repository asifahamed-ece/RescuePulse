#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Traffic Light GPIO Pin Assignments (ESP32-S3)                    */
/*  Verified safe pins avoiding USB (19,20), Flash/PSRAM (26-37),    */
/*  and existing I2S (15-17) + Display (7-12)                        */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  Traffic Lane Enum (maps 1:1 with DoA direction)                  */
/* ------------------------------------------------------------------ */
typedef enum {
    LANE_CENTER = 0,
    LANE_LEFT   = 1,
    LANE_RIGHT  = 2
} lane_t;

/* ------------------------------------------------------------------ */
/*  Detection Message (sent from inference task to traffic ctrl)     */
/* ------------------------------------------------------------------ */
typedef struct {
    bool  siren_active;   /* true if siren detected in this frame */
    lane_t direction;     /* which lane the siren is coming from */
    float confidence;     /* inference confidence (0.0 - 1.0) */
} detection_msg_t;

/* ------------------------------------------------------------------ */
/*  Global Queue Handle (created in traffic_ctrl_init)               */
/* ------------------------------------------------------------------ */
extern QueueHandle_t g_traffic_queue;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/* Initialize traffic controller: create queue, configure GPIOs, launch task.
 * Call once from app_main() after I2S init.
 * Returns ESP_OK on success. */
esp_err_t traffic_ctrl_init(void);

/* Traffic control FreeRTOS task (launched by traffic_ctrl_init).
 * Implements MODE_NORMAL and MODE_EMERGENCY state machine.
 * Do not call directly. */
void traffic_ctrl_task(void *arg);

#ifdef __cplusplus
}
#endif
