#include "traffic_ctrl.h"
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"

static const char *TAG = "traffic_ctrl";

/* ------------------------------------------------------------------ */
/*  State Machine Configuration                                       */
/* ------------------------------------------------------------------ */
#define QUEUE_DEPTH              10
#define TASK_STACK_SIZE          4096
#define TASK_PRIORITY            3
#define TASK_CORE                1

#define NORMAL_GREEN_MS          8000    /* 8 seconds green in normal mode */
#define NORMAL_YELLOW_MS         2000    /* 2 seconds yellow in normal mode */
#define CLEARANCE_MS             2000    /* 2 seconds all-red clearance */
#define EMERGENCY_TIMEOUT_MS     10000   /* 10 seconds no siren -> back to normal */
#define QUEUE_TIMEOUT_MS         100     /* Queue receive timeout for state updates */

/* ------------------------------------------------------------------ */
/*  Traffic Control State Machine                                     */
/* ------------------------------------------------------------------ */
typedef enum {
    MODE_NORMAL,          /* Regular cycling: green→yellow→red across lanes */
    MODE_CLEARANCE,       /* All-red transition phase */
    MODE_EMERGENCY        /* Emergency vehicle detected: hold lane green */
} traffic_mode_t;

typedef struct {
    traffic_mode_t mode;
    lane_t current_lane;        /* Active lane in normal mode or emergency lane */
    int64_t state_start_us;     /* Timestamp when current state started */
    int64_t last_siren_us;      /* Last time siren was detected */
    bool in_yellow;             /* Normal mode: currently in yellow phase */
    lane_t emergency_lane;      /* Which lane has emergency vehicle */
} traffic_state_t;

/* ------------------------------------------------------------------ */
/*  Global Queue Handle                                               */
/* ------------------------------------------------------------------ */
QueueHandle_t g_traffic_queue = NULL;

/* ------------------------------------------------------------------ */
/*  Static State (no heap allocation)                                 */
/* ------------------------------------------------------------------ */
static traffic_state_t s_state;

/* ------------------------------------------------------------------ */
/*  GPIO Helper Functions                                             */
/* ------------------------------------------------------------------ */

/* Set a single lane's RGB lights */
static void set_lane_lights(lane_t lane, bool red, bool yellow, bool green)
{
    switch (lane) {
        case LANE_LEFT:
            gpio_set_level(TL_LEFT_RED, red ? 1 : 0);
            gpio_set_level(TL_LEFT_YELLOW, yellow ? 1 : 0);
            gpio_set_level(TL_LEFT_GREEN, green ? 1 : 0);
            break;
        case LANE_CENTER:
            gpio_set_level(TL_CENTER_RED, red ? 1 : 0);
            gpio_set_level(TL_CENTER_YELLOW, yellow ? 1 : 0);
            gpio_set_level(TL_CENTER_GREEN, green ? 1 : 0);
            break;
        case LANE_RIGHT:
            gpio_set_level(TL_RIGHT_RED, red ? 1 : 0);
            gpio_set_level(TL_RIGHT_YELLOW, yellow ? 1 : 0);
            gpio_set_level(TL_RIGHT_GREEN, green ? 1 : 0);
            break;
    }
}

/* Turn all lights RED */
static void all_red(void)
{
    set_lane_lights(LANE_LEFT, true, false, false);
    set_lane_lights(LANE_CENTER, true, false, false);
    set_lane_lights(LANE_RIGHT, true, false, false);
}

/* Turn all lights OFF */
static void all_off(void)
{
    set_lane_lights(LANE_LEFT, false, false, false);
    set_lane_lights(LANE_CENTER, false, false, false);
    set_lane_lights(LANE_RIGHT, false, false, false);
}

/* ------------------------------------------------------------------ */
/*  State Machine Logic                                               */
/* ------------------------------------------------------------------ */

static const char *lane_name(lane_t lane)
{
    switch (lane) {
        case LANE_LEFT: return "LEFT";
        case LANE_CENTER: return "CENTER";
        case LANE_RIGHT: return "RIGHT";
        default: return "UNKNOWN";
    }
}

static void enter_normal_mode(void)
{
    s_state.mode = MODE_NORMAL;
    s_state.current_lane = LANE_LEFT;
    s_state.in_yellow = false;
    s_state.state_start_us = esp_timer_get_time();

    ESP_LOGI(TAG, "Entering NORMAL mode: starting with LANE_%s GREEN", lane_name(s_state.current_lane));

    /* Start with LANE_LEFT green, others red */
    set_lane_lights(LANE_LEFT, false, false, true);
    set_lane_lights(LANE_CENTER, true, false, false);
    set_lane_lights(LANE_RIGHT, true, false, false);
}

static void enter_clearance_mode(lane_t target_emergency_lane)
{
    s_state.mode = MODE_CLEARANCE;
    s_state.emergency_lane = target_emergency_lane;
    s_state.state_start_us = esp_timer_get_time();

    ESP_LOGI(TAG, "Entering CLEARANCE mode: 2s all-red before emergency LANE_%s",
             lane_name(target_emergency_lane));

    all_red();
}

static void enter_emergency_mode(void)
{
    s_state.mode = MODE_EMERGENCY;
    s_state.state_start_us = esp_timer_get_time();

    ESP_LOGI(TAG, "Entering EMERGENCY mode: LANE_%s GREEN (emergency vehicle)",
             lane_name(s_state.emergency_lane));

    /* Set emergency lane GREEN, all others RED */
    for (int i = 0; i < 3; i++) {
        lane_t lane = (lane_t)i;
        if (lane == s_state.emergency_lane) {
            set_lane_lights(lane, false, false, true);  /* GREEN */
        } else {
            set_lane_lights(lane, true, false, false);  /* RED */
        }
    }
}

static void update_normal_mode(void)
{
    int64_t now_us = esp_timer_get_time();
    int64_t elapsed_ms = (now_us - s_state.state_start_us) / 1000;

    if (!s_state.in_yellow) {
        /* GREEN phase */
        if (elapsed_ms >= NORMAL_GREEN_MS) {
            /* Switch to YELLOW */
            s_state.in_yellow = true;
            s_state.state_start_us = now_us;

            set_lane_lights(s_state.current_lane, false, true, false);  /* YELLOW */
            ESP_LOGD(TAG, "LANE_%s: GREEN → YELLOW", lane_name(s_state.current_lane));
        }
    } else {
        /* YELLOW phase */
        if (elapsed_ms >= NORMAL_YELLOW_MS) {
            /* Switch to RED and advance to next lane */
            set_lane_lights(s_state.current_lane, true, false, false);  /* RED */

            /* Advance to next lane */
            s_state.current_lane = (lane_t)((s_state.current_lane + 1) % 3);
            s_state.in_yellow = false;
            s_state.state_start_us = now_us;

            /* Set new lane to GREEN */
            set_lane_lights(s_state.current_lane, false, false, true);  /* GREEN */
            ESP_LOGI(TAG, "Normal cycle: LANE_%s now GREEN", lane_name(s_state.current_lane));
        }
    }
}

static void update_clearance_mode(void)
{
    int64_t now_us = esp_timer_get_time();
    int64_t elapsed_ms = (now_us - s_state.state_start_us) / 1000;

    if (elapsed_ms >= CLEARANCE_MS) {
        /* Clearance complete, enter emergency mode */
        enter_emergency_mode();
    }
}

static void update_emergency_mode(void)
{
    int64_t now_us = esp_timer_get_time();
    int64_t since_last_siren_ms = (now_us - s_state.last_siren_us) / 1000;

    /* If no siren detected for 10+ seconds, return to normal */
    if (since_last_siren_ms >= EMERGENCY_TIMEOUT_MS) {
        ESP_LOGI(TAG, "No siren for %lld ms, exiting emergency mode", since_last_siren_ms);

        /* Do 2s clearance before returning to normal */
        s_state.mode = MODE_CLEARANCE;
        s_state.state_start_us = now_us;
        s_state.emergency_lane = LANE_CENTER;  /* Unused, will go to normal */
        all_red();

        /* After clearance, will enter normal mode */
        vTaskDelay(pdMS_TO_TICKS(CLEARANCE_MS));
        enter_normal_mode();
    }
}

static void handle_detection_msg(const detection_msg_t *msg)
{
    if (msg->siren_active) {
        /* Update last siren timestamp */
        s_state.last_siren_us = esp_timer_get_time();

        ESP_LOGI(TAG, "🚨 Siren detected: LANE_%s (confidence: %.2f)",
                 lane_name(msg->direction), msg->confidence);

        /* If not already in emergency mode for this lane, enter clearance */
        if (s_state.mode != MODE_EMERGENCY || s_state.emergency_lane != msg->direction) {
            enter_clearance_mode(msg->direction);
        }
    } else {
        /* No siren detected - just update timestamp context */
        ESP_LOGD(TAG, "No siren (confidence: %.2f)", msg->confidence);
    }
}

/* ------------------------------------------------------------------ */
/*  Traffic Control Task                                              */
/* ------------------------------------------------------------------ */
void traffic_ctrl_task(void *arg)
{
    detection_msg_t msg;

    ESP_LOGI(TAG, "Traffic control task started on core %d", xPortGetCoreID());

    /* Initialize state machine to normal mode */
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

/* ------------------------------------------------------------------ */
/*  Initialization                                                    */
/* ------------------------------------------------------------------ */
esp_err_t traffic_ctrl_init(void)
{
    esp_err_t ret;

    ESP_LOGI(TAG, "Initializing Emergency Vehicle Priority Traffic Controller");

    /* Create message queue */
    g_traffic_queue = xQueueCreate(QUEUE_DEPTH, sizeof(detection_msg_t));
    if (g_traffic_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create traffic control queue");
        return ESP_FAIL;
    }

    /* Configure all 9 GPIO pins as outputs */
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << TL_LEFT_RED)   | (1ULL << TL_LEFT_YELLOW)   | (1ULL << TL_LEFT_GREEN) |
                        (1ULL << TL_CENTER_RED) | (1ULL << TL_CENTER_YELLOW) | (1ULL << TL_CENTER_GREEN) |
                        (1ULL << TL_RIGHT_RED)  | (1ULL << TL_RIGHT_YELLOW)  | (1ULL << TL_RIGHT_GREEN),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    ret = gpio_config(&io_conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "GPIO configuration failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Initialize all LEDs to OFF */
    all_off();
    ESP_LOGI(TAG, "GPIO pins configured: 1-6, 13-14, 21");

    /* Launch traffic control task on Core 1 */
    BaseType_t task_ret = xTaskCreatePinnedToCore(
        traffic_ctrl_task,
        "traffic_ctrl",
        TASK_STACK_SIZE,
        NULL,
        TASK_PRIORITY,
        NULL,
        TASK_CORE
    );

    if (task_ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create traffic control task");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Traffic control task launched (priority %d, stack %d bytes)",
             TASK_PRIORITY, TASK_STACK_SIZE);

    return ESP_OK;
}
