#ifndef DETECTION_DATA_H
#define DETECTION_DATA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Direction enum (should match the one in traffic_ctrl.h or main.c)
typedef enum {
    DIR_LEFT = 0,
    DIR_RIGHT = 1,
    DIR_CENTER = 2
} detection_direction_t;

// Detection event structure for logging and transmission
typedef struct {
    int64_t timestamp;        // Unix timestamp in seconds
    detection_direction_t direction;
    float confidence;         // 0.0 to 1.0
    float rms_l;              // Left microphone RMS
    float rms_r;              // Right microphone RMS
    int lag;                  // TDOA lag
    int16_t max_pcm;          // Peak PCM value
} detection_event_t;

// Audio log data structure
typedef struct {
    int64_t timestamp;
    float rms_l;
    float rms_r;
    int lag;
    int16_t max_pcm;
} audio_log_data_t;

// System status structure
typedef struct {
    int64_t timestamp;
    float cpu_usage;          // Percentage
    float memory_usage;       // Percentage
    int wifi_rssi;            // Signal strength in dBm
    float temperature;        // Optional: temperature in Celsius
} system_status_t;

// Log message types for the logging queue
typedef enum {
    LOG_TYPE_DETECTION = 0,
    LOG_TYPE_AUDIO = 1,
    LOG_TYPE_SYSTEM = 2
} log_message_type_t;

// Log message structure for queueing
typedef struct {
    log_message_type_t type;
    union {
        detection_event_t detection;
        audio_log_data_t audio;
        system_status_t system;
    } data;
} log_message_t;

// WebSocket message types (for future use)
typedef enum {
    WS_TYPE_DETECTION = 0,
    WS_TYPE_AUDIO = 1,
    WS_TYPE_SYSTEM = 2,
    WS_TYPE_COMMAND = 3
} ws_message_type_t;

#ifdef __cplusplus
}
#endif

#endif // DETECTION_DATA_H