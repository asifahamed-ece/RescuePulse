#ifndef WEBSOCKET_SERVER_H
#define WEBSOCKET_SERVER_H

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

// Detection event structure
typedef struct {
    int64_t timestamp;        // Unix timestamp in seconds
    detection_direction_t direction;
    float confidence;         // 0.0 to 1.0
} detection_event_t;

// System status structure
typedef struct {
    float cpu_usage;          // Percentage
    float memory_usage;       // Percentage
    int wifi_rssi;            // Signal strength in dBm
} system_status_t;

/**
 * Initialize the WebSocket server.
 * This function sets up the server task and begins listening for connections.
 */
void websocket_server_init(void);

/**
 * Send a detection event to all connected WebSocket clients.
 *
 * @param event Pointer to the detection event to send
 */
void websocket_send_detection(detection_event_t *event);

/**
 * Send audio levels to all connected WebSocket clients.
 *
 * @param rms_l Left microphone RMS value
 * @param rms_r Right microphone RMS value
 */
void websocket_send_audio_levels(float rms_l, float rms_r);

/**
 * Send system status to all connected WebSocket clients.
 *
 * @param status Pointer to the system status to send
 */
void websocket_send_system_status(system_status_t *status);

#ifdef __cplusplus
}
#endif

#endif // WEBSOCKET_SERVER_H