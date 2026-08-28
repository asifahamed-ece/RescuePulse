/*
 * WebSocket Server for RescuePulse Dashboard
 * Handles real-time communication between ESP32 and web dashboard
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"
#include "nvs.h"
#include "nvs_flash.h"
#include "lwip/err.h"
#include "lwip/sockets.h"
#include "lwip/sys.h"
#include <lwip/netdb.h>
#include "websocket_server.h"
#include "sd_logger.h"
#include "detection_data.h"

static const char *TAG = "websocket_server";

// WebSocket server configuration
#define WEBSOCKET_PORT 81
#define MAX_CLIENTS 5

// Client structure
typedef struct {
    int socket;
    struct sockaddr_in addr;
    bool connected;
} websocket_client_t;

// Client list
static websocket_client_t clients[MAX_CLIENTS];
static int client_count = 0;

// Queue for sending messages to clients
static QueueHandle_t ws_msg_queue = NULL;

// Mutex for client list access
static SemaphoreHandle_t client_mutex = NULL;

// Forward declarations
static void websocket_server_task(void *pvParameters);
static void handle_client(int client_socket);
static void broadcast_message(const char *message, size_t length);
static void add_client(int socket, struct sockaddr_in *addr);
static void remove_client(int socket);

/**
 * Initialize the WebSocket server
 */
void websocket_server_init(void) {
    ESP_LOGI(TAG, "Initializing WebSocket server on port %d", WEBSOCKET_PORT);

    // Create mutex for client list
    client_mutex = xSemaphoreCreateMutex();
    if (client_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create client mutex");
        return;
    }

    // Create message queue
    ws_msg_queue = xQueueCreate(10, sizeof(ws_message_t));
    if (ws_msg_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create WebSocket message queue");
        vSemaphoreDelete(client_mutex);
        return;
    }

    // Create server task
    BaseType_t result = xTaskCreatePinnedToCore(
        websocket_server_task,
        "websocket_server",
        4096,
        NULL,
        5, // Priority
        NULL,
        1 // Core 1
    );

    if (result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create WebSocket server task");
        vSemaphoreDelete(client_mutex);
        vQueueDelete(ws_msg_queue);
    } else {
        ESP_LOGI(TAG, "WebSocket server task created");
    }
}

/**
 * WebSocket server task - handles incoming connections and message broadcasting
 */
static void websocket_server_task(void *pvParameters) {
    int listen_socket, client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);
    int opt = 1;

    // Create TCP socket
    listen_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (listen_socket < 0) {
        ESP_LOGE(TAG, "Unable to create socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    // Set socket options
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Set up server address
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_addr.sin_port = htons(WEBSOCKET_PORT);

    // Bind to port
    if (bind(listen_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        ESP_LOGE(TAG, "Socket unable to bind: errno %d", errno);
        close(listen_socket);
        vTaskDelete(NULL);
        return;
    }

    // Start listening
    if (listen(listen_socket, MAX_CLIENTS) < 0) {
        ESP_LOGE(TAG, "Error occurred during listen: errno %d", errno);
        close(listen_socket);
        vTaskDelete(NULL);
        return;
    }

    ESP_LOGI(TAG, "WebSocket server listening on port %d", WEBSOCKET_PORT);

    while (1) {
        // Accept incoming connection
        ESP_LOGI(TAG, "Waiting for client connection...");
        client_socket = accept(listen_socket, (struct sockaddr *)&client_addr, &addr_len);
        if (client_socket < 0) {
            if (errno == EINTR) {
                continue;
            }
            ESP_LOGE(TAG, "Unable to accept connection: errno %d", errno);
            break;
        }

        // Add client to list
        add_client(client_socket, &client_addr);

        // Handle client in a separate task (or we could handle it here)
        // For simplicity, we handle it in the same task but we could create a task per client
        handle_client(client_socket);

        // Remove client after handling
        remove_client(client_socket);
        close(client_socket);
    }

    close(listen_socket);
    vTaskDelete(NULL);
}

/**
 * Add a client to the client list
 */
static void add_client(int socket, struct sockaddr_in *addr) {
    if (xSemaphoreTake(client_mutex, portMAX_DELAY) == pdTRUE) {
        if (client_count < MAX_CLIENTS) {
            clients[client_count].socket = socket;
            clients[client_count].addr = *addr;
            clients[client_count].connected = true;
            client_count++;
            ESP_LOGI(TAG, "Client added. Total clients: %d", client_count);
        } else {
            ESP_LOGW(TAG, "Max clients reached, rejecting new connection");
        }
        xSemaphoreGive(client_mutex);
    }
}

/**
 * Remove a client from the client list
 */
static void remove_client(int socket) {
    if (xSemaphoreTake(client_mutex, portMAX_DELAY) == pdTRUE) {
        for (int i = 0; i < client_count; i++) {
            if (clients[i].socket == socket) {
                // Shift remaining clients down
                for (int j = i; j < client_count - 1; j++) {
                    clients[j] = clients[j + 1];
                }
                client_count--;
                ESP_LOGI(TAG, "Client removed. Total clients: %d", client_count);
                break;
            }
        }
        xSemaphoreGive(client_mutex);
    }
}

/**
 * Handle communication with a single client
 */
static void handle_client(int client_socket) {
    char rx_buffer[128];
    int len;

    // Simple HTTP upgrade to WebSocket (simplified for demonstration)
    // In a real implementation, you would use a proper WebSocket library
    // For this example, we'll just echo messages and send periodic updates

    while (1) {
        len = recv(client_socket, rx_buffer, sizeof(rx_buffer) - 1, 0);
        if (len < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Non-blocking socket, no data available
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            ESP_LOGE(TAG, "recv failed: errno %d", errno);
            break;
        } else if (len == 0) {
            ESP_LOGI(TAG, "Connection closed");
            break;
        } else {
            rx_buffer[len] = 0; // Null-terminate
            ESP_LOGI(TAG, "Received %d bytes: %s", len, rx_buffer);

            // Echo back for testing (in real implementation, parse WebSocket frame)
            send(client_socket, rx_buffer, len, 0);
        }
    }
}

/**
 * Broadcast a message to all connected clients
 */
static void broadcast_message(const char *message, size_t length) {
    if (xSemaphoreTake(client_mutex, portMAX_DELAY) == pdTRUE) {
        for (int i = 0; i < client_count; i++) {
            if (clients[i].connected) {
                int sent = send(clients[i].socket, message, length, 0);
                if (sent < 0) {
                    ESP_LOGE(TAG, "Error sending to client %d: errno %d", clients[i].socket, errno);
                    clients[i].connected = false;
                }
            }
        }
        xSemaphoreGive(client_mutex);
    }
}

/**
 * Send a detection event to all connected WebSocket clients
 */
void websocket_send_detection(detection_event_t *event) {
    // Format as JSON
    char json_buffer[256];
    int len = snprintf(json_buffer, sizeof(json_buffer),
                      "{\"type\":\"detection\",\"timestamp\":%ld,\"direction\":\"%s\",\"confidence\":%.2f}",
                      event->timestamp,
                      event->direction == DIR_LEFT ? "LEFT" :
                                  event->direction == DIR_RIGHT ? "RIGHT" : "CENTER",
                      event->confidence);

    if (len > 0 && len < (int)sizeof(json_buffer)) {
        broadcast_message(json_buffer, len);
    }
}

/**
 * Send audio levels to all connected WebSocket clients
 */
void websocket_send_audio_levels(float rms_l, float rms_r) {
    char json_buffer[128];
    int len = snprintf(json_buffer, sizeof(json_buffer),
                      "{\"type\":\"audio\",\"rms_l\":%.3f,\"rms_r\":%.3f}",
                      rms_l, rms_r);

    if (len > 0 && len < (int)sizeof(json_buffer)) {
        broadcast_message(json_buffer, len);
    }
}

/**
 * Send system status to all connected WebSocket clients
 */
void websocket_send_system_status(system_status_t *status) {
    char json_buffer[256];
    int len = snprintf(json_buffer, sizeof(json_buffer),
                      "{\"type\":\"system\",\"cpu_usage\":%.1f,\"memory_usage\":%.1f,\"wifi_rssi\":%d}",
                      status->cpu_usage,
                      status->memory_usage,
                      status->wifi_rssi);

    if (len > 0 && len < (int)sizeof(json_buffer)) {
        broadcast_message(json_buffer, len);
    }
}