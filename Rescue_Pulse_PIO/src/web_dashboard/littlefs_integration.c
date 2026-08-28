/*
 * LittleFS Integration for RescuePulse Dashboard
 * Handles mounting LittleFS partition and serving web assets via HTTP server
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_vfs.h"
#include "esp_vfs_fat.h"
#include "esp_spiffs.h"
#include "littlefs_integration.h"
#include "websocket_server.h"
#include "sd_logger.h"

static const char *TAG = "littlefs_integration";

// HTTP server handle
static httpd_handle_t http_server = NULL;

// LittleFS configuration
#define LITTLEFS_PARTITION_LABEL "webfs"
#define LITTLEFS_MOUNT_POINT "/littlefs"
#define LITTLEFS_MAX_FILES 5
#define LITTLEFS_FORMAT_ON_FAIL false

// Forward declarations
static esp_err_t mount_littlefs(void);
static esp_err_t start_web_server(void);
static esp_err_t stop_web_server(void);
static esp_err_t unmount_littlefs(void);
static esp_err_t register_uri_handlers(httpd_handle_t server);
static esp_err_t littlefs_file_reader(httpd_req_t *req);

/**
 * Initialize LittleFS and HTTP server for web dashboard
 */
void littlefs_web_init(void) {
    ESP_LOGI(TAG, "Initializing LittleFS web dashboard");

    // Mount LittleFS partition
    esp_err_t ret = mount_littlefs();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount LittleFS partition");
        return;
    }

    // Start HTTP server
    ret = start_web_server();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start web server");
        unmount_littlefs();
        return;
    }

    ESP_LOGI(TAG, "LittleFS web dashboard initialized successfully");
}

/**
 * Deinitialize LittleFS and HTTP server
 */
void littlefs_web_deinit(void) {
    ESP_LOGI(TAG, "Deinitializing LittleFS web dashboard");

    // Stop HTTP server
    stop_web_server();

    // Unmount LittleFS
    unmount_littlefs();

    ESP_LOGI(TAG, "LittleFS web dashboard deinitialized");
}

/**
 * Mount the LittleFS partition
 */
static esp_err_t mount_littlefs(void) {
    ESP_LOGI(TAG, "Mounting LittleFS partition at %s", LITTLEFS_MOUNT_POINT);

    // Configuration for LittleFS
    esp_vfs_littlefs_conf_t conf = {
        .base_path = LITTLEFS_MOUNT_POINT,
        .partition_label = LITTLEFS_PARTITION_LABEL,
        .max_files = LITTLEFS_MAX_FILES,
        .format_if_mount_failed = LITTLEFS_FORMAT_ON_FAIL
    };

    // Use settings defined above to initialize and mount LittleFS filesystem.
    // Note: esp_vfs_littlefs_register is used for mounting LittleFS
    esp_err_t ret = esp_vfs_littlefs_register(&conf);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount or format filesystem");
        } else if (ret == ESP_ERR_NOT_FOUND) {
            ESP_LOGE(TAG, "Failed to find LittleFS partition");
        } else {
            ESP_LOGE(TAG, "Failed to initialize LittleFS (%s)", esp_err_to_name(ret));
        }
        return ret;
    }

    ESP_LOGI(TAG, "LittleFS mounted successfully");
    return ESP_OK;
}

/**
 * Start the HTTP server for serving web dashboard
 */
static esp_err_t start_web_server(void) {
    ESP_LOGI(TAG, "Starting HTTP server");

    // Default HTTP server configuration
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80; // Standard HTTP port
    config.max_uri_handlers = 10;
    config.max_resp_headers = 5;
    config.backlog_conn = 5;
    config.lru_purge_enable = true;
    config.recv_wait_timeout = 5;
    config.send_wait_timeout = 5;

    // Start the HTTP server
    esp_err_t ret = httpd_start(&http_server, &config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP server: %s", esp_err_to_name(ret));
        return ret;
    }

    // Register URI handlers for serving web assets
    ret = register_uri_handlers(http_server);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to register URI handlers");
        httpd_stop(http_server);
        http_server = NULL;
        return ret;
    }

    ESP_LOGI(TAG, "HTTP server started on port %d", config.server_port);
    return ESP_OK;
}

/**
 * Stop the HTTP server
 */
static esp_err_t stop_web_server(void) {
    if (http_server) {
        ESP_LOGI(TAG, "Stopping HTTP server");
        httpd_stop(http_server);
        http_server = NULL;
        return ESP_OK;
    }
    return ESP_FAIL;
}

/**
 * Unmount the LittleFS partition
 */
static esp_err_t unmount_littlefs(void) {
    ESP_LOGI(TAG, "Unmounting LittleFS partition");
    esp_err_t ret = esp_vfs_littlefs_unregister(LITTLEFS_PARTITION_LABEL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to unmount LittleFS: %s", esp_err_to_name(ret));
    }
    return ret;
}

/**
 * Register URI handlers for the HTTP server
 */
static esp_err_t register_uri_handlers(httpd_handle_t server) {
    // Root URI - serve index.html
    httpd_uri_t root_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = littlefs_file_reader,
        .user_ctx  = NULL
    };
    httpd_register_uri_handler(server, &root_uri);

    // Style.css URI
    httpd_uri_t css_uri = {
        .uri       = "/style.css",
        .method    = HTTP_GET,
        .handler   = littlefs_file_reader,
        .user_ctx  = NULL
    };
    httpd_register_uri_handler(server, &css_uri);

    // Script.js URI
    httpd_uri_t js_uri = {
        .uri       = "/script.js",
        .method    = HTTP_GET,
        .handler   = littlefs_file_reader,
        .user_ctx  = NULL
    };
    httpd_register_uri_handler(server, &js_uri);

    // WebSocket endpoint (placeholder - actual WebSocket handling would be different)
    // For now, we'll just note that WebSocket connections go to /ws
    // The actual WebSocket server would be separate from the HTTP server

    ESP_LOGI(TAG, "URI handlers registered");
    return ESP_OK;
}

/**
 * Handler to read files from LittleFS and serve them via HTTP
 */
static esp_err_t littlefs_file_reader(httpd_req_t *req) {
    ESP_LOGI(TAG, "Handling request for URI: %s", req->uri);
    char filepath[256];

    // Construct the full path by prepending the LittleFS mount point
    if (strcmp(req->uri, "/") == 0) {
        strcpy(filepath, LITTLEFS_MOUNT_POINT "/index.html");
    } else {
        // Prevent directory traversal attacks
        if (strstr(req->uri, "..") != NULL) {
            httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid URI");
            return ESP_FAIL;
        }
        snprintf(filepath, sizeof(filepath), "%s%s", LITTLEFS_MOUNT_POINT, req->uri);
    }

    // Check if file exists
    struct stat file_stat;
    if (stat(filepath, &file_stat) != 0) {
        ESP_LOGE(TAG, "Failed to stat file : %s", filepath);
        httpd_resp_send_err(req, HTTPD_404_NOT_FOUND, "File not found");
        return ESP_FAIL;
    }

    // Open the file for reading
    FILE *fd = fopen(filepath, "r");
    if (!fd) {
        ESP_LOGE(TAG, "Failed to read existing file : %s", filepath);
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to read file");
        return ESP_FAIL;
    }

    // Set content type based on file extension
    const char *content_type = "text/plain";
    if (strstr(filepath, ".html") != NULL) {
        content_type = "text/html";
    } else if (strstr(filepath, ".css") != NULL) {
        content_type = "text/css";
    } else if (strstr(filepath, ".js") != NULL) {
        content_type = "application/javascript";
    } else if (strstr(filepath, ".png") != NULL) {
        content_type = "image/png";
    } else if (strstr(filepath, ".jpg") != NULL || strstr(filepath, ".jpeg") != NULL) {
        content_type = "image/jpeg";
    }

    // Set response content type
    httpd_resp_set_type(req, content_type);

    // Read and send the file in chunks
    char chunk[128];
    ssize_t read_bytes;
    do {
        // Read file in chunks
        read_bytes = fread(chunk, 1, sizeof(chunk), fd);
        if (read_bytes > 0) {
            // Send the chunk as HTTP response
            if (httpd_resp_send_chunk(req, chunk, read_bytes) != ESP_OK) {
                fclose(fd);
                ESP_LOGE(TAG, "File sending failed!");
                // Abort sending file
                httpd_resp_sendstr_chunk(req, NULL);
                // Stop responding
                return ESP_FAIL;
            }
        }
    } while (read_bytes > 0);

    // Close the file
    fclose(fd);
    ESP_LOGI(TAG, "File sending complete");

    // Respond with an empty chunk to signal HTTP response completion
    httpd_resp_sendstr_chunk(req, NULL);
    return ESP_OK;
}