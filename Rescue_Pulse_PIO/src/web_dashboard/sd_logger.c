/*
 * SD Card Logger for RescuePulse
 * Handles logging detection data and system information to SD card
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "driver/sdspi_host.h"
#include "sdmmc_cmd.h"
#include "sd_logger.h"
#include "detection_data.h"

static const char *TAG = "sd_logger";

// SD card configuration
#define MOUNT_POINT "/sdcard"
#define MAX_FILE_SIZE (1024 * 1024) // 1MB max file size before rotating

// Logging queue
static QueueHandle_t log_queue = NULL;
static SemaphoreHandle_t fs_mutex = NULL;

// Logging task handle
static TaskHandle_t logging_task_handle = NULL;

// File pointers for current logs
static FILE *detection_log_file = NULL;
static FILE *audio_log_file = NULL;
static FILE *system_log_file = NULL;

// Current file sizes
static size_t detection_file_size = 0;
static size_t audio_file_size = 0;
static size_t system_file_size = 0;

// Forward declarations
static void sd_logger_task(void *pvParameters);
static esp_err_t init_sd_card(void);
static FILE *open_log_file(const char *base_name, const char *mode);
static void rotate_log_file(FILE **file_ptr, size_t *file_size, const char *base_name);
static void write_detection_log(detection_event_t *event, float rms_l, float rms_r);
static void write_audio_log(float rms_l, float rms_r, int lag, int16_t max_pcm);
static void write_system_log(system_status_t *status);

/**
 * Initialize the SD card logger
 */
void sd_logger_init(void) {
    ESP_LOGI(TAG, "Initializing SD card logger");

    // Create mutex for filesystem access
    fs_mutex = xSemaphoreCreateMutex();
    if (fs_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create filesystem mutex");
        return;
    }

    // Create logging queue
    log_queue = xQueueCreate(20, sizeof(log_message_t));
    if (log_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create logging queue");
        vSemaphoreDelete(fs_mutex);
        return;
    }

    // Initialize SD card
    esp_err_t ret = init_sd_card();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize SD card");
        vSemaphoreDelete(fs_mutex);
        vQueueDelete(log_queue);
        return;
    }

    // Create logging task
    BaseType_t result = xTaskCreatePinnedToCore(
        sd_logger_task,
        "sd_logger",
        4096,
        NULL,
        4, // Priority - lower than inference but higher than web server
        &logging_task_handle,
        0 // Core 0
    );

    if (result != pdPASS) {
        ESP_LOGE(TAG, "Failed to create SD logger task");
        // Cleanup
        esp_vfs_fat_sdmmc_unmount();
        vSemaphoreDelete(fs_mutex);
        vQueueDelete(log_queue);
    } else {
        ESP_LOGI(TAG, "SD logger task created");
    }
}

/**
 * Initialize the SD card using SPI or SDMMC peripheral
 */
static esp_err_t init_sd_card(void) {
    esp_err_t ret;

    // Options for mounting the filesystem.
    // If format_if_mount_failed is set to true, the SD card will be partitioned
    // and formatted in case when mounting fails.
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };

    // Use settings defined above to initialize SD card and mount FAT filesystem.
    // Note: esp_vfs_fat_sdmmc_mount is all-in-one convenience function.
    // Please check its source code and implement error recovery when developing
    // production applications.
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();

    // This initializes the slot without card detect (CD) and write protect (WP) signals.
    // Modify slot_config.gpio_cd and slot_config.gpio_wp if your board has these signals.
    sdmmc_slot_config_t slot_config = SDMMC_SLOT_CONFIG_DEFAULT();

    // To use 1-line SD mode, uncomment the following line:
    // slot_config.width = 1;

    // GPIOs used for SD card
    // Change these pins based on your board's wiring
    // For Edgehax ESP32-S3 PRO, check the specific pinout
    slot_config.clk = GPIO_NUM_12;   // CLK
    slot_config.cmd = GPIO_NUM_11;   // CMD
    slot_config.d0 = GPIO_NUM_13;    // D0
    slot_config.d1 = GPIO_NUM_14;    // D1
    slot_config.d2 = GPIO_NUM_15;    // D2
    slot_config.d3 = GPIO_NUM_2;     // D3

    ESP_LOGI(TAG, "Mounting SD card...");
    ret = esp_vfs_fat_sdmmc_mount(MOUNT_POINT, &host, &slot_config, &mount_config, NULL);

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount filesystem. "
                     "If you want the card to be formatted, set format_if_mount_failed = true.");
        } else {
            ESP_LOGE(TAG, "Failed to initialize the card (%s). "
                     "Make sure SD card lines have pull-up resistors in place.", esp_err_to_name(ret));
        }
        return ret;
    }

    // Card has been initialized, print its properties
    sdmmc_card_t *card;
    esp_err_t ret_card = esp_vfs_fat_sdmmc_get_card(MOUNT_POINT, &card);
    if (ret_card == ESP_OK) {
        ESP_LOGI(TAG, "SD Card Manufacturer: %d", card->cid->manufacturer);
        ESP_LOGI(TAG, "SD Card OEM: %04x", (card->cid->oem & 0xFF) << 8 | (card->cid->oem >> 8));
        ESP_LOGI(TAG, "SD Card Volume Name: %.8s", card->cid->vol_name);
        ESP_LOGI(TAG, "SD Card Serial Number: %08x", card->cid->serial_number);
        ESP_LOGI(TAG, "SD Card Manufacturing Year: %d", 2000 + ((card->cid->mfd_year << 8) | (card->cid->mfd_month)));
        ESP_LOGI(TAG, "SD Card Capacity: %lluMB", ((uint64_t)card->csd->capacity) * card->csd->sector_size / (1024 * 1024));
    } else {
        ESP_LOGW(TAG, "Failed to get SD card properties");
    }

    // Open log files
    detection_log_file = open_log_file("detection", "a");
    audio_log_file = open_log_file("audio", "a");
    system_log_file = open_log_file("system", "a");

    if (detection_log_file && audio_log_file && system_log_file) {
        ESP_LOGI(TAG, "Log files opened successfully");
        return ESP_OK;
    } else {
        ESP_LOGE(TAG, "Failed to open log files");
        esp_vfs_fat_sdmmc_unmount();
        return ESP_FAIL;
    }
}

/**
 * Open a log file with date-based naming
 */
static FILE *open_log_file(const char *base_name, const char *mode) {
    time_t now;
    struct tm timeinfo;
    char filename[64];

    time(&now);
    localtime_r(&now, &timeinfo);

    // Format: detection_YYYY-MM-DD.csv
    strftime(filename, sizeof(filename), "%s_%%Y-%%m-%%d.csv", base_name);
    char full_path[128];
    snprintf(full_path, sizeof(full_path), "%s/%s", MOUNT_POINT, filename);

    FILE *file = fopen(full_path, mode);
    if (file) {
        // Get current file size
        fseek(file, 0, SEEK_END);
        size_t size = ftell(file);
        fseek(file, 0, SEEK_SET); // Reset to beginning for appending

        if (strcmp(base_name, "detection") == 0) {
            detection_file_size = size;
        } else if (strcmp(base_name, "audio") == 0) {
            audio_file_size = size;
        } else if (strcmp(base_name, "system") == 0) {
            system_file_size = size;
        }

        ESP_LOGI(TAG, "Opened %s (size: %zu bytes)", full_path, size);
        return file;
    } else {
        ESP_LOGE(TAG, "Failed to open %s", full_path);
        return NULL;
    }
}

/**
 * Rotate log file when it gets too large
 */
static void rotate_log_file(FILE **file_ptr, size_t *file_size, const char *base_name) {
    if (*file_ptr) {
        fclose(*file_ptr);
        *file_ptr = NULL;
    }

    // Close and rename current file, then open new one
    // In a simple implementation, we just close and reopen (which will append)
    // For production, you might want to implement proper rotation with numbering
    *file_ptr = open_log_file(base_name, "a");
    *file_size = 0; // Reset size counter

    if (*file_ptr) {
        ESP_LOGI(TAG, "Rotated log file for %s", base_name);
    } else {
        ESP_LOGE(TAG, "Failed to rotate log file for %s", base_name);
    }
}

/**
 * SD card logger task - processes log messages from the queue
 */
static void sd_logger_task(void *pvParameters) {
    log_message_t msg;

    ESP_LOGI(TAG, "SD logger task started");

    while (1) {
        // Wait for log message (with timeout to allow for cleanup)
        if (xQueueReceive(log_queue, &msg, pdMS_TO_TICKS(100)) == pdTRUE) {
            switch (msg.type) {
                case LOG_TYPE_DETECTION:
                    write_detection_log(&msg.data.detection, msg.data.detection.rms_l, msg.data.detection.rms_r);
                    break;
                case LOG_TYPE_AUDIO:
                    write_audio_log(&msg.data.audio);
                    break;
                case LOG_TYPE_SYSTEM:
                    write_system_log(&msg.data.system);
                    break;
                default:
                    ESP_LOGW(TAG, "Unknown log message type: %d", msg.type);
                    break;
            }

            // Check if we need to rotate files
            if (detection_file_size > MAX_FILE_SIZE) {
                rotate_log_file(&detection_log_file, &detection_file_size, "detection");
            }
            if (audio_file_size > MAX_FILE_SIZE) {
                rotate_log_file(&audio_log_file, &audio_file_size, "audio");
            }
            if (system_file_size > MAX_FILE_SIZE) {
                rotate_log_file(&system_log_file, &system_file_size, "system");
            }
        }
    }

    // Cleanup (should never reach here)
    if (detection_log_file) fclose(detection_log_file);
    if (audio_log_file) fclose(audio_log_file);
    if (system_log_file) fclose(system_log_file);
    esp_vfs_fat_sdmmc_unmount();
    vSemaphoreDelete(fs_mutex);
    vQueueDelete(log_queue);
    vTaskDelete(NULL);
}

/**
 * Write a detection event to the detection log
 */
static void write_detection_log(detection_event_t *event, float rms_l, float rms_r) {
    if (xSemaphoreTake(fs_mutex, portMAX_DELAY) == pdTRUE) {
        if (detection_log_file) {
            // Format: timestamp,direction,confidence,rms_l,rms_r
            fprintf(detection_log_file, "%ld,%d,%.3f,%.3f,%.3f\n",
                    event->timestamp,
                    event->direction,
                    event->confidence,
                    rms_l,
                    rms_r);
            fflush(detection_log_file);
            detection_file_size += fprintf(detection_log_file, "%ld,%d,%.3f,%.3f,%.3f\n",
                                          event->timestamp,
                                          event->direction,
                                          event->confidence,
                                          rms_l,
                                          rms_r);
        }
        xSemaphoreGive(fs_mutex);
    }
}

/**
 * Write audio levels to the audio log
 */
static void write_audio_log(audio_log_data_t *data) {
    if (xSemaphoreTake(fs_mutex, portMAX_DELAY) == pdTRUE) {
        if (audio_log_file) {
            // Format: timestamp,rms_l,rms_r,lag,max_pcm
            fprintf(audio_log_file, "%ld,%.3f,%.3f,%d,%d\n",
                    data->timestamp,
                    data->rms_l,
                    data->rms_r,
                    data->lag,
                    data->max_pcm);
            fflush(audio_log_file);
            audio_file_size += fprintf(audio_log_file, "%ld,%.3f,%.3f,%d,%d\n",
                                      data->timestamp,
                                      data->rms_l,
                                      data->rms_r,
                                      data->lag,
                                      data->max_pcm);
        }
        xSemaphoreGive(fs_mutex);
    }
}

/**
 * Write system status to the system log
 */
static void write_system_log(system_status_t *status) {
    if (xSemaphoreTake(fs_mutex, portMAX_DELAY) == pdTRUE) {
        if (system_log_file) {
            // Format: timestamp,cpu_usage,memory_usage,wifi_rssi
            fprintf(system_log_file, "%ld,%.1f,%.1f,%d\n",
                    status->timestamp,
                    status->cpu_usage,
                    status->memory_usage,
                    status->wifi_rssi);
            fflush(system_log_file);
            system_file_size += fprintf(system_log_file, "%ld,%.1f,%.1f,%d\n",
                                       status->timestamp,
                                       status->cpu_usage,
                                       status->memory_usage,
                                       status->wifi_rssi);
        }
        xSemaphoreGive(fs_mutex);
    }
}

/**
 * Public function to log a detection event
 */
void sd_logger_log_detection(int64_t timestamp, detection_direction_t direction,
                            float confidence, float rms_l, float rms_r) {
    log_message_t msg;
    msg.type = LOG_TYPE_DETECTION;
    msg.data.detection.timestamp = timestamp;
    msg.data.detection.direction = direction;
    msg.data.detection.confidence = confidence;
    msg.data.detection.rms_l = rms_l;
    msg.data.detection.rms_r = rms_r;

    // Try to send to queue (don't block if full)
    if (xQueueSend(log_queue, &msg, pdMS_TO_TICKS(10)) != pdPASS) {
        ESP_LOGW(TAG, "Log queue full, dropping detection log");
    }
}

/**
 * Public function to log audio levels
 */
void sd_logger_log_audio(int64_t timestamp, float rms_l, float rms_r, int lag, int16_t max_pcm) {
    log_message_t msg;
    msg.type = LOG_TYPE_AUDIO;
    msg.data.audio.timestamp = timestamp;
    msg.data.audio.rms_l = rms_l;
    msg.data.audio.rms_r = rms_r;
    msg.data.audio.lag = lag;
    msg.data.audio.max_pcm = max_pcm;

    if (xQueueSend(log_queue, &msg, pdMS_TO_TICKS(10)) != pdPASS) {
        ESP_LOGW(TAG, "Log queue full, dropping audio log");
    }
}

/**
 * Public function to log system status
 */
void sd_logger_log_system(int64_t timestamp, float cpu_usage, float memory_usage, int wifi_rssi) {
    log_message_t msg;
    msg.type = LOG_TYPE_SYSTEM;
    msg.data.system.timestamp = timestamp;
    msg.data.system.cpu_usage = cpu_usage;
    msg.data.system.memory_usage = memory_usage;
    msg.data.system.wifi_rssi = wifi_rssi;

    if (xQueueSend(log_queue, &msg, pdMS_TO_TICKS(10)) != pdPASS) {
        ESP_LOGW(TAG, "Log queue full, dropping system log");
    }
}

/**
 * Deinitialize the SD card logger
 */
void sd_logger_deinit(void) {
    ESP_LOGI(TAG, "Deinitializing SD card logger");

    // Signal task to stop (we'd need a flag for graceful shutdown)
    // For now, just delete the task and cleanup
    if (logging_task_handle) {
        vTaskDelete(logging_task_handle);
        logging_task_handle = NULL;
    }

    if (detection_log_file) {
        fclose(detection_log_file);
        detection_log_file = NULL;
    }
    if (audio_log_file) {
        fclose(audio_log_file);
        audio_log_file = NULL;
    }
    if (system_log_file) {
        fclose(system_log_file);
        system_log_file = NULL;
    }

    esp_vfs_fat_sdmmc_unmount();
    if (fs_mutex) {
        vSemaphoreDelete(fs_mutex);
        fs_mutex = NULL;
    }
    if (log_queue) {
        vQueueDelete(log_queue);
        log_queue = NULL;
    }
}