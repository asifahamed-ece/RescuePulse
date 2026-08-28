#ifndef LITTLEFS_INTEGRATION_H
#define LITTLEFS_INTEGRATION_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize LittleFS and HTTP server for web dashboard.
 * This function mounts the LittleFS partition and starts the HTTP server
 * to serve the web dashboard assets (HTML, CSS, JS).
 */
void littlefs_web_init(void);

/**
 * Deinitialize LittleFS and HTTP server.
 * This function stops the HTTP server and unmounts the LittleFS partition.
 */
void littlefs_web_deinit(void);

#ifdef __cplusplus
}
#endif

#endif // LITTLEFS_INTEGRATION_H