#pragma once
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize I2S Standard Mode (RX only) for INMP441.
 * 16 kHz, 16-bit, mono. GPIO: BCLK=15, WS=16, DIN=17.
 * Returns ESP_OK on success. */
esp_err_t i2s_capture_init(void);

/* Blocking read of n_samples int16 samples into buf.
 * Returns ESP_OK on success. */
esp_err_t i2s_capture_read(int16_t *buf, size_t n_samples);

#ifdef __cplusplus
}
#endif