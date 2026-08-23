#pragma once
#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize I2S Standard Mode (RX only) for Dual INMP441 microphones.
 * 16 kHz, 32-bit slot, stereo. GPIO: BCLK=15, WS=16, DIN=17.
 * Returns ESP_OK on success. */
esp_err_t i2s_capture_init(void);

/* Blocking read of n_samples stereo int16 samples into buf_l and buf_r.
 * De-interleaves 32-bit stereo frames into Left and Right 16-bit PCM channels.
 * buf_l: buffer for Left channel (Mic 1, L/R pin tied to GND)
 * buf_r: buffer for Right channel (Mic 2, L/R pin tied to 3.3V)
 * Returns ESP_OK on success. */
esp_err_t i2s_capture_read_stereo(int16_t *buf_l, int16_t *buf_r, size_t n_samples);

/* Blocking read of n_samples mono int16 samples into buf (Left channel).
 * Returns ESP_OK on success. */
esp_err_t i2s_capture_read(int16_t *buf, size_t n_samples);

#ifdef __cplusplus
}
#endif