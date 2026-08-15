#include "i2s_capture.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"

static const char *TAG = "i2s_capture";

/* INMP441 wiring (safe ESP32-S3 GPIOs) */
#define I2S_BCLK_GPIO  15
#define I2S_WS_GPIO    16
#define I2S_DIN_GPIO   17

#define I2S_SAMPLE_RATE 16000
#define I2S_DMA_DESC_NUM 6
#define I2S_DMA_FRAME_NUM 240

static i2s_chan_handle_t s_rx_chan = NULL;

esp_err_t i2s_capture_init(void)
{
    esp_err_t ret;

    /* ---- 1. Allocate RX channel (Standard Mode) ---- */
    i2s_chan_config_t chan_cfg = {
        .id = I2S_NUM_0,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = I2S_DMA_DESC_NUM,
        .dma_frame_num = I2S_DMA_FRAME_NUM,
        .auto_clear = true,   /* zero DMA buffer on underflow */
    };
    ret = i2s_new_channel(&chan_cfg, NULL, &s_rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* ---- 2. Configure Standard Mode, RX only, 16-bit mono ---- */
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                        I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_GPIO,
            .ws   = I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DIN_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    ret = i2s_channel_init_std_mode(s_rx_chan, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_init_std_mode failed: %s", esp_err_to_name(ret));
        i2s_del_channel(s_rx_chan);
        s_rx_chan = NULL;
        return ret;
    }

    /* ---- 3. Enable RX ---- */
    ret = i2s_channel_enable(s_rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_enable failed: %s", esp_err_to_name(ret));
        i2s_del_channel(s_rx_chan);
        s_rx_chan = NULL;
        return ret;
    }

    ESP_LOGI(TAG, "I2S RX ready: %d Hz, 16-bit, mono (BCLK=%d WS=%d DIN=%d)",
             I2S_SAMPLE_RATE, I2S_BCLK_GPIO, I2S_WS_GPIO, I2S_DIN_GPIO);
    return ESP_OK;
}

esp_err_t i2s_capture_read(int16_t *buf, size_t n_samples)
{
    if (s_rx_chan == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    size_t bytes_read = 0;
    size_t bytes_want = n_samples * sizeof(int16_t);

    esp_err_t ret = i2s_channel_read(s_rx_chan, buf, bytes_want,
                                     &bytes_read, portMAX_DELAY);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* If we got fewer bytes than requested (shouldn't happen with
     * portMAX_DELAY, but guard anyway), zero-fill the remainder. */
    if (bytes_read < bytes_want) {
        size_t samples_read = bytes_read / sizeof(int16_t);
        for (size_t i = samples_read; i < n_samples; i++) {
            buf[i] = 0;
        }
    }
    return ESP_OK;
}