#include "i2s_capture.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"

static const char *TAG = "i2s_capture";

/* INMP441 wiring (safe ESP32-S3 GPIOs) */
#define I2S_BCLK_GPIO  4
#define I2S_WS_GPIO    5
#define I2S_DIN_GPIO   6

#define I2S_SAMPLE_RATE 16000
#define I2S_DMA_DESC_NUM 6
#define I2S_DMA_FRAME_NUM 240

/* Scratch buffer for raw 32-bit I2S words (stereo slots: Left + Right).
 * Needs 2x words for n_samples mono output. */
#define I2S_SCRATCH_MAX_SAMPLES 1024
static int32_t s_scratch[I2S_SCRATCH_MAX_SAMPLES];

static i2s_chan_handle_t s_rx_chan = NULL;

esp_err_t i2s_capture_init(void)
{
    esp_err_t ret;

    i2s_chan_config_t chan_cfg = {
        .id = I2S_NUM_0,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = I2S_DMA_DESC_NUM,
        .dma_frame_num = I2S_DMA_FRAME_NUM,
        .auto_clear = true,
    };
    ret = i2s_new_channel(&chan_cfg, NULL, &s_rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_new_channel failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* ---- Configure Standard Mode, RX only, 32-bit slot / mono ----
     * INMP441 is 24-bit MSB-justified inside a 32-bit frame. In Philips I2S
     * standard mode, there are 2 slots (Left + Right) per frame.
     * With L/R tied to GND, data is in Left slot (Slot 0) and Slot 1 is zero.
     * We capture stereo 32-bit slots and de-interleave in i2s_capture_read(). */
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(I2S_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                        I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
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

    ret = i2s_channel_enable(s_rx_chan);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_enable failed: %s", esp_err_to_name(ret));
        i2s_del_channel(s_rx_chan);
        s_rx_chan = NULL;
        return ret;
    }

    ESP_LOGI(TAG, "I2S RX ready: %d Hz, 32-bit slot -> 16-bit PCM, mono (BCLK=%d WS=%d DIN=%d)",
             I2S_SAMPLE_RATE, I2S_BCLK_GPIO, I2S_WS_GPIO, I2S_DIN_GPIO);
    return ESP_OK;
}

esp_err_t i2s_capture_read(int16_t *buf, size_t n_samples)
{
    if (s_rx_chan == NULL) {
        return ESP_ERR_INVALID_STATE;
    }
    if (n_samples * 2 > I2S_SCRATCH_MAX_SAMPLES) {
        ESP_LOGE(TAG, "n_samples %u exceeds scratch buffer capacity %u",
                 (unsigned)n_samples, (unsigned)(I2S_SCRATCH_MAX_SAMPLES / 2));
        return ESP_ERR_INVALID_ARG;
    }

    size_t bytes_read = 0;
    /* Read 2 slots (Left + Right 32-bit words) per audio frame */
    size_t bytes_want = n_samples * 2 * sizeof(int32_t);

    esp_err_t ret = i2s_channel_read(s_rx_chan, s_scratch, bytes_want,
                                     &bytes_read, portMAX_DELAY);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "i2s_channel_read failed: %s", esp_err_to_name(ret));
        return ret;
    }

    size_t frames_read = bytes_read / (2 * sizeof(int32_t));
    for (size_t i = 0; i < frames_read; i++) {
        /* Slot 0 (even index) contains Left channel from INMP441 */
        buf[i] = (int16_t)(s_scratch[2 * i] >> 16);
    }
    for (size_t i = frames_read; i < n_samples; i++) {
        buf[i] = 0;
    }
    return ESP_OK;
}