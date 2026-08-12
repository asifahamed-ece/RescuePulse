#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "mfcc.h"
#include "mel_tables.h"
#include "test_vectors.h"
#include "model_config.h"
#include "inference.h"

static const char *TAG = "mfcc_parity";

#define N_WIN   64
#define N_MFCC  13
#define N_TOTAL (N_WIN * N_MFCC)   /* 832 */

static float   got[N_WIN][N_MFCC];
static int8_t  q_in[N_TOTAL];
static int8_t  q_out[2];

/* Relative L2 error: ||got-exp|| / ||exp||  (librosa-style) */
static float rel_l2(const float *a, const float *b)
{
    float num = 0.0f, den = 0.0f;
    for (int i = 0; i < N_TOTAL; i++) {
        float d = a[i] - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return sqrtf(num / den);
}

/* z = (mfcc - g_mfcc_mu) / g_mfcc_std ;
 * q = clip( round(z / g_in_scale) + g_in_zp , -128, 127)  */
static void quantize_mfcc(const float mfcc[N_WIN][N_MFCC], int8_t *out)
{
    for (int t = 0; t < N_WIN; t++) {
        for (int c = 0; c < N_MFCC; c++) {
            float z = (mfcc[t][c] - g_mfcc_mu[c]) / g_mfcc_std[c];
            float qf = roundf(z / g_in_scale) + (float)g_in_zp;
            int q = (int)qf;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            out[t * N_MFCC + c] = (int8_t)q;
        }
    }
}

static void run_tag(const char   *tag,
                    const int16_t *pcm,
                    const float   *exp_mfcc,
                    int            expected_class)
{
    int64_t t0 = esp_timer_get_time();
    mfcc_extract_block(pcm, got);
    int64_t t1 = esp_timer_get_time();
    float ms = (float)(t1 - t0) / 1000.0f;

    /* ---- MFCC math parity ----------------------------------------- */
    float rel = rel_l2((const float *)got, exp_mfcc);
    bool pass_l2 = (rel < 0.01f);
    ESP_LOGI(TAG, "%s Test: L2 Error = %.5f (%s)  [%.1f ms]",
             tag, rel, pass_l2 ? "PASS" : "FAIL", ms);

    /* ---- Quantize -> TFLite inference ---------------------------- */
    quantize_mfcc(got, q_in);
    if (!inference_run(q_in, q_out)) {
        ESP_LOGE(TAG, "%s Inference: invoke failed", tag);
        return;
    }

    /* argmax of the dequantized softmax scores.
     * argmax is invariant under the per-tensor affine map, but we dequantize
     * so the probabilities are human-readable on the serial monitor. */
    float p0 = ((float)q_out[0] - (float)g_out_zp) * g_out_scale;
    float p1 = ((float)q_out[1] - (float)g_out_zp) * g_out_scale;
    int pred = (p1 > p0) ? 1 : 0;
    bool pass_inf = (pred == expected_class);
    ESP_LOGI(TAG,
             "%s Inference: Predicted %d (Expected %d) - %s  [scores %.4f / %.4f]",
             tag, pred, expected_class, pass_inf ? "PASS" : "FAIL", p0, p1);
}

void app_main(void)
{
    mfcc_init();
    ESP_LOGI(TAG, "MFCC Init: OK");

    if (!inference_init()) {
        ESP_LOGE(TAG, "TFLite Init: FAIL");
        return;
    }
    ESP_LOGI(TAG, "TFLite Init: OK");

    run_tag("Siren", tv_siren_pcm, tv_siren_mfcc, TV_SIREN_CLASS);
    run_tag("Noise", tv_noise_pcm, tv_noise_mfcc, TV_NOISE_CLASS);

    ESP_LOGI(TAG, "done");
    vTaskDelay(pdMS_TO_TICKS(1000));
}
