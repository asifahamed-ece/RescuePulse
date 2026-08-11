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

static const char *TAG = "mfcc_parity";

#define N_WIN   64
#define N_MFCC  13
#define N_TOTAL (N_WIN * N_MFCC)   /* 832 */

static float got[64][13];

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

static void run_tag(const char *tag,
                    const int16_t *pcm,
                    const float *exp_mfcc,
                    const int8_t *exp_in8)
{
    int64_t t0 = esp_timer_get_time();
    mfcc_extract_block(pcm, got);
    int64_t t1 = esp_timer_get_time();
    float ms = (t1 - t0) / 1000.0f;

    float rel = rel_l2((const float *)got, exp_mfcc);
    ESP_LOGI(TAG, "=== %s ===", tag);
    ESP_LOGI(TAG, "rel_l2 = %.6f  (PASS if < 0.01)  time = %.3f ms", rel, ms);
    ESP_LOGI(TAG, "first 5 coeffs expected: %.4f %.4f %.4f %.4f %.4f",
             exp_mfcc[0], exp_mfcc[1], exp_mfcc[2], exp_mfcc[3], exp_mfcc[4]);
    ESP_LOGI(TAG, "first 5 coeffs got     : %.4f %.4f %.4f %.4f %.4f",
             got[0][0], got[0][1], got[0][2], got[0][3], got[0][4]);

    bool pass_l2 = (rel < 0.01f);

    /* standardize z=(got-g_mfcc_mu)/g_mfcc_std;
       q=clip(round(z/g_in_scale)+g_in_zp,-128,127) */
    int diff_cnt = 0, max_diff = 0;
    for (int t = 0; t < N_WIN; t++) {
        for (int c = 0; c < N_MFCC; c++) {
            float z = (got[t][c] - g_mfcc_mu[c]) / g_mfcc_std[c];
            float qf = roundf(z / g_in_scale) + g_in_zp;
            int q = (int)qf;
            if (q < -128) q = -128;
            if (q > 127)  q = 127;
            int d = abs(q - exp_in8[t * N_MFCC + c]);
            if (d > max_diff) max_diff = d;
            if (d != 0) diff_cnt++;
        }
    }
    float pct = 100.0f * diff_cnt / N_TOTAL;
    bool pass_q = (pct <= 2.0f) && (max_diff <= 1);

    ESP_LOGI(TAG, "int8: %d/%d differ (%.2f%%) max_diff=%d  (PASS if <=2%% and max<=1)",
             diff_cnt, N_TOTAL, pct, max_diff);
    ESP_LOGI(TAG, "RESULT: %s -> L2 %s, int8 %s",
             tag, pass_l2 ? "PASS" : "FAIL", pass_q ? "PASS" : "FAIL");
}

void app_main(void)
{
    mfcc_init();
    ESP_LOGI(TAG, "mfcc_init done");

    run_tag("siren", tv_siren_pcm, tv_siren_mfcc, tv_siren_in8);
    run_tag("noise", tv_noise_pcm, tv_noise_mfcc, tv_noise_in8);

    ESP_LOGI(TAG, "done");
    vTaskDelay(pdMS_TO_TICKS(1000));
}