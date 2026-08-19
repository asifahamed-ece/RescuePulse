#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "mfcc.h"
#include "mel_tables.h"
#include "model_config.h"
#include "inference.h"
#include "i2s_capture.h"

#ifdef RP_PARITY_TEST
#include "test_vectors.h"
#endif

static const char *TAG = "rescuepulse";

/* ------------------------------------------------------------------ */
/*  Audio pipeline constants                                          */
/* ------------------------------------------------------------------ */
#define N_WIN          64
#define N_MFCC         13
#define N_TOTAL        (N_WIN * N_MFCC)          /* 832 */
#define N_SAMPLES      16640                     /* (64-1)*256 + 512 */
#define CHUNK          256                       /* MUST evenly divide N_SAMPLES (16640 / 256 = 65) */
#define VOTE_WINDOWS   5                         /* majority-vote window */
#define VOTE_THRESH    3                         /* >=3 siren -> SIREN DETECTED */
#define RMS_THRESHOLD  0.015f                    /* Skip inference if quieter than this */
#define CONF_THRESHOLD 0.75f                     /* Only count votes with >75% confidence */
#define TASK_STACK_INFERENCE 16384
#define TASK_STACK_CAPTURE   4096

/* ------------------------------------------------------------------ */
/*  Static buffers - NO heap allocation after init                    */
/* ------------------------------------------------------------------ */
static int16_t s_audio_buf[2][N_SAMPLES];
static volatile int s_wr = 0;                 /* buffer being filled */
static volatile int s_rd = 0;                 /* buffer ready for inference */
static float   s_mfcc[N_WIN][N_MFCC];
static int8_t  s_q_in[N_TOTAL];
static int8_t  s_q_out[2];
static SemaphoreHandle_t s_block_ready = NULL;

#ifdef RP_PARITY_TEST
static void run_parity_test(void);
#endif

/* ------------------------------------------------------------------ */
/*  Quantization                                                      */
/* ------------------------------------------------------------------ */
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

/* ------------------------------------------------------------------ */
/*  audio_capture_task  (Core 0)                                      */
/* ------------------------------------------------------------------ */
static void audio_capture_task(void *arg)
{
    static int16_t chunk[CHUNK];
    size_t fill = 0;
    ESP_LOGI(TAG, "audio_capture_task started on core %d", xPortGetCoreID());
    
    while (1) {
        if (i2s_capture_read(chunk, CHUNK) != ESP_OK) {
            ESP_LOGE(TAG, "I2S read failed - retrying");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        
        /* Copy chunk into current write buffer */
        memcpy(&s_audio_buf[s_wr][fill], chunk, sizeof(chunk));
        fill += CHUNK;
        
        if (fill >= N_SAMPLES) {
            /* Buffer full: publish it and switch to the other buffer */
            s_rd = s_wr;
            s_wr = 1 - s_wr;
            fill = 0;
            xSemaphoreGive(s_block_ready);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  inference_task  (Core 1)                                          */
/* ------------------------------------------------------------------ */
static void inference_task(void *arg)
{
    int vote_siren = 0;
    int vote_count = 0;
    ESP_LOGI(TAG, "inference_task started on core %d", xPortGetCoreID());

    while (1) {
        /* Wait for a full block */
        if (xSemaphoreTake(s_block_ready, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        int buf_idx = s_rd;

        /* ---- 1. RMS volume meter ---- */
        float sum_sq = 0.0f;
        for (int i = 0; i < N_SAMPLES; i++) {
            float sample = (float)s_audio_buf[buf_idx][i] / 32768.0f;
            sum_sq += sample * sample;
        }
        float rms = sqrtf(sum_sq / (float)N_SAMPLES);

        /* ---- 2. RMS GATING: Skip inference if too quiet ---- */
        if (rms < RMS_THRESHOLD) {
            ESP_LOGD(TAG, "Silence (RMS=%.4f), skipping", rms);
            vote_siren = 0;     /* Reset vote on silence */
            vote_count = 0;
            continue;
        }

        /* ---- 3. MFCC extraction ---- */
        int64_t t0 = esp_timer_get_time();
        mfcc_extract_block(s_audio_buf[buf_idx], s_mfcc);
        int64_t t1 = esp_timer_get_time();
        float mfcc_ms = (float)(t1 - t0) / 1000.0f;

        /* ---- 4. Quantize ---- */
        quantize_mfcc(s_mfcc, s_q_in);

        /* ---- 5. Inference ---- */
        if (!inference_run(s_q_in, s_q_out)) {
            ESP_LOGE(TAG, "inference_run failed");
            continue;
        }

        /* ---- 6. Dequantize softmax scores ---- */
        float p0 = ((float)s_q_out[0] - (float)g_out_zp) * g_out_scale;
        float p1 = ((float)s_q_out[1] - (float)g_out_zp) * g_out_scale;
        int pred = (p1 > p0) ? 1 : 0;

        /* ---- 7. Confidence threshold ---- */
        float confidence = (pred == 1) ? p1 : p0;

        /* ---- 8. Majority vote (5-window) ---- */
        if (pred == 1 && confidence >= CONF_THRESHOLD) {
            vote_siren++;
        }
        vote_count++;

        if (vote_count >= VOTE_WINDOWS) {
            bool siren = (vote_siren >= VOTE_THRESH);
            if (siren) {
                /* WARNING level ensures this always prints to serial */
                ESP_LOGW(TAG, "🚨 SIREN DETECTED [%d/%d] (conf: %.2f) [RMS: %.4f]",
                         vote_siren, VOTE_WINDOWS, p1, rms);
            } else {
                /* DEBUG level hides normal noise from cluttering the terminal */
                ESP_LOGD(TAG, "NOISE [%d/%d] (conf: %.2f) [RMS: %.4f]",
                         vote_siren, VOTE_WINDOWS, p0, rms);
            }
            vote_siren = 0;
            vote_count = 0;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  app_main                                                          */
/* ------------------------------------------------------------------ */
void app_main(void)
{
    ESP_LOGI(TAG, "BUILD MARKER PHASE8 v3 (Buffer Overflow Fixed)");
    
    mfcc_init();
    ESP_LOGI(TAG, "MFCC Init: OK");

    if (!inference_init()) {
        ESP_LOGE(TAG, "TFLite Init: FAIL");
        return;
    }
    ESP_LOGI(TAG, "TFLite Init: OK");

    if (i2s_capture_init() != ESP_OK) {
        ESP_LOGE(TAG, "I2S Init: FAIL");
        return;
    }
    ESP_LOGI(TAG, "I2S Init: OK");

    s_block_ready = xSemaphoreCreateBinary();
    if (s_block_ready == NULL) {
        ESP_LOGE(TAG, "Semaphore create failed");
        return;
    }

#ifdef RP_PARITY_TEST
    run_parity_test();
#endif

    xTaskCreatePinnedToCore(audio_capture_task, "audio_capture",
                            TASK_STACK_CAPTURE, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(inference_task, "inference",
                            TASK_STACK_INFERENCE, NULL, 4, NULL, 1);

    ESP_LOGI(TAG, "Pipeline started: capture(Core0) -> inference(Core1)");
}

/* ------------------------------------------------------------------ */
/*  Optional offline parity test                                      */
/* ------------------------------------------------------------------ */
#ifdef RP_PARITY_TEST
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

static void run_tag(const char *tag, const int16_t *pcm, const float *exp_mfcc, int expected_class)
{
    int64_t t0 = esp_timer_get_time();
    mfcc_extract_block(pcm, s_mfcc);
    int64_t t1 = esp_timer_get_time();
    float ms = (float)(t1 - t0) / 1000.0f;

    float rel = rel_l2((const float *)s_mfcc, exp_mfcc);
    bool pass_l2 = (rel < 0.01f);
    ESP_LOGI(TAG, "%s Test: L2 Error = %.5f (%s)  [%.1f ms]",
             tag, rel, pass_l2 ? "PASS" : "FAIL", ms);

    quantize_mfcc(s_mfcc, s_q_in);
    if (!inference_run(s_q_in, s_q_out)) {
        ESP_LOGE(TAG, "%s Inference: invoke failed", tag);
        return;
    }

    float p0 = ((float)s_q_out[0] - (float)g_out_zp) * g_out_scale;
    float p1 = ((float)s_q_out[1] - (float)g_out_zp) * g_out_scale;
    int pred = (p1 > p0) ? 1 : 0;

    bool pass_inf = (pred == expected_class);
    ESP_LOGI(TAG, "%s Inference: Predicted %d (Expected %d) - %s  [scores %.4f / %.4f]",
             tag, pred, expected_class, pass_inf ? "PASS" : "FAIL", p0, p1);
}

static void run_parity_test(void)
{
    run_tag("Siren", tv_siren_pcm, tv_siren_mfcc, TV_SIREN_CLASS);
    run_tag("Noise", tv_noise_pcm, tv_noise_mfcc, TV_NOISE_CLASS);
    ESP_LOGI(TAG, "Parity test done");
}
#endif /* RP_PARITY_TEST */