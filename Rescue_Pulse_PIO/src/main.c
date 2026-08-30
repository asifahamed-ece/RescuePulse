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
#include "esp_dsp.h"
#include "mfcc.h"
#include "mel_tables.h"
#include "model_config.h"
#include "inference.h"
#include "i2s_capture.h"
#include "display_st7735s.h"
#include "traffic_ctrl.h"

#ifdef RP_PARITY_TEST
#include "test_vectors.h"
#endif

static const char *TAG = "rescuepulse";

/* ------------------------------------------------------------------ */
/*  Audio pipeline constants                                          */
/* ------------------------------------------------------------------ */
#define N_WIN            64
#define N_MFCC           13
#define N_TOTAL          (N_WIN * N_MFCC)          /* 832 */
#define N_SAMPLES        16640                     /* (64-1)*256 + 512 */
#define CHUNK            256                       /* I2S read chunk size */
#define VOTE_WINDOWS     4                         /* majority-vote window (optimized for latency) */
#define VOTE_THRESH      3                         /* >=3/4 siren votes (75% agreement) -> SIREN DETECTED */
#define RMS_THRESHOLD    0.02f                     /* Skip inference if quieter than this */
#define CONF_THRESHOLD   0.75f                     /* Require 75% confidence to count as a siren vote (stricter) */
#define TASK_STACK_INFERENCE 16384
#define TASK_STACK_CAPTURE   4096

/* ------------------------------------------------------------------ */
/*  DoA / TDoA parameters                                             */
/* ------------------------------------------------------------------ */
#define TDOA_MAX_LAG         32                        /* Search +/- 32 samples (~2ms @ 16kHz) */
#define TDOA_PAT_LEN         1024                      /* Pattern window length for correlation */
#define TDOA_SIG_LEN         (TDOA_PAT_LEN + 2 * TDOA_MAX_LAG) /* 1088 */
#define TDOA_CORR_LEN        (2 * TDOA_MAX_LAG + 1)    /* 65 */
#define TDOA_LAG_THRESHOLD   2                         /* Threshold for Left / Right decision */

typedef enum {
    DOA_CENTER = 0,
    DOA_LEFT,
    DOA_RIGHT
} doa_direction_t;

static const char *doa_to_string(doa_direction_t dir)
{
    switch (dir) {
        case DOA_LEFT:   return "LEFT";
        case DOA_RIGHT:  return "RIGHT";
        case DOA_CENTER:
        default:         return "CENTER";
    }
}

/* ------------------------------------------------------------------ */
/*  Static buffers - NO heap allocation after init                    */
/* ------------------------------------------------------------------ */
/* Dimension 1: Ping-Pong (2), Dimension 2: Channel L/R (2), Dimension 3: Samples */
static int16_t s_audio_buf[2][2][N_SAMPLES];
static volatile int s_wr = 0;                 /* buffer being filled */
static volatile int s_rd = 0;                 /* buffer ready for inference */
static float   s_mfcc[N_WIN][N_MFCC];
static int8_t  s_q_in[N_TOTAL];
static int8_t  s_q_out[2];
static SemaphoreHandle_t s_block_ready = NULL;

/* TDOA scratch buffers (static to prevent runtime heap allocation) */
static float   s_tdoa_sig[TDOA_SIG_LEN];
static float   s_tdoa_pat[TDOA_PAT_LEN];
static float   s_tdoa_corr[TDOA_CORR_LEN];
static float   s_tdoa_accum[TDOA_CORR_LEN];

#ifdef RP_PARITY_TEST
static void run_parity_test(void);
#endif

/* ------------------------------------------------------------------ */
/*  Memory-monitor globals (filled by app_main, read by the monitor)  */
/* ------------------------------------------------------------------ */
static TaskHandle_t g_h_capture   = NULL;
static TaskHandle_t g_h_inference = NULL;

/* ------------------------------------------------------------------ */
/*  memory_monitor_task (Core 0, low priority)                        */
/*  Reports DRAM, PSRAM and per-task high-water marks every 10 s.     */
/*  HWM is returned in WORDS by FreeRTOS, so we multiply by 4.        */
/* ------------------------------------------------------------------ */
static void memory_monitor_task(void *arg)
{
    const TickType_t period = pdMS_TO_TICKS(10000);  /* 10 seconds */
    uint32_t tick = 0;

    while (1) {
        vTaskDelay(period);
        tick += 10;

        UBaseType_t hwm_cap   = (g_h_capture   != NULL)
                              ? uxTaskGetStackHighWaterMark(g_h_capture)   : 0;
        UBaseType_t hwm_inf   = (g_h_inference != NULL)
                              ? uxTaskGetStackHighWaterMark(g_h_inference) : 0;

        /* Stack headroom = allocated stack size - bytes actually used.
         * Allocated size is in BYTES; HWM is in WORDS (4 bytes/word). */
        uint32_t used_cap    = (uint32_t)hwm_cap * 4;
        uint32_t used_inf    = (uint32_t)hwm_inf * 4;
        uint32_t head_cap    = TASK_STACK_CAPTURE   - used_cap;
        uint32_t head_inf    = TASK_STACK_INFERENCE - used_inf;

        ESP_LOGI("MEM", "----- t=%us ------------------------------------------------",
                 (unsigned)tick);
        ESP_LOGI("MEM", "Internal free   : %u B   | min-ever: %u B",
                 (unsigned)esp_get_free_heap_size(),
                 (unsigned)xPortGetMinimumEverFreeHeapSize());
        ESP_LOGI("MEM", "PSRAM free      : %u B   | min-ever: %u B",
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
        ESP_LOGI("MEM", "Capture   task  : used %5u / %5u B   (headroom %5u B)",
                 (unsigned)used_cap, TASK_STACK_CAPTURE,   (unsigned)head_cap);
        ESP_LOGI("MEM", "Inference task  : used %5u / %5u B   (headroom %5u B)",
                 (unsigned)used_inf, TASK_STACK_INFERENCE, (unsigned)head_inf);
        ESP_LOGI("MEM", "----------------------------------------------------------");
    }
}

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
/*  Time Difference of Arrival (TDOA) Direction Estimation            */
/* ------------------------------------------------------------------ */
static doa_direction_t estimate_tdoa_direction(const int16_t *buf_l, const int16_t *buf_r, int *out_lag)
{
    memset(s_tdoa_accum, 0, sizeof(s_tdoa_accum));

    /* Average cross-correlation across 3 segments within the audio block */
    const int num_segments = 3;
    const int segment_offsets[3] = {
        (N_SAMPLES / 4) - (TDOA_SIG_LEN / 2),
        (N_SAMPLES / 2) - (TDOA_SIG_LEN / 2),
        (3 * N_SAMPLES / 4) - (TDOA_SIG_LEN / 2)
    };

    for (int seg = 0; seg < num_segments; seg++) {
        int sig_offset = segment_offsets[seg];
        if (sig_offset < 0) sig_offset = 0;
        if (sig_offset + TDOA_SIG_LEN > N_SAMPLES) sig_offset = N_SAMPLES - TDOA_SIG_LEN;
        int pat_offset = sig_offset + TDOA_MAX_LAG;

        /* Remove DC offset from Left and Right segment slices */
        float sum_sig = 0.0f;
        for (int i = 0; i < TDOA_SIG_LEN; i++) {
            sum_sig += (float)buf_l[sig_offset + i];
        }
        float mean_sig = sum_sig / (float)TDOA_SIG_LEN;

        float sum_pat = 0.0f;
        for (int i = 0; i < TDOA_PAT_LEN; i++) {
            sum_pat += (float)buf_r[pat_offset + i];
        }
        float mean_pat = sum_pat / (float)TDOA_PAT_LEN;

        for (int i = 0; i < TDOA_SIG_LEN; i++) {
            s_tdoa_sig[i] = ((float)buf_l[sig_offset + i] - mean_sig) / 32768.0f;
        }
        for (int i = 0; i < TDOA_PAT_LEN; i++) {
            s_tdoa_pat[i] = ((float)buf_r[pat_offset + i] - mean_pat) / 32768.0f;
        }

        /* ESP-DSP cross-correlation: Signal = Left channel, Pattern = Right channel */
        esp_err_t err = dsps_corr_f32(s_tdoa_sig, TDOA_SIG_LEN, s_tdoa_pat, TDOA_PAT_LEN, s_tdoa_corr);
        if (err == ESP_OK) {
            for (int i = 0; i < TDOA_CORR_LEN; i++) {
                s_tdoa_accum[i] += s_tdoa_corr[i];
            }
        }
    }

    /* Find peak cross-correlation lag */
    float max_corr = -1e30f;
    int peak_idx = TDOA_MAX_LAG;
    for (int i = 0; i < TDOA_CORR_LEN; i++) {
        if (s_tdoa_accum[i] > max_corr) {
            max_corr = s_tdoa_accum[i];
            peak_idx = i;
        }
    }

    /* Lag relative to center (0 lag is at index TDOA_MAX_LAG):
     * lag > 0: Right channel arrived first -> Source on RIGHT
     * lag < 0: Left channel arrived first  -> Source on LEFT
     */
    int lag = peak_idx - TDOA_MAX_LAG;
    if (out_lag != NULL) {
        *out_lag = lag;
    }

    if (lag > TDOA_LAG_THRESHOLD) {
        return DOA_RIGHT;
    } else if (lag < -TDOA_LAG_THRESHOLD) {
        return DOA_LEFT;
    } else {
        return DOA_CENTER;
    }
}

/* ------------------------------------------------------------------ */
/*  audio_capture_task  (Core 0)                                      */
/* ------------------------------------------------------------------ */
static void audio_capture_task(void *arg)
{
    static int16_t chunk_l[CHUNK];
    static int16_t chunk_r[CHUNK];
    size_t fill = 0;
    ESP_LOGI(TAG, "audio_capture_task started on core %d", xPortGetCoreID());
    
    while (1) {
        if (i2s_capture_read_stereo(chunk_l, chunk_r, CHUNK) != ESP_OK) {
            ESP_LOGE(TAG, "I2S stereo read failed - retrying");
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        memcpy(&s_audio_buf[s_wr][0][fill], chunk_l, sizeof(chunk_l));
        memcpy(&s_audio_buf[s_wr][1][fill], chunk_r, sizeof(chunk_r));
        fill += CHUNK;
        if (fill >= N_SAMPLES) {
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
        if (xSemaphoreTake(s_block_ready, portMAX_DELAY) != pdTRUE) {
            continue;
        }
        int buf_idx = s_rd;

        /* ---- 1. RMS calculation for Left & Right channels (AC RMS with DC offset removed) ---- */
        float sum_pcm_l = 0.0f;
        int16_t max_pcm_l = 0;
        for (int i = 0; i < N_SAMPLES; i++) {
            int16_t pcm = s_audio_buf[buf_idx][0][i];
            int16_t abs_pcm = (pcm < 0) ? -pcm : pcm;
            if (abs_pcm > max_pcm_l) max_pcm_l = abs_pcm;
            sum_pcm_l += (float)pcm;
        }
        float mean_pcm_l = sum_pcm_l / (float)N_SAMPLES;

        float sum_sq_l = 0.0f;
        for (int i = 0; i < N_SAMPLES; i++) {
            float ac = ((float)s_audio_buf[buf_idx][0][i] - mean_pcm_l) / 32768.0f;
            sum_sq_l += ac * ac;
        }
        float rms_l = sqrtf(sum_sq_l / (float)N_SAMPLES);

        float sum_pcm_r = 0.0f;
        int16_t max_pcm_r = 0;
        for (int i = 0; i < N_SAMPLES; i++) {
            int16_t pcm = s_audio_buf[buf_idx][1][i];
            int16_t abs_pcm = (pcm < 0) ? -pcm : pcm;
            if (abs_pcm > max_pcm_r) max_pcm_r = abs_pcm;
            sum_pcm_r += (float)pcm;
        }
        float mean_pcm_r = sum_pcm_r / (float)N_SAMPLES;

        float sum_sq_r = 0.0f;
        for (int i = 0; i < N_SAMPLES; i++) {
            float ac = ((float)s_audio_buf[buf_idx][1][i] - mean_pcm_r) / 32768.0f;
            sum_sq_r += ac * ac;
        }
        float rms_r = sqrtf(sum_sq_r / (float)N_SAMPLES);

        float max_rms = (rms_l > rms_r) ? rms_l : rms_r;
        int active_ch = (rms_l >= rms_r) ? 0 : 1;
        int16_t active_max_pcm = (rms_l >= rms_r) ? max_pcm_l : max_pcm_r;

        /* ---- 2. RMS GATING: Skip inference if both channels are too quiet ---- */
        if (max_rms < RMS_THRESHOLD) {
            vote_siren = 0;
            vote_count = 0;
            display_st7735s_show_idle(rms_l, rms_r);
            vTaskDelay(pdMS_TO_TICKS(1)); /* Yield to feed WDT */
            continue;
        }

        /* ---- 3. Time Difference of Arrival (TDOA) Direction of Arrival ---- */
        int lag = 0;
        doa_direction_t doa_dir = estimate_tdoa_direction(
            s_audio_buf[buf_idx][0],
            s_audio_buf[buf_idx][1],
            &lag
        );

        /* ---- 4. MFCC extraction on the louder channel ---- */
        int64_t t0 = esp_timer_get_time();
        mfcc_extract_block(s_audio_buf[buf_idx][active_ch], s_mfcc);
        int64_t t1 = esp_timer_get_time();
        float mfcc_ms = (float)(t1 - t0) / 1000.0f;
        (void)mfcc_ms;

        /* ---- 5. Quantize ---- */
        quantize_mfcc(s_mfcc, s_q_in);

        /* ---- 6. Inference ---- */
        if (!inference_run(s_q_in, s_q_out)) {
            ESP_LOGE(TAG, "inference_run failed");
            continue;
        }

        /* ---- 7. Dequantize softmax scores ---- */
        float p0 = ((float)s_q_out[0] - (float)g_out_zp) * g_out_scale;
        float p1 = ((float)s_q_out[1] - (float)g_out_zp) * g_out_scale;
        int pred = (p1 > p0) ? 1 : 0;
        float confidence = (pred == 1) ? p1 : p0;

        /* ---- 8. Majority vote with Confidence Threshold ---- */
        if (pred == 1 && confidence >= CONF_THRESHOLD) {
            vote_siren++;
        }
        vote_count++;

        if (vote_count >= VOTE_WINDOWS) {
            bool siren = (vote_siren >= VOTE_THRESH);
            if (siren) {
                ESP_LOGW(TAG, "🚨 SIREN DETECTED [%s] (Conf: %.2f) [%d/%d] [RMS L:%.3f R:%.3f, Lag: %d, MaxPCM: %d]",
                         doa_to_string(doa_dir), confidence, vote_siren, VOTE_WINDOWS, rms_l, rms_r, lag, active_max_pcm);
                display_st7735s_show_siren((ui_direction_t)doa_dir, confidence, lag, rms_l, rms_r);

                /* Send detection to traffic controller */
                detection_msg_t msg = {
                    .siren_active = true,
                    .direction = (lane_t)doa_dir,
                    .confidence = confidence
                };
                xQueueSend(g_traffic_queue, &msg, 0);

                vTaskDelay(pdMS_TO_TICKS(1)); /* Yield to feed WDT */
            } else {
                ESP_LOGI(TAG, "🔇 Background Noise [%d/%d] (Conf: %.2f) [RMS L:%.3f R:%.3f]",
                         vote_siren, VOTE_WINDOWS, confidence, rms_l, rms_r);
                display_st7735s_show_noise(confidence, rms_l, rms_r);

                /* Send "no siren" to traffic controller */
                detection_msg_t msg = {
                    .siren_active = false,
                    .direction = (lane_t)doa_dir,
                    .confidence = confidence
                };
                xQueueSend(g_traffic_queue, &msg, 0);

                vTaskDelay(pdMS_TO_TICKS(1)); /* Yield to feed WDT */
            }
            vote_siren = 0;
            vote_count = 0;
        } else {
            /* Intermediate frame update */
            if (pred == 0) {
                display_st7735s_show_noise(confidence, rms_l, rms_r);
                vTaskDelay(pdMS_TO_TICKS(1)); /* Yield to feed WDT */
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  app_main                                                          */
/* ------------------------------------------------------------------ */
void app_main(void)
{
    ESP_LOGI(TAG, "BUILD MARKER PHASE11 (ST7735S SPI TFT + Dual-Mic TDOA DoA Siren Detection)");

    /* ----------------------------------------------------------------
     * BOOT-TIME MEMORY SNAPSHOT
     * Runs before anything is initialized, so it reflects the true
     * static (.bss + .data) footprint of the firmware. After init we
     * will print again to see what got carved out of the heaps.
     * ---------------------------------------------------------------- */
    ESP_LOGI("MEM", "===== BOOT MEMORY SNAPSHOT =====");
    ESP_LOGI("MEM", "Internal free heap (before init) : %u bytes",
             (unsigned)esp_get_free_heap_size());
    ESP_LOGI("MEM", "Min-ever free heap (before init) : %u bytes",
             (unsigned)xPortGetMinimumEverFreeHeapSize());
    ESP_LOGI("MEM", "PSRAM free (before init)         : %u bytes",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI("MEM", "PSRAM min-ever free (before init): %u bytes",
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI("MEM", "=================================");

    /* Initialize Display */
    if (display_st7735s_init() != ESP_OK) {
        ESP_LOGW(TAG, "ST7735S Display Init Failed or Skipped");
    } else {
        ESP_LOGI(TAG, "ST7735S Display Init: OK");
    }

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

    if (traffic_ctrl_init() != ESP_OK) {
        ESP_LOGE(TAG, "Traffic Controller Init: FAIL");
        return;
    }
    ESP_LOGI(TAG, "Traffic Controller Init: OK");

#ifdef RP_PARITY_TEST
    run_parity_test();
#endif

    /* ----------------------------------------------------------------
     * POST-INIT MEMORY SNAPSHOT
     * After all inits (including the 200 KB tensor arena in PSRAM
     * and the traffic_ctrl queue), see what's left in each pool.
     * ---------------------------------------------------------------- */
    ESP_LOGI("MEM", "===== POST-INIT MEMORY SNAPSHOT =====");
    ESP_LOGI("MEM", "Internal free heap (after init) : %u bytes",
             (unsigned)esp_get_free_heap_size());
    ESP_LOGI("MEM", "Min-ever free heap (after init) : %u bytes",
             (unsigned)xPortGetMinimumEverFreeHeapSize());
    ESP_LOGI("MEM", "PSRAM free (after init)         : %u bytes",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI("MEM", "PSRAM min-ever free (after init): %u bytes",
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI("MEM", "Largest free internal block     : %u bytes",
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    ESP_LOGI("MEM", "Largest free PSRAM block        : %u bytes",
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM));
    ESP_LOGI("MEM", "====================================");

    /* Save the task handles so we can query their high-water marks later.
     * (Previously these were NULL because we didn't need them.) */
    TaskHandle_t hCapture   = NULL;
    TaskHandle_t hInference = NULL;

    xTaskCreatePinnedToCore(audio_capture_task, "audio_capture",
                            TASK_STACK_CAPTURE, NULL, 5, &hCapture, 0);
    xTaskCreatePinnedToCore(inference_task, "inference",
                            TASK_STACK_INFERENCE, NULL, 4, &hInference, 1);

    ESP_LOGI(TAG, "Pipeline started: capture(Core0) -> inference(Core1)");

    /* ----------------------------------------------------------------
     * RUNTIME MEMORY MONITOR TASK
     * Dedicated low-priority task that prints memory + HWM stats once
     * every 10 seconds. It also reads each work-task's HWM so you can
     * see how much stack headroom they actually have.
     * ---------------------------------------------------------------- */
    xTaskCreatePinnedToCore(memory_monitor_task, "mem_monitor",
                            4096, (void *)0, 1, NULL, 0);

    /* Keep hCapture / hInference alive so the monitor task can use them
     * via globals (declared below). */
    g_h_capture   = hCapture;
    g_h_inference = hInference;
}

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