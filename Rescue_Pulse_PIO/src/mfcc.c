#include "mfcc.h"
#include "mel_tables.h"
#include <math.h>
#include "esp_dsp.h" 

#define N_FFT      512
#define HOP        256
#define N_MELS     40
#define N_MFCC     13
#define N_WIN      64
#define N_SAMPLES  ((N_WIN - 1) * HOP + N_FFT)   /* 16640 */

/* All buffers static - no heap allocation after init */
static float s_fft[2 * N_FFT];        /* 1024 */
static float s_power[N_FFT / 2 + 1];  /* 257  */
static float s_mel[N_MELS];           /* 40   */
static float s_db[N_WIN][N_MELS];     /* 64x40 */
static float s_y[N_SAMPLES];          /* 16640 pre-emphasized */

void mfcc_init(void)
{
    esp_err_t ret = dsps_fft2r_init_fc32(NULL, N_FFT);
    ESP_LOGI("mfcc", "BUILD MARKER v2");
    if (ret != ESP_OK) {
        ESP_LOGE("mfcc", "FFT init failed: %d", ret);
    }
}

void mfcc_extract_block(const int16_t *pcm, float out[64][13])
{
    /* a) x[n]=pcm[n]/32768f; y[n]=x[n]-0.97f*x[n-1] (y[0]=x[0]) */
    s_y[0] = (float)pcm[0] / 32768.0f;
    for (int n = 1; n < N_SAMPLES; n++) {
        s_y[n] = (float)pcm[n] / 32768.0f - 0.97f * ((float)pcm[n - 1] / 32768.0f);
    }

    /* b) per frame t (start=t*256) */
    for (int t = 0; t < N_WIN; t++) {
        int start = t * HOP;

        /* window + pack real sequence as interleaved complex */
        for (int i = 0; i < N_FFT; i++) {
            s_fft[2 * i]     = s_y[start + i] * g_ham[i];
            s_fft[2 * i + 1] = 0.0f;
        }
        
        // Perform Real FFT. 
        dsps_fft2r_fc32(s_fft, N_FFT);
        dsps_bit_rev_fc32(s_fft, N_FFT);

        /* Unpack the special format into power spectrum */
        s_power[0] = s_fft[0] * s_fft[0];                       /* DC bin, Re[0] */
        for (int k = 1; k < N_FFT / 2; k++) {
            float re = s_fft[2 * k];
            float im = s_fft[2 * k + 1];
            s_power[k] = re * re + im * im;
        }
        s_power[N_FFT / 2] = s_fft[N_FFT] * s_fft[N_FFT];        /* Nyquist bin, Re[N/2] */

        /* c) mel_e[m]=dot(g_mel_fb+m*257, power);
              db=10*log10f(fmaxf(mel_e,1e-10f)) */
        for (int m = 0; m < N_MELS; m++) {
            const float *fb = g_mel_fb + m * (N_FFT / 2 + 1);
            float e = 0.0f;
            for (int k = 0; k <= N_FFT / 2; k++) {
                e += fb[k] * s_power[k];
            }
            s_mel[m] = e;
        }
        for (int m = 0; m < N_MELS; m++) {
            s_db[t][m] = 10.0f * log10f(fmaxf(s_mel[m], 1e-10f));
        }
    }

    /* d) AFTER all 64 frames: mmax over whole 64x40 db block;
          db=fmaxf(db, mmax-80.0f)  <- librosa power_to_db top_db=80 */
    float mmax = s_db[0][0];
    for (int t = 0; t < N_WIN; t++) {
        for (int m = 0; m < N_MELS; m++) {
            if (s_db[t][m] > mmax) mmax = s_db[t][m];
        }
    }
    float floor_val = mmax - 80.0f;
    for (int t = 0; t < N_WIN; t++) {
        for (int m = 0; m < N_MELS; m++) {
            if (s_db[t][m] < floor_val) s_db[t][m] = floor_val;
        }
    }

    /* e) out[t][c]=dot(g_dct+c*40, db+t*40) */
    for (int t = 0; t < N_WIN; t++) {
        for (int c = 0; c < N_MFCC; c++) {
            const float *dct_row = g_dct + c * N_MELS;
            const float *db_row  = s_db[t];
            float v = 0.0f;
            for (int m = 0; m < N_MELS; m++) {
                v += dct_row[m] * db_row[m];
            }
            out[t][c] = v;
        }
    }
}

void mfcc_get_mel_energies(float out[64][40])
{
    /* Copy the mel-band log energies (s_db is already computed in mfcc_extract_block) */
    for (int t = 0; t < N_WIN; t++) {
        for (int m = 0; m < N_MELS; m++) {
            out[t][m] = s_db[t][m];
        }
    }
}