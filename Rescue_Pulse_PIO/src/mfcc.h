#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void mfcc_init(void);
void mfcc_extract_block(const int16_t *pcm, float out[64][13]);
void mfcc_get_mel_energies(float out[64][40]);

#ifdef __cplusplus
}
#endif