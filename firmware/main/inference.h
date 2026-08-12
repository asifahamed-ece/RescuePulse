#pragma once
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the TFLite Micro interpreter (model + arena).
 * Returns true on success. */
bool inference_init(void);

/* Run a single forward pass.
 *  input : 832 int8 values  (1 x 64 x 13)
 *  output: 2 int8 values    (1 x 2)  written by the call
 * Returns true on success. */
bool inference_run(const int8_t *input, int8_t *output);

#ifdef __cplusplus
}
#endif
