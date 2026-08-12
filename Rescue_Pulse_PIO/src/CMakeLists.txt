idf_component_register(SRCS "main.c" "mfcc.c" "inference.cpp" "model_data.cc"
                       INCLUDE_DIRS "."
                       REQUIRES esp-dsp esp-tflite-micro esp_timer)
