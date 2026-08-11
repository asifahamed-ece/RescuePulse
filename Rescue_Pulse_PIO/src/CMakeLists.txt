idf_component_register(SRCS "model_data.cc" "main.c" "mfcc.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp-dsp esp_timer)