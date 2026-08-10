idf_component_register(SRCS "main.c" "mfcc.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp-dsp esp_timer)