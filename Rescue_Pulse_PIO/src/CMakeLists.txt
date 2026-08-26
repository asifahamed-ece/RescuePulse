idf_component_register(SRCS "main.c" "mfcc.c" "inference.cpp" "model_data.cc" "i2s_capture.c" "display_st7735s.c" "traffic_ctrl.c"
                       INCLUDE_DIRS "."
                       REQUIRES esp-dsp esp-tflite-micro esp_timer driver esp_driver_i2s esp_lcd)