#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  ST7735S SPI TFT Pin Configuration (ESP32-S3) — 1.44" 128×128     */
/* ------------------------------------------------------------------ */
#define LCD_PIN_SCLK        12     /* SPI Clock */
#define LCD_PIN_MOSI        11     /* SPI Master Out / Data (SDA) */
#define LCD_PIN_DC          10     /* Data / Command */
#define LCD_PIN_CS           9     /* Chip Select (-1 if tied to GND) */
#define LCD_PIN_RST          8     /* Reset (-1 if tied to EN/3.3V) */
#define LCD_PIN_BL           7     /* Backlight control (-1 if tied to 3.3V) */

/* 1.44" ST7735S 128×128 resolution */
#define LCD_H_RES          128
#define LCD_V_RES          128

/* ST7735S panel pixel offset for 1.44" 128x128 modules */
#define LCD_COL_OFFSET       2
#define LCD_ROW_OFFSET       3

/* ------------------------------------------------------------------ */
/*  Color definitions (RGB565)                                        */
/* ------------------------------------------------------------------ */
#define COLOR_BLACK        0x0000
#define COLOR_WHITE        0xFFFF
#define COLOR_RED          0xF800
#define COLOR_GREEN        0x07E0
#define COLOR_BLUE         0x001F
#define COLOR_CYAN         0x07FF
#define COLOR_MAGENTA      0xF81F
#define COLOR_YELLOW       0xFFE0
#define COLOR_ORANGE       0xFD20
#define COLOR_DARK_GRAY    0x2104
#define COLOR_LIGHT_GRAY   0x8410
#define COLOR_NAVY         0x000F
#define COLOR_DARK_RED     0x7800

/* Direction enum matching main pipeline */
typedef enum {
    UI_DIR_CENTER = 0,
    UI_DIR_LEFT,
    UI_DIR_RIGHT
} ui_direction_t;

/* ------------------------------------------------------------------ */
/*  Display API                                                       */
/* ------------------------------------------------------------------ */

/**
 * @brief Initialize ST7735S SPI display, start async UI task, and render splash screen.
 * @return ESP_OK on success.
 */
esp_err_t display_st7735s_init(void);

/**
 * @brief Show Idle / Listening state with real-time RMS bars (Non-blocking).
 */
void display_st7735s_show_idle(float rms_l, float rms_r);

/**
 * @brief Show High Noise state (Non-blocking).
 */
void display_st7735s_show_noise(float confidence, float rms_l, float rms_r);

/**
 * @brief Show Siren Detected state with Direction of Arrival alert (Non-blocking).
 */
void display_st7735s_show_siren(ui_direction_t dir, float confidence, int lag, float rms_l, float rms_r);

#ifdef __cplusplus
}
#endif
