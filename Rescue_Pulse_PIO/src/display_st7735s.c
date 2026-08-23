#include "display_st7735s.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "esp_lcd_panel_io.h"
#include "esp_lcd_panel_vendor.h"
#include "esp_lcd_panel_ops.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

static const char *TAG = "st7735s";

static esp_lcd_panel_handle_t s_panel_handle = NULL;

/* ------------------------------------------------------------------ */
/*  Full 128x128 RAM Framebuffer (32 KB)                              */
/*  All UI drawing is executed instantly in memory to eliminate DMA   */
/*  racing, font tearing, and SPI line corruption.                    */
/* ------------------------------------------------------------------ */
static uint16_t s_framebuffer[LCD_H_RES * LCD_V_RES];

/* ------------------------------------------------------------------ */
/*  Asynchronous UI Task Queue                                        */
/*  Allows inference_task to post updates without blocking on SPI DMA */
/* ------------------------------------------------------------------ */
typedef enum {
    UI_STATE_SPLASH = 0,
    UI_STATE_IDLE,
    UI_STATE_NOISE,
    UI_STATE_SIREN
} ui_state_type_t;

typedef struct {
    ui_state_type_t state;
    ui_direction_t  dir;
    float           confidence;
    int             lag;
    float           rms_l;
    float           rms_r;
} ui_message_t;

static QueueHandle_t s_ui_queue = NULL;

/* Standard 8x8 ASCII Font (Characters 32 ' ' through 126 '~') */
static const uint8_t font8x8_basic[95][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, /*   */
    {0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00}, /* ! */
    {0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00}, /* " */
    {0x6C,0x6C,0xFE,0x6C,0xFE,0x6C,0x6C,0x00}, /* # */
    {0x18,0x3E,0x60,0x3C,0x06,0x7C,0x18,0x00}, /* $ */
    {0x00,0x63,0x66,0x0C,0x18,0x33,0x63,0x00}, /* % */
    {0x38,0x6C,0x38,0x76,0xDC,0xCC,0x76,0x00}, /* & */
    {0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00}, /* ' */
    {0x0C,0x18,0x30,0x30,0x30,0x18,0x0C,0x00}, /* ( */
    {0x30,0x18,0x0C,0x0C,0x0C,0x18,0x30,0x00}, /* ) */
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00}, /* * */
    {0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00}, /* + */
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x30}, /* , */
    {0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00}, /* - */
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00}, /* . */
    {0x03,0x06,0x0C,0x18,0x30,0x60,0x40,0x00}, /* / */
    {0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0x00}, /* 0 */
    {0x18,0x18,0x38,0x18,0x18,0x18,0x7E,0x00}, /* 1 */
    {0x3C,0x66,0x06,0x0C,0x18,0x30,0x7E,0x00}, /* 2 */
    {0x3C,0x66,0x06,0x1C,0x06,0x66,0x3C,0x00}, /* 3 */
    {0x06,0x0E,0x1E,0x36,0x66,0x7F,0x06,0x00}, /* 4 */
    {0x7E,0x60,0x7C,0x06,0x06,0x66,0x3C,0x00}, /* 5 */
    {0x1C,0x30,0x60,0x7C,0x66,0x66,0x3C,0x00}, /* 6 */
    {0x7E,0x66,0x06,0x0C,0x18,0x18,0x18,0x00}, /* 7 */
    {0x3C,0x66,0x66,0x3C,0x66,0x66,0x3C,0x00}, /* 8 */
    {0x3C,0x66,0x66,0x3E,0x06,0x0C,0x38,0x00}, /* 9 */
    {0x00,0x18,0x18,0x00,0x18,0x18,0x00,0x00}, /* : */
    {0x00,0x18,0x18,0x00,0x18,0x18,0x30,0x00}, /* ; */
    {0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00}, /* < */
    {0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00}, /* = */
    {0x60,0x30,0x18,0x0C,0x18,0x30,0x60,0x00}, /* > */
    {0x3C,0x66,0x06,0x0C,0x18,0x00,0x18,0x00}, /* ? */
    {0x3C,0x66,0x6E,0x6E,0x60,0x62,0x3C,0x00}, /* @ */
    {0x18,0x3C,0x66,0x66,0x7E,0x66,0x66,0x00}, /* A */
    {0x7C,0x66,0x66,0x7C,0x66,0x66,0x7C,0x00}, /* B */
    {0x3C,0x66,0x60,0x60,0x60,0x66,0x3C,0x00}, /* C */
    {0x78,0x6C,0x66,0x66,0x66,0x6C,0x78,0x00}, /* D */
    {0x7E,0x60,0x60,0x7C,0x60,0x60,0x7E,0x00}, /* E */
    {0x7E,0x60,0x60,0x7C,0x60,0x60,0x60,0x00}, /* F */
    {0x3C,0x66,0x60,0x6E,0x66,0x66,0x3C,0x00}, /* G */
    {0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x00}, /* H */
    {0x3C,0x18,0x18,0x18,0x18,0x18,0x3C,0x00}, /* I */
    {0x0F,0x06,0x06,0x06,0x66,0x66,0x3C,0x00}, /* J */
    {0x66,0x6C,0x78,0x70,0x78,0x6C,0x66,0x00}, /* K */
    {0x60,0x60,0x60,0x60,0x60,0x60,0x7E,0x00}, /* L */
    {0x63,0x77,0x7F,0x6B,0x63,0x63,0x63,0x00}, /* M */
    {0x66,0x76,0x7E,0x7E,0x6E,0x66,0x66,0x00}, /* N */
    {0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00}, /* O */
    {0x7C,0x66,0x66,0x7C,0x60,0x60,0x60,0x00}, /* P */
    {0x3C,0x66,0x66,0x66,0x6A,0x6C,0x36,0x00}, /* Q */
    {0x7C,0x66,0x66,0x7C,0x6C,0x66,0x66,0x00}, /* R */
    {0x3C,0x66,0x60,0x3C,0x06,0x66,0x3C,0x00}, /* S */
    {0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x00}, /* T */
    {0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00}, /* U */
    {0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00}, /* V */
    {0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00}, /* W */
    {0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x00}, /* X */
    {0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x00}, /* Y */
    {0x7E,0x06,0x0C,0x18,0x30,0x60,0x7E,0x00}, /* Z */
    {0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00}, /* [ */
    {0x60,0x30,0x18,0x0C,0x06,0x03,0x01,0x00}, /* \\ */
    {0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00}, /* ] */
    {0x18,0x3C,0x66,0x00,0x00,0x00,0x00,0x00}, /* ^ */
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF}, /* _ */
    {0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00}, /* ` */
    {0x00,0x00,0x3C,0x06,0x3E,0x66,0x3B,0x00}, /* a */
    {0x60,0x60,0x7C,0x66,0x66,0x66,0x7C,0x00}, /* b */
    {0x00,0x00,0x3C,0x66,0x60,0x66,0x3C,0x00}, /* c */
    {0x06,0x06,0x3E,0x66,0x66,0x66,0x3E,0x00}, /* d */
    {0x00,0x00,0x3C,0x66,0x7E,0x60,0x3C,0x00}, /* e */
    {0x1C,0x30,0x7C,0x30,0x30,0x30,0x30,0x00}, /* f */
    {0x00,0x00,0x3B,0x66,0x66,0x3E,0x06,0x3C}, /* g */
    {0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x00}, /* h */
    {0x18,0x00,0x38,0x18,0x18,0x18,0x3C,0x00}, /* i */
    {0x0C,0x00,0x0C,0x0C,0x0C,0x0C,0x6C,0x38}, /* j */
    {0x60,0x60,0x66,0x6C,0x78,0x6C,0x66,0x00}, /* k */
    {0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00}, /* l */
    {0x00,0x00,0x66,0x7F,0x7F,0x6B,0x63,0x00}, /* m */
    {0x00,0x00,0x7C,0x66,0x66,0x66,0x66,0x00}, /* n */
    {0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x00}, /* o */
    {0x00,0x00,0x7C,0x66,0x66,0x7C,0x60,0x60}, /* p */
    {0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x06}, /* q */
    {0x00,0x00,0x7C,0x66,0x60,0x60,0x60,0x00}, /* r */
    {0x00,0x00,0x3E,0x60,0x3C,0x06,0x7C,0x00}, /* s */
    {0x18,0x18,0x7E,0x18,0x18,0x18,0x0E,0x00}, /* t */
    {0x00,0x00,0x66,0x66,0x66,0x66,0x3B,0x00}, /* u */
    {0x00,0x00,0x66,0x66,0x66,0x3C,0x18,0x00}, /* v */
    {0x00,0x00,0x63,0x6B,0x7F,0x3E,0x36,0x00}, /* w */
    {0x00,0x00,0x66,0x3C,0x18,0x3C,0x66,0x00}, /* x */
    {0x00,0x00,0x66,0x66,0x66,0x3E,0x06,0x3C}, /* y */
    {0x00,0x00,0x7E,0x0C,0x18,0x30,0x7E,0x00}, /* z */
    {0x0E,0x18,0x18,0x70,0x18,0x18,0x0E,0x00}, /* { */
    {0x18,0x18,0x18,0x18,0x18,0x18,0x18,0x00}, /* | */
    {0x70,0x18,0x18,0x0E,0x18,0x18,0x70,0x00}, /* } */
    {0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00}, /* ~ */
};

/* Swap bytes for ST7735S big-endian SPI transfer format */
static inline uint16_t swap_rgb565(uint16_t color) {
    return (uint16_t)((color << 8) | (color >> 8));
}

/* ------------------------------------------------------------------ */
/*  In-Memory Framebuffer Drawing Primitives                          */
/* ------------------------------------------------------------------ */

static void fb_fill_rect(int x1, int y1, int w, int h, uint16_t color)
{
    if (w <= 0 || h <= 0) return;
    if (x1 < 0) { w += x1; x1 = 0; }
    if (y1 < 0) { h += y1; y1 = 0; }
    if (x1 + w > LCD_H_RES) w = LCD_H_RES - x1;
    if (y1 + h > LCD_V_RES) h = LCD_V_RES - y1;
    if (w <= 0 || h <= 0) return;

    uint16_t c_swapped = swap_rgb565(color);
    for (int y = y1; y < y1 + h; y++) {
        int row_offset = y * LCD_H_RES;
        for (int x = x1; x < x1 + w; x++) {
            s_framebuffer[row_offset + x] = c_swapped;
        }
    }
}

static void fb_draw_char(int x, int y, char c, uint16_t fg, uint16_t bg, int scale)
{
    if (c < 32 || c > 126) c = ' ';
    const uint8_t *bitmap = font8x8_basic[c - 32];
    uint16_t fg_s = swap_rgb565(fg);
    uint16_t bg_s = swap_rgb565(bg);

    for (int row = 0; row < 8; row++) {
        uint8_t bits = bitmap[row];
        int py = y + row * scale;
        if (py >= LCD_V_RES) break;

        for (int col = 0; col < 8; col++) {
            int px = x + col * scale;
            if (px >= LCD_H_RES) break;

            uint16_t color = (bits & (0x80 >> col)) ? fg_s : bg_s;
            for (int sy = 0; sy < scale && (py + sy) < LCD_V_RES; sy++) {
                int row_offset = (py + sy) * LCD_H_RES;
                for (int sx = 0; sx < scale && (px + sx) < LCD_H_RES; sx++) {
                    s_framebuffer[row_offset + px + sx] = color;
                }
            }
        }
    }
}

static void fb_draw_string(int x, int y, const char *str, uint16_t fg, uint16_t bg, int scale)
{
    int cursor_x = x;
    while (*str) {
        if (cursor_x + 8 * scale > LCD_H_RES) break;
        fb_draw_char(cursor_x, y, *str, fg, bg, scale);
        cursor_x += 8 * scale;
        str++;
    }
}

static void fb_draw_centered_string(int y, const char *str, uint16_t fg, uint16_t bg, int scale)
{
    int len = (int)strlen(str);
    int total_w = len * 8 * scale;
    int x = (LCD_H_RES - total_w) / 2;
    if (x < 0) x = 0;
    fb_draw_string(x, y, str, fg, bg, scale);
}

/* ------------------------------------------------------------------ */
/*  UI Component Rendering (128x128)                                  */
/* ------------------------------------------------------------------ */

static void render_header(const char *title, uint16_t bg_color, uint16_t text_color)
{
    fb_fill_rect(0, 0, LCD_H_RES, 16, bg_color);
    fb_draw_centered_string(4, title, text_color, bg_color, 1);
}

static void render_rms_bars(int bar_y, float rms_l, float rms_r, uint16_t bg_color)
{
    const int bar_x = 18;
    const int bar_w = 72;
    const int bar_h = 8;
    const int val_x = 94;

    /* Channel L */
    fb_draw_string(2, bar_y, "L:", COLOR_WHITE, bg_color, 1);
    fb_fill_rect(bar_x, bar_y, bar_w, bar_h, COLOR_DARK_GRAY);
    int fill_l = (int)(rms_l * 500.0f);
    if (fill_l > bar_w) fill_l = bar_w;
    if (fill_l < 0) fill_l = 0;
    uint16_t col_l = (rms_l > 0.08f) ? COLOR_RED : (rms_l > 0.04f ? COLOR_YELLOW : COLOR_GREEN);
    if (fill_l > 0) fb_fill_rect(bar_x, bar_y, fill_l, bar_h, col_l);

    char val_str[8];
    snprintf(val_str, sizeof(val_str), "%.2f", rms_l);
    fb_draw_string(val_x, bar_y, val_str, COLOR_LIGHT_GRAY, bg_color, 1);

    /* Channel R */
    int r_y = bar_y + 14;
    fb_draw_string(2, r_y, "R:", COLOR_WHITE, bg_color, 1);
    fb_fill_rect(bar_x, r_y, bar_w, bar_h, COLOR_DARK_GRAY);
    int fill_r = (int)(rms_r * 500.0f);
    if (fill_r > bar_w) fill_r = bar_w;
    if (fill_r < 0) fill_r = 0;
    uint16_t col_r = (rms_r > 0.08f) ? COLOR_RED : (rms_r > 0.04f ? COLOR_YELLOW : COLOR_GREEN);
    if (fill_r > 0) fb_fill_rect(bar_x, r_y, fill_r, bar_h, col_r);

    snprintf(val_str, sizeof(val_str), "%.2f", rms_r);
    fb_draw_string(val_x, r_y, val_str, COLOR_LIGHT_GRAY, bg_color, 1);
}

/* ------------------------------------------------------------------ */
/*  Scene Renderers (Render to RAM, then push to LCD via DMA)         */
/* ------------------------------------------------------------------ */

static void render_splash_scene(void)
{
    fb_fill_rect(0, 0, LCD_H_RES, LCD_V_RES, COLOR_BLACK);
    render_header("RESCUE PULSE", COLOR_NAVY, COLOR_WHITE);

    fb_draw_centered_string(24, "EDGE AI", COLOR_CYAN, COLOR_BLACK, 2);
    fb_draw_centered_string(44, "SIREN DETECT", COLOR_CYAN, COLOR_BLACK, 1);

    fb_draw_centered_string(60, "ESP32-S3+TFLite", COLOR_WHITE, COLOR_BLACK, 1);
    fb_draw_centered_string(74, "Dual-Mic DoA", COLOR_LIGHT_GRAY, COLOR_BLACK, 1);

    fb_draw_centered_string(100, "INITIALIZING...", COLOR_GREEN, COLOR_BLACK, 1);
}

static void render_idle_scene(float rms_l, float rms_r)
{
    fb_fill_rect(0, 0, LCD_H_RES, LCD_V_RES, COLOR_BLACK);
    render_header("RESCUE PULSE", COLOR_DARK_GRAY, COLOR_CYAN);

    fb_fill_rect(4, 20, LCD_H_RES - 8, 56, COLOR_DARK_GRAY);
    fb_draw_centered_string(28, "LISTEN", COLOR_GREEN, COLOR_DARK_GRAY, 2);
    fb_draw_centered_string(50, "Waiting...", COLOR_LIGHT_GRAY, COLOR_DARK_GRAY, 1);
    fb_draw_centered_string(62, "Ambient OK", COLOR_LIGHT_GRAY, COLOR_DARK_GRAY, 1);

    render_rms_bars(84, rms_l, rms_r, COLOR_BLACK);
}

static void render_noise_scene(float confidence, float rms_l, float rms_r)
{
    fb_fill_rect(0, 0, LCD_H_RES, LCD_V_RES, COLOR_BLACK);
    render_header("NOISE DET.", COLOR_ORANGE, COLOR_BLACK);

    fb_fill_rect(4, 20, LCD_H_RES - 8, 56, COLOR_DARK_GRAY);
    fb_draw_centered_string(24, "TRAFFIC", COLOR_YELLOW, COLOR_DARK_GRAY, 2);

    char conf_buf[16];
    snprintf(conf_buf, sizeof(conf_buf), "Conf: %d%%", (int)(confidence * 100.0f));
    fb_draw_centered_string(46, conf_buf, COLOR_WHITE, COLOR_DARK_GRAY, 1);

    fb_draw_centered_string(60, "Non-Emergency", COLOR_LIGHT_GRAY, COLOR_DARK_GRAY, 1);

    render_rms_bars(84, rms_l, rms_r, COLOR_BLACK);
}

static void render_siren_scene(ui_direction_t dir, float confidence, int lag, float rms_l, float rms_r)
{
    fb_fill_rect(0, 0, LCD_H_RES, LCD_V_RES, COLOR_BLACK);
    render_header("!! SIREN !!", COLOR_RED, COLOR_WHITE);

    fb_fill_rect(4, 20, LCD_H_RES - 8, 56, COLOR_DARK_RED);

    if (dir == UI_DIR_LEFT) {
        fb_draw_centered_string(24, "<< LEFT", COLOR_WHITE, COLOR_DARK_RED, 2);
        fb_draw_centered_string(46, "FROM LEFT", COLOR_YELLOW, COLOR_DARK_RED, 1);
    } else if (dir == UI_DIR_RIGHT) {
        fb_draw_centered_string(24, "RIGHT >>", COLOR_WHITE, COLOR_DARK_RED, 2);
        fb_draw_centered_string(46, "FROM RIGHT", COLOR_YELLOW, COLOR_DARK_RED, 1);
    } else {
        fb_draw_centered_string(24, "CENTER", COLOR_WHITE, COLOR_DARK_RED, 2);
        fb_draw_centered_string(46, "FRONT/BACK", COLOR_YELLOW, COLOR_DARK_RED, 1);
    }

    char detail_buf[20];
    snprintf(detail_buf, sizeof(detail_buf), "C:%d%% Lag:%d", (int)(confidence * 100.0f), lag);
    fb_draw_centered_string(60, detail_buf, COLOR_WHITE, COLOR_DARK_RED, 1);

    render_rms_bars(84, rms_l, rms_r, COLOR_BLACK);
}

/* ------------------------------------------------------------------ */
/*  Dedicated FreeRTOS UI Display Task                                */
/* ------------------------------------------------------------------ */
static void ui_display_task(void *pvParameters)
{
    ui_message_t msg;
    ESP_LOGI(TAG, "ui_display_task started on core %d", xPortGetCoreID());

    while (1) {
        if (xQueueReceive(s_ui_queue, &msg, portMAX_DELAY) == pdTRUE) {
            if (s_panel_handle == NULL) continue;

            switch (msg.state) {
                case UI_STATE_SPLASH:
                    render_splash_scene();
                    break;
                case UI_STATE_IDLE:
                    render_idle_scene(msg.rms_l, msg.rms_r);
                    break;
                case UI_STATE_NOISE:
                    render_noise_scene(msg.confidence, msg.rms_l, msg.rms_r);
                    break;
                case UI_STATE_SIREN:
                    render_siren_scene(msg.dir, msg.confidence, msg.lag, msg.rms_l, msg.rms_r);
                    break;
                default:
                    break;
            }

            /* Single full-frame DMA transfer */
            esp_lcd_panel_draw_bitmap(s_panel_handle, 0, 0, LCD_H_RES, LCD_V_RES, s_framebuffer);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Public Display API                                                */
/* ------------------------------------------------------------------ */

esp_err_t display_st7735s_init(void)
{
    ESP_LOGI(TAG, "Initializing ST7735S SPI LCD (SCLK=%d MOSI=%d DC=%d CS=%d RST=%d BL=%d) %dx%d",
             LCD_PIN_SCLK, LCD_PIN_MOSI, LCD_PIN_DC, LCD_PIN_CS, LCD_PIN_RST, LCD_PIN_BL,
             LCD_H_RES, LCD_V_RES);

    /* Backlight GPIO */
    if (LCD_PIN_BL >= 0) {
        gpio_config_t bl_cfg = {
            .pin_bit_mask = (1ULL << LCD_PIN_BL),
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        gpio_config(&bl_cfg);
        gpio_set_level(LCD_PIN_BL, 1);
    }

    /* SPI Bus */
    spi_bus_config_t buscfg = {
        .sclk_io_num = LCD_PIN_SCLK,
        .mosi_io_num = LCD_PIN_MOSI,
        .miso_io_num = -1,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = LCD_H_RES * LCD_V_RES * sizeof(uint16_t),
    };
    esp_err_t ret = spi_bus_initialize(SPI2_HOST, &buscfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "spi_bus_initialize failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Panel IO */
    esp_lcd_panel_io_handle_t io_handle = NULL;
    esp_lcd_panel_io_spi_config_t io_config = {
        .dc_gpio_num = LCD_PIN_DC,
        .cs_gpio_num = LCD_PIN_CS,
        .pclk_hz = 20 * 1000 * 1000,
        .lcd_cmd_bits = 8,
        .lcd_param_bits = 8,
        .spi_mode = 0,
        .trans_queue_depth = 10,
    };
    ret = esp_lcd_new_panel_io_spi((esp_lcd_spi_bus_handle_t)SPI2_HOST, &io_config, &io_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_io_spi failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* ST7735S Panel Driver */
    esp_lcd_panel_dev_config_t panel_config = {
        .reset_gpio_num = LCD_PIN_RST,
        .rgb_ele_order = LCD_RGB_ELEMENT_ORDER_BGR,
        .bits_per_pixel = 16,
    };
    ret = esp_lcd_new_panel_st7789(io_handle, &panel_config, &s_panel_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_lcd_new_panel_st7789 failed: %s", esp_err_to_name(ret));
        return ret;
    }

    esp_lcd_panel_reset(s_panel_handle);
    esp_lcd_panel_init(s_panel_handle);

    /* Orientation: Mirror X and Y so text reads upright with pins at the top */
    esp_lcd_panel_mirror(s_panel_handle, true, true);

    /* Panel offset for 1.44" 128x128 ST7735S module */
    esp_lcd_panel_set_gap(s_panel_handle, LCD_COL_OFFSET, LCD_ROW_OFFSET);

    esp_lcd_panel_invert_color(s_panel_handle, false);
    esp_lcd_panel_disp_on_off(s_panel_handle, true);

    /* Create UI Message Queue (Length 1 is required for xQueueOverwrite) */
    s_ui_queue = xQueueCreate(1, sizeof(ui_message_t));
    if (s_ui_queue == NULL) {
        ESP_LOGE(TAG, "Failed to create UI queue");
        return ESP_FAIL;
    }

    /* Launch dedicated UI Task on Core 0 (priority 2: below audio capture priority 5) */
    xTaskCreatePinnedToCore(ui_display_task, "ui_display", 4096, NULL, 2, NULL, 0);

    /* Queue Splash Screen */
    ui_message_t splash_msg = { .state = UI_STATE_SPLASH };
    xQueueOverwrite(s_ui_queue, &splash_msg);

    vTaskDelay(pdMS_TO_TICKS(500));
    return ESP_OK;
}

void display_st7735s_show_idle(float rms_l, float rms_r)
{
    if (s_ui_queue == NULL) return;
    ui_message_t msg = {
        .state = UI_STATE_IDLE,
        .rms_l = rms_l,
        .rms_r = rms_r
    };
    xQueueOverwrite(s_ui_queue, &msg);
}

void display_st7735s_show_noise(float confidence, float rms_l, float rms_r)
{
    if (s_ui_queue == NULL) return;
    ui_message_t msg = {
        .state = UI_STATE_NOISE,
        .confidence = confidence,
        .rms_l = rms_l,
        .rms_r = rms_r
    };
    xQueueOverwrite(s_ui_queue, &msg);
}

void display_st7735s_show_siren(ui_direction_t dir, float confidence, int lag, float rms_l, float rms_r)
{
    if (s_ui_queue == NULL) return;
    ui_message_t msg = {
        .state = UI_STATE_SIREN,
        .dir = dir,
        .confidence = confidence,
        .lag = lag,
        .rms_l = rms_l,
        .rms_r = rms_r
    };
    xQueueOverwrite(s_ui_queue, &msg);
}
