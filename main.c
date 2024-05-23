#include "raylib.h"
#include "ctensor.h"
#include <stdlib.h>
#include <stdio.h>

#define SCREEN_W    448
#define SCREEN_H    896
#define IMAGE_W     28
#define IMAGE_H     28
#define NUM_CLASSES 10
#define TARGET_FPS  60
#define PIXEL_SIZE  16


void predict(
    ctensor_tensor_t *w1,
    ctensor_tensor_t *b1,
    ctensor_tensor_t *w2,
    ctensor_tensor_t *b2,
    ctensor_tensor_t *input,
    float *scores
) {
    ctensor_arena_t *arena = ctensor_arena_create(200000);

    ctensor_tensor_t *a1 = ctensor_multiply(arena, input, w1, 1, 0); // [1, 64]
    ctensor_tensor_t *t_b1 = ctensor_permute(arena, b1, (uint64_t[]) { 1, 0 }); // [1, 64]
    ctensor_tensor_t *z1 = ctensor_add(arena, a1, t_b1); // [1, 64]
    ctensor_tensor_t *o1 = ctensor_relu(arena, z1); // [1, 64]
    ctensor_tensor_t *a2 = ctensor_multiply(arena, o1, w2, 1, 0); // [1, 10]
    ctensor_tensor_t *t_b2 = ctensor_permute(arena, b2, (uint64_t[]) { 1, 0 }); // [1, 10]
    ctensor_tensor_t *z2 = ctensor_add(arena, a2, t_b2); // [1, 10]

    float max_value = -10.0f;

    for (uint64_t i = 0; i < z2->num_items; i++) {
        if (z2->data[i] > max_value)
            max_value = z2->data[i];
    }

    for (uint64_t i = 0; i < z2->num_items; i++) {
        scores[i] = z2->data[i] / max_value;
    }
    
    ctensor_arena_destroy(arena);
}

void init_image(uint8_t *inference_image) {
    for (uint8_t i = 0; i < IMAGE_H; i++) {
        for (uint8_t j = 0; j < IMAGE_W; j++) {
            inference_image[i * IMAGE_W + j] = 0;
        }
    }
}

void handle_input(uint8_t *inference_image)
{
    Vector2 mouse_position = GetMousePosition();

    uint8_t col = (uint8_t)(mouse_position.x / PIXEL_SIZE);
    uint8_t row = (uint8_t)(mouse_position.y / PIXEL_SIZE);

    if (col < IMAGE_W && row < IMAGE_H)
    {
        if (IsMouseButtonDown(0))
        {
            inference_image[row * IMAGE_W + col] = 250;
        }
        if (IsMouseButtonDown(1))
        {
            inference_image[row * IMAGE_W + col] = 0;
        }
    }

    if (IsKeyPressed(KEY_R))
    {
        init_image(inference_image);
    }
}

void draw_everything(const uint8_t *inference_image, const float *scores)
{
    BeginDrawing();
    ClearBackground(DARKGRAY);

    // Image Canvas
    for (uint8_t i = 0; i < IMAGE_H; i++)
    {
        for (uint8_t j = 0; j < IMAGE_W; j++)
        {
            uint8_t pixel = inference_image[i * IMAGE_W + j];
            DrawRectangle(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE, (Color){pixel, pixel, pixel, 255});
        }
    }

    DrawText("Draw a digit on the canvas above", 20, 40 + SCREEN_W, 20, LIGHTGRAY);
    DrawText("Press [R] to reset canvas", 20, 80 + SCREEN_W, 20, LIGHTGRAY);
    DrawText("Press [ESC] to exit", 20, 120 + SCREEN_W, 20, LIGHTGRAY);
    DrawText("Prediction:", 20, 200 + SCREEN_W, 20, LIGHTGRAY);

    for (uint8_t i = 0; i < NUM_CLASSES; i++)
    {
        uint8_t c = (uint8_t)(255 * scores[i]);
        char label[2];
        sprintf(label, "%d", i);

        DrawRectangle(20 + 35 * i, 240 + SCREEN_W, 30, 30, (Color){0, c, 0, 255});
        DrawText(label, 24 + 35 * i, 280 + SCREEN_W, 30, (Color){0, c, 0, 255});
    }

    DrawFPS(360, 840);
    EndDrawing();
}

int main() {
    uint8_t inference_image[IMAGE_H * IMAGE_W];
    float scores[10];

    ctensor_arena_t *arena = ctensor_arena_create(300000);

    ctensor_tensor_t *w1 = ctensor_read_file(arena, "./data/export_dense_1_0.ctensor");
    ctensor_tensor_t *b1 = ctensor_read_file(arena, "./data/export_dense_1_1.ctensor");
    ctensor_tensor_t *w2 = ctensor_read_file(arena, "./data/export_dense_2_0.ctensor");
    ctensor_tensor_t *b2 = ctensor_read_file(arena, "./data/export_dense_2_1.ctensor");

    ctensor_tensor_t *input = ctensor_create_tensor(arena, (uint64_t []) { 1, 784 }, 2);

    InitWindow(SCREEN_W, SCREEN_H, "MNIST Inference");
    SetTargetFPS(TARGET_FPS);

    init_image(inference_image);

    while (!WindowShouldClose()) {
        handle_input(inference_image);

        for (uint64_t i=0; i < IMAGE_H * IMAGE_W; i++) {
            input->data[i] = (float) (inference_image[i] / 255.0f);
        }

        predict(w1, b1, w2, b2, input, scores);

        draw_everything(inference_image, scores);
    }

    CloseWindow();
    ctensor_arena_destroy(arena);

    return 0;
}
