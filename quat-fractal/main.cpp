#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <raylib.h>
#include <raymath.h>

#include "kulib/kulib.hpp"

namespace {
    static constexpr int width = 800; 
    static constexpr int height = 450;
    static constexpr int antialias_samples = 2;
    static constexpr int max_iter = 40;

    static constexpr Color bg_color      = {0x1B, 0x1B, 0x1B, 0xFF};
    static constexpr Color teal_accent   = {0x00, 0x80, 0x67, 0xFF};
    static constexpr Color orange_accent = {0xFF, 0x90, 0x00, 0xFF};
}

Color calcPixelValue(int px, int py, float time, kulib::quaternion<float> c) {
    float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
    float total_samples = (float)(antialias_samples * antialias_samples);
    float aspect_ratio = (float)width / (float)height;

    for (int sy = 0; sy < antialias_samples; ++sy) {
        for (int sx = 0; sx < antialias_samples; ++sx) {
            
            float jx = 2.2f * (2.0f * (px + (float)sx / antialias_samples) / width - 1.0f) * aspect_ratio;
            float jy = 2.2f * (2.0f * (py + (float)sy / antialias_samples) / height - 1.0f);

            kulib::quaternion<float> quat = { jx, jy, 0.0f, 0.0f };

            int iter = 0;
            while (quat.norm2() < 4.0f && iter < max_iter) {
                // Quaternion Julia Set: z = z^2 + c
                quat = (quat * quat) + c;
                iter++;
            }

            Color sampleColor;
            if (iter == max_iter) {
                sampleColor = bg_color;
            } else {
                // Smooth escape-time coloring
                float dist = quat.norm();
                float smoothed = (float)iter - log2f(fmaxf(1.0f, log2f(dist)));

                // Animate color phase over time
                float t = 0.5f + 0.5f * sinf(smoothed * 0.15f + time * 0.5f);
                t = Clamp(powf(t, 1.2f), 0.0f, 1.0f);
                
                sampleColor = ColorLerp(teal_accent, orange_accent, t);
                
                // Vignette/Depth effect based on escape speed
                float depth = 0.4f + 0.6f * (1.0f - expf(-smoothed * 0.08f));
                sampleColor.r = (unsigned char)(sampleColor.r * depth);
                sampleColor.g = (unsigned char)(sampleColor.g * depth);
                sampleColor.b = (unsigned char)(sampleColor.b * depth);
            }

            r_acc += sampleColor.r;
            g_acc += sampleColor.g;
            b_acc += sampleColor.b;
        }
    }

    return Color{
        (unsigned char)(r_acc / total_samples),
        (unsigned char)(g_acc / total_samples),
        (unsigned char)(b_acc / total_samples),
        255
    };
}

void renderFrame(std::vector<Color>& frame_buffer, float time, kulib::quaternion<float> c) {
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            frame_buffer[y * width + x] = calcPixelValue(x, y, time, c);
        }
    }
}

int main() {
    InitWindow(width, height, "Raylib - Morphing Quaternion Julia");
    SetTargetFPS(60);

    std::vector<Color> frame_buf(width * height, BLACK);
    Image canvasImage = { frame_buf.data(), width, height, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
    Texture2D canvasTexture = LoadTextureFromImage(canvasImage);

    while (!WindowShouldClose()) {
        float time = (float)GetTime();

        kulib::quaternion<float> dynamicC = {
            -0.745f + sinf(time * 0.4f) * 0.08f,
            0.113f + cosf(time * 0.25f) * 0.06f,
            0.05f * sinf(time * 0.15f),
            0.0f
        };

        renderFrame(frame_buf, time, dynamicC);

        UpdateTexture(canvasTexture, frame_buf.data());

        BeginDrawing();
        
            ClearBackground(BLACK);
            DrawTexture(canvasTexture, 0, 0, WHITE);
            
            DrawFPS(10, 10);
            DrawText("Processing: OpenMP (CPU Float)", 10, 35, 10, GREEN);
            DrawText(TextFormat("C: %.3f, %.3fi, %.3fj", dynamicC.real(), dynamicC.x(), dynamicC.y()), 10, 50, 10, GRAY);
        
        EndDrawing();
    }

    UnloadTexture(canvasTexture);
    CloseWindow();

    return 0;
}