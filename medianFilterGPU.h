#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <sycl/sycl.hpp>
#include "utils.h"



class MedianFilterGPU {
private:
    static float median_7(float arr[7]);
    static uint8_t median_9(uint8_t window[9]);

public:
    static void median_filter_7(const float* input, float* output, size_t length);
    static void median_filter_3x3_gpu(const uint8_t* input, uint8_t* output, size_t width, size_t height, size_t stride);
};

//сортирующа€ сеть на 7 элементов
float MedianFilterGPU::median_7(float arr[7]) {
    cond_swap(arr[0], arr[6]);
    cond_swap(arr[2], arr[3]);
    cond_swap(arr[4], arr[5]);

    cond_swap(arr[0], arr[2]);
    cond_swap(arr[1], arr[4]);
    cond_swap(arr[3], arr[6]);

    arr[1] = get_max(arr[0], arr[1]);
    cond_swap(arr[2], arr[5]);
    cond_swap(arr[3], arr[4]);

    arr[2] = get_max(arr[1], arr[2]);
    arr[4] = get_min(arr[4], arr[6]);

    arr[3] = get_max(arr[2], arr[3]);
    arr[4] = get_min(arr[4], arr[5]);

    arr[3] = get_min(arr[3], arr[4]);

    return arr[3];
}

void MedianFilterGPU::median_filter_7(const float* input, float* output, size_t length) {
    sycl::queue q;

    size_t N = length;
    float* d_input = sycl::malloc_shared<float>(N, q);
    float* d_output = sycl::malloc_shared<float>(N, q);

    q.memcpy(d_input, input, N * sizeof(float)).wait();

    //центральные элементы вычисл€ем на стороне девайса
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(N - 6), [=](sycl::id<1> idx) {
            size_t i = idx[0] + 3;//каждый поток обрабатывает свой индекс
            float window[7];//локальный массив дл€ каждого потока

            for (int j = -3; j <= 3; ++j) window[j + 3] = d_input[i + j];

            d_output[i] = median_7(window);
        });
    });
    q.wait();

    //краевые элементы вычисл€ем на стороне хоста
    float window[7];

    //первые 3 элемента
    for (size_t i = 0; i < 3 && i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    //последние 3 элемента
    for (size_t i = (N > 3 ? N - 3 : 0); i < N; ++i) {
        for (int j = -3; j <= 3; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx < 0) window[j + 3] = d_input[0];
            else if (idx >= static_cast<int>(N)) window[j + 3] = d_input[N - 1];
            else window[j + 3] = d_input[idx];
        }
        d_output[i] = median_7(window);
    }

    //закончили вычислени€
    q.memcpy(output, d_output, N * sizeof(float)).wait();

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

uint8_t MedianFilterGPU::median_9(uint8_t window[9]) {
    cond_swap(window[0], window[3]);
    cond_swap(window[1], window[7]);
    cond_swap(window[2], window[5]);
    cond_swap(window[4], window[8]);

    cond_swap(window[0], window[7]);
    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[8]);
    cond_swap(window[5], window[6]);

    window[2] = get_max(window[0], window[2]);
    cond_swap(window[1], window[3]);
    cond_swap(window[4], window[5]);
    window[7] = get_min(window[7], window[8]);

    window[4] = get_max(window[1], window[4]);
    window[3] = get_min(window[3], window[6]);
    window[5] = get_min(window[5], window[7]);

    cond_swap(window[2], window[4]);
    cond_swap(window[3], window[5]);

    window[3] = get_max(window[2], window[3]);
    window[4] = get_min(window[4], window[5]);

    window[4] = get_max(window[3], window[4]);

    return window[4];
}

void MedianFilterGPU::median_filter_3x3_gpu(const uint8_t* input, uint8_t* output,
    size_t width, size_t height, size_t stride) {

    // –асширенное изображение (padding = 1)
    const size_t pad = 1;
    const size_t pw = width + 2 * pad;
    const size_t ph = height + 2 * pad;
    std::vector<uint8_t> padded(pw * ph, 0);

    //  опируем оригинал в центр
    for (size_t y = 0; y < height; ++y) {
        const uint8_t* src = input + y * stride;
        uint8_t* dst = padded.data() + (y + pad) * pw + pad;
        std::memcpy(dst, src, width);
    }
    // √раницы (дублирование крайних пикселей)
    for (size_t x = 0; x < width; ++x) {
        padded[pad * pw + (x + pad)] = input[0 * stride + x];
        padded[(ph - pad - 1) * pw + (x + pad)] = input[(height - 1) * stride + x];
    }
    for (size_t y = 0; y < height; ++y) {
        padded[(y + pad) * pw + pad] = input[y * stride + 0];
        padded[(y + pad) * pw + pw - pad - 1] = input[y * stride + width - 1];
    }
    padded[0] = input[0];
    padded[pw - 1] = input[width - 1];
    padded[(ph - 1) * pw] = input[(height - 1) * stride];
    padded[(ph - 1) * pw + pw - 1] = input[(height - 1) * stride + width - 1];

    // –азмеры рабочей группы
    constexpr size_t WG_X = 16;
    constexpr size_t WG_Y = 16;
    const size_t grid_x = (width + WG_X - 1) / WG_X;
    const size_t grid_y = (height + WG_Y - 1) / WG_Y;
    const size_t global_x = grid_x * WG_X;
    const size_t global_y = grid_y * WG_Y;

    sycl::queue q;
    uint8_t* d_padded = sycl::malloc_shared<uint8_t>(pw * ph, q);
    uint8_t* d_output = sycl::malloc_shared<uint8_t>(width * height, q);
    q.memcpy(d_padded, padded.data(), pw * ph).wait();

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> tile(sycl::range<2>(WG_Y + 2, WG_X + 2), h);
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(global_y, global_x),
                sycl::range<2>(WG_Y, WG_X)),
            [=](sycl::nd_item<2> it) {
                const int lx = it.get_local_id(1);
                const int ly = it.get_local_id(0);
                const int gx = it.get_global_id(1);
                const int gy = it.get_global_id(0);
                const int bx = it.get_group(1) * WG_X;
                const int by = it.get_group(0) * WG_Y;

                const int tile_w = WG_X + 2;
                const int tile_h = WG_Y + 2;
                const int local_size = WG_X * WG_Y;
                const int lid = ly * WG_X + lx;

                //  ооперативна€ загрузка с проверкой границ
                for (int idx = lid; idx < tile_w * tile_h; idx += local_size) {
                    int ty = idx / tile_w;
                    int tx = idx % tile_w;
                    int img_x = bx + tx - 1;
                    int img_y = by + ty - 1;
                    // Ѕезопасное ограничение индексов
                    img_x = clamp_int(img_x, 0, static_cast<int>(pw) - 1);
                    img_y = clamp_int(img_y, 0, static_cast<int>(ph) - 1);
                    tile[ty][tx] = d_padded[img_y * pw + img_x];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // ¬ычисл€ем медиану дл€ каждого пиксел€
                if (gx < width && gy < height) {
                    const int tx = lx + 1;
                    const int ty = ly + 1;
                    uint8_t window[9] = {
                        tile[ty - 1][tx - 1], tile[ty - 1][tx], tile[ty - 1][tx + 1],
                        tile[ty][tx - 1],   tile[ty][tx],   tile[ty][tx + 1],
                        tile[ty + 1][tx - 1], tile[ty + 1][tx], tile[ty + 1][tx + 1]
                    };
                    d_output[gy * width + gx] = median_9(window);
                }
            }
        );
        }).wait();

    q.memcpy(output, d_output, width * height).wait();
    sycl::free(d_padded, q);
    sycl::free(d_output, q);
}