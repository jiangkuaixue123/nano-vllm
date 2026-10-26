// SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// Derivative of retained mma_utils.cuh; see MIT and Apache-2.0 license files.
#pragma once
#include "vendor/mma_utils.cuh"

namespace deep_gemm {
struct SM90_64x8x32_F32E4M3E4M3_SS {
  __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float& d00,
                               float& d01, float& d02, float& d03, bool scale_d) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %6, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3"
        "{%0,   %1,   %2,   %3},"
        " %4,"
        " %5,"
        " p   , 1,    1;\n"
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03)
        : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
  }

  __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d,
                               bool scale_d) {
    wgmma(desc_a, desc_b, d[0], d[1], d[2], d[3], scale_d);
  }

  static constexpr int M = 64;
  static constexpr int N = 8;
  static constexpr int K = 32;
  static constexpr int kNumAccum = M * N / 128;
};

// The original selector and every original vendor header remain byte-identical.
template <> struct FP8MMASelector<8> {
  using type = SM90_64x8x32_F32E4M3E4M3_SS;
};
}  // namespace deep_gemm
