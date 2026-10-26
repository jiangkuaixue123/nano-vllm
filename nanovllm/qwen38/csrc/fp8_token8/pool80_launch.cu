// SPDX-License-Identifier: MIT
// Inactive compiler discriminator derived from the retained isolated probe binding.
// Host CUDA APIs below are compilation roots only: this package does not authorize loading or launch.
// Device derivatives retain their original licenses.
#include <string>
#include <cstdio>
#include <stdexcept>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#define deep_gemm probe_fp8_token_n8_batch
#include "vendor/nvrtc_cutlass.cuh"
#include "token8_mma.cuh"
#include "vendor/fp8_gemm_impl.cuh"
#include "coupled_kernel.cuh"
#undef deep_gemm

namespace {
namespace dg = probe_fp8_token_n8_batch;
PFN_cuTensorMapEncodeTiled_v12000 encode_tiled = nullptr;
int prepared_device = -1;
constexpr int TILE = 128, THREADS = 384, SMS = 78;
constexpr int SHAPE_N[] = {34816, 16384, 14336};
constexpr int SHAPE_K[] = {5120, 5120, 5120};
constexpr int CASES = 3;

void checked(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template <bool Coupled> constexpr int token_tile() { return 8; }
template <bool Coupled> constexpr int stages() { return Coupled ? 4 : 8; }
template <int Case, bool Coupled> auto kernel() {
  constexpr int N = SHAPE_N[Case], K = SHAPE_K[Case], BN = token_tile<Coupled>();
  using Scheduler = dg::NormalSchedulerSwapAB<N, TILE, BN, 1, 1>;
  if constexpr (Coupled)
    return dg::fp8_gemm_kernel_swapAB_coupled<N, K, TILE, BN, 128, 1, 4, 128, 128, 1,
                                           Scheduler, dg::NormalSchedulerInputSwapAB>;
  else
    return dg::fp8_gemm_kernel_swapAB<N, K, TILE, BN, 128, 1, 8, 128, 128, 1,
                                     Scheduler, dg::NormalSchedulerInputSwapAB>;
}
template <int Case, bool Coupled> constexpr int dynamic_bytes() {
  constexpr int K = SHAPE_K[Case], BN = token_tile<Coupled>();
  return TILE * BN * 2 + stages<Coupled>() * (TILE * 128 + BN * 128 + 128)
      + ((K / 128 * 4 + 7) / 8) * 8 + stages<Coupled>() * 8 * 2;
}

static_assert(dynamic_bytes<0, false>() == 142624 && dynamic_bytes<0, true>() == 72416);
static_assert(dynamic_bytes<1, false>() == 142624 && dynamic_bytes<1, true>() == 72416);
static_assert(dynamic_bytes<2, false>() == 142624 && dynamic_bytes<2, true>() == 72416);

cudaDeviceProp properties() {
  int device = -1;
  checked(cudaGetDevice(&device));
  cudaDeviceProp props{};
  checked(cudaGetDeviceProperties(&props, device));
  if (props.major != 9 || props.minor != 0 || props.multiProcessorCount != SMS)
    throw std::runtime_error("fixed SM90/78-SM target required");
  return props;
}

template <int Case, bool Coupled> void resource_values(unsigned long long* values) {
  auto props = properties();
  cudaFuncAttributes a{};
  checked(cudaFuncGetAttributes(&a, kernel<Case, Coupled>()));
  values[0]=a.numRegs; values[1]=a.sharedSizeBytes; values[2]=a.localSizeBytes;
  values[3]=a.maxThreadsPerBlock; values[4]=a.maxDynamicSharedSizeBytes;
  values[5]=dynamic_bytes<Case, Coupled>(); values[6]=props.sharedMemPerBlockOptin;
  values[7]=props.regsPerBlock; values[8]=props.multiProcessorCount;
  values[9]=a.binaryVersion;
}

template <int Case, bool Coupled> void prepare_kernel() {
  unsigned long long v[10]{};
  resource_values<Case, Coupled>(v);
  if (v[1] + v[5] > v[6] || v[0] * THREADS > v[7] || v[2] != 0 || v[3] < THREADS)
    throw std::runtime_error("actual compiled resource admission failed");
  checked(cudaFuncSetAttribute(kernel<Case, Coupled>(), cudaFuncAttributeMaxDynamicSharedMemorySize,
                              dynamic_bytes<Case, Coupled>()));
}

template <int Case, bool Coupled>
void launch(void* q, void* activation_scales, void* weight, void* weight_scales,
            void* output, cudaStream_t stream) {
  constexpr int N = SHAPE_N[Case], K = SHAPE_K[Case];
  uint64_t weight_dims[2] = {K, N};
  uint32_t weight_box[2] = {128, TILE};
  auto a = dg::make_2d_tma_copy_desc(static_cast<__nv_fp8_e4m3*>(weight),
      weight_dims, K, weight_box, CU_TENSOR_MAP_SWIZZLE_128B, encode_tiled);
  uint64_t activation_dims[2] = {K, 1};
  uint32_t activation_box[2] = {128, token_tile<Coupled>()};
  auto b = dg::make_2d_tma_copy_desc(static_cast<__nv_fp8_e4m3*>(q),
      activation_dims, K, activation_box, CU_TENSOR_MAP_SWIZZLE_128B, encode_tiled);
  uint64_t scale_dims[2] = {4, K / 128};
  uint32_t scale_box[2] = {token_tile<Coupled>(), 1};
  auto bs = dg::make_2d_tma_copy_desc(static_cast<float*>(activation_scales),
      scale_dims, 16, scale_box, CU_TENSOR_MAP_SWIZZLE_NONE, encode_tiled);
  uint64_t output_dims[2] = {N, 1};
  uint32_t output_box[2] = {TILE, 1};
  auto d = dg::make_2d_tma_copy_desc(static_cast<__nv_bfloat16*>(output),
      output_dims, N * 2, output_box, CU_TENSOR_MAP_SWIZZLE_NONE, encode_tiled);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {1, 1, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = Coupled ? 156 : SMS; config.blockDim = THREADS;
  config.dynamicSmemBytes = dynamic_bytes<Case, Coupled>(); config.stream = stream;
  config.attrs = &attr; config.numAttrs = 1;
  dg::NormalSchedulerInputSwapAB input{1, nullptr};
  checked(cudaLaunchKernelEx(&config, kernel<Case, Coupled>(),
      static_cast<__nv_bfloat16*>(output), static_cast<float*>(weight_scales), input, a, b, bs, d));
}

template <class Function>
int boundary(Function function, char* error, size_t capacity) noexcept {
  try { function(); return 0; }
  catch (std::exception const& exc) {
    if (error && capacity) std::snprintf(error, capacity, "%s", exc.what());
  } catch (...) {
    if (error && capacity) std::snprintf(error, capacity, "unknown token_n8 host failure");
  }
  return 1;
}
}
#define API extern "C" __attribute__((visibility("default")))
API int coupled_n8_resources(int case_id, int coupled, unsigned long long* values, char* error, size_t capacity) noexcept {
  return boundary([&] {
    if (!values || case_id < 0 || case_id >= CASES || (coupled != 0 && coupled != 1))
      throw std::runtime_error("fixed resource arguments");
    switch (case_id) {
      case 0: if (coupled) resource_values<0, true>(values); else resource_values<0, false>(values); break;
      case 1: if (coupled) resource_values<1, true>(values); else resource_values<1, false>(values); break;
      case 2: if (coupled) resource_values<2, true>(values); else resource_values<2, false>(values); break;
    }
  }, error, capacity);
}
API int coupled_n8_prepare(char* error, size_t capacity) noexcept {
  return boundary([&] {
    int device=-1; checked(cudaGetDevice(&device));
    if (prepared_device != -1 && prepared_device != device) throw std::runtime_error("device changed");
    prepare_kernel<0, false>(); prepare_kernel<0, true>();
    prepare_kernel<1, false>(); prepare_kernel<1, true>();
    prepare_kernel<2, false>(); prepare_kernel<2, true>();
    encode_tiled=dg::get_cuTensorMapEncodeTiled(); prepared_device=device;
  }, error, capacity);
}
API int coupled_n8_launch(int case_id, int coupled, void* q, void* activation_scales, void* weight,
    void* weight_scales, void* output, void* stream, char* error, size_t capacity) noexcept {
  return boundary([&] {
    int device=-1; checked(cudaGetDevice(&device));
    if (!encode_tiled || device != prepared_device || case_id < 0 || case_id >= CASES || (coupled != 0 && coupled != 1))
      throw std::runtime_error("batch token_n8 runtime not prepared or unknown case");
    if (!q || !activation_scales || !weight || !weight_scales || !output)
      throw std::runtime_error("null tensor pointer");
    switch (case_id) {
      case 0: if (coupled) launch<0, true>(q,activation_scales,weight,weight_scales,output,static_cast<cudaStream_t>(stream));
              else launch<0, false>(q,activation_scales,weight,weight_scales,output,static_cast<cudaStream_t>(stream)); break;
      case 1: if (coupled) launch<1, true>(q,activation_scales,weight,weight_scales,output,static_cast<cudaStream_t>(stream));
              else launch<1, false>(q,activation_scales,weight,weight_scales,output,static_cast<cudaStream_t>(stream)); break;
      case 2: if (coupled) launch<2, true>(q,activation_scales,weight,weight_scales,output,static_cast<cudaStream_t>(stream));
              else launch<2, false>(q,activation_scales,weight,weight_scales,output,static_cast<cudaStream_t>(stream)); break;
    }
  }, error, capacity);
}
