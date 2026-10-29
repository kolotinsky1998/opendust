// Warp-tiled direct softened-Yukawa forces for the JAX XLA FFI.

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kTileSize = 32;
constexpr int kThreadsPerBlock = 256;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

__device__ __forceinline__ float warp_get(float value, int lane) {
  return __shfl_sync(kFullWarpMask, value, lane);
}

__device__ __forceinline__ int warp_get(int value, int lane) {
  return __shfl_sync(kFullWarpMask, value, lane);
}

__device__ __forceinline__ void add_force(float* forces, int atom, float fx,
                                           float fy, float fz) {
  if (fx != 0.0f) atomicAdd(forces + 3 * atom, fx);
  if (fy != 0.0f) atomicAdd(forces + 3 * atom + 1, fy);
  if (fz != 0.0f) atomicAdd(forces + 3 * atom + 2, fz);
}

__device__ __forceinline__ void decode_triangular_tile(
    std::uint64_t tile, int* block_x, int* block_y) {
  // Tiles are enumerated as (0,0), (1,0), (1,1), (2,0), ... .  The
  // correction loops make the mapping exact when sqrtf rounds near a row edge.
  int x = static_cast<int>((sqrtf(8.0f * static_cast<float>(tile) + 1.0f) -
                            1.0f) *
                           0.5f);
  std::uint64_t row_start =
      static_cast<std::uint64_t>(x) * static_cast<std::uint64_t>(x + 1) / 2;
  while (row_start > tile) {
    --x;
    row_start = static_cast<std::uint64_t>(x) *
                static_cast<std::uint64_t>(x + 1) / 2;
  }
  std::uint64_t next_row = static_cast<std::uint64_t>(x + 1) *
                           static_cast<std::uint64_t>(x + 2) / 2;
  while (next_row <= tile) {
    ++x;
    row_start = next_row;
    next_row = static_cast<std::uint64_t>(x + 1) *
               static_cast<std::uint64_t>(x + 2) / 2;
  }
  *block_x = x;
  *block_y = static_cast<int>(tile - row_start);
}

__global__ void SoftenedYukawaKernel(
    const float* __restrict__ positions, const float* __restrict__ prefactor_ptr,
    const float* __restrict__ screening_length_ptr,
    const float* __restrict__ softening_ptr, float* __restrict__ forces,
    int num_atoms, std::uint64_t num_tiles) {
  const int lane = threadIdx.x & (kTileSize - 1);
  const std::uint64_t global_warp =
      (static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
      kTileSize;
  const std::uint64_t total_warps =
      (static_cast<std::uint64_t>(gridDim.x) * blockDim.x) / kTileSize;

  const float prefactor = *prefactor_ptr;
  const float inverse_screening_length = 1.0f / *screening_length_ptr;
  const float softening = *softening_ptr;

  for (std::uint64_t tile = global_warp; tile < num_tiles;
       tile += total_warps) {
    int block_x;
    int block_y;
    decode_triangular_tile(tile, &block_x, &block_y);

    const int atom_i = block_x * kTileSize + lane;
    int atom_j = block_y * kTileSize + lane;
    const bool valid_i = atom_i < num_atoms;

    float ix = 0.0f;
    float iy = 0.0f;
    float iz = 0.0f;
    if (valid_i) {
      ix = positions[3 * atom_i];
      iy = positions[3 * atom_i + 1];
      iz = positions[3 * atom_i + 2];
    }

    float jx = 0.0f;
    float jy = 0.0f;
    float jz = 0.0f;
    if (atom_j < num_atoms) {
      jx = positions[3 * atom_j];
      jy = positions[3 * atom_j + 1];
      jz = positions[3 * atom_j + 2];
    }

    float fix = 0.0f;
    float fiy = 0.0f;
    float fiz = 0.0f;
    float fjx = 0.0f;
    float fjy = 0.0f;
    float fjz = 0.0f;
    const bool diagonal = block_x == block_y;

#pragma unroll
    for (int step = 0; step < kTileSize; ++step) {
      const bool valid_pair = valid_i && atom_j < num_atoms &&
                              (!diagonal || atom_i < atom_j);
      if (valid_pair) {
        const float dx = ix - jx;
        const float dy = iy - jy;
        const float dz = iz - jz;
        const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
        const float inverse_r = rsqrtf(r2);
        const float r = r2 * inverse_r;
        const float denominator = r + softening;
        const float inverse_denominator = 1.0f / denominator;
        const float coefficient =
            prefactor * __expf(-r * inverse_screening_length) * inverse_r *
            (inverse_screening_length * inverse_denominator +
             inverse_denominator * inverse_denominator);
        const float fx = coefficient * dx;
        const float fy = coefficient * dy;
        const float fz = coefficient * dz;
        fix += fx;
        fiy += fy;
        fiz += fz;
        fjx -= fx;
        fjy -= fy;
        fjz -= fz;
      }

      const int source_lane = (lane + 1) & (kTileSize - 1);
      atom_j = warp_get(atom_j, source_lane);
      jx = warp_get(jx, source_lane);
      jy = warp_get(jy, source_lane);
      jz = warp_get(jz, source_lane);
      fjx = warp_get(fjx, source_lane);
      fjy = warp_get(fjy, source_lane);
      fjz = warp_get(fjz, source_lane);
    }

    // After 32 rotations the j data is back in its original lane.  A diagonal
    // tile can combine both roles of an atom into one set of atomic updates.
    if (diagonal) {
      if (valid_i) add_force(forces, atom_i, fix + fjx, fiy + fjy, fiz + fjz);
    } else {
      if (valid_i) add_force(forces, atom_i, fix, fiy, fiz);
      if (atom_j < num_atoms) add_force(forces, atom_j, fjx, fjy, fjz);
    }
  }
}

ffi::Error CudaError(const char* operation, cudaError_t error) {
  return ffi::Error::Internal(std::string(operation) + ": " +
                              cudaGetErrorString(error));
}

ffi::Error SoftenedYukawaHost(
    cudaStream_t stream, ffi::Buffer<ffi::F32> positions,
    ffi::Buffer<ffi::F32> prefactor,
    ffi::Buffer<ffi::F32> screening_length, ffi::Buffer<ffi::F32> softening,
    ffi::ResultBuffer<ffi::F32> forces) {
  const auto position_dims = positions.dimensions();
  const auto force_dims = forces->dimensions();
  if (position_dims.size() != 2 || position_dims[1] != 3) {
    return ffi::Error::InvalidArgument("positions must have shape (N, 3)");
  }
  if (force_dims.size() != 2 || force_dims[0] != position_dims[0] ||
      force_dims[1] != 3) {
    return ffi::Error::InvalidArgument("forces must have shape (N, 3)");
  }
  if (!prefactor.dimensions().empty() ||
      !screening_length.dimensions().empty() ||
      !softening.dimensions().empty()) {
    return ffi::Error::InvalidArgument("force parameters must be scalars");
  }

  const std::int64_t n64 = position_dims[0];
  if (n64 <= 0 || n64 > static_cast<std::int64_t>(INT32_MAX)) {
    return ffi::Error::InvalidArgument("particle count is out of range");
  }
  const int num_atoms = static_cast<int>(n64);
  cudaError_t error = cudaMemsetAsync(forces->typed_data(), 0,
                                      3 * n64 * sizeof(float), stream);
  if (error != cudaSuccess) return CudaError("cudaMemsetAsync failed", error);
  if (num_atoms == 1) return ffi::Error::Success();

  const std::uint64_t num_atom_blocks =
      (static_cast<std::uint64_t>(num_atoms) + kTileSize - 1) / kTileSize;
  const std::uint64_t num_tiles =
      num_atom_blocks * (num_atom_blocks + 1) / 2;

  int device = 0;
  int multiprocessors = 0;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return CudaError("cudaGetDevice failed", error);
  error = cudaDeviceGetAttribute(&multiprocessors,
                                 cudaDevAttrMultiProcessorCount, device);
  if (error != cudaSuccess) {
    return CudaError("cudaDeviceGetAttribute failed", error);
  }
  const int warps_per_block = kThreadsPerBlock / kTileSize;
  const std::uint64_t blocks_for_tiles =
      (num_tiles + warps_per_block - 1) / warps_per_block;
  const int grid_size = static_cast<int>(std::min(
      blocks_for_tiles, static_cast<std::uint64_t>(4 * multiprocessors)));

  SoftenedYukawaKernel<<<grid_size, kThreadsPerBlock, 0, stream>>>(
      positions.typed_data(), prefactor.typed_data(),
      screening_length.typed_data(), softening.typed_data(),
      forces->typed_data(), num_atoms, num_tiles);
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("kernel launch failed", error);
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OpenDustSoftenedYukawa, SoftenedYukawaHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());
