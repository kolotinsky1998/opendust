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
constexpr float kCoulombConstant = 8.9875517923e9f;

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
    int num_atoms, std::uint64_t num_tiles, int* coincident_pairs) {
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
        if (r2 > 0.0f) {
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
        } else {
          atomicAdd(coincident_pairs, 1);
        }
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

__device__ __forceinline__ void polynomial_gradient(
    float x, float y, float z, const float* a, float screening_length,
    float* gx, float* gy, float* gz) {
  const float radius = sqrtf(fmaf(x, x, y * y));
  const float rho = radius / screening_length;
  const float zeta = z / screening_length;
  const float r2 = rho * rho;
  const float z2 = zeta * zeta;
  const float r3 = r2 * rho;
  const float z3 = z2 * zeta;
  const float r4 = r2 * r2;
  const float z4 = z2 * z2;
  const float r5 = r4 * rho;
  const float z5 = z4 * zeta;
  const float r6 = r4 * r2;
  const float z6 = z4 * z2;
  const float r7 = r6 * rho;
  const float z7 = z6 * zeta;
  const float d_rho =
      2.0f * a[1] * rho + 2.0f * a[3] * rho * z2 +
      4.0f * a[4] * r3 + 4.0f * a[6] * r3 * z2 +
      2.0f * a[7] * rho * z4 + 6.0f * a[8] * r5 +
      4.0f * a[10] * r3 * z4 + 6.0f * a[11] * r5 * z2 +
      2.0f * a[12] * rho * z6 + 8.0f * a[13] * r7;
  const float d_zeta =
      2.0f * a[2] * zeta + 2.0f * a[3] * r2 * zeta +
      4.0f * a[5] * z3 + 2.0f * a[6] * r4 * zeta +
      4.0f * a[7] * r2 * z3 + 6.0f * a[9] * z5 +
      4.0f * a[10] * r4 * z3 + 2.0f * a[11] * r6 * zeta +
      6.0f * a[12] * r2 * z5 + 8.0f * a[14] * z7;
  const float inverse_length = 1.0f / screening_length;
  if (radius > 0.0f) {
    *gx = d_rho * x / radius * inverse_length;
    *gy = d_rho * y / radius * inverse_length;
  } else {
    *gx = 0.0f;
    *gy = 0.0f;
  }
  *gz = d_zeta * inverse_length;
}

__global__ void ExternalForcesKernel(
    const float* __restrict__ positions,
    const float* __restrict__ dust_positions,
    const float* __restrict__ dust_charges,
    const float* __restrict__ background_coefficients,
    const float* __restrict__ ion_charge_ptr,
    const float* __restrict__ screening_length_ptr,
    const float* __restrict__ electric_field_ptr,
    float* __restrict__ ion_forces, float* __restrict__ dust_orbit_forces,
    int num_ions, int num_dust) {
  const int ion = blockIdx.x * blockDim.x + threadIdx.x;
  if (ion >= num_ions) return;
  const float ion_charge = *ion_charge_ptr;
  const float screening_length = *screening_length_ptr;
  const float inverse_screening_length = 1.0f / screening_length;
  const float x = positions[3 * ion];
  const float y = positions[3 * ion + 1];
  const float z = positions[3 * ion + 2];
  float fx = 0.0f;
  float fy = 0.0f;
  float fz = ion_charge * (*electric_field_ptr);

  float pgx, pgy, pgz;
  polynomial_gradient(x, y, z, background_coefficients, screening_length,
                      &pgx, &pgy, &pgz);
  const float background_scale =
      -kCoulombConstant * ion_charge * ion_charge * inverse_screening_length;
  fx += background_scale * pgx;
  fy += background_scale * pgy;
  fz += background_scale * pgz;

  for (int dust = 0; dust < num_dust; ++dust) {
    const float dx = x - dust_positions[3 * dust];
    const float dy = y - dust_positions[3 * dust + 1];
    const float dz = z - dust_positions[3 * dust + 2];
    const float r2 = fmaf(dx, dx, fmaf(dy, dy, dz * dz));
    if (r2 <= 0.0f) continue;
    const float inverse_r = rsqrtf(r2);
    const float dust_charge = dust_charges[dust];
    const float bare = kCoulombConstant * ion_charge * dust_charge *
                       inverse_r * inverse_r * inverse_r;
    fx += bare * dx;
    fy += bare * dy;
    fz += bare * dz;

    const float r = r2 * inverse_r;
    const float screened =
        kCoulombConstant * ion_charge * dust_charge *
        __expf(-r * inverse_screening_length) *
        (inverse_r * inverse_r * inverse_r +
         inverse_screening_length * inverse_r * inverse_r);
    atomicAdd(dust_orbit_forces + 3 * dust, -screened * dx);
    atomicAdd(dust_orbit_forces + 3 * dust + 1, -screened * dy);
    atomicAdd(dust_orbit_forces + 3 * dust + 2, -screened * dz);
  }
  ion_forces[3 * ion] = fx;
  ion_forces[3 * ion + 1] = fy;
  ion_forces[3 * ion + 2] = fz;
}

ffi::Error CudaError(const char* operation, cudaError_t error) {
  return ffi::Error::Internal(std::string(operation) + ": " +
                              cudaGetErrorString(error));
}

ffi::Error SoftenedYukawaHost(
    cudaStream_t stream, ffi::Buffer<ffi::F32> positions,
    ffi::Buffer<ffi::F32> prefactor,
    ffi::Buffer<ffi::F32> screening_length, ffi::Buffer<ffi::F32> softening,
    ffi::ResultBuffer<ffi::F32> forces,
    ffi::ResultBuffer<ffi::S32> coincident_pairs) {
  const auto position_dims = positions.dimensions();
  const auto force_dims = forces->dimensions();
  if (position_dims.size() != 2 || position_dims[1] != 3) {
    return ffi::Error::InvalidArgument("positions must have shape (N, 3)");
  }
  if (force_dims.size() != 2 || force_dims[0] != position_dims[0] ||
      force_dims[1] != 3) {
    return ffi::Error::InvalidArgument("forces must have shape (N, 3)");
  }
  if (prefactor.dimensions().size() != 0 ||
      screening_length.dimensions().size() != 0 ||
      softening.dimensions().size() != 0) {
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
  error = cudaMemsetAsync(coincident_pairs->typed_data(), 0, sizeof(int), stream);
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
      forces->typed_data(), num_atoms, num_tiles,
      coincident_pairs->typed_data());
  error = cudaGetLastError();
  if (error != cudaSuccess) return CudaError("kernel launch failed", error);
  return ffi::Error::Success();
}

ffi::Error ExternalForcesHost(
    cudaStream_t stream, ffi::Buffer<ffi::F32> positions,
    ffi::Buffer<ffi::F32> dust_positions,
    ffi::Buffer<ffi::F32> dust_charges,
    ffi::Buffer<ffi::F32> background_coefficients,
    ffi::Buffer<ffi::F32> ion_charge,
    ffi::Buffer<ffi::F32> screening_length,
    ffi::Buffer<ffi::F32> electric_field,
    ffi::ResultBuffer<ffi::F32> ion_forces,
    ffi::ResultBuffer<ffi::F32> dust_orbit_forces) {
  const auto ion_dims = positions.dimensions();
  const auto dust_dims = dust_positions.dimensions();
  if (ion_dims.size() != 2 || ion_dims[1] != 3 ||
      dust_dims.size() != 2 || dust_dims[1] != 3) {
    return ffi::Error::InvalidArgument(
        "positions and dust_positions must have shape (N, 3)");
  }
  if (dust_charges.dimensions().size() != 1 ||
      dust_charges.dimensions()[0] != dust_dims[0]) {
    return ffi::Error::InvalidArgument("dust_charges must have shape (D,)");
  }
  if (background_coefficients.dimensions().size() != 1 ||
      background_coefficients.dimensions()[0] != 15) {
    return ffi::Error::InvalidArgument(
        "background_coefficients must have shape (15,)");
  }
  if (ion_charge.dimensions().size() != 0 ||
      screening_length.dimensions().size() != 0 ||
      electric_field.dimensions().size() != 0) {
    return ffi::Error::InvalidArgument("physical parameters must be scalars");
  }
  if (ion_forces->dimensions().size() != 2 ||
      ion_forces->dimensions()[0] != ion_dims[0] ||
      ion_forces->dimensions()[1] != 3 ||
      dust_orbit_forces->dimensions().size() != 2 ||
      dust_orbit_forces->dimensions()[0] != dust_dims[0] ||
      dust_orbit_forces->dimensions()[1] != 3) {
    return ffi::Error::InvalidArgument("invalid output shapes");
  }
  const std::int64_t n64 = ion_dims[0];
  const std::int64_t d64 = dust_dims[0];
  if (n64 <= 0 || n64 > INT32_MAX || d64 > INT32_MAX) {
    return ffi::Error::InvalidArgument("particle count is out of range");
  }
  cudaError_t error = cudaSuccess;
  if (d64 > 0) {
    error = cudaMemsetAsync(dust_orbit_forces->typed_data(), 0,
                            3 * d64 * sizeof(float), stream);
    if (error != cudaSuccess) return CudaError("cudaMemsetAsync failed", error);
  }
  const int blocks =
      (static_cast<int>(n64) + kThreadsPerBlock - 1) / kThreadsPerBlock;
  ExternalForcesKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      positions.typed_data(), dust_positions.typed_data(),
      dust_charges.typed_data(), background_coefficients.typed_data(),
      ion_charge.typed_data(), screening_length.typed_data(),
      electric_field.typed_data(), ion_forces->typed_data(),
      dust_orbit_forces->typed_data(), static_cast<int>(n64),
      static_cast<int>(d64));
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
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::S32>>());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    OpenDustExternalForces, ExternalForcesHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>());
