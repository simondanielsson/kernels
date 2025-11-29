#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define COARSE_FACTOR 4

template<typename size_t>
__global__ void inclusive_scan_kernel_naive_single_block(const size_t *X, size_t *output, const int size) {
  // Simple version, n log n complexity
  // First sum every other element. Then double the stride, and sum i and i-2 (if in range)
  // continue until done
  uint i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  __shared__ float sX[BLOCK_SIZE];

  // Each thread loads one element from GMEM
  if (i < size) {
    sX[threadIdx.x] = X[i];
  } else {
    sX[threadIdx.x] = 0.0f;
  }

  for (uint stride = 1; stride < size; stride *= 2) {
    // all threads must update their SMEM before we can move on to the next wave
    __syncthreads();

    // Accumulate this stage into register to avoid race condition from other threads reading this value
    float acc;
    if (threadIdx.x >= stride) {
      acc = sX[threadIdx.x] + sX[threadIdx.x - stride];
    }

    // all threads must calculate their accumulator before we can update SMEM
    __syncthreads();
    if (threadIdx.x >= stride) 
      sX[threadIdx.x] = acc;
  }

  // Write output. No need for syncing as each thread only reads its own value
  if (i < size) {
    output[i] = sX[threadIdx.x];
  }
}

template<typename size_t>
__global__ void inclusive_scan_kernel_naive_single_block_coarse(const size_t *X, size_t *output, const int size) {
  // 
  /*
  Each thread is responsible for computing the result on COARSE_FACTOR threads.

  Steps:
  Input:                  0 1 2 3 | 4 5 6 7 | 8 9 10 11
  Local sequential scan:  0 1 3 6 | 4 9 15 22 | 8 17 27 38
  Load local sums:              0 6 22 
  Parallel scan on local sums:  0 6 28
  Increment each local chunk by the local sums: 0 1 3 6 | 10 15 21 28 | 34 ...
  */

  constexpr int NUM_ITERS = COARSE_FACTOR;
  constexpr int NUM_THREADS = BLOCK_SIZE / COARSE_FACTOR;
  uint i = blockIdx.x * blockDim.x + threadIdx.x;  

  __shared__ float sX[BLOCK_SIZE];
  __shared__ float sLocalSums[NUM_THREADS];

  // 0. Cooperatively load from GMEM in coalesced fashion
#pragma unroll
  for (uint iter = 0; iter < NUM_ITERS; ++iter) {
    int idx = NUM_THREADS * iter + i;
    sX[idx] = X[idx];
  }
  __syncthreads();

  // 1. Local sequential scan on each thread's section in regs
  // first copy local values into regs
  float rX[COARSE_FACTOR];

  // This loop could be vectorized
#pragma unroll
  for (uint local_i = 0; local_i < COARSE_FACTOR; ++local_i) {
    rX[local_i] = sX[COARSE_FACTOR*i + local_i];
  }
  __syncthreads();

  for (uint local_i = 1; local_i < COARSE_FACTOR; ++local_i) {
    rX[local_i] += rX[local_i - 1];
  }

  // Store the last value (shifted one position to the right)
  if (i < NUM_THREADS-1) {
    sLocalSums[i+1] = rX[COARSE_FACTOR - 1];
  } else {
    sLocalSums[0] = 0.0f;
  }

  // 2. Perform regular parallel scan (non-coarse) on the local sums
  // This could also be done using parallel scan, but we skip it here
#pragma unroll
  for (uint stride = 1; stride < NUM_THREADS; stride *= 2) {
    __syncthreads();

    float acc;
    if (threadIdx.x >= stride) {
      acc = sLocalSums[threadIdx.x] + sLocalSums[threadIdx.x - stride];
    }

    __syncthreads();
    if (threadIdx.x >= stride) {
      sLocalSums[threadIdx.x] = acc;
    }
  } 
  __syncthreads();

  // 3. Each thread adds the exclusive scan results to its values
  // sLocalSums now contains the offset for which each thread should add to each of its chunks values
#pragma unroll
  for (uint local_i = 0; local_i < COARSE_FACTOR; ++local_i) {
    rX[local_i] += sLocalSums[i];
    // Could be vectorized
    output[COARSE_FACTOR*i + local_i] = rX[local_i];
  }
}

template<typename size_t>
__global__ void inclusive_scan_kernel_naive_single_block_coarse_vectorized(const size_t *X, size_t *output, const int size) {
}
