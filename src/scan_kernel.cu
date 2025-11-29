#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

template<typename T>
__global__ void inclusive_scan_kernel_naive(const T *X, T *output, const int size) {
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
  __syncthreads();

  for (uint stride = 1; stride < size; stride *= 2) {

    // Accumulate this stage into register to avoid race condition from other threads reading this value
    float acc;
    if (threadIdx.x >= stride) {
      acc = sX[threadIdx.x] + sX[threadIdx.x - stride];
    }

    // all threads must calculate their accumulator before we can update SMEM
    __syncthreads();
    if (threadIdx.x >= stride) 
      sX[threadIdx.x] = acc;
    // all threads must update their SMEM before we can move on to the next wave
    __syncthreads();
  }


  // Write output
  if (i < size) {
    output[i] = sX[threadIdx.x];
  }
}

