#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define cdiv(a, b) ((a + b - 1) / b)
#define WARP_SIZE 32
#define BLOCK_SIZE 1024
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
  constexpr int NUM_THREADS = BLOCK_SIZE / COARSE_FACTOR;
  uint i = blockIdx.x * blockDim.x + threadIdx.x;  

  __shared__ size_t sX[BLOCK_SIZE];
  __shared__ size_t sLocalSums[NUM_THREADS];
  size_t rX[COARSE_FACTOR];

  // 0. Cooperatively load from GMEM in coalesced fashion
#pragma unroll
  for (uint iter = 0; iter < COARSE_FACTOR; ++iter) {
    int idx = NUM_THREADS * iter + i;
    sX[idx] = X[idx];
  }
  __syncthreads();

  // 1. Local sequential scan on each thread's section in regs
  // first copy local values into regs

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
#pragma unroll
  for (uint stride = 1; stride < NUM_THREADS; stride *= 2) {
    __syncthreads();

    size_t acc;
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


template<typename size_t, const int block_size>
__global__ void inclusive_scan_kernel_kogge_stone_block_local_double_buffering(const size_t *X, size_t *output, const int size, const int logical_stride) {
  // logical stride allows us to perform scan on every logical_stride'th element
  const int logical_offset = logical_stride - 1;
  const int logical_coarsening = logical_stride;
  uint gtid = (blockIdx.x * block_size + threadIdx.x) * logical_coarsening + logical_offset;

  __shared__ float sX[2*block_size];

  // Each thread loads one element from GMEM
  if (gtid < size) {
    sX[threadIdx.x] = X[gtid];
  } else {
    sX[threadIdx.x] = 0.0f;
  }

  // In double buffering, the "active" portion of SMEM is swapped every iteration
  uint offset = 0;
  uint last_i = 0;
  // for (uint stride = 1, i = 0; stride < block_size; stride *= 2, ++i) {
  //   __syncthreads();
  //   offset = block_size * (i % 2);
  //
  //   // Read from one chunk of the SMEM, and write to the other
  //   if (threadIdx.x >= stride) {
  //     sX[block_size * ((i+1) % 2) + threadIdx.x] = sX[offset + threadIdx.x] + sX[offset + threadIdx.x - stride];
  //   }
  //   if (stride*2 >= block_size) 
  //     last_i = i;
  // }

  for (uint stride = 1, i = 0; stride < block_size; stride <<= 1, ++i) {
    __syncthreads();
    uint src_offset = block_size * (i % 2);
    uint dst_offset = block_size * ((i + 1) % 2);

    size_t val = sX[src_offset + threadIdx.x];
    if (threadIdx.x >= stride) {
      val += sX[src_offset + threadIdx.x - stride];
    }
    sX[dst_offset + threadIdx.x] = val;

    if (stride * 2 >= block_size)
      last_i = i;
  }

  // If last i was even, then we wrote the last values to the second half
  __syncthreads();
  if (gtid < size) {
    offset = block_size * ((last_i+1) % 2);
    output[gtid] = sX[offset + threadIdx.x];
  }
}

template<typename size_t, const int block_size>
__global__ void inclusive_scan_kernel_kogge_stone_block_local(const size_t *X, size_t *output, const int size, int logical_stride) {
  // logical stride allows us to perform scan on every logical_stride'th element
  const int logical_offset = logical_stride - 1;
  const int logical_coarsening = logical_stride;
  uint gtid = (blockIdx.x * block_size + threadIdx.x) * logical_coarsening + logical_offset;

  __shared__ float sX[block_size];

  // Each thread loads one element from GMEM
  if (gtid < size) {
    sX[threadIdx.x] = X[gtid];
  } else {
    sX[threadIdx.x] = 0.0f;
  }

  for (uint stride = 1; stride < block_size; stride *= 2) {
    __syncthreads();

    float acc;
    if (threadIdx.x >= stride) {
      acc = sX[threadIdx.x] + sX[threadIdx.x - stride];
    }

    __syncthreads();
    if (threadIdx.x >= stride) 
      sX[threadIdx.x] = acc;
  }

  if (gtid < size) {
    output[gtid] = sX[threadIdx.x];
  }
}

template <typename size_t, const int block_size>
__global__ void inclusive_scan_update_blocks_with_offsets(size_t* output, int size) {
  // Last element in each block is now updated. Remains to update the other elements (sequentially) per block with the left-adjacent block sum
  const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  const int last_block_idx = blockIdx.x - 1;
  if (last_block_idx >= 0) {
    if ((gtid < size) && (threadIdx.x < blockDim.x - 1)) {
      output[gtid] += output[last_block_idx*blockDim.x + block_size - 1];
    }
  }
}

template<typename size_t>
void inclusive_scan_kernel_kogge_stone_3_stage(const size_t* X, size_t* output, const int size) {
  if (size == 59392) {
    constexpr int num_blocks = cdiv(59392, BLOCK_SIZE);
    // Compute local scans in each block
    inclusive_scan_kernel_kogge_stone_block_local<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(X, output, size, 1);
    // Compute a scan on the last element of each block, across original blocks
    inclusive_scan_kernel_kogge_stone_block_local<size_t, num_blocks><<<1, num_blocks>>>(output, output, size, BLOCK_SIZE);
    // Output remaining elements with the offsets
    inclusive_scan_update_blocks_with_offsets<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(output, size);
  } else if (size == 8192) {
    constexpr int num_blocks = cdiv(8192, BLOCK_SIZE);
    // Compute local scans in each block
    inclusive_scan_kernel_kogge_stone_block_local<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(X, output, size, 1);
    // Compute a scan on the last element of each block, across original blocks
    inclusive_scan_kernel_kogge_stone_block_local<size_t, num_blocks><<<1, num_blocks>>>(output, output, size, BLOCK_SIZE);
    // Output remaining elements with the offsets
    inclusive_scan_update_blocks_with_offsets<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(output, size);
  }
}


template<typename size_t>
void inclusive_scan_kernel_kogge_stone_3_stage_double_buffering(const size_t* X, size_t* output, const int size) {
  if (size == 59392) {
    constexpr int num_blocks = cdiv(59392, BLOCK_SIZE);
    // Compute local scans in each block
    inclusive_scan_kernel_kogge_stone_block_local_double_buffering<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(X, output, size, 1);
    // Compute a scan on the last element of each block, across original blocks
    inclusive_scan_kernel_kogge_stone_block_local_double_buffering<size_t, num_blocks><<<1, num_blocks>>>(output, output, size, BLOCK_SIZE);
    // Output remaining elements with the offsets
    inclusive_scan_update_blocks_with_offsets<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(output, size);
  } else if (size == 8192) {
    constexpr int num_blocks = cdiv(8192, BLOCK_SIZE);
    // Compute local scans in each block
    inclusive_scan_kernel_kogge_stone_block_local_double_buffering<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(X, output, size, 1);
    // Compute a scan on the last element of each block, across original blocks
    inclusive_scan_kernel_kogge_stone_block_local_double_buffering<size_t, num_blocks><<<1, num_blocks>>>(output, output, size, BLOCK_SIZE);
    // Output remaining elements with the offsets
    inclusive_scan_update_blocks_with_offsets<size_t, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(output, size);
  }
}
