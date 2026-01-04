#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>

// on L40
#define NUM_SMS 58
#define NUM_BLOCKS_M 32
#define NUM_BLOCKS_K 32

#define BLOCK_SIZE 32
#define TM 8

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// V1 Naive: (gather-like) one thread per C-element, global only
// This kernel has lots of LG throttle, meaning it spends most of its clock cycles waiting for the global load queue to not be full
void __global__ gemm_v1(int M, int N, int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc) {
  int col_idx0 = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx0 = blockIdx.y * blockDim.y + threadIdx.y;

  int num_threads_x = gridDim.x * blockDim.x;
  int num_threads_y = gridDim.y * blockDim.y;
  
  // grid loop and boundary check
  for (int row_idx = row_idx0; row_idx < M; row_idx += num_threads_y) {
    for (int col_idx = col_idx0; col_idx < N; col_idx += num_threads_x) {
      // k-loop accumulation
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc += A[row_idx * lda + k] * B[k * ldb + col_idx];
      }
      C[row_idx*ldc + col_idx] = acc;
    }
  }
}

// V2: Shared memory
// This solves the issues with global memory stalls, but instead causes MIO pipeline stalls.
// This kernel spends a lot of time loading data from shared memory, causing the number of eligible warps to be extremely low
void __global__ gemm_v2(int M, int N, int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc) {
  __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

  // grid-stride using the block index, striding by the total number of blocks per direction
  for (int by = blockIdx.y; by * BLOCK_SIZE < M; by += gridDim.y) {
    for (int bx = blockIdx.x; bx * BLOCK_SIZE < N; bx += gridDim.x) {
      int grow = by * BLOCK_SIZE + threadIdx.y;
      int gcol = bx * BLOCK_SIZE + threadIdx.x;

      float acc = 0.0f;
      // split the original k-loop into tiles
      for (int k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_SIZE) {
        // cooperatively load a tile into SMEM. each thread loads one element from A and B
        // This way each thread only issues K/BLOCK_SIZE GMEM loads rather than K.
        sA[threadIdx.y][threadIdx.x] = 
          (k_tile_start + threadIdx.x < K && grow < M)
          ? A[grow * lda + k_tile_start + threadIdx.x] 
          : 0.0f;
        sB[threadIdx.y][threadIdx.x] = 
          (k_tile_start + threadIdx.y < K && gcol < N)
          ? B[(k_tile_start + threadIdx.y) * ldb + gcol]
          : 0.0f;
        __syncthreads();

        // k-tile (partial) accumulation. Could remove some loop iterations in edges
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
          acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        // required to avoid writing in the next loop too early
        __syncthreads();
      }
      if (gcol < N && grow < M) {
        C[grow*ldc + gcol] = acc;
      }
    }
  }
}


// V3: block tiling for each thread to calculates a column of output tile elements 
// This way we can reuse the data in the B tile column for each thread, reducing the SMEM load pressure.
void __global__ gemm_v3(int M, int N, int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc) {
  // constexpr int NUM_BLOCKTILES = (BLOCK_SIZE + TM - 1) / TM;

  __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

  int tile_row = blockIdx.y * BLOCK_SIZE;
  int grow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  int gcol = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  // store in a coarsened fashion 
  int out_row0 = tile_row + threadIdx.y * TM;

  // Hold all accumulated values
  float acc[TM] = {0.0f};
  for (int k_tile_start = 0; k_tile_start < K; k_tile_start += BLOCK_SIZE) {
    sA[threadIdx.y][threadIdx.x] = 
      (k_tile_start + threadIdx.x < K && grow < M)
      ? A[grow * lda + k_tile_start + threadIdx.x] 
      : 0.0f;
    sB[threadIdx.y][threadIdx.x] = 
      (k_tile_start + threadIdx.y < K && gcol < N)
      ? B[(k_tile_start + threadIdx.y) * ldb + gcol]
      : 0.0f;
    __syncthreads();

    // Block tiling loop: block tiles are size (TM x BLOCK_SIZE)
    // In each block tile we can accumulate the values for each of the TM outputs value
#pragma unroll
    for (int bk = 0; bk < BLOCK_SIZE; ++bk) { // k-loop accumulation over "outer products"
      // loop swapping enables us to cache the B value in a register
      float B_val = sB[bk][threadIdx.x];
#pragma unroll
      for (int tm = 0; tm < TM; ++tm) { // for each coarsened row
        if (threadIdx.y*TM + tm < BLOCK_SIZE) { // deactivate out of bounds threads
          // coarsen by TM on the rows
          acc[tm] += sA[threadIdx.y*TM + tm][bk] * B_val;
        }
      }
    }
    __syncthreads();
  }
  for (int tm = 0; tm < TM; ++tm) {
    int out_row = out_row0 + tm;
    // ensure we don't accidentally override with 0s from deactivated threads
    int tile_row = threadIdx.y*TM + tm;
    if (tile_row < BLOCK_SIZE && gcol < N && out_row < M) 
      C[out_row * ldc + gcol] = acc[tm];
  }
}

at::Tensor gemm(const at::Tensor& A_h, const at::Tensor& B_h) {
  // auto A = A_cpu.to(torch::kCUDA);
  // auto B = B_cpu.to(torch::kCUDA);
  auto C = torch::empty_like(A_h).to(torch::kCUDA);

  auto A_cpu = A_h.contiguous();
  auto B_cpu = B_h.contiguous();
  const float* A_ptr_h = A_cpu.data_ptr<float>();
  const float* B_ptr_h = B_cpu.data_ptr<float>();
  float* C_d = C.data_ptr<float>();

  int M = A_h.size(0);
  int N = M, K = M;
  // assume all are row-major
  int lda = K, ldb = N, ldc = N;

  float *A_d, *B_d;
  int A_size = sizeof(float)*M*K;
  int B_size = sizeof(float)*K*N;
  checkCuda(cudaMalloc((void **)&A_d, A_size));
  checkCuda(cudaMalloc((void **)&B_d, B_size));
  checkCuda(cudaMemcpy((void *)A_d, (void *)A_ptr_h, A_size, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy((void *)B_d, (void *)B_ptr_h, B_size, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream));

  dim3 blocks(NUM_BLOCKS_M, NUM_BLOCKS_K, 1);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE, 1);
  gemm_v1<<<blocks, threads, 0, stream>>>(M, N, K, A_d, B_d, C_d, lda, ldb, ldc);
  gemm_v2<<<blocks, threads, 0, stream>>>(M, N, K, A_d, B_d, C_d, lda, ldb, ldc);

  dim3 blocks_v3((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  dim3 threads_v3(BLOCK_SIZE, BLOCK_SIZE, 1);
  gemm_v3<<<blocks_v3, threads_v3, 0, stream>>>(M, N, K, A_d, B_d, C_d, lda, ldb, ldc);

  // Sync as device and host code otherwise runs asynchronously
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess)
  {
      std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                << std::endl;
      std::cerr << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
  }
  checkCuda(cudaStreamSynchronize(stream));

  checkCuda(cudaFree(A_d));
  checkCuda(cudaFree(B_d));
  checkCuda(cudaStreamDestroy(stream));

  return C.to(torch::kCPU);
}

at::Tensor reference_gemm(const at::Tensor& A, const at::Tensor& B) {
  TORCH_CHECK(!A.is_cuda(), "input expects CPU tensor");
  TORCH_CHECK(!B.is_cuda(), "input expects CPU tensor");

  auto C_ref64 = torch::matmul(A.to(torch::kFloat64), B.to(torch::kFloat64));
  return C_ref64.to(torch::kFloat32);
}

int run_tests(uint n) {
  auto options_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  auto A_cpu = torch::randn({n, n}, options_cpu);
  auto B_cpu = torch::randn({n, n}, options_cpu);

  auto C_ref = reference_gemm(A_cpu, B_cpu);
  auto C = gemm(A_cpu, B_cpu);

  std::cout << "n = " << n << " - ";
  if (at::allclose(C_ref, C, 1e-2, 5e-2)) {
    std::cout << "OK" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
    if (n <= 16) {
      std::cout << C_ref << std::endl;
      std::cout << C << std::endl;
    }
    return 1;
  }
  return 0;
}

void show_device_props() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
}


int main() {
  int ok;
  show_device_props();
  std::cout << "CUDA available? " << torch::cuda::is_available() << "\n";
  std::cout << "CUDA device count: " << torch::cuda::device_count() << "\n";
  std::vector<int64_t> sizes = {
    // 16,
    // 32,
    // 33,
    // 1023,
    // 1024,
    4096,
  };

  for (auto n : sizes) {
    ok = run_tests(n);
    if (ok != 0) {
      return 1;
    }
  }

  std::cout << "All scan tests passed." << std::endl;
  return 0;
}
