#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>

// on L40
#define NUM_SMS 58
#define NUM_BLOCKS_M 32
#define NUM_BLOCKS_K 32

#define THREADS_PER_BLOCK_M 32
#define THREADS_PER_BLOCK_K 32

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

// Naive: (gather-like) one thread per C-element, global only
void __global__ gemm_v1(int M, int N, int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc) {
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  // grid loop
  while (col_idx < N && row_idx < M) {

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row_idx * lda + k] * B[k * ldb + col_idx];
    }
    C[row_idx*ldc + col_idx] = acc;

    row_idx += gridDim.y;
    col_idx += gridDim.x;
  }
}

// V2: Shared memory
void __global__ gemm_v2(int M, int N, int K, const float* A, const float* B, float* C, int lda, int ldb, int ldc) {
  int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int row_idx = blockIdx.y * blockDim.y + threadIdx.y;

  // grid loop
  while (col_idx < N && row_idx < M) {

    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += A[row_idx * lda + k] * B[k * ldb + col_idx];
    }
    C[row_idx*ldc + col_idx] = acc;

    row_idx += gridDim.y;
    col_idx += gridDim.x;
  }
}

at::Tensor gemm(const at::Tensor& A_h, const at::Tensor& B_h) {
  // auto A = A_cpu.to(torch::kCUDA);
  // auto B = B_cpu.to(torch::kCUDA);
  auto C = torch::empty_like(A_h).to(torch::kCUDA);

  const float* A_ptr_h = A_h.data_ptr<float>();
  const float* B_ptr_h = B_h.data_ptr<float>();
  float* C_d = C.data_ptr<float>();

  int M = A_h.size(0);
  int N = M, K = M;
  // assume all are row-major
  int lda = K, ldb = N, ldc = N;

  float *A_d, *B_d;
  int A_size = sizeof(float)*M*K;
  int B_size = sizeof(float)*K*N;
  checkCuda(cudaMalloc((void **)&A_d, A_size));
  checkCuda(cudaMalloc((void **)&B_d, A_size));
  checkCuda(cudaMemcpy((void *)A_d, (void *)A_ptr_h, A_size, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy((void *)B_d, (void *)B_ptr_h, B_size, cudaMemcpyHostToDevice));

  cudaStream_t stream;
  checkCuda(cudaStreamCreate(&stream));

  dim3 blocks(NUM_BLOCKS_M, NUM_BLOCKS_K, 1);
  dim3 threads(THREADS_PER_BLOCK_M, THREADS_PER_BLOCK_K, 1);
  gemm_v1<<<blocks, threads, 0, stream>>>(M, N, K, A_d, B_d, C_d, lda, ldb, ldc);

  // Sync as device and host code otherwise runs asynchronously
  cudaDeviceSynchronize();
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess)
  {
      std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                << std::endl;
      std::cerr << cudaGetErrorString(err) << std::endl;
      std::exit(EXIT_FAILURE);
  }
  checkCuda(cudaStreamDestroy(stream));

  return C.to(torch::kCPU);
}

at::Tensor reference_gemm(const at::Tensor& A, const at::Tensor& B) {
  TORCH_CHECK(!A.is_cuda(), "input expects CPU tensor");
  TORCH_CHECK(!B.is_cuda(), "input expects CPU tensor");
  return torch::matmul(A, B);
}

int run_tests(uint n) {
  auto options_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  auto A_cpu = torch::randn({n, n}, options_cpu);
  auto B_cpu = torch::randn({n, n}, options_cpu);

  auto C_ref = reference_gemm(A_cpu, B_cpu);
  auto C = gemm(A_cpu, B_cpu);

  std::cout << "n = " << n << " - ";
  if (at::allclose(C_ref, C, 1e-2)) {
    std::cout << "OK" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
    // std::cout << C_ref << std::endl;
    // std::cout << C << std::endl;
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
    1024,
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
