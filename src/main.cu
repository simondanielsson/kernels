#include <torch/torch.h>
#include <iostream>
#include <ATen/ATen.h>
#include "scan.cu"

int run_tests(uint n) {
  auto options_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  auto x_cpu = torch::randn({n}, options_cpu);
  auto x_cuda = x_cpu.to(torch::kCUDA);

  auto out_cpu = reference_inclusive_scan(x_cpu);
  auto out_cuda = inclusive_scan(x_cuda);

  std::cout << "n = " << n << " - ";
  if (at::allclose(out_cpu, out_cuda, 1e-2)) {
    std::cout << "OK" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
    std::cout << out_cpu << std::endl;
    std::cout << out_cuda << std::endl;
    return 1;
  }
  return 0;
}

int main() {
  int ok;
  std::cout << "CUDA available? " << torch::cuda::is_available() << "\n";
  std::cout << "CUDA device count: " << torch::cuda::device_count() << "\n";

  std::vector<int64_t> sizes = {
    //6, 12, 32, //300,
    // 256, 
    //1024, 
    8192,
    59392,
    /*
    0, 1, 2, 3, 4, 7, 8, 15, 16, 17,
    31, 32, 33, 63, 64, 65,
    */
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
