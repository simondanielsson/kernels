#include <torch/torch.h>
#include <ATen/ATen.h>
#include "scan_kernel.cu"

#define cdiv(a, b) ((a + b - 1) / b)

at::Tensor reference_inclusive_scan(const at::Tensor& input) {
  TORCH_CHECK(!input.is_cuda(), "input expects CPU tensor");
  return input.cumsum(0);
}

at::Tensor inclusive_scan(const at::Tensor& input) {
  TORCH_CHECK(input.is_cuda(), "input expects GPU tensor");

  const float* input_data = input.data_ptr<float>();

  auto output = torch::empty_like(input);
  float* output_data = output.data_ptr<float>();
  int size = input.size(0);

  inclusive_scan_kernel_naive<float><<<cdiv(size, BLOCK_SIZE), BLOCK_SIZE>>>(input_data, output_data, size);

  return output.to(torch::kCPU);
}
