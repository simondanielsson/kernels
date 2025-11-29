#include <stdio.h>
#include <cuda.h>

__global__ void vecadd_kernel(float* a, float* b, float* c, unsigned int N) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) {
    c[i] = a[i] + b[i]; 
  }
}

float* vecadd(float* a_h, float* b_h, unsigned int N) {
  // Kernel stub
  float* a_d;
  float* b_d;
  float* c_d;
  float* c_h;

  c_h = (float *) malloc(N * sizeof(float));
  cudaMalloc((void**)&a_d, N * sizeof(float));
  cudaMalloc((void**)&b_d, N * sizeof(float));
  cudaMalloc((void**)&c_d, N * sizeof(float));

  // copy inputs to device
  cudaMemcpy(a_d, a_h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, N*sizeof(float), cudaMemcpyHostToDevice);


  unsigned int BLOCKS_SIZE = 64;
  vecadd_kernel<<<(N + BLOCKS_SIZE - 1) / BLOCKS_SIZE, BLOCKS_SIZE>>>(a_d, b_d, c_d, N);

  // Copy result to host
  cudaMemcpy(c_h, c_d,  N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree((void *)a_d);
  cudaFree((void *)b_d);
  cudaFree((void *)c_d);
  return c_h;
}

int main() {
  // initialize host vectors of size N
  unsigned int N = 20;
  float a_h[20];
  float b_h[20];
  for (unsigned int i = 0; i < N; i++) {
    a_h[i] = i;
    b_h[i] = i;
  }

  float* c = vecadd((float *)a_h, (float *)b_h, N);
  // print the values of c
  for (unsigned int i = 0; i < N; i++) {
    printf("%f ", c[i]);
  }
  free(c);
  return 0;
}
