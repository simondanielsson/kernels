# Example GPU kernel implementations

This repository contains a implementations of some common GPU kernels in CUDA C++.

## Installation

Compile the kernels using 

```bash
mkdir build
make
```

Then run the kernels, for instance using

```bash
./build/scan
```

This have only been tested on nvcc 12.9 on an L4 (compute capability 8.9).

## Profiling

The kernels can be profiled using Nsight Compute or Nsight Systems:

```
# compute
make ncu

# systems
make nsys
```

