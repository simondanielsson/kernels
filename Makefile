default:
	cmake -S . -B build -G Ninja \
		-DCMAKE_C_COMPILER=/usr/bin/clang \
		-DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
		-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
		-DCMAKE_PREFIX_PATH=/home/danielssonsimon/libtorch \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

	cmake --build build -j --config Release --target gemm

memcheck: 
	compute-sanitizer --tool memcheck ./build/scan

ncu:
	sudo /usr/local/cuda/bin/ncu -fo scan_profile --set full --target-processes all build/scan

nsys:
	nsys profile --stats=true build/scan
