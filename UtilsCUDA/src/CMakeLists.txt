# GPU Framework version 14.0.0

set(UTILS_CUDA_SRC_CPP
  src/CUDADriverInfo.cu
  src/CUDAEventTimer.cu
  src/CUDAMemoryPool.cu
  src/CUDAMemoryRegistry.cu
  src/CUDAProcessMemoryPool.cu
  src/CUDAStreamsHandler.cu
  src/CUDAUtilityFunctions.cu

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src" FILES ${UTILS_CUDA_SRC_CPP})