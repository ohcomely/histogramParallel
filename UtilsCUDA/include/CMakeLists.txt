# GPU Framework version 14.0.0

set(UTILS_CUDA_HEADERS
  include/ModuleDLL.h
  include/OutputTypes.h
  include/CUDADriverInfo.h
  include/CUDAEventTimer.h
  include/CUDAGPUComputingAbstraction.h
  include/CUDAKernelLauncher.h
  include/CUDAMemoryHandlers.h
  include/CUDAMemoryPool.h
  include/CUDAMemoryRegistry.h
  include/CUDAMemoryWrappers.h
  include/CUDAParallelFor.h
  include/CUDAProcessMemoryPool.h
  include/CUDAQueue.h
  include/CUDASpinLock.h
  include/CUDAStreamsHandler.h
  include/CUDAUtilityDeviceFunctions.h
  include/CUDAUtilityFunctions.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include" FILES ${UTILS_CUDA_HEADERS})

install(FILES ${UTILS_CUDA_HEADERS} DESTINATION include/${PROJECT_NAME})