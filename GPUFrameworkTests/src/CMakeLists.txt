# GPU Framework version 14.0.0

###################

# set(GPU_FRAMEWORK_TESTS_HOST_SRC_CPP
#   src/Tests/CPUParallelismTest.cpp
#   src/Tests/ColorHistogramTest.cpp  # Add this line

#   #This line is to make CMakeList.txt appear on the VS project tree.
#   src/CMakeLists.txt
# )
# source_group("src\\Tests" FILES ${GPU_FRAMEWORK_TESTS_HOST_SRC_CPP})

###################

set(GPU_FRAMEWORK_TESTS_HOST_SRC_CPP
  src/Tests/CPUParallelismTest.cpp
  ############
  src/Tests/ColorHistogramTest.cpp

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src\\Tests" FILES ${GPU_FRAMEWORK_TESTS_HOST_SRC_CPP})

set(GPU_FRAMEWORK_HOST_SRC_CPP
  src/HostUnitTests.cpp

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src" FILES ${GPU_FRAMEWORK_HOST_SRC_CPP})



set(GPU_FRAMEWORK_TESTS_DEVICE_SRC_CU
  src/Tests/CUDALinearAlgebraGPUComputingTest.cu
  # src/Tests/ColorHistogramGPUTest.cu

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src\\Tests" FILES ${GPU_FRAMEWORK_TESTS_DEVICE_SRC_CU})

set(GPU_FRAMEWORK_DEVICE_SRC_CU
  src/DeviceUnitTests.cu

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src" FILES ${GPU_FRAMEWORK_DEVICE_SRC_CU})



set(GPU_FRAMEWORK_STRESS_TESTS_HOST_SRC_CPP
  src/Tests/LinearAlgebraCPUComputingStressTest.cpp

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src\\Tests" FILES ${GPU_FRAMEWORK_STRESS_TESTS_HOST_SRC_CPP})

set(GPU_FRAMEWORK_STRESS_HOST_SRC_CPP
  src/HostStressTests.cpp

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src" FILES ${GPU_FRAMEWORK_STRESS_HOST_SRC_CPP})



set(GPU_FRAMEWORK_STRESS_TESTS_HOST_DEVICE_SRC_CU
  src/Tests/CUDALinearAlgebraGPUComputingStressTest.cu

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src\\Tests" FILES ${GPU_FRAMEWORK_STRESS_TESTS_HOST_DEVICE_SRC_CU})

set(GPU_FRAMEWORK_STRESS_HOST_DEVICE_SRC_CU
  src/HostDeviceStressTests.cpp

  #This line is to make CMakeList.txt appear on the VS project tree.
  src/CMakeLists.txt
)
source_group("src" FILES ${GPU_FRAMEWORK_STRESS_HOST_DEVICE_SRC_CU})