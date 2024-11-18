# GPU Framework version 14.0.0

#############################

#Ritvik's Test
set(GPU_FRAMEWORK_TESTS_COLOR_HISTOGRAM_HEADER
  include/Tests/ColorHistogramTest.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include\\Tests" FILES ${GPU_FRAMEWORK_TESTS_COLOR_HISTOGRAM_HEADER})

install(FILES ${GPU_FRAMEWORK_TESTS_COLOR_HISTOGRAM_HEADER} DESTINATION include/${PROJECT_NAME})

#############################

set(GPU_FRAMEWORK_TESTS_HOST_HEADER
  include/Tests/CPUParallelismTest.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include\\Tests" FILES ${GPU_FRAMEWORK_TESTS_HOST_HEADER})

install(FILES ${GPU_FRAMEWORK_TESTS_HOST_HEADER} DESTINATION include/${PROJECT_NAME})

set(GPU_FRAMEWORK_HOST_HEADER
  include/HostUnitTests.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include" FILES ${GPU_FRAMEWORK_HOST_HEADER})

install(FILES ${GPU_FRAMEWORK_HOST_HEADER} DESTINATION include/${PROJECT_NAME})



set(GPU_FRAMEWORK_TESTS_DEVICE_HEADER
  include/Tests/CUDALinearAlgebraGPUComputingTest.h
  include/Tests/ColorHistogramGPUTest.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include\\Tests" FILES ${GPU_FRAMEWORK_TESTS_DEVICE_HEADER})

install(FILES ${GPU_FRAMEWORK_TESTS_DEVICE_HEADER} DESTINATION include/${PROJECT_NAME})

set(GPU_FRAMEWORK_DEVICE_HEADER
  include/DeviceUnitTests.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include" FILES ${GPU_FRAMEWORK_DEVICE_HEADER})

install(FILES ${GPU_FRAMEWORK_DEVICE_HEADER} DESTINATION include/${PROJECT_NAME})



set(GPU_FRAMEWORK_STRESS_TESTS_HOST_HEADER
  include/Tests/LinearAlgebraCPUComputingStressTest.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include\\Tests" FILES ${GPU_FRAMEWORK_STRESS_TESTS_HOST_HEADER})

install(FILES ${GPU_FRAMEWORK_STRESS_TESTS_HOST_HEADER} DESTINATION include/${PROJECT_NAME})

set(GPU_FRAMEWORK_STRESS_HOST_HEADER
  include/HostStressTests.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include" FILES ${GPU_FRAMEWORK_STRESS_HOST_HEADER})

install(FILES ${GPU_FRAMEWORK_STRESS_HOST_HEADER} DESTINATION include/${PROJECT_NAME})



set(GPU_FRAMEWORK_STRESS_TESTS_HOST_DEVICE_HEADER
  include/Tests/CUDALinearAlgebraGPUComputingStressTest.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include\\Tests" FILES ${GPU_FRAMEWORK_STRESS_TESTS_HOST_DEVICE_HEADER})

install(FILES ${GPU_FRAMEWORK_STRESS_TESTS_HOST_DEVICE_HEADER} DESTINATION include/${PROJECT_NAME})

set(GPU_FRAMEWORK_STRESS_HOST_DEVICE_HEADER
  include/HostDeviceStressTests.h

  #This line is to make CMakeList.txt appear on the VS project tree.
  include/CMakeLists.txt
)
source_group("include" FILES ${GPU_FRAMEWORK_STRESS_HOST_DEVICE_HEADER})

install(FILES ${GPU_FRAMEWORK_STRESS_HOST_DEVICE_HEADER} DESTINATION include/${PROJECT_NAME})