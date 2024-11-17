# Install script for directory: /home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/Tests/ColorHistogramTest.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/Tests/CPUParallelismTest.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/HostUnitTests.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/Tests/CUDALinearAlgebraGPUComputingTest.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/DeviceUnitTests.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/Tests/LinearAlgebraCPUComputingStressTest.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/HostStressTests.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/Tests/CUDALinearAlgebraGPUComputingStressTest.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPUFrameworkTests" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/HostDeviceStressTests.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/GPUFrameworkTests/include/CMakeLists.txt"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/cmake-build-debug/Utils/cmake_install.cmake")
  include("/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/cmake-build-debug/UtilsCUDA/cmake_install.cmake")
  include("/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/cmake-build-debug/gtest/cmake_install.cmake")

endif()

