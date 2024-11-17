# Install script for directory: /home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/UtilsCUDA" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/ModuleDLL.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/OutputTypes.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDADriverInfo.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAEventTimer.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAGPUComputingAbstraction.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAKernelLauncher.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAMemoryHandlers.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAMemoryPool.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAMemoryRegistry.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAMemoryWrappers.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAParallelFor.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAProcessMemoryPool.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAQueue.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDASpinLock.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAStreamsHandler.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAUtilityDeviceFunctions.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CUDAUtilityFunctions.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/UtilsCUDA/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/build/lib/libUtilsCUDA.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so"
         OLD_RPATH "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/build/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "distributable")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/build/lib/libUtilsCUDA.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so"
         OLD_RPATH "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/build/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtilsCUDA.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "distributable")
endif()

