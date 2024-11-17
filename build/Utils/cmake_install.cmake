# Install script for directory: /home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Utils" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/ModuleDLL.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/lodepng.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/EnvironmentConfig.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/FunctionView.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/MathConstants.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/NewHandlerSupport.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/AccurateTimers.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/Randomizers.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/SIMDVectorizations.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/UnitTests.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/UtilityFunctions.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/VectorTypes.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/Utils/CPUParallelism" TYPE FILE FILES
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/ConcurrentBlockingQueue.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/CPUParallelismNCP.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/CPUParallelismUtilityFunctions.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/ThreadBarrier.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/ThreadGuard.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/ThreadJoiner.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/ThreadOptions.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/ThreadPool.h"
    "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/Utils/include/CPUParallelism/CMakeLists.txt"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/build/lib/libUtils.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "distributable")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/ranjan/ETHZ/personal/ParTec_GPU_Computing_Coding_Challenge_no_OpenGL/GPUFramework_src/build/lib/libUtils.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libUtils.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "distributable")
endif()

