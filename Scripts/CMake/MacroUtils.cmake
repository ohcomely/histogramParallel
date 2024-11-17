# ************************************************************
#
#  Copyright (c) 2009-2018, Thanos Theo. All rights reserved.
#  Released Under a Simplified BSD (FreeBSD) License
#  for academic, personal & non-commercial use.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
#  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  The views and conclusions contained in the software and documentation are those
#  of the author and should not be interpreted as representing official policies,
#  either expressed or implied, of the FreeBSD Project.
#
#  A Commercial License is also available for commercial use with
#  special restrictions and obligations at a one-off fee. See links at:
#  1. http://www.dotredconsultancy.com/openglrenderingenginetoolrelease.php
#  2. http://www.dotredconsultancy.com/openglrenderingenginetoolsourcecodelicence.php
#  Please contact Thanos Theo (thanos.theo@dotredconsultancy.com) for more information.
#
# ************************************************************

# GPU Framework version 14.0.0

# Defines the compiler/linker flags used to build our project.
macro(setupCompilerAndLinker)

  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_STANDARD_EXTENSIONS ON)

  # first set host compiler flags
  if (MSVC)

    # /MP (Build with Multiple Processes)
    if (NOT CMAKE_CXX_FLAGS MATCHES "/MP")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
    endif()
    # /EHsc for not adding exception handling
    if (NOT CMAKE_CXX_FLAGS MATCHES "/EHsc")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    endif()
    # /W3 displays level 1, level 2 and level 3 (production quality) warnings
    if (NOT CMAKE_CXX_FLAGS MATCHES "/W3")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    endif()
    # /WX for treating all compiler warnings as errors
    if (NOT CMAKE_CXX_FLAGS MATCHES "/WX")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
    endif()
    # /D_SCL_SECURE_NO_WARNINGS disables Visual Studio SCL_SECURE STL warnings
    if (NOT CMAKE_CXX_FLAGS MATCHES "/D_SCL_SECURE_NO_WARNINGS")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SCL_SECURE_NO_WARNINGS")
    endif()
    # /D_CRT_SECURE_NO_WARNINGS disables Visual Studio CRT_SECURE STL warnings
    if (NOT CMAKE_CXX_FLAGS MATCHES "/D_CRT_SECURE_NO_WARNINGS")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    endif()
    # /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING disables Google Test deprecation warnings
    if (NOT CMAKE_CXX_FLAGS MATCHES "/D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING")
    endif()

    # set NOMINMAX so the std:min() and std::max() functions are not overwritten with macros
    add_definitions(-DNOMINMAX)

  # -fPIC is in g++ and clang++ by default, but for NVCC separate compilation so we have to explicitly add it
  elseif (CMAKE_COMPILER_IS_GNUCXX)
    if (NOT CMAKE_CXX_FLAGS MATCHES "std")
      if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -mavx2 -mxsave -mxsavec -mxsaves -O0 -Wall -Wuninitialized -Werror -fPIC")
      else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -mavx2 -mxsave -mxsavec -mxsaves -O3 -Wall -Wuninitialized -Werror -fPIC")
      endif()
    endif()

  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (NOT CMAKE_CXX_FLAGS MATCHES "std")
      if (CMAKE_BUILD_TYPE MATCHES Debug)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -mavx2 -mxsave -mxsavec -mxsaves -O0 -Wall -Wuninitialized -Werror -fPIC")
      else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -mavx2 -mxsave -mxsavec -mxsaves -O0 -Wall -Wuninitialized -Werror -fPIC")
      endif()
    endif()

  endif()

  # set NVCC flags, note we do not use " " here in order to make compatible with Dynamic Parallellism, this is
  # the same as appending flags in a list to CUDA_NVCC_FLAGS

  # Note 1: -Wno-deprecated-gpu-targets is for both Windows and Linux OSs to remove the warning for older compute capability GPU hardware (< sm_30)
  # Note 2: -arch=sm_50 to set the minimal GPU compute capability to Maxwell, please enable on >= Maxwell (it also guarantees 1D thread access to reach the grid dimension limit of 2147483647 for >= sm_30)
  # Note 3: IEEE 754 mode (default options except fmad, --use-fast-math inverts first 3 options but not the fmad one)
  # Note 4: --generate-line-info source/line information embedded in optimized executables for profiling (already available in debug '-G' executables)
  if (NOT CUDA_NVCC_FLAGS MATCHES "targets")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets --generate-line-info --expt-extended-lambda --expt-relaxed-constexpr --default-stream per-thread --ftz=false --prec-div=true --prec-sqrt=true --fmad=true)
    if (WIN32)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe "--diag_suppress=base_class_has_different_dll_interface")
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcudafe "--diag_suppress=field_without_dll_interface")
    elseif (UNIX)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 -O3)
    endif()
    if (CMAKE_BUILD_TYPE MATCHES DEBUG OR CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_BUILD_TYPE MATCHES debug)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --debug --device-debug)
      # --debug & --device-debug enable debug mode for the cuda host & device code
      # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --debug)
      # set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --device-debug)
    endif()
  endif()
endmacro(setupCompilerAndLinker)

# configure the GPU Framework build preprocessor environment
macro(setupGPUFrameworkPreprocessorDefines)
  set(GPU_FRAMEWORK_DEBUG                                               ON CACHE BOOL "Remove all debug code & messages from the GPU Framework.")
  set(GPU_FRAMEWORK_PROFILE_NCP_PARALLEL_FOR                           OFF CACHE BOOL "Reporting of NCP parallelFor() profile information for the GPU Framework.")
  set(GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR                       OFF CACHE BOOL "Using system_error for CUDA errors for the GPU Framework.")
  set(GPU_FRAMEWORK_CUDA_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS          OFF CACHE BOOL "use 0 or 1 for single global allocations for the CUDAMemoryPool. (Note: to be enabled for debugging purposes only)")
  set(GPU_FRAMEWORK_CUDA_PROCESS_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS  OFF CACHE BOOL "use 0 or 1 for single global allocations for the CUDAProcessMemoryPool. (Note: to be enabled for debugging purposes only)")

  if (GPU_FRAMEWORK_DEBUG)
    add_definitions(-DGPU_FRAMEWORK_DEBUG)
  endif()

  if (GPU_FRAMEWORK_PROFILE_NCP_PARALLEL_FOR)
    add_definitions(-DGPU_FRAMEWORK_PROFILE_NCP_PARALLEL_FOR)
  endif()

  if (GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR)
    add_definitions(-DGPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR)
  endif()

  if (GPU_FRAMEWORK_CUDA_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS)
    add_definitions(-DGPU_FRAMEWORK_CUDA_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS=1)
  else ()
    add_definitions(-DGPU_FRAMEWORK_CUDA_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS=0)
  endif()

  if (GPU_FRAMEWORK_CUDA_PROCESS_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS)
    add_definitions(-DGPU_FRAMEWORK_CUDA_PROCESS_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS=1)
  else ()
    add_definitions(-DGPU_FRAMEWORK_CUDA_PROCESS_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS=0)
  endif()
endmacro(setupGPUFrameworkPreprocessorDefines)

macro(setupGPUFrameworkExternalLibraries)

  #######################################################################################################################
  # Find Threads package
  # This CMake macro is the to setup Threads for the GPU Framework
  #######################################################################################################################

  find_package(Threads REQUIRED)
  if (NOT Threads_FOUND)
    message(FATAL_ERROR "Threads not found. The GPU Framework can not be built.")
  endif()

  #######################################################################################################################
  # Find OpenGL package
  # This CMake macro is the to setup OpenGL for the GPU Framework
  #######################################################################################################################

  find_package(OpenGL REQUIRED)
  if (NOT OpenGL_FOUND)
    message(FATAL_ERROR "OpenGL not found. The GPU Framework can not be built.")
  endif()

  #######################################################################################################################
  # Find OpenCL package
  # This CMake macro is the to setup OpenCL for the GPU Framework
  #######################################################################################################################

  #find_package(OpenCL REQUIRED)
  #if (NOT OpenCL_FOUND)
    #message(FATAL_ERROR "OpenCL not found. The GPU Framework can not be built.")
  #endif()

  #######################################################################################################################
  # Find CUDA package
  # This CMake macro is the to setup CUDA for the GPU Framework
  #######################################################################################################################

  find_package(CUDA REQUIRED)
  if (NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. The GPU Framework can not be built.")
  endif()

  # Use g++, gcov and lcov to generate code coverage html, only used in Coverage build mode
  if (CMAKE_COMPILER_IS_GNUCXX AND (CMAKE_BUILD_TYPE STREQUAL "Coverage"))
    #  Build a Coverage build, usage:
    #
    #  rm -rf *
    #  cmake -DCUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/NVCC -DCMAKE_BUILD_TYPE=Coverage -DCMAKE_CXX_COMPILER=/usr/bin/g++ ../gpuframework/
    #  make -j8
    #  make codeCoverage
    include(CodeCoverage)
    setupTargetForCoverage(codeCoverage HostUnitTests CodeCoverage)
  endif()
endmacro(setupGPUFrameworkExternalLibraries)

macro(setupDoxygen)
  option(BUILD_DOXYGEN "Use Doxygen to create the HTML based GPU Framework API Documentation." OFF)

  if (BUILD_DOXYGEN AND (CMAKE_SOURCE_DIR))
    # CMAKE_SOURCE_DIR means we execute this part only if setupGPUFrameworkExternalLibraries is called in global build
    # DOXYGEN should not be built in stand alone build case
    find_package(Doxygen)

    if (NOT DOXYGEN_FOUND)
      message(FATAL_ERROR "Doxygen is needed to build the GPU Framework API Documentation. Please install it correctly.")
    endif()

    configure_file(${CMAKE_SOURCE_DIR}/Scripts/doxygen.in ${CMAKE_BINARY_DIR}/Documentation/doxygen @ONLY)

    add_custom_target(Documentation
                      ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Documentation/doxygen
                      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/Documentation
                      COMMENT "Generating the GPU Framework API Documentation with Doxygen"
                      VERBATIM)

    install(DIRECTORY ${CMAKE_BINARY_DIR}/Documentation/html DESTINATION Documentation MESSAGE_NEVER)
  endif()
endmacro(setupDoxygen)

macro(cpackGPUFramework)
  set(CPACK_PACKAGE_NAME GPU_FRAMEWORK)
  set(CPACK_PACKAGE_VENDOR "Dot Red Consultancy Ltd (http://www.dotredconsultancy.com) - Thanos Theo")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU Framework - Installation")
  set(CPACK_GENERATOR ZIP)
  set(CPACK_PACKAGE_DIRECTORY ${CMAKE_INSTALL_PREFIX})
  set(CPACK_PACKAGE_VERSION_MAJOR ${GPU_FRAMEWORK_VERSION_MAJOR})
  set(CPACK_PACKAGE_VERSION_MINOR ${GPU_FRAMEWORK_VERSION_MINOR})
  set(CPACK_PACKAGE_VERSION_PATCH ${GPU_FRAMEWORK_VERSION_PATCH})
  set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
  set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 0)
  if (MSVC)
    set(PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-win64-vs14-${CPACK_PACKAGE_VERSION}-r${BUILD_NUMBER}")
  elseif (UNIX)
    set(PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-lin64-clang38-${CPACK_PACKAGE_VERSION}-r${BUILD_NUMBER}")
  endif()
  set(CPACK_PACKAGE_FILE_NAME ${PACKAGE_FILE_NAME})
  # This must always be last!
  include(CPack)
endmacro(cpackGPUFramework)
