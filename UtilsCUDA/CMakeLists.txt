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

cmake_minimum_required(VERSION 3.9.0)
option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." on)

# check if the externally dependent library exist, report an error if not.
# UtilsCUDA depends on Utils & CUDA headers.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Utils/CMakeLists.txt")
  set(Utils_dir ../Utils)
else()
  message(STATUS FATAL_ERROR "External needed Utils library is not found. UtilsCUDA can not be built.")
endif()

project(UtilsCUDA CXX)

# set the module path so include can find modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Scripts/CMake")

# CMAKE_MODULE_PATH is set, so include looks for modules in that directory
include(MacroUtils)

# if dependency is missing, then build them
if (NOT TARGET Utils)
  #When specifying an out-of-tree source a binary directory must be explicitly specified,
  add_subdirectory(${Utils_dir} ${CMAKE_BINARY_DIR}/Utils)
endif()

if (NOT CUDA_FOUND)
  find_package(CUDA REQUIRED)
endif()

# normally, config_compiler_and_linker is called on the top of the file to set compilation flag.
# but here if it is stand-alone case, then it has to be done after find_package(CUDA REQUIRED),
# since find cuda resset the NVCC compilation flags.

# setup the compiler & linker environment
setupCompilerAndLinker()

include(include/CMakeLists.txt)
include(src/CMakeLists.txt)

# -fPIC is in g++ and clang++ by default, but for NVCC separate compilation, we have to add it explicitly
if (UNIX)
  set(CMAKE_CXX_FLAGS "-mavx2 -O3 -Wall -fPIC")
endif()

# remove -g -O0 --coverage -fprofile-arcs -ftest-coverage from CMAKE_CXX_FLAGS_COVERAGE for NVCC
# because NVCC generates immature gcda files and these files break the whole genhtml process
set(CMAKE_CXX_FLAGS_COVERAGE "")

cuda_add_library(
  UtilsCUDA
  ${UTILS_CUDA_HEADERS} ${UTILS_CUDA_SRC_CPP} ${UTILS_CUDA_CMAKELISTS}
)

# link UtilsCUDA & Utils together
target_link_libraries(UtilsCUDA Utils)

# for both internal build and parent build (ie being pulled in via add_subdirectory() rather than being a standalone build)
target_include_directories(
  UtilsCUDA PUBLIC
  ${UtilsCUDA_SOURCE_DIR}/include
  ${CUDA_TOOLKIT_INCLUDE}
  )

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

if (WIN32)
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .   COMPONENT distributable EXCLUDE_FROM_ALL)
elseif (UNIX)
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib COMPONENT distributable EXCLUDE_FROM_ALL)
endif()