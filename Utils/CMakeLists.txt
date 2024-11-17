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

project(Utils CXX)

# set the module path so include can find modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Scripts/CMake")

# CMAKE_MODULE_PATH is set, so include looks for modules in that directory
include(MacroUtils)

# setup the compiler & linker environment
setupCompilerAndLinker()

include(include/CMakeLists.txt)
include(include/CPUParallelism/CMakeLists.txt)

include(src/CMakeLists.txt)
include(src/CPUParallelism/CMakeLists.txt)

#########################################################################
##
## Defines the Utils libraries.

add_library(Utils
  ${UTILS_SRC_CPP} ${UTILS_HEADERS} ${UTILS_CMAKELISTS}
  ${UTILS_SRC_CPUParallel_CPP} ${UTILS_CPUParallel_HEADERS} ${UTILS_CPUParallel_CMAKELISTS}
  )

if (WIN32)
  # Threads::Threads is for Threads (Windows/Linux, on Linux it is pthreads)
  target_link_libraries(Utils Threads::Threads)
elseif (UNIX)
  # Threads::Threads is for Threads (Windows/Linux, on Linux it is pthreads)
  # stdc++fs is for #include <experimental/filesystem>
  target_link_libraries(Utils Threads::Threads stdc++fs)
endif()

# for both internal build and parent build (ie being pulled in via add_subdirectory() rather than being a standalone build)
target_include_directories(
  Utils PUBLIC
  ${Utils_SOURCE_DIR}/include
  )

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

if (WIN32)
  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .   COMPONENT distributable EXCLUDE_FROM_ALL)
elseif (UNIX)
  install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib COMPONENT distributable EXCLUDE_FROM_ALL)
endif()