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

# move INSTALL and ZERO_CHECK targets into folder
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

cmake_minimum_required(VERSION 3.9.0)
option(BUILD_SHARED_LIBS "Build shared libraries (DLLs)." ON)

# set the project version
project(GPUFramework LANGUAGES CXX VERSION 14.0.0.0 DESCRIPTION "GPU Framework Version 14.0")

# set the module path so include can find modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Scripts/CMake")

# CMAKE_MODULE_PATH is set, so include looks for modules in that directory
include(MacroUtils)

# setup the compiler & linker environment
setupCompilerAndLinker()

if (${CMAKE_SOURCE_DIR} MATCHES ${CMAKE_BINARY_DIR})
  message(STATUS SEND_ERROR)
  message(STATUS "-- ERROR: in-tree-build not allowed.")
  return()
endif()

set(GPU_FRAMEWORK_MAJOR_VERSION 1)
set(GPU_FRAMEWORK_MINOR_VERSION 0)
set(GPU_FRAMEWORK_BUILD_VERSION 0)
set(GPU_FRAMEWORK_VERSION "${GPU_FRAMEWORK_MAJOR_VERSION}.${GPU_FRAMEWORK_MINOR_VERSION}.${GPU_FRAMEWORK_BUILD_VERSION}")

#######################################################################################################################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# setup the build preprocessor environment
setupGPUFrameworkPreprocessorDefines()

# setup all external libraries
setupGPUFrameworkExternalLibraries()

# setup doxygen
setupDoxygen()

#######################################################################################################################

option(BUILD_TESTS "Build all of GPUFramework GoogleTest Unit Tests." on)

if (BUILD_TESTS)
  # GPUFramework GoogleTest Unit Tests
  add_subdirectory (GPUFrameworkTests)
endif()

# perform all the scripts move from source repository to build directories
if (WIN32)
  # copy the Assets directory from the repository
  file(COPY ${CMAKE_SOURCE_DIR}/Assets DESTINATION ${CMAKE_BINARY_DIR}/bin/Debug)
  file(COPY ${CMAKE_SOURCE_DIR}/Assets DESTINATION ${CMAKE_BINARY_DIR}/bin/Release)

  # create the debug environment in VS
  configure_file(${CMAKE_SOURCE_DIR}/Scripts/cmdWindows/Template.vcxproj.user.in ${CMAKE_BINARY_DIR}/GPUFrameworkMain.vcxproj.user COPYONLY)
elseif (UNIX)
  # copy the Assets directory from the repository
  file(COPY ${CMAKE_SOURCE_DIR}/Assets DESTINATION ${CMAKE_BINARY_DIR}/bin)
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/Scripts/CMake DESTINATION . FILES_MATCHING PATTERN "Find*.cmake" PATTERN "*license*" EXCLUDE)

# cpack macro must be last in cmakelist
cpackGPUFramework()
