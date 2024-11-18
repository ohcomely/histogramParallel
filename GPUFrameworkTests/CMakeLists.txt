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

project(GPUFrameworkTests CXX)

# set the module path so include can find modules
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/Scripts/CMake")

# CMAKE_MODULE_PATH is set, so include looks for modules in that directory
include(MacroUtils)

# setup the compiler & linker environment
setupCompilerAndLinker()

if (NOT CUDA_FOUND)
  # setup the build preprocessor environment
  setupGPUFrameworkPreprocessorDefines()

  # setup all external libraries
  setupGPUFrameworkExternalLibraries()

  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# if dependency is missing, then build them
if (NOT TARGET Utils)
  set(Utils_dir ../Utils)
  add_subdirectory(${Utils_dir} ${CMAKE_BINARY_DIR}/Utils)
endif()

# if dependency is missing, then build them
if (NOT TARGET UtilsCUDA)
  set(UtilsCUDA_dir ../UtilsCUDA)
  add_subdirectory(${UtilsCUDA_dir} ${CMAKE_BINARY_DIR}/UtilsCUDA)
endif()

set(gtest_dir ../GoogleTest)
add_subdirectory(${gtest_dir} ${CMAKE_BINARY_DIR}/gtest)

# move gtest target into CMakePredefinedTargets folder, because it is GoogleTest internal lib
set_target_properties (gtest PROPERTIES FOLDER CMakePredefinedTargets)

include_directories(
  ${GPUFrameworkTests_SOURCE_DIR}/include
  )

include(include/CMakeLists.txt)
include(src/CMakeLists.txt)

add_executable(HostUnitTests
  ${GPU_FRAMEWORK_TESTS_HOST_SRC_CPP} ${GPU_FRAMEWORK_TESTS_HOST_HEADER}
  ${GPU_FRAMEWORK_HOST_SRC_CPP}       ${GPU_FRAMEWORK_HOST_HEADER}
  )

target_link_libraries(HostUnitTests gtest Utils)
set_target_properties(HostUnitTests PROPERTIES FOLDER GPUFrameworkTests)

# keep CTest related code for future
# add_test(
#   NAME    HostUnitTests
#   COMMAND HostUnitTests
#   )

cuda_add_executable(DeviceUnitTests
  ${GPU_FRAMEWORK_TESTS_DEVICE_SRC_CU} ${GPU_FRAMEWORK_TESTS_DEVICE_HEADER}
  ${GPU_FRAMEWORK_DEVICE_SRC_CU}       ${GPU_FRAMEWORK_DEVICE_HEADER}  ${GPU_FRAMEWORK_TESTS_HOST_SRC_CPP}
)

target_link_libraries(DeviceUnitTests gtest UtilsCUDA)
set_target_properties(DeviceUnitTests PROPERTIES FOLDER GPUFrameworkTests)

# keep CTest related code for future
# add_test(
#   NAME    DeviceUnitTests
#   COMMAND DeviceUnitTests
#   )

add_executable(HostStressTests
  ${GPU_FRAMEWORK_STRESS_TESTS_HOST_SRC_CPP} ${GPU_FRAMEWORK_STRESS_TESTS_HOST_HEADER}
  ${GPU_FRAMEWORK_STRESS_HOST_SRC_CPP}       ${GPU_FRAMEWORK_STRESS_HOST_HEADER}
  )

target_link_libraries(HostStressTests gtest Utils)
set_target_properties(HostStressTests PROPERTIES FOLDER GPUFrameworkTests)

# keep CTest related code for future
# add_test(
#   NAME    HostStressTests
#   COMMAND HostStressTests
#   )

cuda_add_executable(HostDeviceStressTests
  ${GPU_FRAMEWORK_STRESS_TESTS_HOST_DEVICE_SRC_CU} ${GPU_FRAMEWORK_STRESS_TESTS_HOST_DEVICE_HEADER}
  ${GPU_FRAMEWORK_STRESS_HOST_DEVICE_SRC_CU}       ${GPU_FRAMEWORK_STRESS_HOST_DEVICE_HEADER}
  )

target_link_libraries(HostDeviceStressTests gtest UtilsCUDA)
set_target_properties(HostDeviceStressTests PROPERTIES FOLDER GPUFrameworkTests)

# keep CTest related code for future
# add_test(
#   NAME    HostDeviceStressTests
#   COMMAND HostDeviceStressTests
#   )

if (WIN32)
  # create the debug environment in VS
  configure_file(${GPUFrameworkTests_SOURCE_DIR}/../Scripts/cmdWindows/Template.vcxproj.user.in ${CMAKE_BINARY_DIR}/GPUFrameworkTests/HostUnitTests.vcxproj.user     COPYONLY)
  configure_file(${GPUFrameworkTests_SOURCE_DIR}/../Scripts/cmdWindows/Template.vcxproj.user.in ${CMAKE_BINARY_DIR}/GPUFrameworkTests/DeviceUnitTests.vcxproj.user   COPYONLY)

  configure_file(${GPUFrameworkTests_SOURCE_DIR}/../Scripts/cmdWindows/runAllUnitTests.cmd.in ${CMAKE_BINARY_DIR}/bin/Debug/runAllUnitTests.cmd   @ONLY)
  configure_file(${GPUFrameworkTests_SOURCE_DIR}/../Scripts/cmdWindows/runAllUnitTests.cmd.in ${CMAKE_BINARY_DIR}/bin/Release/runAllUnitTests.cmd @ONLY)
endif()