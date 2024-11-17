# Copyright (c) 2012 - 2015, Lars Bilke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN if ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#
#
# 2012-01-31, Lars Bilke
# - Enable Code Coverage
#
# 2013-09-17, Joakim Söderberg
# - Added support for Clang.
# - Some additional usage instructions.
#
# 2016-12-14, Thanos Theo
# - Improved the robustness of function setupTargetForCoverage.
# - removed unneeded part of this script
# - some comment improved.

# USAGE:

# 0. Copy this file into your cmake modules path.
#
# 1. Add the following line to your CMakeLists.txt:
#      INCLUDE(CodeCoverage)
#
# 2. Set compiler flags to turn off optimization and enable coverage:
#    set(CMAKE_CXX_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
#    set(CMAKE_C_FLAGS "-g -O0 -fprofile-arcs -ftest-coverage")
#
# 3. Use the function SETUP_TARGET_FOR_COVERAGE to create a custom make target
#    which runs your test executable and produces a lcov code coverage report:
#    Example:
#   SETUP_TARGET_FOR_COVERAGE(
#       my_coverage_target  # Name for custom target.
#       test_driver         # Name of the test driver executable that runs the tests.
#                 # NOTE! This should always have a ZERO as exit code
#                 # otherwise the coverage generation will not complete.
#       coverage            # Name of output directory.
#       )
#
# 4. Build a Coverage build:
#    cmake -DCMAKE_BUILD_TYPE=Coverage ..
#    make
#    make my_coverage_target
#

# Check prereqs
find_program( GCOV_PATH gcov)
find_program( LCOV_PATH lcov )
find_program( GENHTML_PATH genhtml )

if (NOT GCOV_PATH)
  message(FATAL_ERROR "gcov not found! Aborting...")
endif() # NOT GCOV_PATH

#this is for mac clang not linux clang
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "(Apple)?[Cc]lang")
  if ("${CMAKE_CXX_COMPILER_VERSION}" VERSION_LESS 3)
    message(FATAL_ERROR "Clang version must be 3.0.0 or greater! Aborting...")
  endif()
elseif (NOT CMAKE_COMPILER_IS_GNUCXX)
  message(FATAL_ERROR "Compiler is not GNU gcc! Aborting...")
endif() # CHECK VALID COMPILER

set(CMAKE_CXX_FLAGS_COVERAGE
  "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
  CACHE STRING "Flags used by the C++ compiler during coverage builds."
  FORCE )
set(CMAKE_C_FLAGS_COVERAGE
  "-g -O0 --coverage -fprofile-arcs -ftest-coverage"
  CACHE STRING "Flags used by the C compiler during coverage builds."
  FORCE )
set(CMAKE_EXE_LINKER_FLAGS_COVERAGE
  ""
  CACHE STRING "Flags used for linking binaries during coverage builds."
  FORCE )
set(CMAKE_SHARED_LINKER_FLAGS_COVERAGE
  ""
  CACHE STRING "Flags used by the shared libraries linker during coverage builds."
  FORCE )
MARK_AS_ADVANCED(
  CMAKE_CXX_FLAGS_COVERAGE
  CMAKE_C_FLAGS_COVERAGE
  CMAKE_EXE_LINKER_FLAGS_COVERAGE
  CMAKE_SHARED_LINKER_FLAGS_COVERAGE )

if (NOT (CMAKE_BUILD_TYPE STREQUAL "Coverage"))
  message( WARNING "Code coverage results with an optimized (non-Coverage) build may be misleading" )
endif() # NOT CMAKE_BUILD_TYPE STREQUAL "Debug"

# Param _targetname     The name of new the custom make target
# Param _testrunner     The name of the target which runs the tests.
#           MUST return ZERO always, even on errors.
#           If not, no coverage report will be created!
# Param _outputname     lcov output is generated as _outputname.info
#                       HTML report is generated in _outputname/index.html
# Optional fourth parameter is passed as arguments to _testrunner
#   Pass them in list form, e.g.: "-j;2" for -j 2
function(setupTargetForCoverage _targetname _testrunner _outputname)
  if (NOT LCOV_PATH)
    message(FATAL_ERROR "lcov not found! Aborting...")
  endif() # NOT LCOV_PATH

  if (NOT GENHTML_PATH)
    message(FATAL_ERROR "genhtml not found! Aborting...")
  endif() # NOT GENHTML_PATH

  set(coverage_info "${CMAKE_BINARY_DIR}/${_outputname}.info")
  set(coverage_cleaned "${coverage_info}.cleaned")

  separate_arguments(test_command UNIX_COMMAND "${_testrunner}")

  add_custom_target(${_targetname}
    #We have to stay in the ${CMAKE_BINARY_DIR}/bin folder to run unittest instead of ${CMAKE_BINARY_DIR}
    #so all athe --directory related argument has to be one level up ../ instead of current dir . .

    #Cleanup lcov
    COMMAND ${LCOV_PATH} --directory ../ --zerocounters

    #Run GPU Framework Unit Tests
    COMMAND ${test_command} ${ARGV3}
    COMMAND HostUnitTests
    COMMAND DeviceUnitTests

    # Capturing lcov counters and generating report
    COMMAND ${LCOV_PATH} --directory ../ --capture --output-file ${coverage_info}
    COMMAND ${LCOV_PATH} --remove ${coverage_info} '/usr/*' 'Assets/*' 'GoogleTest/*' 'Tests/*' '*miniz.h' '*lodepng.h' '*lodepng.cpp' --output-file ${coverage_cleaned}
    COMMAND ${GENHTML_PATH} -o ../${_outputname} ${coverage_cleaned}
    COMMAND ${CMAKE_COMMAND} -E remove ${coverage_info} ${coverage_cleaned}

    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Resetting code coverage counters to zero.\nProcessing code coverage counters and generating report."
    )

  # Show info where to find the report
  add_custom_command(TARGET ${_targetname} POST_BUILD
    COMMAND ;
    COMMENT "Open ./${_outputname}/index.html in your browser to view the coverage report."
    )
endfunction() # SETUP_TARGET_FOR_COVERAGE