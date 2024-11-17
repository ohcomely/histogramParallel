/*

Copyright (c) 2009-2018, Thanos Theo. All rights reserved.
Released Under a Simplified BSD (FreeBSD) License
for academic, personal & non-commercial use.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the author and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

A Commercial License is also available for commercial use with
special restrictions and obligations at a one-off fee. See links at:
1. http://www.dotredconsultancy.com/openglrenderingenginetoolrelease.php
2. http://www.dotredconsultancy.com/openglrenderingenginetoolsourcecodelicence.php
Please contact Thanos Theo (thanos.theo@dotredconsultancy.com) for more information.

*/

#pragma once

#ifndef __EnvironmentConfig_h
#define __EnvironmentConfig_h

#ifdef __cplusplus
extern "C"
{
#endif

/**
*
*  EnvironmentConfig.h:
*  ===================
*  This header-only file contains preprocessor
*  defines to be used throughout the GPU Framework.
*  Note: All the commented-out preprocessor defines
*        below are directly set from the CMake UI.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/

// Note: All the commented-out preprocessor defines below should directly be set from outside.
// Note: comment this line below to remove all debug code & messages from the GPU Framework.
// #define GPU_FRAMEWORK_DEBUG
// Note: comment this line below to remove reporting of NCP parallelFor() call.
// #define GPU_FRAMEWORK_PROFILE_NCP_PARALLEL_FOR
// Note: comment this line below to remove exceptions for CUDA errors and use a C-style exit(errnum).
// #define GPU_FRAMEWORK_USE_EXCEPTION_FOR_CUDA_ERROR
// Note: use 0 or 1 for single global allocations for the CUDAMemoryPool        (Note: to be enabled for debugging purposes only)
#define GPU_FRAMEWORK_CUDA_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS         0
// Note: use 0 or 1 for single global allocations for the CUDAProcessMemoryPool (Note: to be enabled for debugging purposes only)
#define GPU_FRAMEWORK_CUDA_PROCESS_MEMORY_POOL_USE_SEPARATE_ALLOCATIONS 0

// set to 0 or 1 in order to disable or enable
// normally controlled from GPU_FRAMEWORK_DEBUG, override when needed for more fine-grained error output
#ifdef GPU_FRAMEWORK_DEBUG
  #define GPU_FRAMEWORK_DEBUG_CONSOLE       1
  #define GPU_FRAMEWORK_CUDA_CONSOLE        1
  #define GPU_FRAMEWORK_CUDA_ERROR          1
  #define GPU_FRAMEWORK_CUDA_ERROR_DEBUG    1
  #define GPU_FRAMEWORK_CUDA_DRIVER_INFO    1
  #define GPU_FRAMEWORK_GL_CONSOLE          1
#else
  #define GPU_FRAMEWORK_DEBUG_CONSOLE       0
  #define GPU_FRAMEWORK_CUDA_CONSOLE        0
  #define GPU_FRAMEWORK_CUDA_ERROR          0
  #define GPU_FRAMEWORK_CUDA_ERROR_DEBUG    0
  #define GPU_FRAMEWORK_CUDA_DRIVER_INFO    0
  #define GPU_FRAMEWORK_GL_CONSOLE          0
#endif // GPU_FRAMEWORK_DEBUG

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // __EnvironmentConfig_h