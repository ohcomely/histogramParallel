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

#ifndef __DeviceUnitTests_h
#define __DeviceUnitTests_h

/** @brief Namespace Tests for all relevant unit testing host & device (CPU & GPU) code.
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace Tests
{
  /** @brief Device Google Test 01 for the UtilsCUDA::CUDADriverInfo class functionality.
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest01__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 02 for the UtilsCUDA::CUDALinearAlgebraGPUComputing class functionality.
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest02__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 03 for the UtilsCUDA::CUDADriverInfo class CUDA Memory Registry functionality.
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest03__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 04 for the UtilsCUDA::CUDAMemoryHandler set of classes functionality.
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest04__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 05 for the UtilsCUDA::CUDAMemoryHandler (RawDeviceMemory, Span) set of classes functionality.
  * @author David Lenz, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest05__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 06 for the UtilsCUDA::CUDAUtilityFunctions functionality.
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest06__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 07 for the UtilsCUDA::CUDADeviceUtilityFunctions functionality.
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest07__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 08 for the UtilsCUDA::CUDAKernelLauncher functionality.
  * @author David Lenz, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest08__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 09 for the UtilsCUDA::CUDAMemoryPool functionality.
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest09__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 10 for the UtilsCUDA::CUDAUtilityFunctions fast memset functionality.
  * @author Leonid Volnin, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest10__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 11 for the UtilsCUDA::CUDAUtilityFunctions powf(float) & pow(double) functionality.
  * @author Thanos Theo, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest11__UTILS_CUDA_Classes
  {
    void executeTest();
  }

  /** @brief Device Google Test 12 for the UtilsCUDA::CUDAQueue functionality.
  * @author Leonid Volnin, 2019
  * @version 14.0.0.0
  */
  namespace DeviceGoogleTest12__UTILS_CUDA_Classes
  {
    void executeTest();
  }


  //Ritvik's Test
  namespace DeviceGoogleTest13__Color_Histogram_GPU
  {
    void executeTest();
  
  }

  namespace DeviceGoogleTest14__Complete_Histogram_Test
  {
    void executeTest();

  }

} // namespace Tests

#endif // __DeviceUnitTests_h