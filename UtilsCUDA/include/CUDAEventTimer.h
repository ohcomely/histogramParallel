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

#ifndef __CUDAEventTimer_h
#define __CUDAEventTimer_h

#include "ModuleDLL.h"
#include "AccurateTimers.h"
#include <cuda_runtime_api.h>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief This class contains an AccurateTimers encapsulation of CUDA event timers.
  *
  *  CUDAEventTimer.h:
  *  ================
  *  This class contains an AccurateTimers encapsulation of CUDA event timers.
  *  CUDA Events provides a timer with a resolution of around 0.5 microseconds.
  *  Note: No virtual destructor is needed for data-oriented design, no up-casting should ever be used. Using the Curiously Recurring Template Pattern (CRTP).
  *
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  class UTILS_CUDA_MODULE_API CUDAEventTimer final :  private Utils::AccurateTimers::AccurateTimerInterface<CUDAEventTimer>, private Utils::AccurateTimers::AccurateTimerLog // private inheritance used for composition and prohibiting up-casting
  {
  public:

    void startTimer();
    void stopTimer();
    double getElapsedTimeInNanoSecs();
    double getElapsedTimeInMicroSecs();
    double getElapsedTimeInMilliSecs();
    double getElapsedTimeInSecs();
    double getMeanTimeInNanoSecs();
    double getMeanTimeInMicroSecs();
    double getMeanTimeInMilliSecs();
    double getMeanTimeInSecs();
    double getDecimalElapsedTimeInMicroSecs();
    double getDecimalElapsedTimeInMilliSecs();
    double getDecimalElapsedTimeInSecs();
    double getDecimalMeanTimeInMicroSecs();
    double getDecimalMeanTimeInMilliSecs();
    double getDecimalMeanTimeInSecs();

    CUDAEventTimer(int device = 0, const cudaStream_t& cudaStream = nullptr) noexcept;
    ~CUDAEventTimer() noexcept; // no virtual destructor for data-oriented design (no up-casting should ever be used)
    CUDAEventTimer(const CUDAEventTimer&) = delete;
    CUDAEventTimer(CUDAEventTimer&&)      = delete;
    CUDAEventTimer& operator=(const CUDAEventTimer&) = delete;
    CUDAEventTimer& operator=(CUDAEventTimer&&)      = delete;

  private:

    const cudaStream_t cudaStream_{};
    cudaEvent_t start_{};
    cudaEvent_t stop_{};

    float getElapsedTime();
  };

  /** @brief ProfileGPUTimer profiling helper class using RAII.
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  class ProfileGPUTimer final
  {
  public:

    ProfileGPUTimer(const std::string& message = "", int device = 0, const cudaStream_t& cudaStream = nullptr)
      : message_(message)
      , timer_(device, cudaStream)
    {
      timer_.startTimer();
    }

    double getElapsedTimeInMilliSecs()
    {
      return timer_.getElapsedTimeInMilliSecs();
    }

    ~ProfileGPUTimer()
    {
      timer_.stopTimer();
      DebugConsole_consoleOutLine(message_, " ", timer_.getDecimalElapsedTimeInMilliSecs(), " ms.");
    }

    ProfileGPUTimer(const ProfileGPUTimer&) = delete; // copy-constructor delete
    ProfileGPUTimer(ProfileGPUTimer&&)      = delete; // move-constructor delete
    ProfileGPUTimer& operator=(const ProfileGPUTimer&) = delete; //      assignment operator delete
    ProfileGPUTimer& operator=(ProfileGPUTimer&&)      = delete; // move-assignment operator delete

  private:

    std::string message_;
    CUDAEventTimer timer_;
  };
} // namespace UtilsCUDA

#endif // __CUDAEventTimer_h