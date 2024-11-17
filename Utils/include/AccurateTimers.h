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

#ifndef __AccurateTimers_h
#define __AccurateTimers_h

#include "ModuleDLL.h"
#include "UtilityFunctions.h"
#include <array>
#include <chrono>
#include <cstdint>
#include <string>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace AccurateTimers contains utility classes for accurate timer logging.
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace AccurateTimers
  {
    /** @brief The AccurateTimerInterface struct encapsulates a basic interface for a generic high resolution timer using the Curiously Recurring Template Pattern (CRTP).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    template <typename Derived>
    class CRTP_MODULE_API AccurateTimerInterface
    {
    public:

      void startTimer()                         {        asDerived()->startTimer(); }
      void stopTimer()                          {        asDerived()->stopTimer();  }
      double getElapsedTimeInNanoSecs()         { return asDerived()->getElapsedTimeInNanoSecs();  }
      double getElapsedTimeInMicroSecs()        { return asDerived()->getElapsedTimeInMicroSecs(); }
      double getElapsedTimeInMilliSecs()        { return asDerived()->getElapsedTimeInMilliSecs(); }
      double getElapsedTimeInSecs()             { return asDerived()->getElapsedTimeInSecs();      }
      double getMeanTimeInNanoSecs()            { return asDerived()->getMeanTimeInNanoSecs();  }
      double getMeanTimeInMicroSecs()           { return asDerived()->getMeanTimeInMicroSecs(); }
      double getMeanTimeInMilliSecs()           { return asDerived()->getMeanTimeInMilliSecs(); }
      double getMeanTimeInSecs()                { return asDerived()->getMeanTimeInSecs();      }
      double getDecimalElapsedTimeInMicroSecs() { return asDerived()->getDecimalElapsedTimeInMicroSecs(); }
      double getDecimalElapsedTimeInMilliSecs() { return asDerived()->getDecimalElapsedTimeInMilliSecs(); }
      double getDecimalElapsedTimeInSecs()      { return asDerived()->getDecimalElapsedTimeInSecs();      }
      double getDecimalMeanTimeInMicroSecs()    { return asDerived()->getDecimalMeanTimeInMicroSecs(); }
      double getDecimalMeanTimeInMilliSecs()    { return asDerived()->getDecimalMeanTimeInMilliSecs(); }
      double getDecimalMeanTimeInSecs()         { return asDerived()->getDecimalMeanTimeInSecs();      }

      AccurateTimerInterface()  = default;
      ~AccurateTimerInterface() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      AccurateTimerInterface(const AccurateTimerInterface&) = delete; // copy-constructor delete
      AccurateTimerInterface(AccurateTimerInterface&&)      = delete; // move-constructor delete
      AccurateTimerInterface& operator=(const AccurateTimerInterface&) = delete; //      assignment operator delete
      AccurateTimerInterface& operator=(AccurateTimerInterface&&)      = delete; // move-assignment operator delete

    private:

            Derived* asDerived()       { return reinterpret_cast<      Derived*>(this); }
      const Derived* asDerived() const { return reinterpret_cast<const Derived*>(this); }
    };

    /** @brief The AccurateTimerLog struct is to be used for composition in timer related sub-classes through private inheritance.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct UTILS_MODULE_API AccurateTimerLog
    {
      enum class TimerTypes : std::size_t { NANOSECS = 0, MICROSECS = 1, MILLISECS = 2, SECS = 3 };

      // use the nanosecs timer divided with the numbers below to have decimal accuracy
      static constexpr double NANO_TO_MICROSECS_CONVERSION =        1000.0;
      static constexpr double NANO_TO_MILLISECS_CONVERSION =     1000000.0;
      static constexpr double NANO_TO_SECS_CONVERSION      =  1000000000.0;

      // book-keeping variables for the mean timers
      static constexpr std::size_t NUMBER_OF_TIMER_FORMATS  = 4;
      static constexpr std::size_t TIMERS_BOOK_KEEPING_SIZE = 11;

      /** @brief The implementation below is based on BitSquid's Time Step Smoothing article.
      *
      *  The implementation below is based on BitSquid's Time Step Smoothing article:
      *  http://bitsquid.blogspot.se/2010/10/time-step-smoothing.html
      *  It does it in 4 main steps:
      *  1) Keep a history of the time step for the last 11 frames.
      *  2) Throw away the outliers, the two highest and the two lowest values.
      *  3) Calculate the mean of the remaining 7 values.
      *  4) Lerp from the time step for the last frame to the calculated mean (adding more smoothness)
      *
      * @author Thanos Theo, 2009-2018
      * @version 14.0.0.0
      */
      static double calculateMeanTime(double currentTime, double* __restrict timersBookKeeping, std::int64_t& timersBookKeepingIndex, bool& firstTimersBookKeepingIterationCompleted);

      double timersBookKeeping_[NUMBER_OF_TIMER_FORMATS][TIMERS_BOOK_KEEPING_SIZE] = { { 0.0 } };
      std::array<std::int64_t, NUMBER_OF_TIMER_FORMATS> timersBookKeepingIndex_{ { 0 } };               // double braces because we initialize an array inside an std::array object
      std::array<bool, NUMBER_OF_TIMER_FORMATS> firstTimersBookKeepingIterationCompleted_{ { false } }; // double braces because we initialize an array inside an std::array object

      bool stopped_ = false;

      AccurateTimerLog() noexcept;
      ~AccurateTimerLog() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      AccurateTimerLog(const AccurateTimerLog&) = delete; // copy-constructor delete
      AccurateTimerLog(AccurateTimerLog&&)      = delete; // move-constructor delete
      AccurateTimerLog& operator=(const AccurateTimerLog&) = delete; //      assignment operator delete
      AccurateTimerLog& operator=(AccurateTimerLog&&)      = delete; // move-assignment operator delete
    };

    /** @brief The AccurateCPUTimer class provides a concrete implementation of a high resolution CPU timer using the 'chrono' C++11 namespace.
    *
    *  Note: No virtual destructor is needed for data-oriented design, ie no up-casting should ever be used. Using the Curiously Recurring Template Pattern (CRTP).
    *
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class UTILS_MODULE_API AccurateCPUTimer final : private AccurateTimerInterface<AccurateCPUTimer>, private AccurateTimerLog // private inheritance used for composition and prohibiting up-casting
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

      static std::uint64_t getNanosecondsTimeSinceEpoch();
      static std::uint64_t getMicrosecondsTimeSinceEpoch();
      static std::uint64_t getMillisecondsTimeSinceEpoch();
      static std::uint64_t getSecondsTimeSinceEpoch();

      AccurateCPUTimer()  = default;
      ~AccurateCPUTimer() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      AccurateCPUTimer(const AccurateCPUTimer&) = delete; // copy-constructor delete
      AccurateCPUTimer(AccurateCPUTimer&&)      = delete; // move-constructor delete
      AccurateCPUTimer& operator=(const AccurateCPUTimer&) = delete; //      assignment operator delete
      AccurateCPUTimer& operator=(AccurateCPUTimer&&)      = delete; // move-assignment operator delete

    private:

      std::chrono::high_resolution_clock::time_point start_ = std::chrono::high_resolution_clock::now();
      std::chrono::high_resolution_clock::time_point stop_  = std::chrono::high_resolution_clock::now();

      template<typename ChronoType> double getElapsedTime();
    };

    /** @brief ProfileCPUTimer profiling helper class using RAII.
    *
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class ProfileCPUTimer final
    {
    public:

      ProfileCPUTimer(const std::string& message = "")
        : message_(message)
      {
        timer_.startTimer();
      }

      double getElapsedTimeInMilliSecs()
      {
        return timer_.getElapsedTimeInMilliSecs();
      }

      ~ProfileCPUTimer()
      {
        timer_.stopTimer();
        DebugConsole_consoleOutLine(message_, " ", timer_.getDecimalElapsedTimeInMilliSecs(), " ms.");
      }

      ProfileCPUTimer(const ProfileCPUTimer&) = delete; // copy-constructor delete
      ProfileCPUTimer(ProfileCPUTimer&&)      = delete; // move-constructor delete
      ProfileCPUTimer& operator=(const ProfileCPUTimer&) = delete; //      assignment operator delete
      ProfileCPUTimer& operator=(ProfileCPUTimer&&)      = delete; // move-assignment operator delete

    private:

      std::string message_;
      AccurateCPUTimer timer_;
    };
  } // namespace AccurateTimers
} // namespace Utils

#endif // __AccurateTimers_h