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

#ifndef __LinearAlgebraCPUComputingStressTest_h
#define __LinearAlgebraCPUComputingStressTest_h

#include "UnitTests.h"
#include "AccurateTimers.h"
#include <cstdint>
#include <memory>

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
    /** @brief This class contains a basic Linear Algebra CPU Computing stress test case in the host. Using the Curiously Recurring Template Pattern (CRTP).
    *
    *  LinearAlgebraCPUComputingStressTest.h:
    *  =====================================
    *  This class contains a basic Linear Algebra CPU Computing stress test case in the host. Using the Curiously Recurring Template Pattern (CRTP).
    *
    * @author Thanos Theo, 2019
    * @version 14.0.0.0
    */
    class LinearAlgebraCPUComputingStressTest final : private UnitTests::UnitTestsInterface<LinearAlgebraCPUComputingStressTest>, private UnitTests::UnitTestUtilityFunctions_flt // private inheritance used for composition and prohibiting up-casting
    {
    public:

        // IUnitTests -> for Google Tests
        void resetTests();
        bool conductTests();
        void reportTestResults();

        LinearAlgebraCPUComputingStressTest(std::size_t arraySize = 16384, size_t numberOfCPUKernelIterations = 160) noexcept;
        ~LinearAlgebraCPUComputingStressTest() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
        LinearAlgebraCPUComputingStressTest(const LinearAlgebraCPUComputingStressTest&) = delete; // copy-constructor delete
        LinearAlgebraCPUComputingStressTest(LinearAlgebraCPUComputingStressTest&&)      = delete; // move-constructor delete
        LinearAlgebraCPUComputingStressTest& operator=(const LinearAlgebraCPUComputingStressTest&) = delete; //      assignment operator delete
        LinearAlgebraCPUComputingStressTest& operator=(LinearAlgebraCPUComputingStressTest&&)      = delete; // move-assignment operator delete

    private:

        std::size_t arraySize_                   = 16384 * 16384;
        std::size_t numberOfCPUKernelIterations_ = 0;

        std::unique_ptr<std::int32_t[]> arrayA_ = nullptr;
        std::unique_ptr<std::int32_t[]> arrayB_ = nullptr;
        std::unique_ptr<std::int32_t[]> arrayC_ = nullptr;

        AccurateTimers::AccurateCPUTimer cpuTimer_;
        double totalTimeTakenInMs_ = 0.0;
    };
} // namespace Utils

#endif // __LinearAlgebraCPUComputingStressTest_h