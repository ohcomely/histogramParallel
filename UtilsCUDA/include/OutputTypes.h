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

#ifndef __OutputTypes_h
#define __OutputTypes_h

#include <cstdint>

/** @brief namespace UtilsCUDA for encapsulating all the CUDA related code compiled by the NVCC compiler.
*
* @author Thanos Theo, 2018
* @version 14.0.0.0
*/
namespace UtilsCUDA
{
  /** @brief Usage of a C-style enum (not typesafe C++11 enum class) to be able to use a viz-style bitwise flag OR API on enum values.
  *
  *  OutputTypes.h:
  *  =============
  *  Usage of a C-style enum (not typesafe C++11 enum class) to be able to use a viz-style bitwise flag OR API on enum values.
  *
  * @author Thanos Theo, 2018
  * @version 14.0.0.0
  */
  struct OutputTypes
  {
    enum OutputType : std::uint32_t
    {
      WRITE_TO_NOTHING     = (1 <<  0),
      WRITE_TO_CPU_MEMORY  = (1 <<  1),
      WRITE_TO_BINARY      = (1 <<  2),
      WRITE_TO_ZIP         = (1 <<  3),
      WRITE_TO_TEXT        = (1 <<  4),
      WRITE_TO_GPU0_MEMORY = (1 <<  5),
      WRITE_TO_GPU1_MEMORY = (1 <<  6),
      WRITE_TO_GPU2_MEMORY = (1 <<  7),
      WRITE_TO_GPU3_MEMORY = (1 <<  8),
      WRITE_TO_GPU4_MEMORY = (1 <<  9),
      WRITE_TO_GPU5_MEMORY = (1 << 10),
      WRITE_TO_GPU6_MEMORY = (1 << 11),
      WRITE_TO_GPU7_MEMORY = (1 << 12)
    };

    OutputTypes()  = delete;
    ~OutputTypes() = delete;
    OutputTypes(const OutputTypes&) = delete;
    OutputTypes(OutputTypes&&)      = delete;
    OutputTypes& operator=(const OutputTypes&) = delete;
    OutputTypes& operator=(OutputTypes&&)      = delete;
  };
} // namespace UtilsCUDA

#endif // __OutputTypes_h