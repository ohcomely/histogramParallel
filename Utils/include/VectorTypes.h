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

#ifndef __VectorTypes_h
#define __VectorTypes_h

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace VectorTypes provides float2-3-4 functionality.
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace VectorTypes
  {
    /** @brief The float2 class provides float2 functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct float2 final
    {
      float x = 0.0f;
      float y = 0.0f;

      float2()  = default;
      float2(float x, float y) noexcept : x(x), y(y) {}
      ~float2() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      float2(const float2&)  = default;
      float2(float2&& other) = default;
      float2& operator=(const float2&)  = default;
      float2& operator=(float2&& other) = default;
    };

    /** @brief The float3 class provides float3 functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct float3 final
    {
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;

      float3()  = default;
      float3(float x, float y, float z) noexcept : x(x), y(y), z(z) {}
      ~float3() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      float3(const float3&)  = default;
      float3(float3&& other) = default;
      float3& operator=(const float3&)  = default;
      float3& operator=(float3&& other) = default;
    };

    /** @brief The float4 class provides float4 functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct float4 final
    {
      float x = 0.0f;
      float y = 0.0f;
      float z = 0.0f;
      float w = 0.0f;

      float4()  = default;
      float4(float x, float y, float z, float w) noexcept : x(x), y(y), z(z), w(w) {}
      ~float4() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      float4(const float4&)  = default;
      float4(float4&& other) = default;
      float4& operator=(const float4&)  = default;
      float4& operator=(float4&& other) = default;
    };

    /** @brief The double2 class provides double2 functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct double2 final
    {
      double x = 0.0;
      double y = 0.0;

      double2()  = default;
      double2(double x, double y) noexcept : x(x), y(y) {}
      ~double2() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      double2(const double2&)  = default;
      double2(double2&& other) = default;
      double2& operator=(const double2&)  = default;
      double2& operator=(double2&& other) = default;
    };

    /** @brief The double3 class provides double3 functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct double3 final
    {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;

      double3()  = default;
      double3(double x, double y, double z) noexcept : x(x), y(y), z(z) {}
      ~double3() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      double3(const double3&)  = default;
      double3(double3&& other) = default;
      double3& operator=(const double3&)  = default;
      double3& operator=(double3&& other) = default;
    };

    /** @brief The double4 class provides double4 functionality.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    struct double4 final
    {
      double x = 0.0;
      double y = 0.0;
      double z = 0.0;
      double w = 0.0;

      double4()  = default;
      double4(double x, double y, double z, double w) noexcept : x(x), y(y), z(z), w(w) {}
      ~double4() = default; // no virtual destructor for data-oriented design (no up-casting should ever be used)
      double4(const double4&)  = default;
      double4(double4&& other) = default;
      double4& operator=(const double4&)  = default;
      double4& operator=(double4&& other) = default;
    };
  } // namespace VectorTypes
} // namespace Utils

#endif // __VectorTypes_h