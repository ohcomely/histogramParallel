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

#ifndef __SIMDVectorizations_h
#define __SIMDVectorizations_h

#ifndef __aarch64__
  #include <immintrin.h> // SSE + AVX intrinsics header
  #include <array>
  #include <iostream>

  #ifdef _WIN32
    #include <intrin.h>           // needed for VS2017+ compatibility
    #include "UtilityFunctions.h" // needed for the reportCPUCapabilities() function
    #include <sstream>            // needed for the reportCPUCapabilities() function
    #include <string>             // needed for InstructionSetInternal class
    #include <bitset>             // needed for InstructionSetInternal class
    #include <vector>             // needed for InstructionSetInternal class
  #endif // _WIN32
#endif // __aarch64__

/** @brief Namespace Utils contains utility classes with mainly static CPU related methods.
* @author Thanos Theo, 2009-2018
* @version 14.0.0.0
*/
namespace Utils
{
  /** @brief Namespace SIMDVectorizations contains utility classes for SIMD vectorizations.
  *
  *  SIMDVectorizations.h:
  *  ====================
  *  These classes encapsulate the SSE/AVX SIMD instructions on Intel Hardware in an syntactical GLSL-friendly way.\n
  *  Originally based on with further extensions: https://www.cs.uaf.edu/2011/fall/cs441/lecture/09_29_SSE.html.   \n
  *
  * @author Thanos Theo, 2009-2018
  * @version 14.0.0.0
  */
  namespace SIMDVectorizations
  {
#ifndef __aarch64__
  #ifdef _WIN32
    /** @brief The InstructionSet class is a Windows-specific detection CPU Host mechanism implementation based on cpuid (from: https://msdn.microsoft.com/en-us/library/hskdteyh.aspx).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class InstructionSet
    {
      // forward declarations
      struct InstructionSetInternal;

    public:
      // getters
      static std::string Vendor() { return CPURepresentation.vendor; }
      static std::string Brand()  { return CPURepresentation.brand;  }

      static bool SSE3()       { return CPURepresentation.f_1_ECX[0]; }
      static bool PCLMULQDQ()  { return CPURepresentation.f_1_ECX[1]; }
      static bool MONITOR()    { return CPURepresentation.f_1_ECX[3]; }
      static bool SSSE3()      { return CPURepresentation.f_1_ECX[9]; }
      static bool FMA()        { return CPURepresentation.f_1_ECX[12]; }
      static bool CMPXCHG16B() { return CPURepresentation.f_1_ECX[13]; }
      static bool SSE41()      { return CPURepresentation.f_1_ECX[19]; }
      static bool SSE42()      { return CPURepresentation.f_1_ECX[20]; }
      static bool MOVBE()      { return CPURepresentation.f_1_ECX[22]; }
      static bool POPCNT()     { return CPURepresentation.f_1_ECX[23]; }
      static bool AES()        { return CPURepresentation.f_1_ECX[25]; }
      static bool XSAVE()      { return CPURepresentation.f_1_ECX[26]; }
      static bool OSXSAVE()    { return CPURepresentation.f_1_ECX[27]; }
      static bool AVX()        { return CPURepresentation.f_1_ECX[28]; }
      static bool F16C()       { return CPURepresentation.f_1_ECX[29]; }
      static bool RDRAND()     { return CPURepresentation.f_1_ECX[30]; }

      static bool MSR()   { return CPURepresentation.f_1_EDX[5]; }
      static bool CX8()   { return CPURepresentation.f_1_EDX[8]; }
      static bool SEP()   { return CPURepresentation.f_1_EDX[11]; }
      static bool CMOV()  { return CPURepresentation.f_1_EDX[15]; }
      static bool CLFSH() { return CPURepresentation.f_1_EDX[19]; }
      static bool MMX()   { return CPURepresentation.f_1_EDX[23]; }
      static bool FXSR()  { return CPURepresentation.f_1_EDX[24]; }
      static bool SSE()   { return CPURepresentation.f_1_EDX[25]; }
      static bool SSE2()  { return CPURepresentation.f_1_EDX[26]; }

      static bool FSGSBASE() { return CPURepresentation.f_7_EBX[0]; }
      static bool BMI1()     { return CPURepresentation.f_7_EBX[3]; }
      static bool HLE()      { return CPURepresentation.isIntel && CPURepresentation.f_7_EBX[4]; }
      static bool AVX2()     { return CPURepresentation.f_7_EBX[5]; }
      static bool BMI2()     { return CPURepresentation.f_7_EBX[8]; }
      static bool ERMS()     { return CPURepresentation.f_7_EBX[9]; }
      static bool INVPCID()  { return CPURepresentation.f_7_EBX[10]; }
      static bool RTM()      { return CPURepresentation.isIntel && CPURepresentation.f_7_EBX[11]; }
      static bool AVX512F()  { return CPURepresentation.f_7_EBX[16]; }
      static bool RDSEED()   { return CPURepresentation.f_7_EBX[18]; }
      static bool ADX()      { return CPURepresentation.f_7_EBX[19]; }
      static bool AVX512PF() { return CPURepresentation.f_7_EBX[26]; }
      static bool AVX512ER() { return CPURepresentation.f_7_EBX[27]; }
      static bool AVX512CD() { return CPURepresentation.f_7_EBX[28]; }
      static bool SHA()      { return CPURepresentation.f_7_EBX[29]; }

      static bool PREFETCHWT1() { return CPURepresentation.f_7_ECX[0]; }

      static bool LAHF()  { return CPURepresentation.f_81_ECX[0]; }
      static bool LZCNT() { return CPURepresentation.isIntel && CPURepresentation.f_81_ECX[5]; }
      static bool ABM()   { return CPURepresentation.isAMD   && CPURepresentation.f_81_ECX[5]; }
      static bool SSE4a() { return CPURepresentation.isAMD   && CPURepresentation.f_81_ECX[6]; }
      static bool XOP()   { return CPURepresentation.isAMD   && CPURepresentation.f_81_ECX[11]; }
      static bool TBM()   { return CPURepresentation.isAMD   && CPURepresentation.f_81_ECX[21]; }

      static bool SYSCALL()   { return CPURepresentation.isIntel && CPURepresentation.f_81_EDX[11]; }
      static bool MMXEXT()    { return CPURepresentation.isAMD   && CPURepresentation.f_81_EDX[22]; }
      static bool RDTSCP()    { return CPURepresentation.isIntel && CPURepresentation.f_81_EDX[27]; }
      static bool _3DNOWEXT() { return CPURepresentation.isAMD   && CPURepresentation.f_81_EDX[30]; }
      static bool _3DNOW()    { return CPURepresentation.isAMD   && CPURepresentation.f_81_EDX[31]; }

    private:
      static const InstructionSetInternal CPURepresentation;

      struct InstructionSetInternal
      {
        InstructionSetInternal()
        {
          std::array<int, 4> cpui{ { 0 } }; // double braces because we initialize an array inside an std::array object

          // calling __cpuid with 0x0 as the function_id argument
          // gets the number of the highest valid function ID.
          __cpuid(cpui.data(), 0);
          nIds = cpui[0];

          for (int i = 0; i <= nIds; ++i)
          {
            __cpuidex(cpui.data(), i, 0);
            data.push_back(cpui);
          }

          // capture vendor string
          vendor = std::string(32, ' ');
          *reinterpret_cast<int*>(&vendor[0]    ) = data[0][1];
          *reinterpret_cast<int*>(&vendor[0] + 4) = data[0][3];
          *reinterpret_cast<int*>(&vendor[0] + 8) = data[0][2];
          vendor = std::string(vendor.c_str()); // forced conversion from a possible null-terminated C-string above

          if (vendor == "GenuineIntel")
          {
            isIntel = true;
          }
          else if (vendor == "AuthenticAMD")
          {
            isAMD = true;
          }

          // load bitset with flags for function 0x00000001
          if (nIds >= 1)
          {
            f_1_ECX = data[1][2];
            f_1_EDX = data[1][3];
          }

          // load bitset with flags for function 0x00000007
          if (nIds >= 7)
          {
            f_7_EBX = data[7][1];
            f_7_ECX = data[7][2];
          }

          // calling __cpuid with 0x80000000 as the function_id argument
          // gets the number of the highest valid extended ID.
          __cpuid(cpui.data(), 0x80000000);
          nExIds = cpui[0];

          for (int i = 0x80000000; i <= nExIds; ++i)
          {
            __cpuidex(cpui.data(), i, 0);
            extdata.push_back(cpui);
          }

          // load bitset with flags for function 0x80000001
          if (nExIds >= int(0x80000001))
          {
            f_81_ECX = extdata[1][2];
            f_81_EDX = extdata[1][3];
          }

          // capture brand string
          brand = std::string(64, ' ');
          // interpret CPU brand string if reported
          if (nExIds >= int(0x80000004))
          {
            memcpy(&brand[0],      extdata[2].data(), sizeof(cpui));
            memcpy(&brand[0] + 16, extdata[3].data(), sizeof(cpui));
            memcpy(&brand[0] + 32, extdata[4].data(), sizeof(cpui));
            brand = std::string(brand.c_str()); // forced conversion from a possible null-terminated C-string above
          }
        }

        std::string vendor;
        std::string brand;
        bool isIntel = false;
        bool isAMD   = false;
        int nIds     = 0;
        int nExIds   = 0;
        std::bitset<32> f_1_ECX  = 0;
        std::bitset<32> f_1_EDX  = 0;
        std::bitset<32> f_7_EBX  = 0;
        std::bitset<32> f_7_ECX  = 0;
        std::bitset<32> f_81_ECX = 0;
        std::bitset<32> f_81_EDX = 0;
        std::vector<std::array<int, 4>> data;
        std::vector<std::array<int, 4>> extdata;
      };
    };

    // initialize static member data
    const InstructionSet::InstructionSetInternal InstructionSet::CPURepresentation;
  #endif // _WIN32

    /** @brief The not_vec4 class is an internal class: not be used directly.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class not_vec4 final
    {
      private:

        __m128 v_; // bitwise inverse of our value (!!)

      public:

        not_vec4(__m128 value) { v_ = value; }
        __m128 get() const { return v_; } // returns INVERSE of our value (!!)
    };

    /** @brief The vec4 class is the main SIMD float4 class using the GLSL nomenclature.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class vec4
    {
      private:

        __m128 v_;

      public:

        vec4(__m128 value) { v_ = value; }
        vec4(const float* __restrict src) { v_ = _mm_load_ps(src); }
        explicit vec4(float x) { v_ = _mm_set1_ps(x); }
        vec4(const vec4&) = default; // default copy-constructor
        vec4& operator=(const vec4&) = default; // default assignment operator

        vec4 operator+ (const vec4& rhs) const { return _mm_add_ps(v_, rhs.v_); }
        vec4 operator- (const vec4& rhs) const { return _mm_sub_ps(v_, rhs.v_); }
        vec4 operator* (const vec4& rhs) const { return _mm_mul_ps(v_, rhs.v_); }
        vec4 operator/ (const vec4& rhs) const { return _mm_div_ps(v_, rhs.v_); }
        vec4 operator& (const vec4& rhs) const { return _mm_and_ps(v_, rhs.v_); }
        vec4 operator| (const vec4& rhs) const { return _mm_or_ps(v_, rhs.v_); }
        vec4 operator^ (const vec4& rhs) const { return _mm_xor_ps(v_, rhs.v_); }
        vec4 operator==(const vec4& rhs) const { return _mm_cmpeq_ps(v_, rhs.v_); }
        vec4 operator!=(const vec4& rhs) const { return _mm_cmpneq_ps(v_, rhs.v_); }
        vec4 operator< (const vec4& rhs) const { return _mm_cmplt_ps(v_, rhs.v_); }
        vec4 operator<=(const vec4& rhs) const { return _mm_cmple_ps(v_, rhs.v_); }
        vec4 operator> (const vec4& rhs) const { return _mm_cmpgt_ps(v_, rhs.v_); }
        vec4 operator>=(const vec4& rhs) const { return _mm_cmpge_ps(v_, rhs.v_); }
        float &operator[](int index) { return reinterpret_cast<float*>(&v_)[index]; }
        float  operator[](int index) const { return reinterpret_cast<const float* __restrict>(&v_)[index]; }

        not_vec4 operator~() const { return not_vec4(v_); }

        bool if_any_not_true() const { return (_mm_movemask_ps(v_) != 0xf); }

        __m128 get() const { return v_; }

        float* store(float* __restrict ptr) const
        {
          _mm_store_ps(ptr, v_);
          return ptr;
        }

        float* store_unaligned(float* __restrict ptr) const
        {
          _mm_storeu_ps(ptr, v_);
          return ptr;
        }

        friend std::ostream& operator<<(std::ostream& o, const vec4& y)
        {
          return (o << y[0] << " " << y[1] << " " << y[2] << " " << y[3]);
        }

        friend vec4 operator&(const vec4& lhs, const not_vec4& rhs)
        {
          return _mm_andnot_ps(rhs.get(), lhs.get());
        }

        friend vec4 operator&(const not_vec4& lhs, const vec4& rhs)
        {
          return _mm_andnot_ps(lhs.get(), rhs.get());
        }

        vec4 if_then_else(const vec4& then, const vec4& else_part) const
        {
          return _mm_or_ps(_mm_and_ps(v_, then.v_), _mm_andnot_ps(v_, else_part.v_));
        }
    };

    /** @brief The vec4_unaligned class is the main unaligned SIMD float4 class using the GLSL nomenclature.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class vec4_unaligned final : public vec4
    {
      public:

        vec4_unaligned(const float* __restrict src) : vec4(_mm_loadu_ps(src)) {}
    };

    inline vec4 sqrt(const vec4& v)
    {
      return _mm_sqrt_ps(v.get());
    }

    inline vec4 rsqrt(const vec4& v)
    {
      return _mm_rsqrt_ps(v.get());
    }

    /** @brief Return value = dot product of a & b, replicated 4 times.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline vec4 dot(const vec4& a, const vec4& b)
    {
      vec4 t = a * b;
      __m128 vt = _mm_hadd_ps(t.get(), t.get());
      return _mm_hadd_ps(vt, vt);
    }

  #ifdef _WIN32
    #pragma warning (push)
    #pragma warning (disable : 4752) // for AVX intrinsics
  #endif // _WIN32
    /** @brief The not_vec8 class is an internal class: not be used directly.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class not_vec8 final
    {

      private:

        __m256 v_; // bitwise inverse of our value (!!)

      public:

        not_vec8(__m256 value) { v_ = value; }
        __m256 get() const { return v_; } // returns INVERSE of our value (!!)
    };

    /** @brief The vec8 class is the main SIMD float8 class using the GLSL nomenclature.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class vec8
    {
      private:

        __m256 v_;

      public:

        vec8(__m256 value) { v_ = value; }
        vec8(const float* __restrict src) { v_ = _mm256_load_ps(src); }
        explicit vec8(float x) { v_ = _mm256_broadcast_ss(&x); }
        vec8(const vec8&) = default; // default copy-constructor
        vec8& operator=(const vec8&) = default; // default assignment operator

        vec8 operator+ (const vec8 &rhs) const { return _mm256_add_ps(v_, rhs.v_); }
        vec8 operator- (const vec8 &rhs) const { return _mm256_sub_ps(v_, rhs.v_); }
        vec8 operator* (const vec8 &rhs) const { return _mm256_mul_ps(v_, rhs.v_); }
        vec8 operator/ (const vec8 &rhs) const { return _mm256_div_ps(v_, rhs.v_); }
        vec8 operator& (const vec8 &rhs) const { return _mm256_and_ps(v_, rhs.v_); }
        vec8 operator| (const vec8 &rhs) const { return _mm256_or_ps(v_, rhs.v_); }
        vec8 operator^ (const vec8 &rhs) const { return _mm256_xor_ps(v_, rhs.v_); }
        vec8 operator==(const vec8 &rhs) const { return _mm256_cmp_ps(v_, rhs.v_, _CMP_EQ_OQ); }
        vec8 operator!=(const vec8 &rhs) const { return _mm256_cmp_ps(v_, rhs.v_, _CMP_NEQ_OQ); }
        vec8 operator< (const vec8 &rhs) const { return _mm256_cmp_ps(v_, rhs.v_, _CMP_LT_OQ); }
        vec8 operator<=(const vec8 &rhs) const { return _mm256_cmp_ps(v_, rhs.v_, _CMP_LE_OQ); }
        vec8 operator> (const vec8 &rhs) const { return _mm256_cmp_ps(v_, rhs.v_, _CMP_GT_OQ); }
        vec8 operator>=(const vec8 &rhs) const { return _mm256_cmp_ps(v_, rhs.v_, _CMP_GT_OQ); }
        float &operator[](int index) { return reinterpret_cast<float*>(&v_)[index]; }
        float  operator[](int index) const { return reinterpret_cast<const float* __restrict>(&v_)[index]; }

        not_vec8 operator~() const { return not_vec8(v_); }

        bool if_any_not_true() const { return (_mm256_movemask_ps(v_) != 0xff); }

        __m256 get() const { return v_; }

        float* store(float* __restrict ptr) const
        {
          _mm256_store_ps(ptr, v_);
          return ptr;
        }

        float* store_unaligned(float* __restrict ptr) const
        {
          _mm256_storeu_ps(ptr, v_);
          return ptr;
        }

        friend std::ostream& operator<<(std::ostream& o, const vec8& y)
        {
          return (o << y[0] << " " << y[1] << " " << y[2] << " " << y[3] << " " << y[4] << " " << y[5] << " " << y[6] << " " << y[7]);
        }

        friend vec8 operator&(const vec8& lhs, const not_vec8& rhs)
        {
          return _mm256_andnot_ps(rhs.get(), lhs.get());
        }

        friend vec8 operator&(const not_vec8& lhs, const vec8& rhs)
        {
          return _mm256_andnot_ps(lhs.get(), rhs.get());
        }

        vec8 if_then_else(const vec8& then, const vec8& else_part) const
        {
          return _mm256_or_ps(_mm256_and_ps(v_, then.v_), _mm256_andnot_ps(v_, else_part.v_));
        }
    };

    /** @brief The vec8_unaligned class is the main unaligned SIMD float8 class using the GLSL nomenclature.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    class vec8_unaligned final : public vec8
    {

      public:

        vec8_unaligned(const float* __restrict src) :vec8(_mm256_loadu_ps(src)) {}


        float* store(float* __restrict ptr) const
        {
          _mm256_storeu_ps(ptr, vec8::get());
          return ptr;
        }
    };

    inline vec8 sqrt(const vec8& v)
    {
      return _mm256_sqrt_ps(v.get());
    }

    inline vec8 rsqrt(const vec8& v)
    {
      return _mm256_rsqrt_ps(v.get());
    }

    /** @brief Return value = dot product of a & b, replicated 8 times.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline vec8 dot(const vec8& a, const vec8& b)
    {
      vec8 t = a * b;
      __m256 vt = _mm256_hadd_ps(t.get(), t.get());
      return _mm256_hadd_ps(vt, vt);
    }

  #ifdef _WIN32
    #pragma warning (pop)
  #endif // _WIN32

  #ifdef _WIN32
    /** @brief Function to report the CPU capabilities in detail.
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline void reportCPUCapabilities()
    {
      std::ostringstream ss;
      ss << '\n';
      ss << "   --- Report CPU device Capabilities ---" << '\n';
      ss << "Vendor: " << Utils::UtilityFunctions::StringAuxiliaryFunctions::trim(InstructionSet::Vendor()) << '\n';
      ss << "Brand:  " << Utils::UtilityFunctions::StringAuxiliaryFunctions::trim(InstructionSet::Brand())  << '\n';
      ss << '\n';

      const auto& supportMessage = [&ss](std::string isAFeature, bool isSupported)
      {
        ss << isAFeature << Utils::UtilityFunctions::StringAuxiliaryFunctions::toString<bool>(isSupported) << '\n';
      };

      ss << '\n';
      ss << "   --- General Information for CPU device ---" << '\n';
      supportMessage("3DNOW:       ", InstructionSet::_3DNOW());
      supportMessage("3DNOWEXT:    ", InstructionSet::_3DNOWEXT());
      supportMessage("ABM:         ", InstructionSet::ABM());
      supportMessage("ADX:         ", InstructionSet::ADX());
      supportMessage("AES:         ", InstructionSet::AES());
      supportMessage("AVX:         ", InstructionSet::AVX());
      supportMessage("AVX2:        ", InstructionSet::AVX2());
      supportMessage("AVX512CD:    ", InstructionSet::AVX512CD());
      supportMessage("AVX512ER:    ", InstructionSet::AVX512ER());
      supportMessage("AVX512F:     ", InstructionSet::AVX512F());
      supportMessage("AVX512PF:    ", InstructionSet::AVX512PF());
      supportMessage("BMI1:        ", InstructionSet::BMI1());
      supportMessage("BMI2:        ", InstructionSet::BMI2());
      supportMessage("CLFSH:       ", InstructionSet::CLFSH());
      supportMessage("CMPXCHG16B:  ", InstructionSet::CMPXCHG16B());
      supportMessage("CX8:         ", InstructionSet::CX8());
      supportMessage("ERMS:        ", InstructionSet::ERMS());
      supportMessage("F16C:        ", InstructionSet::F16C());
      supportMessage("FMA:         ", InstructionSet::FMA());
      supportMessage("FSGSBASE:    ", InstructionSet::FSGSBASE());
      supportMessage("FXSR:        ", InstructionSet::FXSR());
      supportMessage("HLE:         ", InstructionSet::HLE());
      supportMessage("INVPCID:     ", InstructionSet::INVPCID());
      supportMessage("LAHF:        ", InstructionSet::LAHF());
      supportMessage("LZCNT:       ", InstructionSet::LZCNT());
      supportMessage("MMX:         ", InstructionSet::MMX());
      supportMessage("MMXEXT:      ", InstructionSet::MMXEXT());
      supportMessage("MONITOR:     ", InstructionSet::MONITOR());
      supportMessage("MOVBE:       ", InstructionSet::MOVBE());
      supportMessage("MSR:         ", InstructionSet::MSR());
      supportMessage("OSXSAVE:     ", InstructionSet::OSXSAVE());
      supportMessage("PCLMULQDQ:   ", InstructionSet::PCLMULQDQ());
      supportMessage("POPCNT:      ", InstructionSet::POPCNT());
      supportMessage("PREFETCHWT1: ", InstructionSet::PREFETCHWT1());
      supportMessage("RDRAND:      ", InstructionSet::RDRAND());
      supportMessage("RDSEED:      ", InstructionSet::RDSEED());
      supportMessage("RDTSCP:      ", InstructionSet::RDTSCP());
      supportMessage("RTM:         ", InstructionSet::RTM());
      supportMessage("SEP:         ", InstructionSet::SEP());
      supportMessage("SHA:         ", InstructionSet::SHA());
      supportMessage("SSE:         ", InstructionSet::SSE());
      supportMessage("SSE2:        ", InstructionSet::SSE2());
      supportMessage("SSE3:        ", InstructionSet::SSE3());
      supportMessage("SSE4.1:      ", InstructionSet::SSE41());
      supportMessage("SSE4.2:      ", InstructionSet::SSE42());
      supportMessage("SSE4a:       ", InstructionSet::SSE4a());
      supportMessage("SSSE3:       ", InstructionSet::SSSE3());
      supportMessage("SYSCALL:     ", InstructionSet::SYSCALL());
      supportMessage("TBM:         ", InstructionSet::TBM());
      supportMessage("XOP:         ", InstructionSet::XOP());
      supportMessage("XSAVE:       ", InstructionSet::XSAVE());
      ss << '\n';
      DebugConsole_consoleOutLine(ss.str());
    }
  #endif // _WIN32

    /** @brief Function to test for SSE3 support (x86 architecture).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline bool isSupportedSSE3()
    {
    #ifdef _WIN32
      return InstructionSet::SSE3();
    #else
      #ifdef __SSE3__
        return true;
      #else
        return false;
      #endif // __SSE3__
    #endif // _WIN32
    }

    /** @brief Function to test for SSE41 support (x86 architecture).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline bool isSupportedSSE41()
    {
    #ifdef _WIN32
      return InstructionSet::SSE41();
    #else
      #ifdef __SSE4_1__
        return true;
      #else
        return false;
      #endif // __SSE4_1__
    #endif // _WIN32
    }

    /** @brief Function to test for SSE42 support (x86 architecture).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline bool isSupportedSSE42()
    {
    #ifdef _WIN32
      return InstructionSet::SSE42();
    #else
      #ifdef __SSE4_2__
        return true;
      #else
        return false;
      #endif // __SSE4_2__
    #endif // _WIN32
    }

    /** @brief Function to test for AVX support (x86 architecture).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline bool isSupportedAVX()
    {
    #ifdef _WIN32
      return InstructionSet::AVX();
    #else
      #ifdef __AVX__
        return true;
      #else
        return false;
      #endif // __AVX__
    #endif // _WIN32
    }

    /** @brief Function to test for AVX2 support (x86 architecture).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline bool isSupportedAVX2()
    {
    #ifdef _WIN32
      return InstructionSet::AVX2();
    #else
      #ifdef __AVX2__
        return true;
      #else
        return false;
      #endif // __AVX2__
    #endif // _WIN32
    }

    /** @brief Function to test for AVX512F support (x86 architecture).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline bool isSupportedAVX512F()
    {
    #ifdef _WIN32
      return InstructionSet::AVX512F();
    #else
      #ifdef __AVX512F__
        return true;
      #else
        return false;
      #endif // __AVX512F__
    #endif // _WIN32
    }
#endif // NOT __aarch64__

    /** @brief Function to test for NEON support (ARM NEON SIMD architecture).
    * @author Thanos Theo, 2009-2018
    * @version 14.0.0.0
    */
    inline bool isSupportedNEON()
    {
      // advanced SIMD (aka NEON) is mandatory for aarch64 (make sure to use the compiler flag '-ftree-vectorize')
    #ifdef __aarch64__
      return true;
    #else
      return false;
    #endif // __aarch64__
    }

#ifndef __aarch64__
    inline void memcpy_GL_matrices_SSE(float* __restrict destination, const float* __restrict source)
    {
      vec4(source     ).store(destination     );
      vec4(source +  4).store(destination +  4);
      vec4(source +  8).store(destination +  8);
      vec4(source + 12).store(destination + 12);
    }

    inline void memcpy_unaligned_GL_matrices_SSE(float* __restrict destination, const float* __restrict source)
    {
      vec4(source     ).store_unaligned(destination     );
      vec4(source +  4).store_unaligned(destination +  4);
      vec4(source +  8).store_unaligned(destination +  8);
      vec4(source + 12).store_unaligned(destination + 12);
    }

    inline void memcpy_GL_matrices_AVX(float* __restrict destination, const float* __restrict source)
    {
      vec8(source    ).store(destination    );
      vec8(source + 8).store(destination + 8);
    }

    inline void memcpy_unaligned_GL_matrices_AVX(float* __restrict destination, const float* __restrict source)
    {
      vec8(source    ).store_unaligned(destination    );
      vec8(source + 8).store_unaligned(destination + 8);
    }

    inline std::array<float, 16> convert_to_float_GL_matrix_SSE(const double* __restrict source)
    {
      std::array<float, 16> destination{ { 0.0f } }; // double braces because we initialize an array inside an std::array object
      vec4(_mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(source     )), _mm_cvtpd_ps(_mm_loadu_pd(source +  2)))).store(destination.data()     );
      vec4(_mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(source +  4)), _mm_cvtpd_ps(_mm_loadu_pd(source +  6)))).store(destination.data() +  4);
      vec4(_mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(source +  8)), _mm_cvtpd_ps(_mm_loadu_pd(source + 10)))).store(destination.data() +  8);
      vec4(_mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(source + 12)), _mm_cvtpd_ps(_mm_loadu_pd(source + 14)))).store(destination.data() + 12);
      return destination;
    }
#endif // NOT __aarch64__
  } // namespace SIMDVectorizations
} // namespace Utils

#endif // __SIMDVectorizations_h