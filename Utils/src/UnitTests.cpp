#include "UnitTests.h"
#include "UtilityFunctions.h"

using namespace std;
using namespace Utils::UnitTests;
using namespace Utils::UtilityFunctions;

template <typename T>
void UnitTestUtilityFunctions<T>::parseComplexArrayFromTextRowMajor(const list<string>& dataLines, complex<T>* __restrict complexArray, uint32_t dataSize)
{
  uint32_t i = 0;
  for (const auto& line : dataLines)
  {
    if (!line.empty()) // skip empty lines
    {
      // default delimiter of tokenize is space
      vector<string> results = StringAuxiliaryFunctions::tokenize<vector<string>>(line);
      if (i == 0)
      {
        for (uint32_t j = 0; j < dataSize; ++j)
        {
          complexArray[j].real(complexArray[j].real() + StringAuxiliaryFunctions::fromString<float>(results[j]));
        }
      }
      else if (i == 1)
      {
        for (uint32_t j = 0; j < dataSize; ++j)
        {
          complexArray[j].imag(complexArray[j].imag() + StringAuxiliaryFunctions::fromString<float>(results[j]));
        }
      }
      ++i;
    }
  }
}

template <typename T>
void UnitTestUtilityFunctions<T>::parseComplexArrayFromTextColumnMajor(const list<string>& dataLines, complex<T>* __restrict complexArray)
{
  uint32_t i = 0;
  for (const auto& line : dataLines)
  {
    if (line.empty() || (line[0] == '#')) // skip empty or commented lines
    {
      continue;
    }

    // default delimiter of tokenize is space
    vector<string> results = StringAuxiliaryFunctions::tokenize<vector<string>>(line);

    complexArray[i].real(StringAuxiliaryFunctions::fromString<T>(results[0]));
    complexArray[i].imag(StringAuxiliaryFunctions::fromString<T>(results[1]));

    ++i;
  }
}

template void UnitTestUtilityFunctions<float >::parseComplexArrayFromTextRowMajor(   const list<string>& dataLines, complex<float >* __restrict complexArray, uint32_t dataSize); // explicit instantiation definition of function template
template void UnitTestUtilityFunctions<double>::parseComplexArrayFromTextRowMajor(   const list<string>& dataLines, complex<double>* __restrict complexArray, uint32_t dataSize); // explicit instantiation definition of function template
template void UnitTestUtilityFunctions<float >::parseComplexArrayFromTextColumnMajor(const list<string>& dataLines, complex<float >* __restrict complexArray); // explicit instantiation definition of function template
template void UnitTestUtilityFunctions<double>::parseComplexArrayFromTextColumnMajor(const list<string>& dataLines, complex<double>* __restrict complexArray); // explicit instantiation definition of function template