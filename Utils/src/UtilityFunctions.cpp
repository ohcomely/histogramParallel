#include "UtilityFunctions.h"
#include <filesystem>
#include <limits>
#include <unordered_set>

using namespace std;
using namespace std::filesystem;
using namespace Utils::UtilityFunctions;

namespace // anonymous namespace used instead of deprecated 'static' keyword used for cpp variable locality
{
  const string ALL_BASE64_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const string TRIM_DELIMITER        = " \t\r\n\0";
  const string XOR_SWAP_KEY_PHRASE   = "GPU_FMWK_XOR_SWP"; // Note: has to be a power-of-2 string size for this to work with bitshift modulo
  const string EMPTY_STRING          = "";

  // used as static variables in local compilation unit for DebugConsole class to avoid exposing them thus causing export issues to other components/dlls
  bool   useLogFile  = false;
  string logFileName = "Logfile.log";

  inline bool isBase64(char c)
  {
    return (::isalnum(c) || (c == '+') || (c == '/'));
  }
}

string Base64CompressorScrambler::encodeBase64String(const string& str)
{
  string::size_type stringLength = str.size();
  string encodedString;
  array<char, 3> charArray3{ { ' ' } }; // double braces because we initialize an array inside an std::array object
  array<char, 4> charArray4{ { ' ' } }; // double braces because we initialize an array inside an std::array object
  int i = 0;
  int j = 0;
  int index = 0;

  while (stringLength--)
  {
    charArray3[i++] = str[index++];
    if (i == 3)
    {
      charArray4[0] =  char((charArray3[0] & 0xfc) >> 2);
      charArray4[1] = char(((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4));
      charArray4[2] = char(((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6));
      charArray4[3] =   char(charArray3[2] & 0x3f);

      for (i = 0; i < 4; ++i)
      {
        encodedString += ALL_BASE64_CHARACTERS[charArray4[i]];
      }
      i = 0;
    }
  }

  if (i)
  {
    for (j = i; j < 3; ++j)
    {
      charArray3[j] = '\0';
    }

    charArray4[0] =  char((charArray3[0] & 0xfc) >> 2);
    charArray4[1] = char(((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4));
    charArray4[2] = char(((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6));
    charArray4[3] =   char(charArray3[2] & 0x3f);

    for (j = 0; j < i + 1; ++j)
    {
      encodedString += ALL_BASE64_CHARACTERS[charArray4[j]];
    }

    while (i++ < 3)
    {
      encodedString += '=';
    }
  }

  return encodedString;
}

string Base64CompressorScrambler::decodeBase64String(const string& str)
{
  string::size_type stringLength = str.size();
  string decodedString;
  array<char, 3> charArray3{ { ' ' } }; // double braces because we initialize an array inside an std::array object
  array<char, 4> charArray4{ { ' ' } }; // double braces because we initialize an array inside an std::array object
  int i = 0;
  int j = 0;
  int index = 0;

  while (stringLength-- && (str[index] != '=') && isBase64(str[index]))
  {
    charArray4[i++] = str[index++];
    if (i == 4)
    {
      for (i = 0; i < 4; ++i)
      {
        charArray4[i] = char(ALL_BASE64_CHARACTERS.find(charArray4[i]));
      }

      charArray3[0] =  char((charArray4[0] << 2)        + ((charArray4[1] & 0x30) >> 4));
      charArray3[1] = char(((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2));
      charArray3[2] = char(((charArray4[2] & 0x3) << 6) +   charArray4[3]);

      for (i = 0; i < 3; ++i)
      {
        decodedString += charArray3[i];
      }

      i = 0;
    }
  }

  if (i)
  {
    for (j = i; j < 4; ++j)
    {
      charArray4[j] = 0;
    }

    for (j = 0; j < 4; ++j)
    {
      charArray4[j] = char(ALL_BASE64_CHARACTERS.find(charArray4[j]));
    }

    charArray3[0] =  char((charArray4[0] << 2)        + ((charArray4[1] & 0x30) >> 4));
    charArray3[1] = char(((charArray4[1] & 0xf) << 4) + ((charArray4[2] & 0x3c) >> 2));
    charArray3[2] = char(((charArray4[2] & 0x3) << 6) +   charArray4[3]);

    for (j = 0; j < i - 1; ++j)
    {
      decodedString += charArray3[j];
    }
  }

  return decodedString;
}

string Base64CompressorScrambler::flipString(const string& line)
{
  string flippedLine = line;
  for (string::size_type i = 0; i < line.size(); ++i)
  {
    flippedLine[i] = ~line[i];
  }

  return flippedLine;
}

string Base64CompressorScrambler::xorSwapString(const string& line)
{
  string xorSwappedLine = line;
  const string::size_type XORSwapKeyPhraseSize = XOR_SWAP_KEY_PHRASE.size();
  for (string::size_type i = 0; i < line.size(); ++i)
  {
    xorSwappedLine[i] ^= XOR_SWAP_KEY_PHRASE[i & (XORSwapKeyPhraseSize - 1)];
  }

  return xorSwappedLine;
}

int BitManipulationFunctions::getLowestBitPositionOfPowerOfTwoNumber(int value)
{
  int r = (value & 0xAAAAAAAA) != 0;
  r |= ((value & 0xFFFF0000) != 0) << 4;
  r |= ((value & 0xFF00FF00) != 0) << 3;
  r |= ((value & 0xF0F0F0F0) != 0) << 2;
  r |= ((value & 0xCCCCCCCC) != 0) << 1;

  return r;
}

int BitManipulationFunctions::countTurnedOnBitsOfNumber(int value)
{
  int r = value - ((value >> 1) & 0x55555555);
  r = (r & 0x33333333) + ((r >> 2) & 0x33333333);
  r = (((r + (r >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;

  return r;
}

unsigned int BitManipulationFunctions::getPrevPowerOfTwo(unsigned int value)
{
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;

  return value - (value >> 1);
}

unsigned int BitManipulationFunctions::getNextPowerOfTwo(unsigned int value)
{
  --value;
  value |= value >> 1;
  value |= value >> 2;
  value |= value >> 4;
  value |= value >> 8;
  value |= value >> 16;
  ++value;

  return value;
}

bool StringAuxiliaryFunctions::startsWith(const string& str, const string& starting)
{
  if (starting.size() > str.size())
  {
    return false;
  }

  return equal(starting.begin(), starting.end(), str.begin());
}

bool StringAuxiliaryFunctions::endsWith(const string& str, const string& ending)
{
  if (ending.size() > str.size())
  {
    return false;
  }

  return equal(ending.rbegin(), ending.rend(), str.rbegin());
}

string StringAuxiliaryFunctions::trimLeft(const string& str)
{
  if (str.empty())
  {
    return str;
  }

  const string::size_type startPosition = str.find_first_not_of(TRIM_DELIMITER);

  return (startPosition == string::npos) ? EMPTY_STRING : str.substr(startPosition);
}

string StringAuxiliaryFunctions::trimRight(const string& str)
{
  if (str.empty())
  {
    return str;
  }

  const string::size_type endPosition = str.find_last_not_of(TRIM_DELIMITER);

  return (endPosition == string::npos) ? EMPTY_STRING : str.substr(0, endPosition + 1);
}

string StringAuxiliaryFunctions::trim(const string& str)
{
  if (str.empty())
  {
    return str;
  }

  const string::size_type b = str.find_first_not_of(TRIM_DELIMITER);
  const string::size_type e = str.find_last_not_of(TRIM_DELIMITER);

  return (b == string::npos) ? EMPTY_STRING : str.substr(b, e - b + 1);
}

string StringAuxiliaryFunctions::toUpperCase(const string& str)
{
  if (str.empty())
  {
    return str;
  }

  string copyString(str);
  for (char& character : copyString)
  {
    character = char(toupper(character));
  }

  return copyString;
}

string StringAuxiliaryFunctions::toLowerCase(const string& str)
{
  if (str.empty())
  {
    return str;
  }

  string copyString(str);
  for (char& character : copyString)
  {
    character = char(tolower(character));
  }

  return copyString;
}

string StringAuxiliaryFunctions::formatNumberString(size_t number, size_t totalNumbers)
{
  const size_t trailingZeros = toString(totalNumbers).length();
  vector<char> resultString(trailingZeros + 1, ' ');
  snprintf(resultString.data(), resultString.size(), string("%0" + toString(trailingZeros) + "d").c_str(), number);

  return string(resultString.data());
}

bool StdReadWriteFileFunctions::assure(const ios& stream, const string& fullpathWithFileName)
{
  if (stream)
  {
    return true;
  }

  DebugConsole_consoleOutLine("Could not open file:\n", fullpathWithFileName);

  return false;
}

bool StdReadWriteFileFunctions::assure(size_t numberOfElements, const string& fullpathWithFileName)
{
  if (numberOfElements > 0)
  {
    return true;
  }

  DebugConsole_consoleOutLine("File is empty: ", fullpathWithFileName);

  return false;
}

list<string> StdReadWriteFileFunctions::readTextFile(const string& fullpathWithFileName, bool trimString)
{
  list<string> lineString;
  ifstream in;
  in.open(fullpathWithFileName, ios::in);
  if (assure(in, fullpathWithFileName))
  {
    string line;
    while (getline(in, line)) // in.getline() removes '\n'
    {
      lineString.emplace_back(trimString ? StringAuxiliaryFunctions::trim(line) : move(line));
    }
  }
  in.close();

  return lineString;
}

void StdReadWriteFileFunctions::writeTextFile(const string& fullpathWithFileName, const string& textToWrite, ios_base::openmode mode)
{
  ofstream out;
  out.open(fullpathWithFileName, mode);
  if (assure(out, fullpathWithFileName))
  {
    out << textToWrite;
    out.flush();
  }
  out.close();
}

void StdReadWriteFileFunctions::writeTextFile(const string& fullpathWithFileName, const list<string>& textToWrite, ios_base::openmode mode)
{
  ofstream out;
  out.open(fullpathWithFileName, mode);
  if (assure(out, fullpathWithFileName))
  {
    for (const auto& text : textToWrite)
    {
      out << text << '\n';
    }
  }
  out.flush();
  out.close();
}

bool StdReadWriteFileFunctions::pathExists(const string& fullpath)
{
  try
  {
    const path filePath(fullpath);
    return exists(filePath);
  }
  catch (const filesystem_error& error)
  {
    DebugConsole_consoleOutLine("pathExists() error: ", error.what());
    return false;
  }
}

size_t StdReadWriteFileFunctions::getFileSize(const string& fullpathWithFileName)
{
  try
  {
    const path filePath(fullpathWithFileName);
    if (!is_regular_file(filePath))
    {
      DebugConsole_consoleOutLine("getFileSize() error: given input '", fullpathWithFileName, "' is not a regular file.");
      return 0;
    }

    return exists(filePath) ? size_t(file_size(filePath)) : 0;
  }
  catch (const filesystem_error& error)
  {
    DebugConsole_consoleOutLine("getFileSize() error: ", error.what());
    return 0;
  }
}

string StdReadWriteFileFunctions::getCurrentPath()
{
  try
  {
    string currentPath = current_path().string();
    // make sure to always use a Unix-friendly forward slash path, which works fine with STL on all platforms
    replace(currentPath.begin(), currentPath.end(), '\\', '/');
    return currentPath;
  }
  catch (const filesystem_error& error)
  {
    DebugConsole_consoleOutLine("getCurrentPath() error: ", error.what());
    return EMPTY_STRING;
  }
}

bool StdReadWriteFileFunctions::removeFile(const string& fullpathWithFileName)
{
  try
  {
    const path filePath(fullpathWithFileName);
    if (!is_regular_file(filePath))
    {
      DebugConsole_consoleOutLine("removeFile() error: given input '", fullpathWithFileName, "' is not a regular file.");
      return false;
    }

    return remove(filePath);
  }
  catch (const filesystem_error& error)
  {
    DebugConsole_consoleOutLine("removeFile() error: ", error.what());
    return false;
  }
}

bool StdReadWriteFileFunctions::removeAllFilesWithExtension(const string& fullpath, const string& fileExtension)
{
  try
  {
    const path filePath(fullpath);
    if (is_regular_file(filePath))
    {
      DebugConsole_consoleOutLine("removeAllFilesWithExtension() error: given input '", fullpath, "' is a regular file.");
      return false;
    }

    unordered_set<string> filesToRemove;
    for (auto& pathIterator : recursive_directory_iterator(fullpath))
    {
      const path& currentPath = pathIterator.path();
      if (!is_directory(currentPath)) // skip directory
      {
        const string currentFileName      = currentPath.filename().string();
        const string currentFileExtension = currentPath.extension().string();
        if (currentFileExtension == ("." + fileExtension))
        {
          filesToRemove.emplace(fullpath + currentFileName);
        }
      }
    }

    for (auto& fileToRemove : filesToRemove)
    {
      removeFile(fileToRemove);
    }

    return !filesToRemove.empty();
  }
  catch (const filesystem_error& error)
  {
    DebugConsole_consoleOutLine("removeAllFilesWithExtension() error: ", error.what());
    return false;
  }
}

bool StdReadWriteFileFunctions::createDirectory(const string& fullpath)
{
  try
  {
    const path filePath(fullpath);
    if (is_regular_file(filePath))
    {
      DebugConsole_consoleOutLine("createDirectory() error: given input '", fullpath, "' is a regular file.");
      return false;
    }

    return create_directory(filePath);
  }
  catch (const filesystem_error& error)
  {
    DebugConsole_consoleOutLine("createDirectory() error: ", error.what());
    return false;
  }
}

uintmax_t StdReadWriteFileFunctions::removeDirectory(const string& fullpath)
{
  try
  {
    const path filePath(fullpath);
    if (is_regular_file(filePath))
    {
      DebugConsole_consoleOutLine("removeDirectory() error: given input '", fullpath, "' is a regular file.");
      // error on remove_all() returns uintmax_t>(-1)->ie numeric_limits<uintmax_t>::max())
      return numeric_limits<uintmax_t>::max();
    }

    return remove_all(filePath);
  }
  catch (const filesystem_error& error)
  {
    DebugConsole_consoleOutLine("removeDirectory() error: ", error.what());
    // error on remove_all() returns uintmax_t>(-1)->ie numeric_limits<uintmax_t>::max())
    return numeric_limits<uintmax_t>::max();
  }
}

void DebugConsole::setLogFileName(const string& givenLogFileName)
{
  logFileName = givenLogFileName;
}

void DebugConsole::setUseLogFile(bool givenUseLogFile)
{
  useLogFile = givenUseLogFile;
}

string DebugConsole::getLogFileName()
{
  return logFileName;
}

bool DebugConsole::getUseLogFile()
{
  return useLogFile;
}

void DebugConsole::checkAndWriteLogFileImpl(const string& msg)
{
  if (useLogFile)
  {
    writeLogFileImpl(msg);
  }
}

void DebugConsole::writeLogFileImpl(const string& msg)
{
  StdReadWriteFileFunctions::writeTextFile(logFileName, msg, ios::app);
}