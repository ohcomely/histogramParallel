#pragma once

#ifndef __UTILS_MODULEDLL_H_INCL__
#define __UTILS_MODULEDLL_H_INCL__

#ifdef __cplusplus
extern "C"
{
#endif

/**
*  ModuleDLL.h
*  Definitions for Module DLL
*/

#if defined _WIN32
  /** Definitions for exporting or importing the Module DLL
  * In MSVC projects, DLL projects should add BUILD_MODULE_DLL to Preprocessor definitions.
  * The default is MODULE_EXPORTS so that should be replaced.
  * For projects linking with this dll, they should add LINK_DLL to Preprocessor definitions.
  * This includes DLL projects which link with other DLL projects.
  */
  #if defined Utils_EXPORTS
    #define UTILS_MODULE_API __declspec(dllexport)
    #pragma warning (push)
    #pragma warning (disable : 4251) // for std member dll export
    #pragma warning (push)
    #pragma warning (disable : 4275) // for dll-interface struct/class export
  #elif defined LINK_DLL
    #define UTILS_MODULE_API __declspec(dllimport)
  #else
    #define UTILS_MODULE_API
  #endif
  /** Definitions for the Curiously Recurring Template Pattern (CRTP)
  * In MSVC projects and in NVCC mode, DLL projects should export the CRTP interface.
  */
  #if defined __CUDACC__
    #define CRTP_MODULE_API __declspec(dllexport)
  #else
    #define CRTP_MODULE_API
  #endif
#else /* non-Windows OSs don't need all this */
  #define UTILS_MODULE_API
  #define CRTP_MODULE_API
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  /* __UTILS_MODULEDLL_H_INCL__ */