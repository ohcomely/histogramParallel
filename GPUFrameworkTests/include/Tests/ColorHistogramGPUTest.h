#pragma once

#ifndef __ColorHistogramGPUTest_h
#define __ColorHistogramGPUTest_h

#include "CUDAGPUComputingAbstraction.h"
#include "CUDAMemoryHandlers.h"
#include "CUDAProcessMemoryPool.h"
#include "CUDAStreamsHandler.h"
#include "CUDAEventTimer.h"
#include "AccurateTimers.h"
#include <cstdint>
#include <array>

namespace UtilsCUDA {

class ColorHistogramGPUTest final : private CUDAGPUComputingAbstraction<ColorHistogramGPUTest> {
public:
    static constexpr size_t NUM_CHANNELS = 3; // RGB
    static constexpr size_t NUM_BINS = 256;  // 8-bit per channel
    
    void initializeGPUMemory();
    void performGPUComputing();
    void retrieveGPUResults();
    bool verifyComputingResults();
    void releaseGPUComputingResources();

    const std::array<std::array<uint32_t, NUM_BINS>, NUM_CHANNELS>& getHistogram() const { 
        return histogram_;
    }

    double getTotalTime() const { return totalTimeTakenInMs_; }

    ColorHistogramGPUTest(const CUDADriverInfo& cudaDriverInfo, int device = 0);
    ~ColorHistogramGPUTest() = default;
    ColorHistogramGPUTest(const ColorHistogramGPUTest&) = delete;
    ColorHistogramGPUTest(ColorHistogramGPUTest&&) = delete;
    ColorHistogramGPUTest& operator=(const ColorHistogramGPUTest&) = delete;
    ColorHistogramGPUTest& operator=(ColorHistogramGPUTest&&) = delete;

    void initializeFromImage(const uint8_t* imageData, size_t width, size_t height);
    void compute();

private:
    size_t width_ = 0;
    size_t height_ = 0;
    double totalTimeTakenInMs_ = 0.0;

    // Final histogram result
    std::array<std::array<uint32_t, NUM_BINS>, NUM_CHANNELS> histogram_;

    // GPU Memory handlers
    DeviceMemory<uint8_t> imageData_;  // Input image data
    DeviceMemory<uint32_t> histogramRed_;   // Histogram for red channel
    DeviceMemory<uint32_t> histogramGreen_; // Histogram for green channel 
    DeviceMemory<uint32_t> histogramBlue_;  // Histogram for blue channel

    CUDAProcessMemoryPool cudaProcessMemoryPool_;
    const CUDAStreamsHandler cudaStreamsHandler_;
    CUDAEventTimer gpuTimer_;
};

} // namespace UtilsCUDA

#endif // __ColorHistogramGPUTest_h