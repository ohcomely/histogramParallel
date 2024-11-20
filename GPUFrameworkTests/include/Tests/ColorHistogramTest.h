#pragma once

#include "AccurateTimers.h"
#include "CPUParallelism/CPUParallelismNCP.h"
#include <array>
#include <memory>

namespace Tests {

class ColorHistogramTest {
public:
    // Constants
    static constexpr size_t NUM_CHANNELS = 3;  // RGB
    static constexpr size_t NUM_BINS = 256;    // 8-bit per channel
    static constexpr size_t BLOCK_SIZE = 1024; // Cache-friendly block size

    ColorHistogramTest() noexcept;

    // Load image data 
    void initializeFromImage(const uint8_t* imageData, size_t width, size_t height);

    // Single-core computation
    void computeSingleCore();

    // Multi-core computation 
    void computeParallel(size_t numThreads);

    // Save results to CSV
    void saveHistogramCSV(const char* filename) const;

    // Getters
    const std::array<std::array<uint32_t, NUM_BINS>, NUM_CHANNELS>& getHistogram() const { 
        return histogram_; 
    }
    double getTotalTime() const { 
        return totalTimeTakenInMs_; 
    }

private:
    // Image data
    std::unique_ptr<uint8_t[]> imageData_;
    size_t width_;
    size_t height_;

    // Histogram results for each channel
    std::array<std::array<uint32_t, NUM_BINS>, NUM_CHANNELS> histogram_;

    // Timing
    mutable Utils::AccurateTimers::AccurateCPUTimer timer_;
    mutable double totalTimeTakenInMs_;
};

} // namespace Tests