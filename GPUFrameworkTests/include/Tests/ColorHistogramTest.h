#pragma once

#include "AccurateTimers.h"
#include "CPUParallelism/ThreadPool.h"
#include <array>
#include <memory>
#include <vector>
#include <cstdint>

namespace Tests {

class ColorHistogramTest {
public:
    // Constants
    static constexpr size_t NUM_CHANNELS = 3;  // RGB
    static constexpr size_t NUM_BINS = 256;    // 8-bit per channel

    ColorHistogramTest() noexcept;
    
    // Main interface
    void initializeFromImage(const uint8_t* imageData, size_t width, size_t height);
    void computeSingleCore();  // Original single-core implementation for comparison
    void computeParallel();    // New parallel implementation
    void saveHistogramCSV(const char* filename) const;

    // Getters
    const std::array<std::array<uint32_t, NUM_BINS>, NUM_CHANNELS>& getHistogram() const { return histogram_; }
    double getTotalTime() const { return totalTimeTakenInMs_; }

private:
    // Image data
    size_t width_;
    size_t height_;
    std::unique_ptr<uint8_t[]> imageData_;

    // Histogram data
    std::array<std::array<uint32_t, NUM_BINS>, NUM_CHANNELS> histogram_;
    std::vector<std::array<std::array<uint32_t, NUM_BINS>, NUM_CHANNELS>> threadLocalHists_;

    // Timing variables - mutable to allow updates in const methods
    mutable Utils::AccurateTimers::AccurateCPUTimer timer_;
    mutable double totalTimeTakenInMs_;

    // Private merge implementations
    void mergeHistogramsAVX2(Utils::CPUParallelism::ThreadPool& threadPool);
    void mergeHistogramsSSE4(Utils::CPUParallelism::ThreadPool& threadPool);
    void mergeHistogramsScalar(Utils::CPUParallelism::ThreadPool& threadPool);
    void mergeHistograms(Utils::CPUParallelism::ThreadPool& threadPool);
};

} // namespace Tests