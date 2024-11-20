#include "Tests/ColorHistogramTest.h"
#include "UtilityFunctions.h"
#include <fstream>
#include <algorithm>

#include "CPUParallelism/ThreadPool.h"

using namespace std;
using namespace Tests;
using namespace Utils::CPUParallelism;
using namespace Utils::UtilityFunctions;

ColorHistogramTest::ColorHistogramTest() noexcept 
    : width_(0)
    , height_(0)
    , totalTimeTakenInMs_(0.0) {
    // Initialize histogram arrays to zero
    for (auto& channel : histogram_) {
        channel.fill(0);
    }
}

void ColorHistogramTest::initializeFromImage(const uint8_t* imageData, size_t width, size_t height) {
    timer_.startTimer();

    // Store dimensions
    width_ = width;
    height_ = height;

    // Copy image data
    const size_t numPixels = width * height;
    const size_t dataSize = numPixels * NUM_CHANNELS;
    imageData_ = make_unique<uint8_t[]>(dataSize);
    copy(imageData, imageData + dataSize, imageData_.get());

    timer_.stopTimer();
    totalTimeTakenInMs_ += timer_.getElapsedTimeInMilliSecs();
}

void ColorHistogramTest::computeSingleCore() {
    timer_.startTimer();

    // Reset histogram
    for (auto& channel : histogram_) {
        channel.fill(0);
    }

    // Process each pixel's RGB values
    const size_t numPixels = width_ * height_;
    for (size_t i = 0; i < numPixels; ++i) {
        const size_t baseIndex = i * NUM_CHANNELS;
        
        // Process each channel
        histogram_[0][imageData_[baseIndex + 0]]++; // R
        histogram_[1][imageData_[baseIndex + 1]]++; // G
        histogram_[2][imageData_[baseIndex + 2]]++; // B
    }

    timer_.stopTimer();
    totalTimeTakenInMs_ += timer_.getElapsedTimeInMilliSecs();
}

void ColorHistogramTest::computeParallel(size_t num) {
    timer_.startTimer();

    // Get number of threads for parallelization
    const size_t numThreads = (num==0)?numberOfHardwareThreads():num;
    const size_t numPixels = width_ * height_;

    // Create thread-local histograms
    vector<array<array<uint32_t, NUM_BINS>, NUM_CHANNELS>> threadLocalHists(numThreads);

    // Initialize thread-local histograms to zero
    for (auto& threadHist : threadLocalHists) {
        for (auto& channel : threadHist) {
            channel.fill(0);
        }
    }

    // Process pixels in parallel using thread-local histograms
    ThreadPool threadPool(numThreads);
    
    // Process blocks of pixels for better cache utilization
    parallelForThreadLocal(
        0, numPixels,
        [&](size_t i, size_t threadIdx) {
            auto& localHist = threadLocalHists[threadIdx];
            const size_t baseIndex = i * NUM_CHANNELS;

            // Update local histogram for each channel
            localHist[0][imageData_[baseIndex + 0]]++; // R
            localHist[1][imageData_[baseIndex + 1]]++; // G
            localHist[2][imageData_[baseIndex + 2]]++; // B
        },
        threadPool
    );

    // Reset final histogram
    for (auto& channel : histogram_) {
        channel.fill(0);
    }

    // Merge thread-local histograms into final histogram
    for (const auto& threadHist : threadLocalHists) {
        for (size_t c = 0; c < NUM_CHANNELS; ++c) {
            for (size_t bin = 0; bin < NUM_BINS; ++bin) {
                histogram_[c][bin] += threadHist[c][bin];
            }
        }
    }

    timer_.stopTimer();
    totalTimeTakenInMs_ += timer_.getElapsedTimeInMilliSecs();
}

void ColorHistogramTest::saveHistogramCSV(const char* filename) const {
    timer_.startTimer();

    // Open output file
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        DebugConsole_consoleOutLine("Error opening file for writing: ", filename);
        return;
    }

    // Write CSV header
    outFile << "Bin,Red,Green,Blue\n";

    // Write histogram data
    for (size_t bin = 0; bin < NUM_BINS; ++bin) {
        outFile << bin << ","
                << histogram_[0][bin] << "," // Red
                << histogram_[1][bin] << "," // Green  
                << histogram_[2][bin] << "\n"; // Blue
    }

    timer_.stopTimer();
    totalTimeTakenInMs_ += timer_.getElapsedTimeInMilliSecs();
}