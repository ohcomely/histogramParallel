#include "Tests/ColorHistogramTest.h"
#include "SIMDVectorizations.h"
#include "UtilityFunctions.h"
#include <fstream>

#ifdef __aarch64__
    #include <arm_neon.h>
#else
    #include <immintrin.h>
#endif

using namespace std;
using namespace Tests;
using namespace Utils::CPUParallelism;
using namespace Utils::UtilityFunctions;
using namespace Utils::SIMDVectorizations;

ColorHistogramTest::ColorHistogramTest() noexcept
    : width_(0)
    , height_(0)
    , totalTimeTakenInMs_(0.0)
{
    // Initialize histogram arrays to zero
    for (auto& channel : histogram_) {
        channel.fill(0);
    }
}

void ColorHistogramTest::initializeFromImage(const uint8_t* imageData, size_t width, size_t height) 
{
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

void ColorHistogramTest::computeSingleCore() 
{
    timer_.startTimer();

    // Reset histogram
    for (auto& channel : histogram_) {
        channel.fill(0);
    }

    // Process each pixel
    const size_t numPixels = width_ * height_;
    for (size_t i = 0; i < numPixels; ++i) {
        const size_t baseIndex = i * NUM_CHANNELS;
        // Increment counts for each RGB channel
        for (size_t channel = 0; channel < NUM_CHANNELS; ++channel) {
            const uint8_t value = imageData_[baseIndex + channel];
            histogram_[channel][value]++;
        }
    }

    timer_.stopTimer();
    totalTimeTakenInMs_ += timer_.getElapsedTimeInMilliSecs(); 
}

void ColorHistogramTest::computeParallel()
{
    timer_.startTimer();

    // Initialize thread-local histograms
    // const size_t numThreads = numberOfHardwareThreads();
    const size_t numThreads = 8;
    DebugConsole_consoleOutLine("Number of hardware threads: ", numThreads);
    threadLocalHists_.resize(numThreads);
    for (auto& threadHist : threadLocalHists_) {
        for (auto& channel : threadHist) {
            channel.fill(0);
        }
    }

    // Create thread pool
    ThreadPool threadPool(numThreads);

    // Parallel histogram computation with thread-local storage
    const size_t numPixels = width_ * height_;
    parallelForThreadLocal(0, numPixels, [&](size_t i, size_t threadIdx) {
        const size_t baseIndex = i * NUM_CHANNELS;
        threadLocalHists_[threadIdx][0][imageData_[baseIndex + 0]]++;  // R
        threadLocalHists_[threadIdx][1][imageData_[baseIndex + 1]]++;  // G
        threadLocalHists_[threadIdx][2][imageData_[baseIndex + 2]]++;  // B
    }, threadPool);

    // Reset final histogram before merging
    for (auto& channel : histogram_) {
        channel.fill(0);
    }

    // Merge results using the best available method
    mergeHistograms(threadPool);

    timer_.stopTimer();
    totalTimeTakenInMs_ += timer_.getElapsedTimeInMilliSecs();
}

void ColorHistogramTest::mergeHistograms(ThreadPool& threadPool) 
{
#ifdef __aarch64__
    // On ARM, use NEON implementation
    mergeHistogramsNEON(threadPool);
#else
    // On x86, check for available SIMD support
    if (isSupportedAVX2()) {
        DebugConsole_consoleOutLine("Using AVX2 for histogram merging");
        mergeHistogramsAVX2(threadPool);
    }
    else if (isSupportedSSE41()) {
        mergeHistogramsSSE4(threadPool);
    }
    else {
        mergeHistogramsScalar(threadPool);
    }
#endif
}

void ColorHistogramTest::mergeHistogramsAVX2(ThreadPool& threadPool)
{
    const size_t numThreads = threadLocalHists_.size();
    
    // Process each channel in parallel
    parallelFor(0, NUM_CHANNELS, [&](size_t channel) {
        // Use AVX2 to merge bins within each channel
        for (size_t bin = 0; bin < NUM_BINS; bin += 8) {
            __m256i sum = _mm256_setzero_si256();
            // Sum across thread-local histograms
            for (size_t t = 0; t < numThreads; t++) {
                __m256i vals = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(&threadLocalHists_[t][channel][bin]));
                sum = _mm256_add_epi32(sum, vals);
            }
            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(&histogram_[channel][bin]), sum);
        }
    }, threadPool);
}

void ColorHistogramTest::mergeHistogramsSSE4(ThreadPool& threadPool)
{
    const size_t numThreads = threadLocalHists_.size();
    
    // Process each channel in parallel
    parallelFor(0, NUM_CHANNELS, [&](size_t channel) {
        // Use SSE4 to merge bins within each channel
        for (size_t bin = 0; bin < NUM_BINS; bin += 4) {
            __m128i sum = _mm_setzero_si128();
            // Sum across thread-local histograms
            for (size_t t = 0; t < numThreads; t++) {
                __m128i vals = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(&threadLocalHists_[t][channel][bin]));
                sum = _mm_add_epi32(sum, vals);
            }
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(&histogram_[channel][bin]), sum);
        }
    }, threadPool);
}

void ColorHistogramTest::mergeHistogramsScalar(ThreadPool& threadPool)
{
    const size_t numThreads = threadLocalHists_.size();
    
    // Process each channel in parallel
    parallelFor(0, NUM_CHANNELS, [&](size_t channel) {
        // Regular scalar operations for each bin
        for (size_t bin = 0; bin < NUM_BINS; ++bin) {
            uint32_t sum = 0;
            for (size_t t = 0; t < numThreads; t++) {
                sum += threadLocalHists_[t][channel][bin];
            }
            histogram_[channel][bin] = sum;
        }
    }, threadPool);
}

void ColorHistogramTest::saveHistogramCSV(const char* filename) const 
{
    timer_.startTimer();

    // Open file for writing
    ofstream outFile(filename);
    if (!outFile.is_open()) {
        DebugConsole_consoleOutLine("Error opening file for writing: ", filename);
        return;
    }

    // Write header
    outFile << "BinValue,Red,Green,Blue\n";

    // Write data
    for (size_t bin = 0; bin < NUM_BINS; ++bin) {
        outFile << bin << ","
                << histogram_[0][bin] << ","  // Red
                << histogram_[1][bin] << ","  // Green
                << histogram_[2][bin] << "\n"; // Blue
    }

    timer_.stopTimer();
    totalTimeTakenInMs_ += timer_.getElapsedTimeInMilliSecs();
}