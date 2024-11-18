#include "CUDAParallelFor.h"
#include "CUDAUtilityFunctions.h"
#include "Tests/ColorHistogramGPUTest.h"
#include <algorithm>

using namespace std;
using namespace UtilsCUDA;
using namespace UtilsCUDA::CUDAParallelFor;

namespace {
constexpr size_t BLOCK_SIZE = 256;
constexpr size_t NUM_CUDA_STREAMS = 3; // One per color channel

inline void startMemoryPool(const CUDADriverInfo &cudaDriverInfo,
                            CUDAProcessMemoryPool &cudaProcessMemoryPool,
                            int device, size_t arraySize,
                            bool useUnifiedMemory) {
  const size_t hostBytesToAllocate =
      (3 * arraySize + 3 * cudaDriverInfo.getTextureAlignment(device)) *
      sizeof(int32_t); // Note: allocation is in bytes, plus some padding space
  if (useUnifiedMemory) {
    // use the Device Memory Pool for allocation of device memory (with Unified
    // Memory enabled) for the given device
    array<size_t, CUDAProcessMemoryPool::MAX_DEVICES>
        deviceBytesToAllocatePerDevice = {
            {0}}; // default-initialize all devices to zero bytes usage
    deviceBytesToAllocatePerDevice[device] =
        hostBytesToAllocate; // allocate bytes in given device
    cudaProcessMemoryPool.allocateDeviceMemoryPool(
        deviceBytesToAllocatePerDevice, 1ull << device);
  } else {
    // use the Host/Device Memory Pool for allocation of host/device memory for
    // the given device
    array<size_t, CUDAProcessMemoryPool::MAX_DEVICES>
        deviceBytesToAllocatePerDevice = {
            {0}}; // default-initialize all devices to zero bytes usage
    deviceBytesToAllocatePerDevice[device] =
        hostBytesToAllocate; // allocate bytes in given device
    cudaProcessMemoryPool.allocateHostDeviceMemoryPool(
        hostBytesToAllocate, deviceBytesToAllocatePerDevice);
  }
}

// CUDA kernel for computing histogram for one channel
    __global__ void computeHistogramKernel(const uint8_t *imageData,
                                          uint32_t *histogram,
                                          size_t width,
                                          size_t height,
                                          size_t channelOffset) {
    // Shared memory for per-block histogram
    __shared__ uint32_t sharedHist[256];

    // Initialize shared memory
    const uint32_t tid = threadIdx.x;
    sharedHist[tid] = 0;
    __syncthreads();

    // Process pixels with striding
    const size_t numPixels = width * height;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numPixels;
         i += stride) {
        const uint8_t pixelValue = imageData[i * 3 + channelOffset];
        atomicAdd(&sharedHist[pixelValue], 1);
         }

    // Make sure all threads finished updating shared memory
    __syncthreads();

    // Write shared memory to global memory
    atomicAdd(&histogram[tid], sharedHist[tid]);
}
} // namespace

ColorHistogramGPUTest::ColorHistogramGPUTest(
    const CUDADriverInfo &cudaDriverInfo, int device)
    : CUDAGPUComputingAbstraction(cudaDriverInfo, device),
      cudaProcessMemoryPool_(cudaDriverInfo, false),
      cudaStreamsHandler_(cudaDriverInfo, device, NUM_CUDA_STREAMS),
      gpuTimer_(device, cudaStreamsHandler_[0]) {
  // Choose GPU device
  CUDAError_checkCUDAError(cudaSetDevice(device));
}

void ColorHistogramGPUTest::initializeFromImage(const uint8_t *imageData,
                                                size_t width, size_t height) {
  width_ = width;
  height_ = height;

  gpuTimer_.startTimer();

  startMemoryPool(cudaDriverInfo_, cudaProcessMemoryPool_, device_,
                  width * height, false);
  // Initialize GPU memory

  // Copy image data to GPU
  const size_t imageSize = width * height * NUM_CHANNELS;
  cudaProcessMemoryPool_.reserve(imageData_, imageSize, device_);

  // Copy input data to GPU
  CUDAError_checkCUDAError(cudaMemcpy(imageData_.device(), imageData, imageSize,
                                      cudaMemcpyHostToDevice));

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

void ColorHistogramGPUTest::initializeGPUMemory() {
  gpuTimer_.startTimer();

  // Initialize histogram arrays in device memory
  cudaProcessMemoryPool_.reserve(histogramRed_, NUM_BINS, device_);
  cudaProcessMemoryPool_.reserve(histogramGreen_, NUM_BINS, device_);
  cudaProcessMemoryPool_.reserve(histogramBlue_, NUM_BINS, device_);

  // Initialize histograms to zero
  CUDAError_checkCUDAError(
      cudaMemset(histogramRed_.device(), 0, NUM_BINS * sizeof(uint32_t)));
  CUDAError_checkCUDAError(
      cudaMemset(histogramGreen_.device(), 0, NUM_BINS * sizeof(uint32_t)));
  CUDAError_checkCUDAError(
      cudaMemset(histogramBlue_.device(), 0, NUM_BINS * sizeof(uint32_t)));

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

void ColorHistogramGPUTest::performGPUComputing() {
    gpuTimer_.startTimer();

    // Reset histograms to zero before computing
    CUDAError_checkCUDAError(
        cudaMemset(histogramRed_.device(), 0, NUM_BINS * sizeof(uint32_t)));
    CUDAError_checkCUDAError(
        cudaMemset(histogramGreen_.device(), 0, NUM_BINS * sizeof(uint32_t)));
    CUDAError_checkCUDAError(
        cudaMemset(histogramBlue_.device(), 0, NUM_BINS * sizeof(uint32_t)));

    // Calculate grid dimensions for kernel launch
    const int threadsPerBlock = 256;
    const int numBlocks = min(
        (width_ * height_ + threadsPerBlock - 1) / threadsPerBlock,
        size_t(cudaDriverInfo_.getMaxGridSize(device_)[0])
    );

    DebugConsole_consoleOutLine(
        "Launching kernel with ", numBlocks, " blocks of ",
        threadsPerBlock, " threads each");
    DebugConsole_consoleOutLine(
        "Processing ", width_, "x", height_, "=", width_*height_, " pixels");

    // Verify device memory pointers
    if (!imageData_.device()) {
        DebugConsole_consoleOutLine("Error: Null image data pointer");
        return;
    }

    // Launch kernels for each color channel
    uint32_t* histPtrs[3] = {
        histogramRed_.device(),
        histogramGreen_.device(),
        histogramBlue_.device()
    };

    for (int i = 0; i < 3; i++) {
        if (!histPtrs[i]) {
            DebugConsole_consoleOutLine("Error: Null histogram pointer for channel ", i);
            return;
        }
    }

    for (size_t channel = 0; channel < NUM_CHANNELS; channel++) {
        computeHistogramKernel<<<numBlocks, threadsPerBlock,
                                0, cudaStreamsHandler_[channel]>>>(
            imageData_.device(),
            histPtrs[channel],
            width_, height_,
            channel
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            DebugConsole_consoleOutLine(
                "Kernel launch failed for channel ", channel, ": ",
                cudaGetErrorString(err));
            return;
        }
    }

    // Wait for all channels to complete
    for (size_t i = 0; i < NUM_CHANNELS; i++) {
        CUDAError_checkCUDAError(
            cudaStreamSynchronize(cudaStreamsHandler_[i]));
    }

    totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

void ColorHistogramGPUTest::retrieveGPUResults() {
  gpuTimer_.startTimer();

  // Copy results back from GPU
  CUDAError_checkCUDAError(
      cudaMemcpy(histogram_[0].data(), histogramRed_.device(),
                 NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDAError_checkCUDAError(
      cudaMemcpy(histogram_[1].data(), histogramGreen_.device(),
                 NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDAError_checkCUDAError(
      cudaMemcpy(histogram_[2].data(), histogramBlue_.device(),
                 NUM_BINS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  totalTimeTakenInMs_ += gpuTimer_.getElapsedTimeInMilliSecs();
}

bool ColorHistogramGPUTest::verifyComputingResults() {
    // Verify total counts match image dimensions
    const size_t totalPixels = width_ * height_;

    // Print verification info
    for (size_t channel = 0; channel < NUM_CHANNELS; ++channel) {
        size_t gpuSum = 0;

        for (size_t bin = 0; bin < NUM_BINS; ++bin) {
            gpuSum += histogram_[channel][bin];
        }

        DebugConsole_consoleOutLine(
            "Channel ", channel, " total pixels: GPU=", gpuSum,
            " Expected=", totalPixels);

        if (gpuSum != totalPixels) {
            DebugConsole_consoleOutLine(
                "Channel ", channel, " total count mismatch: GPU=", gpuSum,
                " Expected=", totalPixels);
            return false;
        }
    }

    // Optionally add histogram distribution checks
    for (size_t channel = 0; channel < NUM_CHANNELS; ++channel) {
        // Verify no single bin has more pixels than total
        for (size_t bin = 0; bin < NUM_BINS; ++bin) {
            if (histogram_[channel][bin] > totalPixels) {
                DebugConsole_consoleOutLine(
                    "Invalid bin count in channel ", channel,
                    " bin ", bin, ": ", histogram_[channel][bin],
                    " exceeds total pixels ", totalPixels);
                return false;
            }
        }

        // Verify bins have some reasonable distribution
        // This is image dependent but helps catch major issues
        size_t emptyBins = 0;
        size_t fullBins = 0;
        for (size_t bin = 0; bin < NUM_BINS; ++bin) {
            if (histogram_[channel][bin] == 0) emptyBins++;
            if (histogram_[channel][bin] == totalPixels) fullBins++;
        }

        // Warn if distribution looks suspicious
        if (emptyBins > 200) { // More than ~80% empty
            DebugConsole_consoleOutLine(
                "Warning: Channel ", channel, " has ",
                emptyBins, " empty bins out of 256");
        }
        if (fullBins > 0) {
            DebugConsole_consoleOutLine(
                "Warning: Channel ", channel, " has ",
                fullBins, " bins containing all pixels");
        }
    }

    return true;
}

void ColorHistogramGPUTest::releaseGPUComputingResources() {
  // Memory will be automatically freed by RAII
  for (auto &channel : histogram_) {
    channel.fill(0);
  }
}