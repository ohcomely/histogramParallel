#!/bin/bash

mkdir -p build
cd build
cmake ..
make
cd bin
# ./DeviceUnitTests --gtest_filter=DeviceGoogleTest14__Complete_Histogram_Test.CompleteHistogramTest
./HostUnitTests --gtest_filter=HostGoogleTest10__ColorHistogram.SingleAndMultiCore  
cd ../../
python3 plot.py