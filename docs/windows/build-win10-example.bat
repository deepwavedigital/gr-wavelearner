mkdir build
cd build

cmake .. -G "Visual Studio 15 2017 Win64" -DTensorRT_ROOT=D:\TensorRT-5.0.4.3 -DBOOST_ROOT=C:\local\boost_1_60_0 -DPYTHON_EXECUTABLE="C:\Program Files\GNURadio-3.7\gr-python27\python.exe" -DSWIG_EXECUTABLE=D:\swigwin-3.0.12\swig.exe -DGNURADIO_RUNTIME_INCLUDE_DIRS="C:\Program Files\GNURadio-3.7\include" -DGNURADIO_RUNTIME_LIBRARIES_gnuradio-runtime="C:\Program Files\GNURadio-3.7\lib\gnuradio-runtime.lib" -DGNURADIO_PMT_INCLUDE_DIRS="C:\Program Files\GNURadio-3.7\include" -DGNURADIO_PMT_LIBRARIES_gnuradio-pmt="C:\Program Files\GNURadio-3.7\lib\gnuradio-pmt.lib" -DBoost_ROOT= -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0"

cmake --build . --config Release
cmake --build . --target install --config Release
