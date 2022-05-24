#!/bin/bash
echo "Building Larix software using CMake..."
echo "<<< Make sure you've got cython and nvcc compiler installed >>>"
rm -r build
mkdir build
cd build
#make clean
export VERSION=0.1.3
# install Python modules without CUDA
#cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
# install Python modules with CUDA
cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
############### Python(linux)###############
cp install/lib/liblarix.so install/python/larix/methods
# copy testing script and data
cp ../tests/build_testing.py install/python/
cp ../data/sino_noisy.npy install/python/
cp ../data/sino_denoiseCPU.npy install/python/
cp ../data/data3D_to_crop.npy.bz2 install/python/
cp ../data/volume_filteredCPU.npy.bz2 install/python/
cd install/python/
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
ipython build_testing.py

# another option is to run spyder and go the folder larix/build/install/python to be able to import modules
#spyder --new-instance

# All built shared objects will be placed into larix/build/install/python/larix/methods
