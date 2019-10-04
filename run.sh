#!/bin/bash
echo "Building i23-segmentation modules using CMake"
rm -r build
# Requires Cython, install it first:
# pip install cython
mkdir build
cd build
#make clean
export VERSION=0.1.1
# install Python modules without CUDA
cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
############### Python(linux)###############
cp install/lib/libi23seg.so install/python/i23/methods
cd install/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
spyder --new-instance
