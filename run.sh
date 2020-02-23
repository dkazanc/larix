#!/bin/bash
echo "Building COLLEDA software using CMake"
rm -r build
# Requires Cython, install it first:
# pip install cython
mkdir build
cd build
#make clean
export VERSION=0.1.3
# install Python modules without CUDA
cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
############### Python(linux)###############
cp install/lib/libcolleda.so install/python/colleda/methods
cd install/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
spyder --new-instance
