#!/bin/bash
echo "Building Larix software using CMake"
rm -r build
# Requires Cython, install it first:
# pip install cython
mkdir build
cd build
#make clean
export VERSION=0.1.3
# install Python modules without CUDA
cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
# install Python modules with CUDA
#cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
############### Python(linux)###############
cp install/lib/liblarix.so install/python/larix/methods
cd install/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
spyder --new-instance

# All built shared objects will be placed into larix/build/install/python/larix/methods
# in order to be able import modules you need to go to the folder larix/build/install/python to be able to import
# modules
