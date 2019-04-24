#!/bin/bash
echo "Building i23-segmentation modules using CMake"
rm -r ../build_proj
# Requires Cython, install it first:
# pip install cython
mkdir ../build_proj
cd ../build_proj/
#make clean
export i23seg_VERSION=0.1
# install Python modules without CUDA
cmake ../ -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
############### Python(linux)###############
#cp install/lib/libi23seg.so install/python/i23/methods
#cd install/python
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
#spyder
