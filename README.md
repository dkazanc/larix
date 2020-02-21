| Master | Anaconda binaries |
|--------|-------------------|
| [![Build Status](https://travis-ci.org/dkazanc/i23seg.svg?branch=master)](https://travis-ci.org/dkazanc/i23seg.svg?branch=master) | ![conda version](https://anaconda.org/dkazanc/i23-seg/badges/version.svg) ![conda last release](https://anaconda.org/dkazanc/i23-seg/badges/latest_release_date.svg) [![conda platforms](https://anaconda.org/dkazanc/i23-seg/badges/platforms.svg) ![conda dowloads](https://anaconda.org/dkazanc/i23-seg/badges/downloads.svg)](https://anaconda.org/dkazanc/i23-seg/) |

# IPOLS: Image Processing toOLS software consists of various modules for data processing tasks

** Various novel and already existing routines are implemented at [Diamond Light Source](https://www.diamond.ac.uk/Home.html) to help with the processing of collected data (not only limited to the synchrotron data). Most of the modules are implemented in C language with OpenMP multithreading capability which ensures faster data processing. The modules are wrapped for Python and continious integration with Travis helps with easy installation of software through conda install.**

## Software includes:
 * Mask evolving segmentation method with mask initialisation (**2D/3D CPU**)
 * Morphological processing of segmented image/volume by removing various gaps and misclassified regions (**2D/3D CPU**)

## Installation
Here an example of build on Linux (see also `run.sh` for additional info):

```bash
git clone https://github.com/vais-ral/CCPi-Regularisation-Toolkit.git
mkdir build
cd build
cmake .. -DCONDA_BUILD=OFF -DBUILD_MATLAB_WRAPPER=ON -DBUILD_PYTHON_WRAPPER=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
make install
cd install/python
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../lib
```

### Installation in Python (conda):
Install from the channel with `conda install -c dkazanc i23-seg` or build with:
```
export VERSION=`date +%Y.%m`
conda build recipe/ --numpy 1.15 --python 2.7  
conda install i23-seg --use-local --force-reinstall
```
Additionaly you can also run/modify `conda_install.sh` for automatic conda build/install

#### Other usefull software: 
 * [ASTRA-toolbox](https://www.astra-toolbox.com/) versatile CPU/GPU toolbox for tomography
 * [ToMoBAR](https://github.com/dkazanc/ToMoBAR) TOmographic iterative MOdel-BAsed Reconstruction
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) for regularisation of IR
 * [TomoPhantom](https://github.com/dkazanc/TomoPhantom) for tomographic simulation

