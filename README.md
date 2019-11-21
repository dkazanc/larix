| Master | Anaconda binaries |
|--------|-------------------|
| [![Build Status](https://travis-ci.org/dkazanc/i23seg.svg?branch=master)](https://travis-ci.org/dkazanc/i23seg.svg?branch=master) | ![conda version](https://anaconda.org/dkazanc/i23-seg/badges/version.svg) ![conda last release](https://anaconda.org/dkazanc/i23-seg/badges/latest_release_date.svg) [![conda platforms](https://anaconda.org/dkazanc/i23-seg/badges/platforms.svg) ![conda dowloads](https://anaconda.org/dkazanc/i23-seg/badges/downloads.svg)](https://anaconda.org/dkazanc/i23-seg/) |

# i23-beamline reconstruction and segmentation routines for full-field tomography

**Scripts to perform regularised iterative image reconstruction (IIR) and segmentation for tomographic data of [i23 beamline](https://www.diamond.ac.uk/Instruments/Mx/I23.html) of [Diamond Light Source](http://diamond.ac.uk/). For reconstruction the [ToMoBAR](https://github.com/dkazanc/ToMoBAR) and [CCPi-Regularisation](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) software is used. The segmentation is performed in two steps: 1) clustering by Gaussian Mixtures, 2) Model-based iterative processing of the GMM mask by resolving spatial inconsistencies between classes** 

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
