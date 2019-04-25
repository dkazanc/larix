# i23 reconstruction and segmentation routines

**Scripts to perform regularised iterative image reconstruction (IIR) and segmentation for tomographic data of [i23 beamline](https://www.diamond.ac.uk/Instruments/Mx/I23.html) of [Diamond Light Source](http://diamond.ac.uk/). For reconstruction the [ToMoBAR](https://github.com/dkazanc/ToMoBAR) and [CCPi-Regularisation](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) software is used. The segmentation is performed in two steps: 1) clustering by Gaussian Mixtures, 2) Model-based iterative processing of the GMM mask by incorporating structural information (classes spatial inconsistencies)** 

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

### Python (conda-build)
```
export i23seg_VERSION=0.1.0
conda build recipe/ --numpy 1.14 --python 2.7  
conda install i23-seg=${i23seg_VERSION} --use-local --force-reinstall
```
