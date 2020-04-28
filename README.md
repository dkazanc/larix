<table>
    <tr>
        <td>
        <div align="left">
          <img src="docs/images/larix_logo.png" width="1150"><br>
        </div>
        </td>
        <td>
        <font size="5"><b> Larix: is a set of data and image processing tools </b></font>
        <br><font size="3" face="verdana" color="green"><b> Larix</b> is an open-source software written in C/CUDA languages with Python wrappers. The software consists of novel and already existing methods for various image processing tasks, e.g. filtering, inpainting, segmentation, morphological processing, etc.  Most of the modules are implemented with OpenMP multithreading capability in C or on GPU to ensure faster data processing. Larix is being developed at Diamond Light Source synchrotron (UK) with the main purpose to process collected data effectively and efficiently.
        </font></br>
        </td>
    </tr>
</table>

| Master | Anaconda binaries |
|--------|-------------------|
| [![Build Status](http://192.168.0.9:8080/buildStatus/icon?job=larix%2Fmaster)](http://192.168.0.9:8080/job/larix/job/master/) | ![conda version](https://anaconda.org/dkazanc/larix/badges/version.svg) ![conda last release](https://anaconda.org/dkazanc/larix/badges/latest_release_date.svg) [![conda platforms](https://anaconda.org/dkazanc/larix/badges/platforms.svg) ![conda dowloads](https://anaconda.org/dkazanc/larix/badges/downloads.svg)](https://anaconda.org/dkazanc/larix/) |

## Larix software includes:
 * Mask evolving segmentation method (RegionGrow) with mask initialisation (**2D/3D CPU**)
 * Morphological processing of the segmented image/volume (e.g. the result of RegionGrow)  using the line segments to remove gaps and misclassified regions (**2D/3D CPU**)
 * Auto cropping for tomographic projections or reconstructed images when the object is within the FOV (**2D/3D CPU**)
 * Median filtration and median-based dezinger to remove broken pixels aka outliers  (**2D/3D CPU/GPU**)
 * Inpainting using linear/non-linear diffusion and non-local marching method  (**2D CPU**)

 <div align="center">
   <img src="docs/images/demo_larix.png" width="650">
 </div>

## Installation in Python (conda):
Install from the anaconda cloud: `conda install -c dkazanc larix` or build with:
```
export VERSION=`date +%Y.%m`
conda build recipe/ --numpy 1.15 --python 3.5
conda install -c file://${CONDA_PREFIX}/conda-bld/ larix
```
Additionally you can also run/modify `conda_install.sh` for automatic conda build/install

### Installation using Cmake
See `run.sh` for information how to build and install software.

#### Other useful software:
 * [Savu](https://github.com/DiamondLightSource/Savu) Tomographic reconstruction pipeline at DLS to which Larix has been incorporated
 * [ASTRA-toolbox](https://www.astra-toolbox.com/) Versatile CPU/GPU toolbox for tomography
 * [ToMoBAR](https://github.com/dkazanc/ToMoBAR) TOmographic iterative MOdel-BAsed Reconstruction software
 * [CCPi-RegularisationToolkit](https://github.com/vais-ral/CCPi-Regularisation-Toolkit) Adds regularisation to ToMoBAR
 * [TomoPhantom](https://github.com/dkazanc/TomoPhantom) Tomographic simulation and phantoms
