# test_larix

import unittest
import pytest

def test_import_cpu_modules():
    try:
        from larix.methods.segmentation import REGION_GROW, MORPH_PROC_LINE
    except NameError:
        print('____! Larix has not been installed properly (import error) !____')
        raise

def test_import_gpu_modules():
    try:
        from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_DEZING_GPU
    except NameError:
        print('____! GPU modules of Larix have not been installed !____')
        raise

###############################################################################

#if __name__ == '__main__':
#    unittest.main()
