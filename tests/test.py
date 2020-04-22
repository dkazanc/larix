import unittest
import numpy as np
from larix.methods.segmentation import REGION_GROW, MORPH_PROC_LINE

try:
    from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_DEZING_GPU
except:
    print('____! GPU modules of Larix have not been installed !____')

###############################################################################

if __name__ == '__main__':
    unittest.main()
