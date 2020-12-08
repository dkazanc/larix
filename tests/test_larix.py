import unittest
import pytest
import numpy as np

class TestLarix(unittest.TestCase):
    
    def test_import_cpu_modules(self):
        try:
            from larix.methods.segmentation import REGION_GROW, MORPH_PROC_LINE
        except NameError:
            print('____! Larix has not been installed properly (import error) !____')
            raise
    
    def test_import_gpu_modules(self):
        try:
            from larix.methods.misc_gpu import MEDIAN_FILT_GPU, MEDIAN_DEZING_GPU
        except NameError:
            print('____! GPU modules of Larix have not been installed !____')
            raise
        """
        # enable if there is a GPU available
        image_test = np.ones((128,128))
        image_test[64,64] = 5
        image_test[74,74] = 15
        image_test[94,94] = 25
        pars = {'input_data' : np.float32(image_test), # input grayscale image
            'kernel_size' : 5,
            'mu_threshold': 1.0}
        gpu_dezingered = MEDIAN_DEZING_GPU(pars['input_data'], pars['kernel_size'], pars['mu_threshold'])
        self.assertLessEqual(np.max(gpu_dezingered) , 1.0)
        """

if __name__ == '__main__':
    unittest.main()
