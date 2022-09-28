import numpy as np
import timeit
import bz2
import IPython

from larix.methods.misc import MEDIAN_FILT
from larix.methods.misc_gpu import MEDIAN_FILT_GPU

import tomophantom
from tomophantom import TomoP3D
import os

model = 13 # select a model
N_size = (32*30,32*30,32) # set dimension of the phantom
# one can specify an exact path to the parameters file
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
phantom_3D = TomoP3D.Model(model, N_size, path_library3D)

print("Applying Median Filter in 3D using CPU...")
pars = {'input_data' : np.float32(phantom_3D), # input grayscale image
        'kernel_size' : 3}


print("Applying Median Filter in 3D using GPU...")
start_time = timeit.default_timer()
volume_filteredGPU = MEDIAN_FILT_GPU(pars['input_data'], pars['kernel_size'])
txtstr = "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
print (txtstr)
IPython.embed()