#%%

import matplotlib.pyplot as plt
import numpy as np
import os

"""
filename = os.path.join( "data" ,"lena_gray_512.tif")

# read image
Im = plt.imread(filename)
Im = np.asarray(Im, dtype='float32')

Im = Im/255.0
perc = 0.05
u0 = Im + np.random.normal(loc = 0 ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
u_ref = Im + np.random.normal(loc = 0 ,
                                  scale = 0.01 * Im , 
                                  size = np.shape(Im))
(N,M) = np.shape(u0)
u0 = u0.astype('float32')
u_ref = u_ref.astype('float32')

mask = np.ones([512,512])

mask[160:260,160:260] = 2
mask[160:260,261:262] = 2
mask[280:281,280] = 2


mask = mask.astype(np.float64) / np.max(mask)
mask = 255 * mask # Now scale by 255
mask = mask.astype(np.uint8)
"""
u0 = np.load("ReconFISTA.npy")
mask = np.load("GMM_SEGM.npy")

#%%
from ccpi.filters.regularisers import MASK_CORR

pars = {'algorithm' : MASK_CORR, \
        'maskdata' : mask,\
        'select_classes': np.uint8(np.array([2,3])),\
        'total_classesNum': 5,\
        'CorrectionWindow' : 8}

(upd_mask,correct_matrix) = MASK_CORR(pars['maskdata'],
              pars['select_classes'],
              pars['total_classesNum'],
              pars['CorrectionWindow'],'cpu')

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(upd_mask)

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(correct_matrix, vmin=0, vmax=255)
#%%
reg_param_scalar = 0.0001
reg_param = np.float32(correct_matrix)/np.float32(np.max(correct_matrix))*reg_param_scalar
reg_param = reg_param+reg_param_scalar
reg_param[(np.where(reg_param > reg_param_scalar))] = reg_param[(np.where(reg_param > reg_param_scalar))]*100

# smooth it 
import scipy.signal
# make some kind of kernel, there are many ways to do this...
t = 1 - np.abs(np.linspace(-1, 1, 21))
kernel = t.reshape(21, 1) * t.reshape(1, 21)
kernel /= kernel.sum()   # kernel should sum to 1!  :) 

# convolve 2d the kernel with each channel
reg_param_smooth = scipy.signal.convolve2d(reg_param, kernel, mode='same')


#%%
from ccpi.filters.regularisers import NDF_MASK

pars = {'algorithm' : NDF_MASK, \
        'input' : u0,\
        'maskdata' : upd_mask,\
        'diffuswindow' : 1,\
        'regularisation_parameter':reg_param , \
        'number_of_iterations' :1000 ,\
        'time_marching_parameter':0.01,\
        'edge_parameter':0.002,\
        'penalty_type':1,\
        'tolerance_constant':0.0}

(ndf_cpu,info_vec_cpu) = NDF_MASK(pars['input'], 
              pars['maskdata'],
              pars['diffuswindow'],
              pars['regularisation_parameter'],
              pars['edge_parameter'], 
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['penalty_type'],
              pars['tolerance_constant'],'cpu')

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(ndf_cpu,  cmap="gray")
#%%

from ccpi.filters.regularisers import ROF_TV
# set parameters
pars = {'algorithm' : ROF_TV, \
        'input' : u0,\
        'regularisation_parameter': 0.00005, \
        'number_of_iterations' :1500 ,\
        'time_marching_parameter':0.003,\
        'tolerance_constant':0.0}
        
(rof_cpu_m,info_vec_cpu) = ROF_TV(pars['input'], 
              pars['regularisation_parameter'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'], 
              pars['tolerance_constant'],'cpu')

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(rof_cpu_m,  cmap="gray")
#%%
from sklearn.mixture import GaussianMixture
N_size = 700
inputdata = RecFISTA_Huber_TV_os.reshape(N_size**2, 1)/np.max(RecFISTA_Huber_TV_os)
classes_number = 5
#hist, bin_edges = np.histogram(inputdata, bins=100)
classif = GaussianMixture(n_components=classes_number, covariance_type="tied")
classif.fit(inputdata)
cluster = classif.predict(inputdata)
segm = classif.means_[cluster]
segm = segm.reshape(N_size, N_size)
mask = segm.astype(np.float64) / np.max(segm)
mask = 255 * mask # Now scale by 255
mask = mask.astype(np.uint8) # we obtain the mask 

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) image')
plt.imshow(mask)

#%%
# Now we process the mask 
import numpy as np
from i23.methods.segmentation import MASK_CORR_CPU
import matplotlib.pyplot as plt
mask = np.load("upd_mask.npy")
upd_mask_input = mask.copy()
classes_number = 5

pars = {'maskdata' : upd_mask_input,\
        'select_classes': (3,0,1),\
        'total_classesNum': classes_number,\
        'CorrectionWindow' : 10,\
        'iterationsNumb' : 25}

(upd_mask_input,correct_matrix) = MASK_CORR_CPU(pars['maskdata'], pars['select_classes'],
pars['total_classesNum'], pars['CorrectionWindow'], pars['iterationsNumb'])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Segmented image (postproc GMM)')
plt.imshow(upd_mask_input)
#%%

