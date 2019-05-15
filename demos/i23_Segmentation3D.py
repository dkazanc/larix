# Run Reconstruction script first
#%%
# GMM classification and segmentation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

slices, NxSize, NySize = np.shape(RecFISTA)
inputdata = RecFISTA.reshape(slices*NxSize*NySize, 1)/np.max(RecFISTA)
classes_number = 4
#hist, bin_edges = np.histogram(inputdata, bins=100)
classif = GaussianMixture(n_components=classes_number, covariance_type="tied")
classif.fit(inputdata)
cluster = classif.predict(inputdata)
segm = classif.means_[cluster]
segm = segm.reshape(slices, NxSize, NySize)
mask = segm.astype(np.float64) / np.max(segm)
mask = 255 * mask # Now scale by 255
mask = mask.astype(np.uint8) # we obtain the mask 

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM segmented (clustered) image')
plt.imshow(mask[10,:,:])
# np.save('13068_GMM_100slices.npy', mask)
#%%
import numpy as np
import matplotlib.pyplot as plt
from i23.methods.segmentation import MASK_CORR
# Now we process the mask 
#mask = np.load('/scratch/DATA_TEMP/13068_GMM_100slices.npy')
mask = np.load('/scratch/DATA_TEMP/13076_GMM_50slices.npy')
mask_input = mask.copy()
total_classesNum = 5

#%%
# 13068
pars = {'maskdata' : mask_input[50,:,:],\
        'class_names': ('liquor','air','loop'),\
        'total_classesNum': total_classesNum,\
        'restricted_combinations': (('loop','crystal','liquor','loop'),
                                    ('air','artifacts','liquor','liquor'),
                                    ('air','loop','liquor','liquor'),
                                    ('air','artifacts','loop','loop'),
                                    ('air','crystal','loop','loop'),
                                    ('air','loop','crystal','crystal')),\
        'CorrectionWindow' : 9,\
        'iterationsNumb' : 5}

upd_mask_input = MASK_CORR(pars['maskdata'], pars['class_names'], \
                pars['total_classesNum'], pars['restricted_combinations'],\
                pars['CorrectionWindow'], pars['iterationsNumb'])


slicetovis = 10
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('GMM clustered mask')
plt.imshow(mask[slicetovis,:,:])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Segmented image (postproc GMM)')
plt.imshow(upd_mask_input[slicetovis,:,:])
#%%
#13076 (4 classes)!

pars = {'maskdata' : mask_4classes[25,:,:],\
        'class_names': ('crystal','air','loop'),\
        'total_classesNum': 4,\
        'restricted_combinations': (('loop','crystal','liquor','loop'),
                                    ('air','loop','liquor','liquor'),
                                    ('air','loop','crystal','liquor'),
                                    ('air','crystal','loop','loop'),
                                    ('air','crystal','liquor','liquor'),
                                    ('air','liquor','loop','loop')),\
        'CorrectionWindow' : 10,\
        'iterationsNumb' : 20}

upd_mask_input = MASK_CORR(pars['maskdata'], pars['class_names'], \
                pars['total_classesNum'], pars['restricted_combinations'],\
                pars['CorrectionWindow'], pars['iterationsNumb'])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Segmented image (postproc GMM)')
plt.imshow(upd_mask_input)

#%%
#%%
# a 2D loop over 3D volume
upd_mask_input2D = np.uint8(np.zeros(np.shape(mask)))

for i in range(0, 100):
    pars = {'maskdata' : mask_input[i,:,:],\
        'class_names': ('liquor','air','loop'),\
        'total_classesNum': total_classesNum,\
        'restricted_combinations': (('loop','crystal','liquor','loop'),
                                    ('air','artifacts','liquor','liquor'),
                                    ('air','loop','liquor','liquor'),
                                    ('air','artifacts','loop','loop'),
                                    ('air','crystal','loop','loop'),
                                    ('air','loop','crystal','crystal')),\
        'CorrectionWindow' : 9,\
        'iterationsNumb' : 5}

    upd_mask_input2D[i,:,:] = MASK_CORR(pars['maskdata'], pars['class_names'], \
                pars['total_classesNum'], pars['restricted_combinations'],\
                pars['CorrectionWindow'], pars['iterationsNumb'])

slicetovis = 10
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('3D processed 3D mask')
plt.imshow(upd_mask_input[slicetovis,:,:])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('2D processed 3D mask')
plt.imshow(upd_mask_input2D[slicetovis,:,:])
#%%
"""
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
"""

