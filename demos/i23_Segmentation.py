# Run Reconstruction script first
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
#N_size = 700
inputdata = RecFISTA.reshape(N_size**2, 1)/np.max(RecFISTA)
total_classesNum = 5
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
import numpy as np
import matplotlib.pyplot as plt
from i23.methods.segmentation import MASK_CORR
# Now we process the mask 
mask = np.load("/home/kjy41806/Documents/SOFT/i23seg/demos/upd_mask.npy")
mask_input = mask.copy()
total_classesNum = 5

pars = {'maskdata' : mask_input,\
        'class_names': ('liquor','air','loop'),\
        'total_classesNum': total_classesNum,\
        'restricted_combinations': (('loop','crystal','liquor','loop'),
                                    ('air','artifacts','liquor','liquor'),
                                    ('air','loop','liquor','liquor'),
                                    ('air','artifacts','loop','loop'),
                                    ('air','crystal','loop','loop'),
                                    ('air','loop','crystal','crystal')),\
        'CorrectionWindow' : 10,\
        'iterationsNumb' : 6}

upd_mask_input = MASK_CORR(pars['maskdata'], pars['class_names'], \
                pars['total_classesNum'], pars['restricted_combinations'],\
                pars['CorrectionWindow'], pars['iterationsNumb'])

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.title('Segmented image (postproc GMM)')
plt.imshow(upd_mask_input)
#%%
"""
        'restricted_combinations': (('loop','crystal','liquor','loop'),
                                    ('air','artifacts','liquor','liquor'),
                                    ('air','loop','liquor','liquor'),
                                    ('air','artifacts','loop','loop'),
                                    ('air','crystal','loop','loop'),
                                    ('air','loop','crystal','crystal')),\
"""

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

