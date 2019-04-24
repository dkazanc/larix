"""
script which assigns a proper device core function based on a flag ('cpu' or 'gpu')
"""

from i23.methods.cpu_segmentation import MASK_CORR_CPU

def MASK_CORR(maskdata, select_classes, total_classesNum, CorrectionWindow, iterationsNumb, device='cpu'):
    if device == 'cpu':
        return MASK_CORR_CPU(maskdata, select_classes, total_classesNum, CorrectionWindow, iterationsNumb)
    elif device == 'gpu' and gpu_enabled:
        return MASK_CORR_CPU(maskdata, select_classes, total_classesNum, CorrectionWindow, iterationsNumb)
    else:
        if not gpu_enabled and device == 'gpu':
    	    raise ValueError ('GPU is not available')
        raise ValueError('Unknown device {0}. Expecting gpu or cpu'\
                         .format(device))
