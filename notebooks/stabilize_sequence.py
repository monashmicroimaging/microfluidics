import imreg
import numpy as np
from skimage import transform
from concurrent.futures import ThreadPoolExecutor

def stabilize_getshifts(im, workers=1):
    """get shifts for sequence
    
    Parameters
    ----------
    seq : numpy array time series (t, ...)
        finds the pairwise shifts
    returns np.array of shifts
    """
    with ThreadPoolExecutor() as p:
            shifts= list(p.map(lambda n: imreg.translation(im[n,...], im[n+1,...]), range(im.shape[0]-1)))
    shifts=[[0,0]] + shifts
    return np.array(shifts)

def stabilize_apply_shifts(seq, shifts):
    """apply shifts to a sequence possibly with multiple channels  
    Parameters
    ----------
    seq : numpy array time series (tyx or tcyx)
        [description]
    shifts : np array as returned from stabilize_getshifts
        [description]
    """

    input_is_single_ch = False
    if seq.ndim == 3:
        seq = seq[:,np.newaxis, ...]
        input_is_single_ch = True

    cumulative_shifts=np.cumsum(shifts,axis=0)
    cummax = np.max(cumulative_shifts, axis=0)
    cummin = np.min(cumulative_shifts, axis=0)

    cumulative_shifts -= cummin
    enlarge_shape_by = cummax - cummin
    
    nt, nc, ny, nx = seq.shape
    padded_shape =  [nt, nc] + list(np.array([ny,nx])+enlarge_shape_by)
    padded_series = np.zeros(padded_shape, dtype=seq.dtype)
    for i, d in enumerate(cumulative_shifts):
        padded_series[i, 0:1, d[0]:d[0]+ny, d[1]:d[1]+nx] = seq[i,  ...]
    if input_is_single_ch:
        return padded_series[:,0,...]
    else:
        return padded_series

    def stabilize_crop_to_overlap(seq):
        tmp = np.where(np.min(seq, axis=0)>0)
        miny, maxy = np.min(tmp[-2]),np.max(tmp[-2])
        minx, maxx = np.min(tmp[-1]),np.max(tmp[-1])
        cropped = seq[..., miny:maxy, minx:maxx]
        return cropped