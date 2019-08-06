import numpy as np
import tifffile
import matplotlib.pyplot as plt
from .stabilize_sequence import stabilize_getshifts, stabilize_apply_shifts, stabilize_crop_to_overlap
from .detect_and_label_chambers import detect_circles
from skimage.measure import label, regionprops_table
from skimage.color import label2rgb
import skimage.filters
import skimage.morphology
import pathlib
from typing import Dict, Optional, Tuple

def process_folder(folder: pathlib.Path):
    """[summary]
    
    Parameters
    ----------
    folder : pathlib.Path
        [description]
    replace : [type], optional
        [description], by default ("/scratch/"
    """
    # find original nd2 file based on which folder has
    # been created with bfconvert (use string replacement)

    # inspect metadata of orignal nd2 file to identify which channel corresponds
    # to which dye

    # inspect nd2 metadata to identify which channel m
    # process each 
    
    # find all tiff series (corresponding to timeseries for a position) in folder
    # process all timeseries (pass foldername and metadata mappings)

    pass


def bg_estimate_initial(timeseq: np.ndarray):
    """generate an initial background estimate based on the minimun intensity
    seen for each pixel over time. 
    
    Arguments:
        timeseq {np.ndarray} -- a 3D (t,y,x) time series

    NOTE: if there are areas where cells are present over the whole time course
    (e.g. stationary cells, or a big lump of cells), those will merge into the background.
    The model assumption is that at each pixel we see the background at least once
    over the time course! This assumption holds for the images I've seen so far,
    and is what I would anticipate for this type of experiment. BUT THIS
    NEEDS TO BE KEPT IN MIND !!!
    """
    return skimage.filters.gaussian(np.min(timeseq, axis=0))


def segment_cells(dapiseq: np.ndarray, min_thresh: Optional[float]=None, dilate_by: int=20) -> Tuple[np.ndarray, np.ndarray]:
    """given a time series of dapi images, calculate labels and background mask
    
    Arguments:
        dapiseq {np.ndarray} -- dapi image sequence()
        min_tresh {float} -- optional, minimum threshold. 
        dilate_by {int} -- how much to dilate the nuclei to determine the background mask
    Returns:
        Tuple[np.ndarray, np.ndarray] -- (label sequence, background mask sequence)

    Segmentation is performed using otsu thresholding for each time point, using the larger of 
    the determined otsu threshold or min_thresh. The segmented regions are labelled. 
    Also the nuclei areas will be dilated and then inverted to create a mask of the background.
    The resulting label sequence and background mask binary sequence are returned as a tuple
    """

    labels = np.zeros(dapiseq.shape, dtype=np.uint16)
    masks = np.zeros(dapiseq.shape, dtype=np.bool)
    def process_frame(i):
        th = skimage.filters.threshold_otsu(dapiseq[i,...])
        if min_thresh is not None:
            th = max(th, min_thresh)
        cellmask = dapiseq[i, ...] > th
        dilated_mask = skimage.morphology.dilation(cellmask, skimage.morphology.disk(dilate_by))
        bg = ~dilated_mask
        # TODO see whether we observe
        # any clumping and implement distance-transform-based splitting
        # if necessary
        labeled_frame = label(cellmask)
        labels[i, ...] = labeled_frame
        masks[i, ...] = bg
        return(i)

    # using a map and a nested function may look odd,
    # but should allow replacing map with a ThreadPool().map
    # for multithreaded performance. More elegant suggestions welcome. 
    res = list(map(process_frame, range(dapiseq.shape[0])))
    return labels, masks


def process_timeseries(input_tif: str, metadata: Dict, output_csv = Optional[str], output_overlay = Optional[str]):
    """Provided an input image sequence correspoding to a time series at a given position perform
    egg chamber detection/segmentation/ 

    
    Parameters
    ----------
    input_tif : str
        input filename
    metadata : Dict
        dictionary that contains metadata which should be added to output table
    output_csv : [type], optional
        file name for the output csv file, will be generated automatically based on input_tif if None
    output_overlay : [type], optional
        file name for the segmentation results,  will be generated automatically based on input_tif if None
    """
    
    seq = tifffile.imread(input_tif)

    if seq.ndim == 5: # we sometimes get duplicate dimensions from bfconvert
        seq = seq[0, ... ]
    assert seq.ndim == 4

    
    # determine channels based on metadata
    # fill in later 
    # we definitely need DAPI channel for nuclei and brightfield 
    # for egg chambers

    ch_dapi = 0
    ch_bf = 1

    # TODO: swap dimensions as necessary (according to metadata np.)
    # assume from here on that we have the dimension order
    # from here onwards (t, ch, y, x)

    # Drift correction of sequence
    shifts = stabilize_getshifts(seq[:, ch_bf, ...])
    seq = stabilize_apply_shifts(seq, shifts)
    seq = stabilize_crop_to_overlap(seq)

    # Detect egg chambers
    # take pixel-wise median (over time) to remove moving cells
    bf_median = np.median(seq[:,ch_bf,...], axis=2)
    chamber_labels = detect_circles(bf_median)

    # first round of background subtraction
    # apply to all fluorescence channels

    all_channels = tuple(range(seq.shape[1]))
    fluo_channels = tuple(set(all_channels) - set([ch_bf]))

    bg_corr = list(map(lambda ch: seq[:, ch, ...] - bg_estimate_intial(seq[:,ch,...], fluo_channels)))

    # single out dapi channel for nuclei segmentation
    bg_corr_dapi = bg_corr[fluo_channels.index(ch_dapi)]


    

    # for each fluorescence channel (use a map)
    #    smooth
    #    find per-pixel min over time as initial background estimate
    #    subtract per-pixel min 

    # segment and label cells in DAPI channel. Return mask and label image
    # create mask of background areas by dilating cell maks and inverting
    # for each fluorescence channel
    #     calc median intensity of background area and subtract to obtain final bgcorrected intensity image
    # concatenate
    #    for [each bgcorrected fluorescence channel] + [egg chamber label image]:
    #         calc desired regionprops
    #         rename regionprop columns (prepend_channel)
    # add a metadata column that gives the total size of the mask to which the cell belongs
    # this will allow filtering out cells that were in egg chambers that were not fully in the field of view
    # and for the background mask it records the areas outside of the egg chambers after cropping (small caveat:
    # the chambers are actually larger than the cirlces)
    # generate output_csv filename from input filename unless provided
    # write csv
    # generate output_overlay filename/s from overlay filename unless provided
    # write image series (png/jpg)
    # return 

    pass
    

def segment_cells(im: np.ndarray) -> np.ndarray:
    """[summary]
    
    Parameters
    ----------
    im : np.ndarray
        Input image (cell nuclei with a fluorescent marker)
    
    Returns
    -------
    np.ndarray
        label image
    """
    pass


