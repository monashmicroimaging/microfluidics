import numpy as np
import tifffile
import matplotlib.pyplot as plt
from stabilize_sequence import stabilize_getshifts, stabilize_apply_shifts, stabilize_crop_to_overlap
from detect_and_label_chambers import detect_circles
from skimage.measure import label, regionprops_table
from skimage.color import label2rgb
from skimage.filters import gaussian
import skimage.filters
import skimage.morphology
import pathlib
from typing import Dict, Optional, Tuple
from functools import partial
from scipy.ndimage import find_objects
from concurrent.futures import ThreadPoolExecutor

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
    return gaussian(np.min(timeseq, axis=0))

def subtract_mask_median_bg(seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """for each frame of a time seq, subtract the median intensity in the mask
    
    Arguments:
        seq {np.ndarray} -- time seq (t, y, x)
        mask {np.ndarray} -- mask (y,x)
    
    Returns:
        np.ndarray -- 
    """
    subtracted_frames = map(lambda frame: frame-np.median(frame[mask], seq))
    return np.array(list(subtracted_frames))

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
        frame = dapiseq[i, ...]
        #frame = gaussian(frame)
        th = skimage.filters.threshold_otsu(frame)
        if min_thresh is not None:
            th = max(th, min_thresh)
        cellmask = frame > th
        dilated_mask = skimage.morphology.dilation(cellmask, skimage.morphology.disk(dilate_by))
        bg = ~dilated_mask
        # TODO see whether we observe
        # any clumping and implement distance-transform-based splitting
        # if necessary
        labelled_frame = label(cellmask)
        labels[i, ...] = labelled_frame
        masks[i, ...] = bg
        return(i)

    # using a map and a nested function may look odd here
    # but should allow for replacing map with ThreadPool().map
    # in the future for multithreaded performance. 
    # More elegant suggestions welcome. 
    with ThreadPoolExecutor() as p:
        res = list(p.map(process_frame, range(dapiseq.shape[0])))
    return labels, masks


def create_overlay(int_channels, label_channels, outline_channels):
    # TODO 
    # this is just a placeholder
    # create a generic function to create rgb overlays 
    #
    # may just implement something specific to the task here
    pass

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

    for i in range(seq.shape[1]):
        plt.imshow(np.median(seq[:,i,...], axis=0))
        plt.show()

    # Detect egg chambers
    # take pixel-wise median (over time) to remove moving cells
    bf_median = np.median(seq[:,ch_bf,...], axis=0)
    chamber_labels = detect_circles(bf_median)

    plt.imshow(chamber_labels)
    plt.show()

    # first round of background subtraction
    # apply to all fluorescence channels

    all_channels = tuple(range(seq.shape[1]))
    fluo_channels = tuple(set(all_channels) - set([ch_bf]))

    with ThreadPoolExecutor() as p:
        print("performing initial background correction")
        bg_corr = list(p.map(lambda ch: gaussian(seq[:, ch, ...]) - bg_estimate_initial(seq[:,ch,...]), fluo_channels))
        print(len(bg_corr))
        print(bg_corr[0].shape)

        # single out dapi channel for nuclei segmentation
        print("segmeting nuclei ...")
        bg_corr_dapi = bg_corr[fluo_channels.index(ch_dapi)]
        labels, bg_mask = segment_cells(bg_corr_dapi)

        # TODO add tqdm progress bars for the maps
        
        # refined, per-frame background estimate (for each frame, remove
        # median intensity in background mask).
        correct_frame = lambda frame, mask: frame-np.median(frame[mask])
        correct_ch = lambda ch: np.array(list(map(correct_frame, zip(ch, bg_mask))))
        bg_corr_refined = list(map(correct_ch, bg_corr_dapi))
        


    

    # concatenate
    #    for [each bgcorrected fluorescence channel] + [egg chamber label image]:
    #         calc desired regionprops
    #         rename regionprop columns (prepend_channel)
    p = ('label', 'centroid', 'max_intensity', 'mean_intensity', 'min_intensity', 'area')
    objects_seq = list(map(find_objects, labels)) # we want to re-use the objects for each ch
    measure_frame = lambda l, insty, o: regionprops_table(l, insty, properties=p, objects=o)
    measure_ch = lambda ch: np.array(measure_frame, zip(labels, ch))
    measurements = list(map(measure_ch, bg_corr_refined))
    
    
    return measurements # just an early exit point while debugging
    # add 'timepoint' to each dict

    # add 'metadata' to each dict

    
    # add a metadata column that gives the total size of the mask to which the cell belongs
    # this will allow filtering out cells that were in egg chambers that were not fully in the field of view
    # and for the background mask it records the areas outside of the egg chambers after cropping (small caveat:
    # the chambers are actually larger than the cirlces)

    # combine list of dicts into a pd Dataframe

    # generate output_csv filename from input filename unless provided
    # write csv
    # generate output_overlay filename/s from overlay filename unless provided
    # write image series (png/jpg)
    # return 