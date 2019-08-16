import numpy as np
import pathlib
import tifffile
import matplotlib.pyplot as plt
from stabilize_sequence import stabilize_getshifts, stabilize_apply_shifts, stabilize_crop_to_overlap
from detect_and_label_chambers import detect_circles
from skimage.measure import label, regionprops_table
from skimage.color import label2rgb
from skimage.filters import gaussian, threshold_otsu, threshold_li
from skimage.exposure import rescale_intensity
from skimage.morphology import dilation, erosion, disk
from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from typing import Dict, Optional, Tuple
from functools import partial
from scipy.ndimage import find_objects
from concurrent.futures import ThreadPoolExecutor
from itertools import product, repeat
import pandas as pd
from cv2 import VideoWriter, VideoWriter_fourcc


def contour_from_label_single_frame(label: np.ndarray) -> np.ndarray: 
    """Creates a boolean mask with label contours

    Arguments:
        label {np.ndarray} -- label image as ndarray

    Returns:
        np.ndarray -- boolean area in which the contours around labels are true

    Note: these contours extend into and out of the label, i.e. they are
    not useful for very thin objects.
    """
    return (dilation(label,disk(1))-erosion(label,disk(1)))>0


def get_random_colors(n_entries=256, zero="white"):
    tmp = np.random.rand(n_entries,3)
    # set first entry to white or black if requested:
    if zero == "white":
        tmp[0,:] = (1, 1, 1)
    elif zero == "black":
        tmp[0,:] = (0, 0, 0)
    return tmp


def create_overlay_sequence(label_seq: np.ndarray, im_seq: np.ndarray, chamber_labels: Optional[np.ndarray]=None) -> np.ndarray:
    """Creates a sequence of overlays with cells as colored labels on im_seq. Chambers as outlines.
    
    Arguments:
        label_seq {np.ndarray} -- sequence of object label images [nt, ny, nx]
        im_seq {np.ndarray} -- sequence of image on which to overlay the labels [nt, ny, nx]
    
    Keyword Arguments:
        chamber_labels {Optional[np.ndarray]} -- label image with egg_chambers(for outlines) (default: {None})
    
    Returns:
        np.ndarray -- [nt, ny, nx, 3] color image sequence with overlays

        
    """
    def label_cells(l):
        label,im = l
        rcolors = get_random_colors()
        return label2rgb(label, im, bg_label=0, bg_color=None,alpha=0.3, colors=rcolors, kind='overlay')

    with ThreadPoolExecutor() as p:
        overlay = np.array(list(p.map(label_cells, zip(label_seq, im_seq))))

    if chamber_labels is not None:
        c = contour_from_label_single_frame(chamber_labels)
        overlay[:,c,:] = (1.0,1.0,1.0)

    return overlay


def save_seq_as_avi(seq: np.ndarray, filename: str, fps: int=3):
    """Saves seq as an avi movie
    
    Arguments:
        seq {np.ndarray} -- numpy nd.array float type with range 0.0,1.0 and shape [nt, ny, nx, 3]  
        filename {str} -- filename of the output avi file
    
    Keyword Arguments:
        fps {int} -- frames per second (default: {3})
    """
    height, width = seq[0,...,0].shape
    size=(width,height)
    out = VideoWriter(filename, VideoWriter_fourcc(*'DIVX'), fps, size)
    for im in seq:
        out.write((255*im[:,:,:]).astype(np.uint8))
    out.release()

def process_folder(folder: pathlib.Path, basepath_tif: pathlib.Path, basepath_nd2: pathlib.Path):
    """Finds all the .tif files in a folder and triggers the processing
    
    Parameters
    ----------
    folder : pathlib.Path
        relative Path to the folder, below basepath
    basepath_tif : basepath for the tif files
    basepath_nd2 : basepath for the corresponding nd2 files (from which the tif files
                were created using bfconvert)

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


def bg_estimate_initial(timeseq: np.ndarray, sigma=3):
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
    return gaussian(np.min(timeseq, axis=0), sigma)

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
        # frame = gaussian(frame, 3)
        # plt.imshow(frame)
        # plt.show()
        th = threshold_otsu(frame)
        # alternative suggested by @jni
        #initial_guess= np.quantile(frame, 0.99)
        #th  = threshold_li(frame, initial_guess=initial_guess)
        if min_thresh is not None:
            th = max(th, min_thresh)
        # print(f"threshold for frame {i} is {th}") # debug code
        cellmask = frame > th 
        #plt.imshow(cellmask) # debug code
        #plt.show()
        dilated_mask = dilation(cellmask, disk(dilate_by))
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

def process_measurements(measurements, fluo_channels):
    """helper function that combines measurements from 
    individual time points, adds a timepoint column and 
    prepends a string identifying the channel where a
    measurement was taken.
    Reslt is a list of dataframes, one per channel.
    """
    outer = []
    for ch_number, ch_measurements in zip(fluo_channels, measurements):
        inner = []
        for i,m in enumerate(ch_measurements):
            tmp = pd.DataFrame(m)
            tmp["timepoint"] = i
            inner.append(tmp)
        ch_df = pd.concat(inner)
        ch_df = ch_df.rename(columns = lambda cname: cname  if cname in  ("timepoint", "label") else f"ch_{ch_number}_{cname}")
        outer.append(ch_df)
    return outer # Should the data frame merging happen here ? (Currently in calling function)
    
def process_timeseries(input_tif: str, metadata, output_csv: Optional[str]=None, output_overlay: Optional[str]=None):
    """Process a single time series, write measurements to csv and overlay to MP4 
       
       Parameters
        ----------
       input_tif: [str], str with path to input tifffile.
       metadata: [dict] ? metadate from the corresponding nd2 file
       output_csv : [str], optional
       file name for the output csv file, will be generated automatically based on input_tif if None
       output_overlay : [str], optional
       file name for the segmentation results,  will be generated automatically based on input_tif if None
    """

    #########################
    # Call actual processing
    #########################
    print("Drift correction, circle detection and cell segmentation")
    results = segment_and_measure_timeseries(input_tif, {})

    ####################
    # Save measurements
    ####################
    print("Saving csv")
    if output_csv is None:
        output_csv=input_tif.replace(".tif",".csv")
    df = results["measurements"]
    print(f"Saving measurements as {output_csv}")
    df.to_csv(output_csv) #TODO catch PermissionError  and what else?

    ######################################
    # Create and save segemenation overlay
    ######################################
    if output_overlay is None:
        output_overlay=input_tif.replace(".tif",".mp4")
    
    # rescale intensities
    # TODO: extract from metadata and remove this section
    bf_ch=1 
    dapi_ch=0
    bf_seq = rescale_intensity(results["seq"][:,bf_ch,...] ,out_range=np.uint8).astype(np.uint8)
    dapi_seq = rescale_intensity(results["seq"][:,dapi_ch,...], out_range=np.uint8).astype(np.uint8)
    label_seq = results["cell_label_seq"]
    egg_chambers = results["chamber_labels"]
    overlay = create_overlay_sequence(label_seq, bf_seq, egg_chambers)
    save_seq_as_avi(overlay, output_overlay)

    return results # pass results up, useful when using notebook

def segment_and_measure_timeseries(input_tif: str, metadata: Dict, sigma: int=3):
    """Provided an input image sequence correspoding to a time series at a given position perform
    egg chamber detection/segmentation/ 

    
    Parameters
    ----------
    input_tif : str
        input filename
    metadata : Dict
        dictionary that contains metadata which should be added to output table
   
    sigma: [int]: 
        passed on to gaussian smoothing
    """
    
    seq = tifffile.imread(input_tif)

    if seq.ndim == 5: # we sometimes get duplicate dimensions from bfconvert
        seq = seq[0, ... ]
    assert seq.ndim == 4

    ##################
    # TODO: REMOVE !!!!!!!!!!! 
    # for debugging only ... create an additional fluorescence channel by copying dapi
    # this is because I don't have a suitbale multi-channel image on my laptop
    #seq = seq[:40,...]
    shape = list(seq.shape)
    shape[1] = shape[1] + 1
    tmp = np.zeros(shape, dtype = seq.dtype)
    for i in range(shape[0]):
        for j in (0,1):
            tmp[i, j, ... ] = seq[i,j,...]
        tmp[i,2] = seq[i,1,...]
    seq=tmp.copy()

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

    ##########################################
    # Detect egg chambers  
    #
    # clean up brightfield image first by
    # taking the pixel-wise median (over time) 
    # to remove moving cells
    bf_median = np.median(seq[:,ch_bf,...], axis=0)
    chamber_labels = detect_circles(bf_median)

    plt.imshow(chamber_labels)
    plt.show()

    # first round of background subtraction
    # apply to all fluorescence channels

    all_channels = tuple(range(seq.shape[1]))
    fluo_channels = tuple(set(all_channels) - set([ch_bf]))

    print(f"all channes {all_channels}")
    print(f"fluo channels {fluo_channels}")

    with ThreadPoolExecutor() as p:
        #################################
        #  Initial Background Estimation
        # 
        print("performing initial background correction")
        bg_corr = list(p.map(lambda ch: gaussian(seq[:, ch, ...],sigma) - bg_estimate_initial(seq[:,ch,...], sigma), fluo_channels))

        ################################
        # Segment Nuclei in DAPI channel
        #
        print("segmeting nuclei ...")
        bg_corr_dapi = bg_corr[fluo_channels.index(ch_dapi)]
        #for i, tmpim  in enumerate(bg_corr_dapi):
        #    print(f"bg_corr_dapi {i}")
        #    plt.imshow(tmpim)
        #    plt.show()
        labels, bg_mask = segment_cells(bg_corr_dapi)

        #################################################
        # Per-frame refinemant of background subtraction, 
        # 
        # (for each frame, remove
        # median intensity in background mask).
        # After this, the median intensity of the background
        # is 0 by construction.
        print("refining background correction estimation")
        def correct_frame(intuple):
            frame, mask = intuple
            bgval = np.median(frame[mask])
            return frame - bgval
        correct_ch = lambda ch: np.array(list(p.map(correct_frame, zip(ch, bg_mask))))
        bg_corr_refined = list(p.map(correct_ch, bg_corr))
        
        ############################################################################
        # Feature extraction : measure intensity statistics in fluorescence channels
        # 
        # TODO: add other metrics as needed (e.g. std deviation)
        print("Calculating intensity statistics")
        props_intensity = ('label', 'centroid', 'max_intensity', 'mean_intensity', 'min_intensity', 'area')
        objects_seq = list(map(find_objects, labels)) # we want to re-use the objects for each ch
        def measure_frame(intuple, props):
            l, intensity, obs = intuple
            return regionprops_table(l, intensity, properties=props,objects=obs)

        measure_frame_int = partial(measure_frame, props=props_intensity)
        measure_ch = lambda ch: np.array(list(p.map(measure_frame_int, zip(labels, ch, objects_seq))))
        measurements_intensity = list(p.map(measure_ch, bg_corr_refined))
        tmp = process_measurements(measurements_intensity, fluo_channels) # modify column names to reflect channel
        
        ######################################
        # Merge tables for different channels
        # 
        merged = pd.merge(tmp[0], tmp[1], how = "outer", on=('label','timepoint'))
        
        ################################################################
        # Determine location of each cell (egg chamber or background)
        #
        # In order to figure out which egg chamber a cell belongs to we simply measure the 
        # intensity for each cell label in the chamber label image.  
        # Max intensity will do here, we only need the single readout
        props_mask = ('label', 'max_intensity')
        measure_frame_mask = partial(measure_frame, props=props_mask)
        measure_mask = np.array(list(p.map(measure_frame_mask, zip(labels, repeat(chamber_labels) , objects_seq))))
    
    # add egg chamber label as column to merged data frame
    merged = merged.assign(eggchamber=pd.concat(map(pd.DataFrame, measure_mask))["max_intensity"].astype(np.int).values)
    merged["filename"] = input_tif
    # add area in pixels for each egg chamber
    # as we know the approximate radius and area of each chamber
    # this will allow filtering for egg chambers that are only partially in
    # the field of view
    chamber_sizes = {} # build a dictionary to help translate between chamber label and area
    for chamber in merged["eggchamber"].unique():   
        area_pixels=np.sum(chamber_labels==chamber)
        chamber_sizes[chamber]=area_pixels
    merged["chamberarea"] = merged["eggchamber"].apply(lambda x: chamber_sizes[x])

    # TODO:
    # add 'metadata' to each dict

    result = {"measurements" : merged, "seq" : seq, "cell_label_seq": labels, "chamber_labels": chamber_labels }
    return result
 
