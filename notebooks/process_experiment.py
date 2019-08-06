import numpy as np
import tifffile
import matplotlib.pyplot as plt
from stabilize_sequence import stabilize_getshifts, stabilize_apply_shifts, stabilize_crop_to_overlap
from detect_and_label_chambers import detect_circles
from skimage.color import label2rgb
import skimage.filters
import skimage.morphology
import pathlib
from typing import Dict, Optional

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


def process_timeseries(input_tif: str, metadata: Dict, output_csv = Optional[str], output_overlay = Optional[str]):
    #
    # read time series using tifffile
    # determine channels based on metadata
    # reshape array if necessary (according to metadata np.)
    # stabilize time series
    # crop to overlap
    # find circles in cropped -> label image of egg chambers
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


