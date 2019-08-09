import os
import re
import pims


PATTERN = r'(?P<base>.*)(?P<series>_s(?P<seriesid>\d+))\.(?P<extension>tif)'


def channel_metadata(series_filename,
        *,
        scratch_prefix='/scratch/dk49',
        project_prefix='/projects/dk49',
        return_timepoints=False):
    """Get metadata from a series filename.
    
    Parameters
    ----------
    series_filename : str
        The filename of the series to process.
    scratch_prefix : str, optional
        The base location of the tiff series in /scratch.
    project_prefix : str, optional
        The base location of the nd2 files in /projects.
    return_timepoints : bool, optional
        Whether to return the timestamps in milliseconds. This *could*
        crash the process depending on the nd2 file.

    Returns
    -------
    series_id : int
        The series ID within the original file.
    stage_coords : dict with keys 'x_um' and 'y_um'.
        The stage coordinates for this series.
    channel_info : list of dict with keys 'name', 'emission_nm', 'rgb_value'
        The information for each channel in the series.
    timepoints_ms : list of float, or None
        The timepoint of each image in the series, in ms. Is None if
        ``return_timepoints`` is False.

    Examples
    --------
    >>> filename = ('/scratch/dk49/eggs/2018_MarchtoMay Microfluidics '
    ...             'rawdata/20180503 JenEggLinkerCurve_s10.tif')
    >>> seriesid, stage_coords, channel_info, _ = channel_metadata(filename)
    >>> seriesid
    10
    >>> n_channels = len(channel_info)
    >>> n_channels
    2
    >>> channel_info[0]['name']
    'Q-DAPI'
    >>> channel_info[1]
    {'name': 'BF', 'emission_nm': 640.0, 'rgb_value': (1.0, 1.0, 1.0)}
    """
    series_filename = os.path.abspath(series_filename)
    original_filename = series_filename.replace(scratch_prefix,
                                                project_prefix)
    match = re.match(PATTERN, original_filename)
    original_filename = match.group('base') + '.nd2'
    image = pims.open(original_filename)
    image.iter_axes = 't'
    series_id = int(match.group('seriesid'))
    im.default_coords['m'] = series_id
    channel_keys = [f'plane_{i}' for i in range(image.metadata['plane_count'])]
    channel_info = [{k: image.metadata[chk][k]
                     for k in ['name', 'emission_nm', 'rgb_value']}
                    for chk in channel_keys]
    stage_coords = {k: image[0].metadata[k] for k in ['x_um', 'y_um']}
    if return_timepoints:
        n_timepoints = len(image)
        timepoints_ms = [image[i].metadata['t_ms'] for i in range(n_timepoints)]
    else:
        timepoints_ms = None
    return series_id, stage_coords, channel_info, timepoints_ms
