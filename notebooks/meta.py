import os
import re
import pims


PATTERN = r'(?P<base>.*)(?P<series>_s(?P<seriesid>\d+))\.(?P<extension>tif)'


def channel_metadata(series_filename,
        *,
        scratch_prefix='/scratch/dk49',
        project_prefix='/projects/dk49',
        return_timepoints=False):
    """Get metadata from a series filename."""
    series_filename = os.path.abspath(series_filename)
    original_filename = series_filename.replace(scratch_prefix,
                                                project_prefix)
    match = re.match(PATTERN, original_filename)
    original_filename = match.group('base') + '.nd2'
    image = pims.open(original_filename)
    image.iter_axes = 't'
    series_id = int(match.group('seriesid'))
    im.default_coords['m'] = series_id
    channel_keys = sorted(k for k in im.metadata.keys()
                          if k.startswith('plane_'))
    channel_info = [{k: im.metadata[chk][k]
                     for k in ['name', 'emission_nm', 'rgb_value']}
                    for chk in channel_keys]
    stage_coords = {k: im[0].metadata[k] for k in ['x_um', 'y_um']}
    if return_timepoints:
        n_timepoints = len(im)
        timepoints_ms = [im[i].metadata['t_ms'] for i in range(n_timepoints)]
    else:
        timepoints_ms = None
    return series_id, stage_coords, channel_info, timepoints_ms
