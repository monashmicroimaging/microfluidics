import sys
import numpy as np
import vispy.color
import napari
from dask import array as da, delayed
from pims import ND2_Reader as nd2


def nd2dask(filename):
    frames = nd2(filename)
    num_channels = frames.sizes['c']
    num_expts = frames.sizes['m']
    num_timepoints = frames.sizes['t']
    channel_colors = [frames.metadata[f'plane_{i}']['rgb_value']
                      for i in range(num_channels)]
    channel_names = [frames.metadata[f'plane_{i}']['name']
                     for i in range(num_channels)]
    frames.bundle_axes = 'yx'
    frames.iter_axes = 'cmt'
    nframes = len(frames)
    arr_list = [da.from_delayed(delayed(frames.__getitem__)(i),
                                shape=frames.frame_shape,
                                dtype=frames.pixel_type)
                for i in range(nframes)]
    array = da.stack(arr_list, axis=0).reshape((num_channels, num_expts,
                                                num_timepoints) +
                                               frames.frame_shape)
    return list(zip(channel_names, channel_colors, array))


with napari.util.app_context():
    v = napari.ViewerApp()

    for name, color, image in nd2dask(sys.argv[1]):
        layer = v.add_image(image, name=name, multichannel=False)
        layer.blending = 'additive'
        cmap = vispy.color.Colormap([[0., 0., 0.], color])
        layer.colormap = cmap
