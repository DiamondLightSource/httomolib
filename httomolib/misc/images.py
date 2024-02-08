#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2022 Diamond Light Source Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------
# Created By  : <scientificsoftware@diamond.ac.uk>
# Created Date: 27/October/2022
# ---------------------------------------------------------------------------
""" Module for loading/saving images """

import os
import pathlib
from typing import Optional, Union

import numpy as np
from numpy import ndarray
from PIL import Image
from skimage import exposure

__all__ = [
    "save_to_images",
]

def save_to_images(
    data: ndarray,
    out_dir: Union[str, os.PathLike],
    subfolder_name: str = "images",
    axis: int = 1,
    file_format: str = "tif",
    bits: int = 8,
    perc_range_min: float = 0.0,
    perc_range_max: float = 100.0,
    jpeg_quality: int = 95,
    glob_stats: Optional[tuple] = None,
    offset: int = 0,
):
    """
    Saves data as 2D images. If the data type is not already one of the integer types
    uint8, uint16, or uint32, the data is rescaled and converted first.
    Otherwise it leaves the data as is.

    Parameters
    ----------
    data : np.ndarray
        Required input NumPy ndarray.
    out_dir : str
        The main output directory for images.
    subfolder_name : str, optional
        Subfolder name within the main output directory.
        Defaults to 'images'.
    axis : int, optional
        Specify the axis to use to slice the data (if `data` is a 3D array).
    file_format : str, optional
        Specify the file format to use, e.g. "png", "jpeg", or "tif".
        Defaults to "tif".
    bits : int, optional
        Specify the number of bits to use (8, 16, or 32-bit).
    perc_range_min: float, optional
        lower cutoff point: min + perc_range_min * (max-min)/100
        Defaults to 0.0
    perc_range_max: float, optional
        upper cutoff point: min + perc_range_max * (max-min)/100
        Defaults to 100.0
    jpeg_quality : int, optional
        Specify the quality of the jpeg image.
    glob_stats: tuple, optional
        Global statistics of the input data in a tuple format: (min, max, mean, total_elements_number).
        If None, then it will be calculated based on the input data.
    offest: int, optional
        The offset to start file indexing from, e.g. if offset is 100, images will start at
        00100.tif. This is used when executed in parallel context and only partial data is 
        passed in this run. 
    """
    if data.dtype in [np.uint8, np.uint16, np.uint32]:
        # do not touch the data if it's already in integers, but set the bits in order to 
        # create the right folder
        bits = data.dtype.itemsize * 8
    elif bits not in [8, 16, 32]:
            bits = 32
            print(
                "The selected bit type %s is not available, "
                "resetting to 32 bit \n" % str(bits)
            )

    # create the output folder
    subsubfolder_name = f"images{str(bits)}bit_{str(file_format)}"
    path_to_images_dir = pathlib.Path(out_dir) / subfolder_name / subsubfolder_name
    path_to_images_dir.mkdir(parents=True, exist_ok=True)

    do_rescale = False
    if data.dtype not in [np.uint8, np.uint16, np.uint32]:
        do_rescale = True
        
        
        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)
        
        if glob_stats is None or glob_stats is False:
            min_percentile = np.nanpercentile(data, perc_range_min)
            max_percentile = np.nanpercentile(data, perc_range_max)
        else:
            # calculate the range here based on global max and min
            range_intensity = glob_stats[1] - glob_stats[0]
            min_percentile = (perc_range_min * (range_intensity) / 100) + glob_stats[0]
            max_percentile = (perc_range_max * (range_intensity) / 100) + glob_stats[0]

    if data.ndim == 3:
        slice_dim_size = np.shape(data)[axis]
        for idx in range(slice_dim_size):
            
            filename = f"{idx + offset:05d}.{file_format}"
            filepath = os.path.join(path_to_images_dir, f"{filename}")
            # note: data.take call is far more time consuming
            if axis == 0:
                d = data[idx, :, :] 
            elif axis == 1:
                d = data[:, idx, :] 
            else:
                d = data[:, :, idx]
                
            if do_rescale:
                d = _rescale_2d(d, bits, min_percentile, max_percentile)
            
            Image.fromarray(d).save(filepath, quality=jpeg_quality)

    else:
        filename = f"{1:05d}.{file_format}"
        filepath = os.path.join(path_to_images_dir, f"{filename}")
        if do_rescale:
                data = _rescale_2d(data, bits, min_percentile, max_percentile)
        Image.fromarray(data).save(filepath, quality=jpeg_quality)


def _rescale_2d(d: np.ndarray, bits: int, min_percentile, max_percentile):
    if bits == 8:
        d = exposure.rescale_intensity(
            d, in_range=(min_percentile, max_percentile), out_range=(0, 255)
        ).astype(np.uint8)

    elif bits == 16:
        d = exposure.rescale_intensity(
            d, in_range=(min_percentile, max_percentile), out_range=(0, 65535)
        ).astype(np.uint16)

    else:
        d = exposure.rescale_intensity(
            d, in_range=(min_percentile, max_percentile), out_range=(min_percentile, max_percentile)
        ).astype(np.uint32)
        
    return d