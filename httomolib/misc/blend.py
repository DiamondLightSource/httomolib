#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Copyright 2023 Diamond Light Source Ltd.
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
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>
# Created Date: 30 April 2026
# ---------------------------------------------------------------------------
"""Module for data blending functions"""

import numpy as np
from typing import Optional

from httomolib.misc.utils import (
    __check_variable_type,
    __check_if_data_3D_array,
    __check_if_data_correct_type,
)

__all__ = [
    "seam_blend_stitched_data",
]


def seam_blend_stitched_data(
    data: np.ndarray,
    seam_index: Optional[int] = None,
    blending_width: Optional[int] = None,
    path_to_stiched_params_file: Optional[str] = None,
    shift_seam_index: int = 0,
) -> np.ndarray:
    """
    Function blends the seam present in the stitched projection data. It uses the redundant by blending_width data present on both sides of the seam.
    Used in HTTomo for seamless stitching of datasets coming from two different PCO cameras.

    Parameters
    ----------
    data : np.ndarray
        3d array of the stitched data, assuming the following axis ["angles", "detY", "detX"].
    seam_index : Optional, int
        The horizontal index of the seam along the 'detX' axis. If None and 'path_to_stiched_params_file' is provided, it will be taken from the file, otherwise middle of the horizontal axis.
    blending_width : Optional, int
        The area for symmetric blending (e.g. with the ramp filter) around the seam position (seam_index) of the stitched data. If None and 'path_to_stiched_params_file' is provided, it will be taken from the file, otherwise 0.
    path_to_stiched_params_file : Optional, str
            Path to the text file with the stiching parameters. If provided 'seam_index' and 'blending_width' parameters will be overridden by the ones provided in the file.
    shift_seam_index : int
        performs a shift of the seam index with seam_index - shift_seam_index. This is purely an HTTomo related feature and should be ignored by users. 
    Raises
    ----------
        ValueError: When data is not 3D.

    Returns
    ----------
        np.ndarray: stitched data without the seam.
    """
    ### Data and parameters checks ###
    methods_name = "seam_blender"
    __check_if_data_3D_array(data, methods_name)
    __check_if_data_correct_type(
        data,
        accepted_type=["float64", "float32", "uint8", "uint16", "uint32"],
        methods_name=methods_name,
    )
    __check_variable_type(seam_index, [int, type(None)], "seam_index", [], methods_name)
    __check_variable_type(
        blending_width, [int, type(None)], "blending_width", [], methods_name
    )
    __check_variable_type(
        path_to_stiched_params_file,
        [str, type(None)],
        "path_to_stiched_params_file",
        [],
        methods_name,
    )
    ###################################

    angles_dim, detY, detX = data.shape

    if path_to_stiched_params_file is not None:
        params = {}
        with open(path_to_stiched_params_file) as f:
            for line in f:
                key, value = line.split()
                params[key] = int(value)

        blending_width = params.get("blending_width")
        seam_index = params.get("seam_index")

    if blending_width is None:
        blending_width = 0
    if seam_index is None:
        seam_index = int(detX // 2)

    blending_width *= 2
    seam_index -= shift_seam_index

    if seam_index >= detX - blending_width:
        err_str = f"Seam index given as '{seam_index}' must be smaller than the horizontal dimension of the data '{detX}' minus blending width."
        raise ValueError(err_str)

    # Split regions of data
    left_part = data[:, :, 0 : (seam_index - blending_width)]
    right_part = data[:, :, (seam_index + blending_width) : :]

    # Overlap regions
    left_overlap = data[:, :, (seam_index - blending_width) : seam_index]
    right_overlap = data[:, :, seam_index : (seam_index + blending_width)]

    # Create ramp weights (0 → 1)
    ramp = np.float32(np.linspace(0, 1, blending_width))
    ramp = np.tile(ramp, (detY, 1))

    # Blend
    blended_overlap = (1 - ramp) * left_overlap + ramp * right_overlap

    return np.dstack([left_part, blended_overlap, right_part])
