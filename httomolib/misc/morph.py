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
# Created Date: 23 March 2023
# ---------------------------------------------------------------------------
"""Module for data type morphing functions"""

import numpy as xp

cupy_run = False

try:
    import cupy as cp

    try:
        cp.cuda.Device(0).compute_capability
        cupy_run = True

    except cp.cuda.runtime.CUDARuntimeError:
        import numpy as np
except ImportError:
    import numpy as np

from typing import Literal

from httomolib.misc.utils import (
    __check_variable_type,
    __check_if_data_3D_array,
    __check_if_data_correct_type,
)

__all__ = [
    "data_reducer",
]


def data_reducer(
    data: xp.ndarray,
    axis: Literal[0, 1, 2] = 0,
    method: Literal["mean", "median"] = "mean",
) -> xp.ndarray:
    """
    Reduce the data along the given axis dimension using the preferred method.

    Parameters
    ----------
    data : xp.ndarray
        3d np or cp array.
    axis : int
        Axis along which the reduction is applied. Defaults to 0.
    method : str
        Selection of the reduction method. Defaults to 'mean'.
    Raises
    ----------
        ValueError: When data is not 3D.
        ValueError: When the method is not mean or median.
        ValueError: Only 0,1,2 values for axes are supported.

    Returns
    ----------
        xp.ndarray: data reduced 3d array where the reduced dimension is equal to one.
    """
    if cupy_run:
        xp = cp.get_array_module(data)

    ### Data and parameters checks ###
    methods_name = "data_reducer"
    __check_if_data_3D_array(data, methods_name)
    __check_if_data_correct_type(
        data,
        accepted_type=["float64", "float32", "uint8", "uint16", "uint32"],
        methods_name=methods_name,
    )
    __check_variable_type(axis, [int], "axis", [0, 1, 2], methods_name)
    __check_variable_type(method, [str], "method", ["mean", "median"], methods_name)
    ###################################
    N, M, Z = data.shape

    if axis == 0:
        if cupy_run:
            reduced_data = xp.empty((1, M, Z), dtype=xp.float32)
        else:
            reduced_data = np.empty((1, M, Z), dtype=np.float32)
        if method == "mean":
            if cupy_run:
                xp.mean(data, axis=axis, dtype=xp.float32, out=reduced_data[0, :, :])
            else:
                np.mean(data, axis=axis, dtype=np.float32, out=reduced_data[0, :, :])
        else:
            if cupy_run:
                xp.median(data, axis=axis, out=reduced_data[0, :, :])
            else:
                np.median(data, axis=axis, out=reduced_data[0, :, :])
    if axis == 1:
        if cupy_run:
            reduced_data = xp.empty((N, 1, Z), dtype=xp.float32)
        else:
            reduced_data = np.empty((N, 1, Z), dtype=np.float32)
        if method == "mean":
            if cupy_run:
                xp.mean(data, axis=axis, dtype=xp.float32, out=reduced_data[:, 0, :])
            else:
                np.mean(data, axis=axis, dtype=np.float32, out=reduced_data[:, 0, :])
        else:
            if cupy_run:
                xp.median(data, axis=axis, out=reduced_data[:, 0, :])
            else:
                np.median(data, axis=axis, out=reduced_data[:, 0, :])
    if axis == 2:
        if cupy_run:
            reduced_data = xp.empty((N, M, 1), dtype=xp.float32)
        else:
            reduced_data = np.empty((N, M, 1), dtype=np.float32)
        if method == "mean":
            if cupy_run:
                xp.mean(data, axis=axis, dtype=xp.float32, out=reduced_data[:, :, 0])
            else:
                np.mean(data, axis=axis, dtype=np.float32, out=reduced_data[:, :, 0])
        else:
            if cupy_run:
                xp.median(data, axis=axis, out=reduced_data[:, :, 0])
            else:
                np.median(data, axis=axis, out=reduced_data[:, :, 0])
    return reduced_data
