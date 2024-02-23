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

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability

    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp
except ImportError:
    import numpy as xp

__all__ = [
    "data_reducer",
]


def data_reducer(data: xp.ndarray, axis: int = 0, method: str = "mean") -> xp.ndarray:
    """Reduce the data along the given dimension by axis number and the method.

    Args:
        data (xp.ndarray): 3d np or cp array.
        axis (int, optional): Axis along which reduction is applied. Defaults to 1.
        method (str, optional): Selection of the reduction method. Defaults to 'mean'.

    Raises:
        ValueError: When data is not 3D.
        ValueError: When the method is not mean or median.
        ValueError: Only 0,1,2 values for axes are supported.

    Returns:
        xp.ndarray: data reduced 3d array where the reduced dimension is equal to one.
    """

    if data.ndim != 3:
        raise ValueError("only 3D data is supported")
    if method not in ["mean", "median"]:
        raise ValueError("Supported methods are mean and median")

    N, M, Z = xp.shape(data)
    if axis == 0:
        reduced_data = xp.empty((1, M, Z), dtype=xp.float32)
        if method == "mean":
            xp.mean(data, axis=axis, dtype=xp.float32, out=reduced_data[axis])
        else:
            xp.median(data, axis=axis, out=reduced_data[axis])
    elif axis == 1:
        reduced_data = xp.empty((N, 1, Z), dtype=xp.float32)
        if method == "mean":
            xp.mean(data, axis=axis, dtype=xp.float32, out=reduced_data[:, 0, :])
        else:
            xp.median(data, axis=axis, out=reduced_data[:, 0, :])
    elif axis == 2:
        reduced_data = xp.empty((N, M, 1), dtype=xp.float32)
        if method == "mean":
            xp.mean(data, axis=axis, dtype=xp.float32, out=reduced_data[:, :, 0])
        else:
            xp.median(data, axis=axis, out=reduced_data[:, :, 0])
    else:
        raise ValueError("Only 0,1,2 values for axes are supported")

    return reduced_data
