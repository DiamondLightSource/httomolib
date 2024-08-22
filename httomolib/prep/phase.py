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
# Created By  : Tomography Team at DLS <scientificsoftware@diamond.ac.uk>
# Created Date: 23 June 2023
# ---------------------------------------------------------------------------
"""Modules for phase retrieval and phase-contrast enhancement"""

import math
from typing import Tuple
import numpy as np
import scipy
from scipy.fft import fft, fft2, ifft2, fftshift

__all__ = [
    "paganin_filter",
]

# Define constants used in phase retrieval method
BOLTZMANN_CONSTANT = 1.3806488e-16  # [erg/k]
SPEED_OF_LIGHT = 299792458e2  # [cm/s]
PI = 3.14159265359
PLANCK_CONSTANT = 6.58211928e-19  # [keV*s]


## %%%%%%%%%%%%%%%%%%%%% Retrieve phase / Paganin %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##
# Adaptation with some corrections of retrieve_phase (Paganin filter) from TomoPy
def paganin_filter(
    tomo: np.ndarray,
    pixel_size: float = 1e-4,
    dist: float = 50.0,
    energy: float = 53.0,
    alpha: float = 1e-3,
) -> np.ndarray:
    """
    Perform single-material phase retrieval from flats/darks corrected tomographic measurements.

    Parameters
    ----------
    tomo : np.ndarray
        3D array of f/d corrected tomographic projections.
    pixel_size : float, optional
        Detector pixel size in cm.
    dist : float, optional
        Propagation distance of the wavefront in cm.
    energy : float, optional
        Energy of incident wave in keV.
    alpha : float, optional
        Regularization parameter, the ratio of delta/beta. Larger values lead to more smoothing.

    Returns
    -------
    np.ndarray
        The 3D array of Paganin phase-filtered projection images.
    """

    # Check the input data is valid
    if tomo.ndim != 3:
        raise ValueError(
            f"Invalid number of dimensions in data: {tomo.ndim},"
            " please provide a stack of 2D projections."
        )

    dz_orig, dy_orig, dx_orig = np.shape(tomo)

    # Perform padding to the power of 2 as FFT is O(n*log(n)) complexity
    # NOTE: Need to convert to float32 as FFT produces complex128 array from uint16
    # TODO: adding other options of padding?
    padded_tomo, pad_tup = _pad_projections_to_second_power(np.float32(tomo))

    dz, dy, dx = np.shape(padded_tomo)

    # 3D FFT of tomo data
    fft_tomo = fft2(padded_tomo, axes=(-2, -1), overwrite_x=True)

    # Compute the reciprocal grid.
    w2 = _reciprocal_grid(pixel_size, (dy, dx))

    # Build filter in the Fourier space.
    phase_filter = fftshift(_paganin_filter_factor(energy, dist, alpha, w2))
    phase_filter = phase_filter / phase_filter.max()  # normalisation

    # Apply filter and take inverse FFT
    ifft_filtered_tomo = (
        ifft2(phase_filter * fft_tomo, axes=(-2, -1), overwrite_x=True)
    ).real

    # slicing indices for cropping
    slc_indices = (
        slice(pad_tup[0][0], pad_tup[0][0] + dz_orig, 1),
        slice(pad_tup[1][0], pad_tup[1][0] + dy_orig, 1),
        slice(pad_tup[2][0], pad_tup[2][0] + dx_orig, 1),
    )

    # crop the padded filtered data:
    tomo = ifft_filtered_tomo[slc_indices]

    # taking the negative log
    # tomo = -np.log(tomo)/(4*PI/_wavelength(energy))
    # as implemented in TomoPy (no scaling)
    return -np.log(tomo)


def _shift_bit_length(x):
    return 1 << (x - 1).bit_length()


def _pad_projections_to_second_power(tomo: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Performs padding of each projection to the next power of 2.
    If the shape is not even we also care of that before padding.

    Parameters
    ----------
    tomo : cp.ndarray
        3d projection data

    Returns
    -------
    ndarray: padded 3d projection data
    tuple: a tuple with padding dimensions
    """
    full_shape_tomo = np.shape(tomo)

    pad_tup = []
    for index, element in enumerate(full_shape_tomo):
        if index == 0:
            pad_width = (0, 0)  # do not pad the slicing dim
        else:
            diff = _shift_bit_length(element + 1) - element
            if element % 2 == 0:
                pad_width = diff // 2
                pad_width = (pad_width, pad_width)
            else:
                # need an uneven padding for odd-number lengths
                left_pad = diff // 2
                right_pad = diff - left_pad
                pad_width = (left_pad, right_pad)

        pad_tup.append(pad_width)

    padded_tomo = np.pad(tomo, tuple(pad_tup), "edge")

    return padded_tomo, pad_tup


def _wavelength(energy):
    return 2 * PI * PLANCK_CONSTANT * SPEED_OF_LIGHT / energy


def _paganin_filter_factor(energy, dist, alpha, w2):
    # Alpha represents the ratio of delta/beta.
    # return 1 / (1 + (dist * alpha * _wavelength(energy) * w2/(4*PI)))
    # as implemented in TomoPy
    return 1 / (_wavelength(energy) * dist * w2 / (4 * PI) + alpha)


def _reciprocal_grid(pixel_size, shape_proj):
    """
    Calculate 2d reciprocal grid.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    shape_proj : tuple
        Sizes of the reciprocal grid along x and y axes.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    # Sampling in reciprocal space.
    indx = _reciprocal_coord(pixel_size, shape_proj[0])
    indy = _reciprocal_coord(pixel_size, shape_proj[1])
    np.square(indx, out=indx)
    np.square(indy, out=indy)
    return np.add.outer(indx, indy)


def _reciprocal_coord(pixel_size, num_grid):
    """
    Calculate reciprocal grid coordinates for a given pixel size
    and discretization.

    Parameters
    ----------
    pixel_size : float
        Detector pixel size in cm.
    num_grid : int
        Size of the reciprocal grid.

    Returns
    -------
    ndarray
        Grid coordinates.
    """
    n = num_grid - 1
    rc = np.arange(-n, num_grid, 2, dtype=np.float32)
    rc *= 2 * PI / (n * pixel_size)
    return rc


## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ##
