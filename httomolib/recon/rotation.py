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
# Created Date: 01 November 2022
# ---------------------------------------------------------------------------
"""Modules for finding the axis of rotation for 180 and 360 degrees scans"""

import numpy as np
import scipy.ndimage as ndi
import pyfftw.interfaces.scipy_fftpack as fft
import logging
from typing import Optional, Literal


__all__ = [
    "find_center_vo",
    "find_center_360",
]


def find_center_vo(
    data: np.ndarray,
    ind: Optional[int] = None,
    cor_initialisation_value: Optional[float] = None,
    smin: int = -100,
    smax: int = 100,
    search_radius: float = 6.0,
    search_step: float = 0.25,
    ratio: float = 0.5,
    drop: int = 20,
) -> float:
    """
    This method is taken from Savu
    """

    if data.ndim == 2:
        data = np.expand_dims(data, 1)
        ind = 0

    if len(data.shape) > 2:
        sino = np.mean(data, axis=1)
    else:
        sino = data[:, ind, :]

    (nrow, ncol) = sino.shape
    dsp_row = 1
    dsp_col = 1
    if ncol > 2000:
        dsp_col = 4
    if nrow > 2000:
        dsp_row = 2
    # Denoising
    sino_csearch = ndi.gaussian_filter(sino, (3, 1), mode="reflect")
    sino_fsearch = ndi.gaussian_filter(sino, (2, 2), mode="reflect")
    sino_dsp = _downsample(sino_csearch, dsp_row, dsp_col)
    fine_srange = max(search_radius, dsp_col)
    off_set = 0.5 * dsp_col if dsp_col > 1 else 0.0
    if cor_initialisation_value is None:
        cor_initialisation_value = (ncol - 1.0) / 2.0
    start_cor = np.int16(np.floor(1.0 * (cor_initialisation_value + smin) / dsp_col))
    stop_cor = np.int16(np.ceil(1.0 * (cor_initialisation_value + smax) / dsp_col))
    raw_cor = _coarse_search(sino_dsp, start_cor, stop_cor, ratio, drop)
    cor = _fine_search(
        sino_fsearch,
        raw_cor * dsp_col + off_set,
        fine_srange,
        search_step,
        ratio,
        drop,
    )
    return cor


def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    cen_row = np.int16(np.ceil(nrow / 2.0) - 1)
    cen_col = np.int16(np.ceil(ncol / 2.0) - 1)
    drop = min(drop, np.int16(np.ceil(0.05 * nrow)))
    mask = np.zeros((nrow, ncol), dtype="float32")
    for i in range(nrow):
        pos = np.int16(np.round(((i - cen_row) * dv / radius) / du))
        (pos1, pos2) = np.clip(np.sort((-pos + cen_col, pos + cen_col)), 0, ncol - 1)
        mask[i, pos1 : pos2 + 1] = 1.0
    mask[cen_row - drop : cen_row + drop + 1, :] = 0.0
    mask[:, cen_col - 1 : cen_col + 2] = 0.0
    return mask


def _coarse_search(sino, start_cor, stop_cor, ratio, drop):
    """
    Coarse search for finding the rotation center.
    """
    (nrow, ncol) = sino.shape
    start_cor, stop_cor = np.sort((start_cor, stop_cor))
    start_cor = np.int16(np.clip(start_cor, 0, ncol - 1))
    stop_cor = np.int16(np.clip(stop_cor, 0, ncol - 1))
    cen_fliplr = (ncol - 1.0) / 2.0
    flip_sino = np.fliplr(sino)
    comp_sino = np.flipud(sino)
    list_cor = np.arange(start_cor, stop_cor + 1.0)
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    sino_sino = np.vstack((sino, flip_sino))
    for i, cor in enumerate(list_cor):
        shift = np.int16(2.0 * (cor - cen_fliplr))
        _sino = sino_sino[nrow:]
        _sino[...] = np.roll(flip_sino, shift, axis=1)
        if shift >= 0:
            _sino[:, :shift] = comp_sino[:, :shift]
        else:
            _sino[:, shift:] = comp_sino[:, shift:]
        list_metric[i] = np.mean(np.abs(np.fft.fftshift(fft.fft2(sino_sino))) * mask)
    minpos = np.argmin(list_metric)
    if minpos == 0:
        error_msg_1 = (
            "!!! WARNING !!! Global minimum is out of "
            "the searching range. Please extend smin"
        )
        logging.warning(error_msg_1)
    if minpos == len(list_metric) - 1:
        error_msg_2 = (
            "!!! WARNING !!! Global minimum is out of "
            "the searching range. Please extend smax"
        )
        logging.warning(error_msg_2)
    rot_centre = list_cor[minpos]
    return rot_centre


def _fine_search(sino, start_cor, search_radius, search_step, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    # Denoising
    (nrow, ncol) = sino.shape
    flip_sino = np.fliplr(sino)
    search_radius = np.clip(np.abs(search_radius), 1, ncol // 10 - 1)
    search_step = np.clip(np.abs(search_step), 0.1, 1.1)
    start_cor = np.clip(start_cor, search_radius, ncol - search_radius - 1)
    cen_fliplr = (ncol - 1.0) / 2.0
    list_cor = start_cor + np.arange(
        -search_radius, search_radius + search_step, search_step
    )
    comp_sino = np.flipud(sino)  # Used to avoid local minima
    list_metric = np.zeros(len(list_cor), dtype=np.float32)
    mask = _create_mask(2 * nrow, ncol, 0.5 * ratio * ncol, drop)
    for i, cor in enumerate(list_cor):
        shift = 2.0 * (cor - cen_fliplr)
        sino_shift = ndi.interpolation.shift(
            flip_sino, (0, shift), order=3, prefilter=True
        )
        if shift >= 0:
            shift_int = np.int16(np.ceil(shift))
            sino_shift[:, :shift_int] = comp_sino[:, :shift_int]
        else:
            shift_int = np.int16(np.floor(shift))
            sino_shift[:, shift_int:] = comp_sino[:, shift_int:]
        mat1 = np.vstack((sino, sino_shift))
        list_metric[i] = np.mean(np.abs(np.fft.fftshift(fft.fft2(mat1))) * mask)
    min_pos = np.argmin(list_metric)
    cor = list_cor[min_pos]
    return cor


def _downsample(image, dsp_fact0, dsp_fact1):
    """Downsample an image by averaging.

    Parameters
    ----------
        image : 2D array.
        dsp_fact0 : downsampling factor along axis 0.
        dsp_fact1 : downsampling factor along axis 1.

    Returns
    ---------
        image_dsp : Downsampled image.
    """
    (height, width) = image.shape
    dsp_fact0 = np.clip(np.int16(dsp_fact0), 1, height // 2)
    dsp_fact1 = np.clip(np.int16(dsp_fact1), 1, width // 2)
    height_dsp = height // dsp_fact0
    width_dsp = width // dsp_fact1
    if dsp_fact0 == 1 and dsp_fact1 == 1:
        image_dsp = image
    else:
        image_dsp = image[0 : dsp_fact0 * height_dsp, 0 : dsp_fact1 * width_dsp]
        image_dsp = (
            image_dsp.reshape(height_dsp, dsp_fact0, width_dsp, dsp_fact1)
            .mean(-1)
            .mean(1)
        )
    return image_dsp


def find_center_360(
    data: np.ndarray,
    ind: Optional[int] = None,
    win_width: int = 10,
    side: Optional[Literal[0, 1]] = None,
    denoise: bool = True,
    norm: bool = False,
    use_overlap: bool = False,
) -> tuple[float, float, Optional[Literal[0, 1]], float]:
    """
    Numpy implementation - for reference in testing the cupy
    production version.
    """
    if data.ndim != 3:
        raise ValueError("A 3D array must be provided")

    # this method works with a 360-degree sinogram.
    if ind is None:
        _sino = data[:, 0, :]
    else:
        _sino = data[:, ind, :]

    (nrow, ncol) = _sino.shape
    nrow_180 = nrow // 2 + 1
    sino_top = _sino[0:nrow_180, :]
    sino_bot = np.fliplr(_sino[-nrow_180:, :])
    (overlap, side, overlap_position) = _find_overlap(
        sino_top, sino_bot, win_width, side, denoise, norm, use_overlap
    )
    if side == 0:
        cor = overlap / 2.0 - 1.0
    else:
        cor = ncol - overlap / 2.0 - 1.0

    return float(cor), float(overlap), side, float(overlap_position)


def _find_overlap(
    mat1, mat2, win_width, side=None, denoise=True, norm=False, use_overlap=False
):
    ncol1 = mat1.shape[1]
    ncol2 = mat2.shape[1]
    win_width = int(np.clip(win_width, 6, min(ncol1, ncol2) // 2))

    if side == 1:
        (list_metric, offset) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=side,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = ncol1 - overlap_position + win_width // 2
    elif side == 0:
        (list_metric, offset) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=side,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        overlap_position = _calculate_curvature(list_metric)[1]
        overlap_position += offset
        overlap = overlap_position + win_width // 2
    else:
        (list_metric1, offset1) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=1,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )
        (list_metric2, offset2) = _search_overlap(
            mat1,
            mat2,
            win_width,
            side=0,
            denoise=denoise,
            norm=norm,
            use_overlap=use_overlap,
        )

        (curvature1, overlap_position1) = _calculate_curvature(list_metric1)
        overlap_position1 += offset1
        (curvature2, overlap_position2) = _calculate_curvature(list_metric2)
        overlap_position2 += offset2

        if curvature1 > curvature2:
            side = 1
            overlap_position = overlap_position1
            overlap = ncol1 - overlap_position + win_width // 2
        else:
            side = 0
            overlap_position = overlap_position2
            overlap = overlap_position + win_width // 2

    return overlap, side, overlap_position


def _search_overlap(
    mat1, mat2, win_width, side, denoise=True, norm=False, use_overlap=False
):
    if denoise is True:
        mat1 = ndi.gaussian_filter(mat1, (2, 2), mode="reflect")
        mat2 = ndi.gaussian_filter(mat2, (2, 2), mode="reflect")
    (nrow1, ncol1) = mat1.shape
    (nrow2, ncol2) = mat2.shape

    if nrow1 != nrow2:
        raise ValueError("Two images are not at the same height!!!")

    win_width = int(np.clip(win_width, 6, min(ncol1, ncol2) // 2 - 1))
    offset = win_width // 2
    win_width = 2 * offset  # Make it even
    ramp_down = np.linspace(1.0, 0.0, win_width, dtype=np.float32)
    ramp_down = ramp_down.reshape((1, ramp_down.size))
    ramp_up = 1.0 - ramp_down

    if side == 1:
        mat2_roi = mat2[:, 0:win_width]
        mat2_roi_wei = mat2_roi * ramp_up
    else:
        mat2_roi = mat2[:, ncol2 - win_width :]
        mat2_roi_wei = mat2_roi * ramp_down

    list_mean2 = np.mean(np.abs(mat2_roi), axis=1, keepdims=True)  # (Nx1)
    list_pos = np.arange(offset, ncol1 - offset)
    num_metric = len(list_pos)
    list_metric = np.empty(num_metric, dtype=np.float32)

    mat1_roi = np.empty((mat1.shape[0], 2 * offset), dtype=np.float32)

    for i, pos in enumerate(list_pos):
        mat1_roi[:] = mat1[:, pos - offset : pos + offset]
        if norm is True:
            list_fact = np.mean(np.abs(mat1_roi), axis=1, keepdims=True)
            np.divide(list_mean2, list_fact, out=list_fact)
            mat1_roi *= list_fact

        if use_overlap is True:
            if side == 1:
                mat_comb = mat1_roi * ramp_down
            else:
                mat_comb = mat1_roi * ramp_up
            mat_comb += mat2_roi_wei
            list_metric[i] = (
                _correlation_metric(mat1_roi, mat2_roi)
                + _correlation_metric(mat1_roi, mat_comb)
                + _correlation_metric(mat2_roi, mat_comb)
            ) / 3.0
        else:
            list_metric[i] = _correlation_metric(mat1_roi, mat2_roi)
    min_metric = np.min(list_metric)
    if min_metric != 0.0:
        list_metric = list_metric / min_metric

    return list_metric, offset


def _calculate_curvature(list_metric):
    radi = 2
    num_metric = list_metric.size
    min_metric_idx = int(np.argmin(list_metric))
    min_pos = int(np.clip(min_metric_idx, radi, num_metric - radi - 1))

    # work mostly on CPU here - we have very small arrays here
    list1 = list_metric[min_pos - radi : min_pos + radi + 1]
    afact1 = np.polyfit(np.arange(0, 2 * radi + 1), list1, 2)[0]
    list2 = list_metric[min_pos - 1 : min_pos + 2]
    (afact2, bfact2, _) = np.polyfit(np.arange(min_pos - 1, min_pos + 2), list2, 2)

    curvature = np.abs(afact1)
    if afact2 != 0.0:
        num = -bfact2 / (2 * afact2)
        if (num >= min_pos - 1) and (num <= min_pos + 1):
            min_pos = num

    return curvature, np.float32(min_pos)


def _correlation_metric(mat1, mat2):
    # pearsonr coefficient
    assert mat1.size == mat2.size, "matrices must be same size"
    X = np.vstack((mat1.reshape((1, mat1.size)), mat2.reshape((1, mat2.size))))
    r = np.corrcoef(X)
    return float(np.abs(1.0 - r[0, 1]))
