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
# Created Date: 17 November 2025
# ---------------------------------------------------------------------------
"""Various utilities for data inspection and correction"""

import numpy as np
from typing import Optional

from httomolib.core.modules import (
    count_zeros_16bit_data_C,
    count_zeros_32bit_float_data_C    
)

__all__ = [
    "data_checker",
]

def data_checker(data: np.ndarray, infsnans_correct: bool= True, zeros_warning: bool = False, data_to_method_name: Optional[str] = None, verbosity: bool = True) -> np.ndarray:
    """Function that performs checks on input data to ensure its validity, performs corrections and prints the warnings.
    Currently it checks for the presence of Infs and NaNs in the data and corrects them.

    Parameters
    ----------
    data : np.ndarray
        Numpy array either float32 or uint16 data type.
    infsnans_correct: bool
        Perform correction of NaNs and Infs if they are present in the data.
    zeros_warning: bool
        Count the number of zeros in the data and produce a warning if more half of the data are zeros.
    verbosity : bool
        Print the warnings.
    data_to_method_name : str, optional.
        Method's name for which the tested data is intended. This is for printing purposes when the method runs in HTTomo.

    Returns
    -------
    np.ndarray
        Returns corrected Numpy array.
    """
    if data.dtype not in ["uint16", "float32"]:
        raise ValueError(
            "The input data of `uint16` and `float32` data types is accepted only."
        )

    if infsnans_correct:
        data = __naninfs_check(data, verbosity=verbosity, method_name=data_to_method_name)\
    
    if zeros_warning:
        __zeros_check(data, verbosity=verbosity, percentage_threshold = 50, method_name=data_to_method_name)

    return data


def __naninfs_check(
    data: np.ndarray,
    verbosity: bool = True,
    method_name: Optional[str] = None,
) -> np.ndarray:
    """
    This function corrects for NaN's, +-Inf's in the input data, corrects it and then prints the warnings if verbosity is enabled.
    """

    if not np.all(np.isfinite(data)):
        np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if verbosity:
            print(
                f"Warning!!! Input data to method: {method_name} contains Inf's or/and NaN's. This will be corrected but it is recommended to check the validity of input to the method."
            )

    return data

def __zeros_check(
    data: np.ndarray,
    verbosity: bool,
    percentage_threshold: float,
    method_name: Optional[str] = None,
) -> int:
    """
    This function finds all zeros present in the data. If the amount of zeros is larger than percentage_threshold it prints the warning.
    """
    nonzero_elements_total = 1
    for tot_elements_mult in data.shape:
        nonzero_elements_total *= tot_elements_mult

    if data.dtype == "uint16":
        zeros_counted = count_zeros_16bit_data_C(np.asarray(data, order="C"))
    else:
        zeros_counted = count_zeros_32bit_float_data_C(np.asarray(data, order="C"))

    zero_elements_total = int(zeros_counted[0])

    if (zero_elements_total / nonzero_elements_total) * 100 >= percentage_threshold:
        if verbosity:
            print(
                f"Warning!!! Input data to method: {method_name} contains more than {percentage_threshold} percent of zeros."
            )

    return zero_elements_total