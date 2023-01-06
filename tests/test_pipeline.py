import cupy as cp
import numpy as np
from cupy.testing import assert_allclose

from httomolib.prep.normalize import normalize_cupy
from httomolib.prep.stripe import remove_stripe_based_sorting_cupy
from httomolib.recon.rotation import find_center_vo_cupy


def test_cpu_vs_gpu():
    #: GPU pipeline tested on `tomo_standard`

    in_file = 'tests/test_data/tomo_standard.npz'
    datafile = np.load(in_file)
    host_data = datafile['data']
    host_flats = datafile['flats']
    host_darks = datafile['darks']

    data = cp.asarray(host_data)
    flats = cp.asarray(host_flats)
    darks = cp.asarray(host_darks)

    #: Normalize the data first
    data_normalize_cupy = normalize_cupy(data, flats, darks, minus_log=True)

    #: Now do the stripes removal
    corrected_data = remove_stripe_based_sorting_cupy(data_normalize_cupy)

    #: Apply Fresnel/Paganin filtering

    #: Set data collection angles as equally spaced between 0-180 degrees.
    cor = find_center_vo_cupy(corrected_data)

    #: Correct distortion
