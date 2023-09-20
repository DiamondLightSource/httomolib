import time
import numpy as np
import pytest
from httomolib.prep.phase import paganin_filter
from numpy.testing import assert_allclose

eps = 1e-6

def test_paganin_filter(host_data):
    # --- testing the Paganin filter from TomoPy on tomo_standard ---#
    filtered_data = paganin_filter(host_data)

    assert filtered_data.ndim == 3
    assert_allclose(np.mean(filtered_data), -6.74213, rtol=eps)
    assert_allclose(np.max(filtered_data), -6.496699, rtol=eps)

    #: make sure the output is float32
    assert filtered_data.dtype == np.float32


def test_paganin_filter_energy100(host_data):
    filtered_data = paganin_filter(host_data, energy=100.0)

    assert_allclose(np.mean(filtered_data), -6.73455, rtol=1e-05)
    assert_allclose(np.min(filtered_data), -6.909582, rtol=eps)

    assert filtered_data.ndim == 3
    assert filtered_data.dtype == np.float32


def test_paganin_filter_dist75(host_data):
    filtered_data = paganin_filter(host_data, dist=75.0, alpha=1e-6)
    
    assert_allclose(np.sum(np.mean(filtered_data, axis=(1, 2))), -1215.4985, rtol=1e-6)
    assert_allclose(np.sum(filtered_data), -24893412., rtol=1e-6)
    assert_allclose(np.mean(filtered_data[0, 60:63, 90]), -6.645878, rtol=1e-6)
    assert_allclose(np.sum(filtered_data[50:100, 40, 1]), -343.5908, rtol=1e-6)
