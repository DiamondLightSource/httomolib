import pathlib
from unittest.mock import ANY

import numpy as np
import pytest
from pytest_mock import MockerFixture
from httomolib.misc.images import save_to_images
from PIL import Image


@pytest.mark.parametrize("bits", [8, 16, 32])
def test_save_to_images(host_data: np.ndarray, tmp_path: pathlib.Path, bits: int):
    # --- Test for bits=8
    save_to_images(host_data[:, :10, :], tmp_path / "save_to_images", bits=bits)

    folder = tmp_path / "save_to_images" / "images" / f"images{bits}bit_tif"
    assert folder.exists()
    files = list(folder.glob("*"))

    assert len(files) == 10
    for f in files:
        assert f.name[-3:] == "tif"
        assert Image.open(f).size == (host_data.shape[2], host_data.shape[0])


def test_save_to_images_2D(host_data: np.ndarray, tmp_path: pathlib.Path):
    save_to_images(
        np.squeeze(host_data[:, 1, :]),
        tmp_path / "save_to_images",
        bits=8,
        file_format="tif",
    )

    folder = tmp_path / "save_to_images" / "images" / "images8bit_tif"
    assert folder.exists()
    files = [f.name for f in folder.glob("*")]

    assert files == ["00001.tif"]


@pytest.mark.parametrize("offset", [0, 10, 35])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_save_to_images_offset_axis(
    host_data: np.ndarray, tmp_path: pathlib.Path, offset: int, axis: int
):
    save_to_images(
        host_data[:10, :10, :10],
        tmp_path / "save_to_images",
        bits=8,
        offset=offset,
        file_format="tif",
        axis=axis,
    )

    folder = tmp_path / "save_to_images" / "images" / "images8bit_tif"
    assert folder.exists()
    # convert file names without extension to numbers and sort them
    files = sorted([int(f.name[:-4]) for f in folder.glob("*")])

    assert len(files) == 10
    assert files == list(range(offset, offset + len(files)))


@pytest.mark.parametrize("bits", [19, 242, 4432])
def test_save_to_images_other_bits_default_to_32(
    host_data, tmp_path: pathlib.Path, bits: int
):
    save_to_images(
        host_data[:, 1:3, :],
        tmp_path / "save_to_images",
        subfolder_name="test",
        file_format="png",
        bits=bits,
    )

    folder = tmp_path / "save_to_images" / "test" / "images32bit_png"
    assert folder.exists()
    assert len(list(folder.glob("*.png"))) == 2


def test_glob_stats_percentile_computation(
    host_data: np.ndarray, tmp_path: pathlib.Path, mocker: MockerFixture
):
    save_single_mock = mocker.patch("httomolib.misc.images._save_single_img")
    save_to_images(
        np.squeeze(host_data[:, 1, :]),
        tmp_path / "save_to_images",
        bits=8,
        file_format="tif",
        glob_stats=(20., 60., 40., 123),
        perc_range_min=10.0,
        perc_range_max=90.0,
    )

    min_percentile = 10.0 * 40.0 / 100.0 + 20
    max_percentile = 90.0 * 40.0 / 100.0 + 20

    save_single_mock.assert_called_once_with(
        ANY, min_percentile, max_percentile, 8, ANY, ANY
    )
