import pathlib
from unittest.mock import ANY

import numpy as np
import pytest
from pytest_mock import MockerFixture
from httomolib.misc.images import save_to_images
from PIL import Image
import time


@pytest.mark.parametrize("bits", [8, 16, 32])
def test_save_to_images(host_data: np.ndarray, tmp_path: pathlib.Path, bits: int):
    if bits == 8:
        dtype = np.uint8
    elif bits == 16:
        dtype = np.uint16
    else:
        dtype = np.uint32

    images = host_data[:, :10, :].astype(dtype)
    save_to_images(images, tmp_path / "save_to_images")

    folder = tmp_path / "save_to_images" / "images" / f"images{bits}bit_tif"
    assert folder.exists()
    files = list(folder.glob("*"))

    assert len(files) == 10
    for f in files:
        assert f.name[-3:] == "tif"
        assert Image.open(f).size == (host_data.shape[2], host_data.shape[0])


@pytest.mark.parametrize("bits", [8, 16, 32])
def test_save_to_images_watermark(
    host_data: np.ndarray, tmp_path: pathlib.Path, bits: int
):
    if bits == 8:
        dtype = np.uint8
    elif bits == 16:
        dtype = np.uint16
    else:
        dtype = np.uint32

    images = host_data[:, :10, :].astype(dtype)
    watermark_vals = tuple(range(0, 10))
    save_to_images(images, tmp_path / "save_to_images", watermark_vals=watermark_vals)

    folder = tmp_path / "save_to_images" / "images" / f"images{bits}bit_tif"
    assert folder.exists()
    files = list(folder.glob("*"))

    assert len(files) == 10
    for f in files:
        assert f.name[-3:] == "tif"
        assert Image.open(f).size == (host_data.shape[2], host_data.shape[0])


def test_save_to_images_2D(host_data: np.ndarray, tmp_path: pathlib.Path):
    DTYPE = np.uint8
    bits = np.dtype(DTYPE).itemsize * 8
    save_to_images(
        np.squeeze(host_data[:, 1, :]).astype(DTYPE),
        tmp_path / "save_to_images",
        file_format="tif",
    )

    folder = tmp_path / "save_to_images" / "images" / f"images{bits}bit_tif"
    assert folder.exists()
    files = [f.name for f in folder.glob("*")]

    assert files == ["00001.tif"]


def test_save_to_images_watermark_2D(host_data: np.ndarray, tmp_path: pathlib.Path):
    DTYPE = np.uint8
    bits = np.dtype(DTYPE).itemsize * 8
    watermark_vals = tuple(range(0, 1))

    save_to_images(
        np.squeeze(host_data[:, 1, :]).astype(DTYPE),
        tmp_path / "save_to_images",
        file_format="tif",
        watermark_vals=watermark_vals,
    )

    folder = tmp_path / "save_to_images" / "images" / f"images{bits}bit_tif"
    assert folder.exists()
    files = [f.name for f in folder.glob("*")]

    assert files == ["00001.tif"]


@pytest.mark.parametrize("offset", [0, 10, 35])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_save_to_images_offset_axis(
    host_data: np.ndarray, tmp_path: pathlib.Path, offset: int, axis: int
):
    DTYPE = np.uint8
    bits = np.dtype(DTYPE).itemsize * 8
    save_to_images(
        host_data[:10, :10, :10].astype(DTYPE),
        tmp_path / "save_to_images",
        offset=offset,
        file_format="tif",
        axis=axis,
    )

    folder = tmp_path / "save_to_images" / "images" / f"images{bits}bit_tif"
    assert folder.exists()
    # convert file names without extension to numbers and sort them
    files = sorted([int(f.name[:-4]) for f in folder.glob("*")])

    assert len(files) == 10
    assert files == list(range(offset, offset + len(files)))


@pytest.mark.parametrize("dtype", [np.int8, np.float32, np.float64])
def test_save_to_images_unsupported_dtype_raises_error(
    host_data, tmp_path: pathlib.Path, dtype: np.dtype
):
    with pytest.raises(ValueError) as e:
        save_to_images(
            host_data[:, 1:3, :].astype(dtype),
            tmp_path / "save_to_images",
            subfolder_name="test",
            file_format="png",
        )

    assert "input data must be in uint(8,16,32 bit) data type" in str(e)

    bits = np.dtype(dtype).itemsize * 8
    folder = tmp_path / "save_to_images" / "test" / f"images{bits}bit_png"
    assert not folder.exists()


def test_glob_stats_percentage_computation(
    host_data: np.ndarray, tmp_path: pathlib.Path, mocker: MockerFixture
):
    save_single_mock = mocker.patch(
        "httomolib.misc.images._rescale_2d", return_value=host_data[:, 1, :]
    )
    save_to_images(
        np.squeeze(host_data[:, 1, :]).astype(np.float32),
        tmp_path / "save_to_images",
        bits=8,
        rescale_method="percentage",
        file_format="tif",
        glob_stats=(20.0, 60.0, 40.0, 123),
        perc_range_min=10.0,
        perc_range_max=90.0,
    )

    min_perc = 10.0 * 40.0 / 100.0 + 20
    max_perc = 90.0 * 40.0 / 100.0 + 20

    save_single_mock.assert_called_once_with(ANY, 8, min_perc, max_perc)


def test_glob_stats_percentile_computation(
    host_data: np.ndarray, tmp_path: pathlib.Path, mocker: MockerFixture
):
    save_single_mock = mocker.patch(
        "httomolib.misc.images._rescale_2d", return_value=host_data[:, 1, :]
    )
    save_to_images(
        np.squeeze(host_data[:, 1, :]).astype(np.float32),
        tmp_path / "save_to_images",
        bits=8,
        rescale_method="percentile",
        file_format="tif",
        glob_stats=(20.0, 60.0, 40.0, 123),
        perc_range_min=10.0,
        perc_range_max=90.0,
    )

    min_perc = 941.0
    max_perc = 1021.0

    save_single_mock.assert_called_once_with(ANY, 8, min_perc, max_perc)


def test_integer_input_does_not_rescale(
    host_data: np.ndarray, tmp_path: pathlib.Path, mocker: MockerFixture
):
    save_single_mock = mocker.patch("httomolib.misc.images._rescale_2d")
    save_to_images(
        np.squeeze(host_data[:, 0:3, :]).astype(np.uint8),
        tmp_path / "save_to_images",
        bits=8,
        rescale_method="Off",
        file_format="tif",
        glob_stats=(20.0, 60.0, 40.0, 123),
        perc_range_min=10.0,
        perc_range_max=90.0,
    )

    save_single_mock.assert_not_called()
    folder = tmp_path / "save_to_images" / "images" / "images8bit_tif"
    assert folder.exists()


@pytest.mark.perf
def test_save_to_images_performance(tmp_path: pathlib.Path):
    data = np.random.randint(low=0, high=255, size=(160, 1800, 160), dtype=np.uint8)

    # uncomment and adapt to save / test different locations
    # tmp_path = pathlib.Path(
    #     '/mnt/gpfs03/scratch/data/imaging/tomography/tmp/image-saver-perf-test'
    # )

    start = time.perf_counter_ns()
    save_to_images(
        data, tmp_path / "save_to_images1", bits=8, axis=1, asynchronous=False
    )
    end = time.perf_counter_ns()
    duration_ms_old = (end - start) * 1e-6

    start = time.perf_counter_ns()
    save_to_images(
        data, tmp_path / "save_to_images2", bits=8, axis=1, asynchronous=True
    )
    end = time.perf_counter_ns()
    duration_ms = (end - start) * 1e-6

    assert duration_ms_old == duration_ms
