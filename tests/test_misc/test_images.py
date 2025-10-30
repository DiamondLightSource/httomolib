import pathlib
from unittest.mock import ANY

import numpy as np
import pytest
from httomolib.misc.images import save_to_images
from PIL import Image
import time


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_save_to_images(host_data: np.ndarray, tmp_path: pathlib.Path, dtype: np.dtype):

    images = host_data[:, 50:60, :].astype(dtype)
    save_to_images(images, tmp_path / "save_to_images")

    folder = tmp_path / "save_to_images" / f"images{dtype(0).nbytes * 8}bit_tif"
    assert folder.exists()
    files = list(folder.glob("*"))

    assert len(files) == 10
    for f in files:
        assert f.name[-3:] == "tif"
        assert Image.open(f).size == (host_data.shape[2], host_data.shape[0])


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float32, np.float64])
def test_save_to_images_watermark(
    host_data: np.ndarray, tmp_path: pathlib.Path, dtype: np.dtype
):

    images = host_data[:, 50:53, :].astype(dtype)
    watermark_vals = (0.1, 0.343, 10)
    save_to_images(images, tmp_path / "save_to_images", watermark_vals=watermark_vals)

    folder = tmp_path / "save_to_images" / f"images{dtype(0).nbytes * 8}bit_tif"
    assert folder.exists()
    files = list(folder.glob("*"))

    assert len(files) == 3
    for f in files:
        assert f.name[-3:] == "tif"
        assert Image.open(f).size == (host_data.shape[2], host_data.shape[0])


@pytest.mark.parametrize("dtype", [np.uint8])
@pytest.mark.parametrize("file_format", ['jpeg', 'png'])
def test_save_to_images_jpeg_png_uint8(
    host_data: np.ndarray, tmp_path: pathlib.Path, file_format: str, dtype: np.dtype
):
    images = host_data[:, 50:53, :].astype(dtype)
    save_to_images(images, tmp_path / "save_to_images", file_format=file_format)

    folder = tmp_path / "save_to_images" / f"images{dtype(0).nbytes * 8}bit_{file_format}"
    assert folder.exists()
    files = list(folder.glob("*"))
    assert len(files) == 3
    for f in files:
        assert Image.open(f).size == (host_data.shape[2], host_data.shape[0])    


@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.float32, np.float64])
@pytest.mark.parametrize("file_format", ['jpeg', 'png'])
def test_save_to_images_jpeg_png_other_bit_type_error(
    host_data: np.ndarray, tmp_path: pathlib.Path, file_format: str, dtype: np.dtype
):
    images = host_data[:, 50:53, :].astype(dtype)
    with pytest.raises(ValueError):
        save_to_images(images, tmp_path / "save_to_images", file_format=file_format)


def test_save_to_images_2D(host_data: np.ndarray, tmp_path: pathlib.Path):
    DTYPE = np.uint8
    bits = np.dtype(DTYPE).itemsize * 8
    save_to_images(
        np.squeeze(host_data[:, 1, :]).astype(DTYPE),
        tmp_path / "save_to_images",
        file_format="tif",
    )

    folder = tmp_path / "save_to_images" / f"images{bits}bit_tif"
    assert folder.exists()
    files = [f.name for f in folder.glob("*")]

    assert files == ["00001.tif"]


def test_save_to_images_tiff_float32(host_data: np.ndarray, host_flats: np.ndarray, tmp_path: pathlib.Path):
    host_data_norm = np.float32(host_data / np.mean(host_flats,axis=0))

    bits = host_data_norm.dtype.itemsize * 8
    save_to_images(
        host_data_norm,
        tmp_path / "save_to_images",
        file_format="tif",
    )

    folder = tmp_path / "save_to_images" / f"images{bits}bit_tif"
    assert folder.exists()
    files = list(folder.glob("*"))
    assert len(files) == 128


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

    folder = tmp_path / "save_to_images" / f"images{bits}bit_tif"
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

    folder = tmp_path / "save_to_images" / f"images{bits}bit_tif"
    assert folder.exists()
    # convert file names without extension to numbers and sort them
    files = sorted([int(f.name[:-4]) for f in folder.glob("*")])

    assert len(files) == 10
    assert files == list(range(offset, offset + len(files)))


@pytest.mark.perf
def test_save_to_images_performance(tmp_path: pathlib.Path):
    data = np.random.randint(low=0, high=255, size=(160, 1800, 160), dtype=np.uint8)

    start = time.perf_counter_ns()
    save_to_images(data, tmp_path / "save_to_images1", axis=1, asynchronous=False)
    end = time.perf_counter_ns()
    duration_ms_old = (end - start) * 1e-6

    start = time.perf_counter_ns()
    save_to_images(data, tmp_path / "save_to_images2", axis=1, asynchronous=True)
    end = time.perf_counter_ns()
    duration_ms = (end - start) * 1e-6

    assert duration_ms_old == duration_ms
