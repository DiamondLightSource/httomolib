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
# Created By  : <scientificsoftware@diamond.ac.uk>
# Created Date: 27/October/2022
# ---------------------------------------------------------------------------
""" Module for loading/saving images """

import asyncio
from io import BytesIO
import os
import pathlib
from typing import List, Optional, Union
import httomolib

import numpy as np
from numpy import ndarray
from PIL import Image, ImageDraw, ImageFont

from skimage import exposure
import aiofiles

__all__ = [
    "save_to_images",
]

# number of asyncio workers to use to process saving images
# 40-ish seems to be the sweet spot, but it doesn't matter much
NUM_WORKERS = 40


def save_to_images(
    data: ndarray,
    out_dir: Union[str, os.PathLike],
    subfolder_name: str = "images",
    axis: int = 1,
    file_format: str = "tif",
    bits: int = 8,
    perc_range_min: float = 0.0,
    perc_range_max: float = 100.0,
    jpeg_quality: int = 95,
    glob_stats: Optional[tuple] = None,
    offset: int = 0,
    asynchronous: bool = False,
    watermark_txt: Optional[str] = None,
):
    """
    Saves data as 2D images. If the data type is not already one of the integer types
    uint8, uint16, or uint32, the data is rescaled and converted first.
    Otherwise it leaves the data as is.

    Parameters
    ----------
    data : np.ndarray
        Required input NumPy ndarray.
    out_dir : str
        The main output directory for images.
    subfolder_name : str, optional
        Subfolder name within the main output directory.
        Defaults to 'images'.
    axis : int, optional
        Specify the axis to use to slice the data (if `data` is a 3D array).
    file_format : str, optional
        Specify the file format to use, e.g. "png", "jpeg", or "tif".
        Defaults to "tif".
    bits : int, optional
        Specify the number of bits for unsigned integer data type to use. The options are: [8, 16, 32].
    perc_range_min: float, optional
        lower cutoff point: min + perc_range_min * (max-min)/100
        Defaults to 0.0
    perc_range_max: float, optional
        upper cutoff point: min + perc_range_max * (max-min)/100
        Defaults to 100.0
    jpeg_quality : int, optional
        Specify the quality of the jpeg image.
    glob_stats: tuple, optional
        Global statistics of the input data in a tuple format: (min, max, mean, total_elements_number).
        If None, then it will be calculated based on the input data.
    offset: int, optional
        The offset to start file indexing from, e.g. if offset is 100, images will start at
        00100.tif. This is used when executed in parallel context and only partial data is
        passed in this run.
    asynchronous: bool, optional
        Perform write operations synchronously or asynchronously.
    """
    if data.dtype in [np.uint8, np.uint16, np.uint32]:
        # do not touch the data if it's already in integers, but set the bits in order to
        # create the right folder
        bits = data.dtype.itemsize * 8
    elif bits not in [8, 16, 32]:
        bits = 32
        print(
            "The selected bit type %s is not available, "
            "resetting to 32 bit floating point \n" % str(bits)
        )

    # create the output folder
    subsubfolder_name = f"images{str(bits)}bit_{str(file_format)}"
    path_to_images_dir = pathlib.Path(out_dir) / subfolder_name / subsubfolder_name
    path_to_images_dir.mkdir(parents=True, exist_ok=True)

    queue: Optional[asyncio.Queue] = None
    if asynchronous:
        # async task queue - we push our tasks for every 2D image here
        queue = asyncio.Queue()

    do_rescale = False
    if data.dtype not in [np.uint8, np.uint16, np.uint32]:
        do_rescale = True

        data = np.nan_to_num(data, copy=False, nan=0.0, posinf=0, neginf=0)

        if glob_stats is None or glob_stats is False:
            min_percentile = np.nanpercentile(data, perc_range_min)
            max_percentile = np.nanpercentile(data, perc_range_max)
        else:
            # calculate the range here based on global max and min
            range_intensity = glob_stats[1] - glob_stats[0]
            min_percentile = (perc_range_min * (range_intensity) / 100) + glob_stats[0]
            max_percentile = (perc_range_max * (range_intensity) / 100) + glob_stats[0]

    if data.ndim == 3:
        slice_dim_size = np.shape(data)[axis]
        for idx in range(slice_dim_size):

            filename = f"{idx + offset:05d}.{file_format}"
            filepath_name = os.path.join(path_to_images_dir, f"{filename}")
            # note: data.take call is far more time consuming
            if axis == 0:
                d = data[idx, :, :]
            elif axis == 1:
                d = data[:, idx, :]
            else:
                d = data[:, :, idx]

            if do_rescale:
                d = _rescale_2d(d, bits, min_percentile, max_percentile)

            if asynchronous:
                # give the actual saving to the background task
                assert queue is not None
                queue.put_nowait(
                    (
                        d,
                        jpeg_quality,
                        "TIFF" if file_format == "tif" else file_format,
                        filepath_name,
                    )
                )
            else:
                Image.fromarray(d).save(filepath_name, quality=jpeg_quality)

            # after saving the image we check if the watermark needs to be added to that image
            if watermark_txt is not None:
                _add_watermark(filepath_name, watermark_txt)

    else:
        filename = f"{1:05d}.{file_format}"
        filepath_name = os.path.join(path_to_images_dir, f"{filename}")
        if do_rescale:
            data = _rescale_2d(data, bits, min_percentile, max_percentile)

        if asynchronous:
            # give the actual saving to the background task
            assert queue is not None
            queue.put_nowait(
                (
                    data,
                    jpeg_quality,
                    "TIFF" if file_format == "tif" else file_format,
                    filepath_name,
                )
            )
        else:
            Image.fromarray(data).save(filepath_name, quality=jpeg_quality)

        # after saving the image we check if the watermark needs to be added to that image
        if watermark_txt is not None:
            _add_watermark(filepath_name, watermark_txt)

    if asynchronous:
        # Start the event loop to save the images - and wait until it's done
        assert queue is not None
        asyncio.run(_waiting_loop(queue))


def _rescale_2d(d: np.ndarray, bits: int, min_percentile, max_percentile):
    if bits == 8:
        d = exposure.rescale_intensity(
            d, in_range=(min_percentile, max_percentile), out_range=(0, 255)
        ).astype(np.uint8)

    elif bits == 16:
        d = exposure.rescale_intensity(
            d, in_range=(min_percentile, max_percentile), out_range=(0, 65535)
        ).astype(np.uint16)

    else:
        d = exposure.rescale_intensity(
            d,
            in_range=(min_percentile, max_percentile),
            out_range=(min_percentile, max_percentile),
        ).astype(np.uint32)

    return d


def _add_watermark(
    filepath_name: str,
    watermark_txt: str,
    font_size_perc: int = 5,
    margin_perc: int = 10,
):
    """Adding two watermarks in the bottom left and the bottom right corners"""
    original_image = Image.open(filepath_name)
    draw = ImageDraw.Draw(original_image)
    image_width, image_height = original_image.size  # the image can be a non-square one
    font_size_relative = int(image_height / 100 * font_size_perc)  # relative to height
    margin_relative_w = int(image_width / 100 * margin_perc)
    margin_relative_h = int(image_height / 100 * margin_perc)

    # as pillow doesn't provide fonts and the default one cannot be scaled,
    # we need to ship the font with httomolib ourselves
    path_to_font = os.path.dirname(httomolib.__file__)
    font = ImageFont.truetype(
        path_to_font + "/misc" + "/DejaVuSans.ttf", font_size_relative
    )
    text_width, text_height = draw.textlength(watermark_txt, font)

    # Calculating positions
    position_left = (margin_relative_w, image_height - margin_relative_h - text_height)
    position_right = (
        image_width - margin_relative_w - text_width,
        image_height - margin_relative_h - text_height,
    )
    draw.text(
        position_left, watermark_txt, fill="white", stroke_fill="black", font=font
    )
    draw.text(
        position_right, watermark_txt, fill="black", stroke_fill="white", font=font
    )
    original_image.save(filepath_name)


async def _save_single_image(data: np.ndarray, quality: float, format: str, path: str):
    # We need a binary buffer in order to use aiofiles to write - PIL does not have
    # async methods itself.
    # So we convert image into a bytes array synchronously first
    buffer = BytesIO()
    Image.fromarray(data).save(buffer, quality=quality, format=format)

    # and then we write the buffer asynchronously to a file
    async with aiofiles.open(path, "wb") as file:
        await file.write(buffer.getbuffer())


async def _image_save_worker(queue):
    """Asynchronous worker task that waits on the given queue for tasks to save images"""
    while True:
        # Get a "work item" out of the queue - this is a suspend point for the task
        data, quality, format, path = await queue.get()

        await _save_single_image(data, quality, format, path)

        # Notify the queue that the "work item" has been processed.
        queue.task_done()


async def _waiting_loop(queue) -> None:
    """Async loop that assigns workers to process queue tasks and
    waits for them to finish"""

    # First, create  worker tasks to process the queue concurrently.
    tasks: List[asyncio.Task] = []
    for _ in range(NUM_WORKERS):
        task = asyncio.create_task(_image_save_worker(queue))
        tasks.append(task)

    # Wait until the queue is fully processed.
    await queue.join()

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()

    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)
