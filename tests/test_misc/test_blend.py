import numpy as np

from httomolib.misc.blend import seam_blend_stitched_data


def test_seam_blend_stitched_data(data_stitched):

    result = seam_blend_stitched_data(data_stitched, seam_index=151, blending_width=30)

    assert result.flags.c_contiguous
    assert result.dtype == np.float32
    assert result.shape == (300, 4, 240)


def test_seam_blend_stitched_data_txt(data_stitched, data_stitched_txt):

    result = seam_blend_stitched_data(
        data_stitched, path_to_stiched_params_file=data_stitched_txt
    )

    assert result.flags.c_contiguous
    assert result.dtype == np.float32
    assert result.shape == (300, 4, 240)
