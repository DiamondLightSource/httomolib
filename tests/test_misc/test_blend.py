import numpy as np
import pytest

from httomolib.misc.blend import seam_blend_stitched_data



def test_seam_blend_stitched_data(data_stitched):

    result = seam_blend_stitched_data(data_stitched,seam_index=151,blending_width=30)
    
    assert result.flags.c_contiguous
    assert result.dtype == np.float32
    assert result.shape == (300, 4, 240)
