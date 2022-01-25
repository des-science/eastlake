import subprocess
import tempfile
import os

import pytest
import numpy as np
import fitsio

from ..utils import unpack_fits_file_if_needed


@pytest.mark.parametrize("do_rm", [False, True])
def test_unpack_fits_file_if_needed(do_rm):
    with tempfile.TemporaryDirectory() as tmpdir:
        d = np.zeros((100, 100), dtype="f4")
        pth = os.path.join(tmpdir, "data.fits")
        pth_fz = os.path.join(tmpdir, "data.fits.fz")
        fitsio.write(pth, d, clobber=True)
        subprocess.run("fpack data.fits", shell=True, check=True, cwd=tmpdir)
        assert os.path.exists(pth_fz)

        if do_rm:
            subprocess.run("rm -f " + pth, shell=True, check=True)

        opth, oext = unpack_fits_file_if_needed(pth_fz, 1)

        assert opth == pth
        assert os.path.exists(opth)
        assert oext == 0
