import os

import numpy as np
import galsim

import pytest

from ..des_piff import DES_Piff


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'DES_Piff can only be tested if '
        'test data is at TEST_DESDATA'))
def test_des_piff_smoke():
    piff_fname = os.path.join(
        os.environ['TEST_DESDATA'],
        "PIFF_DATA_DIR",
        "y3a1-v29",
        "232321",
        "D00232321_i_c55_r2357p01_piff.fits",
    )
    piff = DES_Piff(piff_fname)
    psf_im = piff.getPSF(galsim.PositionD(20, 10)).drawImage(nx=53, ny=53, scale=0.263).array

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(psf_im)
        import pdb
        pdb.set_trace()

    assert np.all(np.isfinite(psf_im))
    assert np.allclose(np.sum(psf_im), 1)

    y, x = np.unravel_index(np.argmax(psf_im), psf_im.shape)
    cen = (psf_im.shape[0]-1)/2
    assert y == cen
    assert x == cen


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'DES_Piff can only be tested if '
        'test data is at TEST_DESDATA'))
def test_des_piff_smooth():
    piff_fname = os.path.join(
        os.environ['TEST_DESDATA'],
        "PIFF_DATA_DIR",
        "y3a1-v29",
        "232321",
        "D00232321_i_c55_r2357p01_piff.fits",
    )
    piff = DES_Piff(piff_fname)
    psf_im = piff.getPSF(
        galsim.PositionD(20, 10),
    ).drawImage(nx=53, ny=53, scale=0.263).array

    piff = DES_Piff(piff_fname)
    psf_im1 = piff.getPSF(
        galsim.PositionD(20, 10),
        smooth=True,
    ).drawImage(nx=53, ny=53, scale=0.263).array

    piff = DES_Piff(piff_fname, smooth=True)
    psf_im2 = piff.getPSF(
        galsim.PositionD(20, 10),
    ).drawImage(nx=53, ny=53, scale=0.263).array

    for im in [psf_im, psf_im1, psf_im2]:
        assert np.all(np.isfinite(im))
        assert np.allclose(np.sum(im), 1)
        y, x = np.unravel_index(np.argmax(im), im.shape)
        cen = (im.shape[0]-1)/2
        assert y == cen
        assert x == cen

    assert np.allclose(psf_im1, psf_im2)
    assert not np.allclose(psf_im, psf_im1)
    assert not np.allclose(psf_im, psf_im2)
