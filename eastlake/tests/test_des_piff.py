import os

import numpy as np
import galsim
import yaml

import pytest

from ..des_piff import PSF_KWARGS, DES_Piff


def _get_piff_file():
    with open(
        os.path.join(os.environ['TEST_DESDATA'], "source_info.yaml"),
        "r",
    ) as fp:
        return os.path.join(
            os.environ['TEST_DESDATA'],
            "DESDATA",
            yaml.safe_load(fp)["piff_path"]
        )


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'DES_Piff can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize("x_offset", [-0.3, 0.0, 0.3])
@pytest.mark.parametrize("y_offset", [-0.3, 0.0, 0.3])
@pytest.mark.parametrize("cls", [DES_Piff])
def test_des_piff_centering(x_offset, y_offset, cls):
    piff_fname = _get_piff_file()
    piff = cls(piff_fname)
    atol = 0.01

    # test the image it makes
    psf_im = piff.getPSFImage(
        galsim.PositionD(20 + x_offset, 10 + y_offset),
        **PSF_KWARGS["g"],
    )

    cen = (53-1)/2 + 1
    admom = psf_im.FindAdaptiveMom()

    assert np.allclose(
        admom.moments_centroid.x,
        cen + x_offset,
        rtol=0,
        atol=atol,
    )

    assert np.allclose(
        admom.moments_centroid.y,
        cen + y_offset,
        rtol=0,
        atol=atol,
    )

    # test how well it recenters
    psf_im = piff.getPSF(
        galsim.PositionD(20 + x_offset, 10 + y_offset),
        **PSF_KWARGS["g"],
    ).drawImage(nx=53, ny=53, scale=0.263, offset=(x_offset, y_offset))
    cen = (53-1)/2 + 1
    admom = psf_im.FindAdaptiveMom()

    assert np.allclose(
        admom.moments_centroid.x,
        cen + x_offset,
        rtol=0,
        atol=atol,
    )

    assert np.allclose(
        admom.moments_centroid.y,
        cen + y_offset,
        rtol=0,
        atol=atol,
    )


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'DES_Piff can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize("cls", [DES_Piff])
def test_des_piff_color(cls):
    piff_fname = _get_piff_file()
    piff = cls(piff_fname)
    psf_im1 = piff.getPSF(
        galsim.PositionD(20, 10),
        GI_COLOR=1.3,
        IZ_COLOR=0.4,
    ).drawImage(nx=53, ny=53, scale=0.263).array

    psf_im2 = piff.getPSF(
        galsim.PositionD(20, 10),
        GI_COLOR=0.7,
        IZ_COLOR=0.1,
    ).drawImage(nx=53, ny=53, scale=0.263).array

    assert not np.allclose(psf_im1, psf_im2)

    psf_im2 = piff.getPSF(
        galsim.PositionD(20, 10),
        GI_COLOR=1.3,
        IZ_COLOR=0.4,
    ).drawImage(nx=53, ny=53, scale=0.263).array

    assert np.array_equal(psf_im1, psf_im2)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'DES_Piff can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize("cls", [DES_Piff])
def test_des_piff_raises(cls):
    piff_fname = _get_piff_file()
    piff = cls(piff_fname)
    with pytest.raises(Exception) as e:
        piff.getPSF(
            galsim.PositionD(20, 10),
        ).drawImage(nx=53, ny=53, scale=0.263).array

    assert "_COLOR" in str(e.value)

    with pytest.raises(Exception) as e:
        piff.getPSF(
            galsim.PositionD(20, 10),
            IZ_COLOR=0.4,
        ).drawImage(nx=53, ny=53, scale=0.263).array

    assert "_COLOR" in str(e.value)


@pytest.mark.skipif(
    os.environ.get('TEST_DESDATA', None) is None,
    reason=(
        'DES_Piff can only be tested if '
        'test data is at TEST_DESDATA'))
@pytest.mark.parametrize("cls", [DES_Piff])
def test_des_piff_smoke(cls):
    piff_fname = _get_piff_file()
    piff = cls(piff_fname)
    psf_im = piff.getPSF(
        galsim.PositionD(20, 10),
        **PSF_KWARGS["g"],
    ).drawImage(nx=53, ny=53, scale=0.263).array

    if False:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(psf_im)
        import pdb
        pdb.set_trace()

    assert np.all(np.isfinite(psf_im))
    assert np.allclose(np.sum(psf_im), 1, rtol=1e-4, atol=0)

    y, x = np.unravel_index(np.argmax(psf_im), psf_im.shape)
    cen = (psf_im.shape[0]-1)/2
    assert y == cen
    assert x == cen
