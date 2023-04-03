import os
import logging
import functools

import galsim
import galsim.config

import piff

logger = logging.getLogger(__name__)


PSF_KWARGS = {
    "g": {"GI_COLOR": 1.1},
    "r": {"GI_COLOR": 1.1},
    "i": {"GI_COLOR": 1.1},
    "z": {"IZ_COLOR": 0.34},
}


@functools.lru_cache(maxsize=200)
def _read_piff(file_name):
    return piff.read(
        os.path.expanduser(os.path.expandvars(file_name))
    )


class DES_Piff(object):
    """A wrapper for Piff to use with Galsim.

    Parameters
    ----------
    file_name : str
        The file with the Piff psf solution.
    """
    _req_params = {'file_name': str}
    _opt_params = {}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name):
        self.file_name = file_name
        self._piff = _read_piff(file_name)
        self._chipnum = list(set(self._piff.wcs.keys()))[0]
        assert len(list(set(self._piff.wcs.keys()))) == 1

    def _draw(
        self, image_pos, wcs=None, n_pix=None,
        gsparams=None, **psf_kwargs
    ):
        """Get an image of the PSF at the given location.

        Parameters
        ----------
        image_pos : galsim.Position
            The image position for the PSF.
        wcs : galsim.BaseWCS or subclass, optional
            The WCS to use to draw the PSF.
        n_pix : int, optional
            The image size to use when drawing without smoothing. Defaults to
            53 pixels if not given
        gsparams : galsim.GSParams, optional
            Optional galsim configuration data to pass along.
        psf_kwargs : extra keyword arguments
            These are passed to the Piff model as needed.

        Returns
        -------
        psf_img : galsim.ImageD
            The Galsim image of the PSF.
        pixel_wcs : galsim.BaseWCS
            The WCS of the image
        offset : tuple of floats
            The offset from the true center of the PSF centroid.
        """
        if n_pix is None:
            n_pix = 53

        if wcs is not None:
            pixel_wcs = wcs.local(image_pos)
        else:
            pixel_wcs = galsim.PixelScale(0.263)

        # nice and big image size here cause this has been a problem
        image = galsim.ImageD(ncol=n_pix, nrow=n_pix, wcs=pixel_wcs)

        if "GI_COLOR" in self.getPiff().interp_property_names:
            psf_kwargs.pop("IZ_COLOR", None)
            if (
                "GI_COLOR" in psf_kwargs and (
                    psf_kwargs["GI_COLOR"] is None or
                    psf_kwargs["GI_COLOR"] == "None"
                )
            ):
                psf_kwargs["GI_COLOR"] = 1.1

        elif "IZ_COLOR" in self.getPiff().interp_property_names:
            psf_kwargs.pop("GI_COLOR", None)

            if (
                "IZ_COLOR" in psf_kwargs and (
                    psf_kwargs["IZ_COLOR"] is None or
                    psf_kwargs["IZ_COLOR"] == "None"
                )
            ):
                psf_kwargs["IZ_COLOR"] = 0.34

        offset = (
            image_pos.x - int(image_pos.x + 0.5),
            image_pos.y - int(image_pos.y + 0.5),
        )

        psf_img = self.getPiff().draw(
            image_pos.x,
            image_pos.y,
            image=image,
            chipnum=self._chipnum,
            center=True,
            offset=offset,
            **psf_kwargs,
        )

        return psf_img, pixel_wcs, offset

    def getPiff(self):
        return self._piff

    def getPSF(
        self, image_pos, wcs=None,
        n_pix=None, depixelize=False,
        gsparams=None,
        **kwargs
    ):
        """Get an image of the PSF at the given location.

        Parameters
        ----------
        image_pos : galsim.Position
            The image position for the PSF.
        wcs : galsim.BaseWCS or subclass, optional
            The WCS to use to draw the PSF. If not given, the WCS in the Piff model is used.
        n_pix : int, optional
            The image size to use when drawing without smoothing.
        depixelize : bool, optional
            If True, the interpolated image will be depixelized. Default is False.
        gsparams : galsim.GSParams, optional
            Optional galsim configuration data to pass along.
        **kwargs : extra keyword arguments
            These are all passed on to the Piff model when drawing.

        Returns
        -------
        psf : galsim.GSObject
            The PSF at the image position.
        """

        psf_img, pixel_wcs, offset = self._draw(
            image_pos,
            wcs=wcs,
            n_pix=n_pix,
            gsparams=gsparams,
            **kwargs
        )

        psf = galsim.InterpolatedImage(
            galsim.ImageD(psf_img.array),  # make sure galsim is not keeping state
            wcs=pixel_wcs,
            gsparams=gsparams,
            x_interpolant='lanczos15',
            depixelize=depixelize,
            offset=offset,
        ).withFlux(
            1.0
        )
        return psf

    def getPSFImage(
        self, image_pos, wcs=None,
        n_pix=None,
        gsparams=None,
        **kwargs
    ):
        """Get an image of the PSF at the given location.

        Parameters
        ----------
        image_pos : galsim.Position
            The image position for the PSF.
        wcs : galsim.BaseWCS or subclass, optional
            The WCS to use to draw the PSF. If not given, the WCS in the Piff model is used.
        n_pix : int, optional
            The image size to use when drawing without smoothing.
        gsparams : galsim.GSParams, optional
            Optional galsim configuration data to pass along.
        **kwargs : extra keyword arguments
            These are all passed on to the Piff model when drawing.

        Returns
        -------
        psf : galsim.Image
            The PSF image at the image position.
        """
        psf_img, pixel_wcs, offset = self._draw(
            image_pos,
            wcs=wcs,
            n_pix=n_pix,
            gsparams=gsparams,
            **kwargs
        )
        return psf_img


class PiffLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {'file_name': str}
        opt = {}
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req, opt=opt
        )

        return kwargs, safe


# add a config input section
galsim.config.RegisterInputType('des_piff', PiffLoader(DES_Piff))


# and a builder
def BuildDES_Piff(config, base, ignore, gsparams, logger):
    opt = {
        'flux': float,
        'image_pos': galsim.PositionD,
        'n_pix': int,
        'gi_color': float,
        'iz_color': float,
        'depixelize': bool,
        'file_name': str,
    }
    params, safe = galsim.config.GetAllParams(
        config, base, opt=opt, ignore=ignore
    )

    if params.get("file_name", None) is None:
        des_piff = galsim.config.GetInputObj('des_piff', config, base, 'DES_Piff')
    else:
        des_piff = DES_Piff(params.get("file_name", None))

    if 'image_pos' in params:
        image_pos = params['image_pos']
    elif 'image_pos' in base:
        image_pos = base['image_pos']
    else:
        raise galsim.GalSimConfigError(
            "DES_Piff requested, but no image_pos defined in base.")

    if 'wcs' not in base:
        raise galsim.GalSimConfigError(
            "DES_Piff requested, but no wcs defined in base.")
    wcs = base['wcs']

    if gsparams:
        gsparams = galsim.GSParams(**gsparams)
    else:
        gsparams = None

    psf = des_piff.getPSF(
        image_pos,
        wcs=wcs,
        n_pix=params.get("n_pix", None),
        gi_color=params.get("gi_color", None),
        iz_color=params.get("iz_color", None),
        depixelize=params.get("depixelize", False),
        gsparams=gsparams,
    )

    if 'flux' in params:
        psf = psf.withFlux(params['flux'])

    # we make sure to declare the returned object as not safe for reuse
    can_be_reused = False
    return psf, can_be_reused


galsim.config.RegisterObjectType(
    'DES_Piff', BuildDES_Piff, input_type='des_piff')
