import os
import logging

import galsim
import galsim.config

import piff

import numpy as np
import ngmix
if ngmix.__version__[0:2] == "v1":
    NGMIX_V2 = False
    from ngmix.fitting import LMSimple
    from ngmix.admom import Admom
else:
    NGMIX_V2 = True
    from ngmix.fitting import Fitter
    from ngmix.admom import AdmomFitter

from scipy.interpolate import CloughTocher2DInterpolator

logger = logging.getLogger(__name__)

# pixel scale used for fitting the Piff models
PIFF_SCALE = 0.25


PSF_KWARGS = {
    "g": {"GI_COLOR": 1.1},
    "r": {"GI_COLOR": 1.1},
    "i": {"GI_COLOR": 1.1},
    "z": {"IZ_COLOR": 0.34},
}


class DES_Piff(object):
    """A wrapper for Piff to use with Galsim.

    This wrapper uses ngmix to fit smooth models to the Piff PSF images. The
    parameters of these models are then interpolated across the SE image
    and used to generate a smooth approximation to the PSF.

    Parameters
    ----------
    file_name : str
        The file with the Piff psf solution.
    smooth : bool, optional
        If True, then smooth the Piff PSFs. Default of False.
    """
    _req_params = {'file_name': str}
    _opt_params = {}
    _single_params = []
    _takes_rng = False

    def __init__(self, file_name, smooth=False, psf_kwargs=None):
        self.file_name = file_name
        self.psf_kwargs = psf_kwargs or {}
        # Read the Piff file. This may fail if the Piff
        # file is missing. We catch this and continue
        # since if we're substituting in some different
        # PSF model for rejectlisted piff files, we'll
        # never actually use self._piff
        try:
            self._piff = piff.read(
                os.path.expanduser(os.path.expandvars(file_name)))

            self._chipnum = list(set(self._piff.wcs.keys()))[0]
            assert len(list(set(self._piff.wcs.keys()))) == 1
        except IOError as e:
            print("failed to load Piff file, hopefully it's rejectlisted: %s" % repr(e))
            self._piff = None
        self._did_fit = False
        self.smooth = smooth

    def _fit_smooth_model(self):
        dxy = 256
        ny = 4096 // dxy + 1
        nx = 2048 // dxy + 1

        xloc = np.empty((ny, nx), dtype=np.float64)
        yloc = np.empty((ny, nx), dtype=np.float64)
        pars = np.empty((ny, nx, 3), dtype=np.float64)
        for yi, yl in enumerate(np.linspace(1, 4096, ny)):
            for xi, xl in enumerate(np.linspace(1, 2048, nx)):
                rng = np.random.RandomState(seed=yi + nx * xi)
                xloc[yi, xi] = xl
                yloc[yi, xi] = yl

                pos = galsim.PositionD(x=xl, y=yl)
                gs_img = self._draw(pos).drawImage(
                    nx=19, ny=19, scale=PIFF_SCALE, method='sb')
                img = gs_img.array
                nse = np.std(
                    np.concatenate([img[0, :], img[-1, :]]))
                obs = ngmix.Observation(
                    image=img,
                    weight=np.ones_like(img)/nse**2,
                    jacobian=ngmix.jacobian.DiagonalJacobian(
                        x=9, y=9, scale=PIFF_SCALE))

                _g1 = np.nan
                _g2 = np.nan
                _T = np.nan

                # there are some nutty PSFs
                if gs_img.calculateFWHM() > 0.5:
                    for _ in range(5):
                        try:
                            if NGMIX_V2:
                                am = AdmomFitter(rng=rng)
                                res = am.go(obs, 0.3)
                                if res['flags'] != 0:
                                    continue

                                lm = Fitter(model='turb')
                                lm_res = lm.go(obs, res['pars'])
                                if lm_res['flags'] == 0:
                                    _g1 = lm_res['pars'][2]
                                    _g2 = lm_res['pars'][3]
                                    _T = lm_res['pars'][4]
                                    break
                            else:
                                am = Admom(obs, rng=rng)
                                am.go(0.3)
                                res = am.get_result()
                                if res['flags'] != 0:
                                    continue

                                lm = LMSimple(obs, 'turb')
                                lm.go(res['pars'])
                                lm_res = lm.get_result()
                                if lm_res['flags'] == 0:
                                    _g1 = lm_res['pars'][2]
                                    _g2 = lm_res['pars'][3]
                                    _T = lm_res['pars'][4]
                                    break
                        except ngmix.gexceptions.GMixRangeError:
                            pass

                    try:
                        irr, irc, icc = ngmix.moments.g2mom(_g1, _g2, _T)
                        # this is a fudge factor that gets the overall PSF FWHM
                        # correct
                        # the naive correction for the pixel size is
                        # a bit too small
                        pixel_var = PIFF_SCALE * PIFF_SCALE / 12 * 1.73
                        irr -= pixel_var
                        icc -= pixel_var
                        _g1, _g2, _T = ngmix.moments.mom2g(irr, irc, icc)
                    except Exception:
                        _g1 = np.nan
                        _g2 = np.nan
                        _T = np.nan

                pars[yi, xi, 0] = _g1
                pars[yi, xi, 1] = _g2
                pars[yi, xi, 2] = _T

        xloc = xloc.ravel()
        yloc = yloc.ravel()
        pos = np.stack([xloc, yloc], axis=1)
        assert pos.shape == (xloc.shape[0], 2)

        # make interps
        g1 = pars[:, :, 0].ravel()
        msk = np.isfinite(g1)
        if len(msk) < 10:
            raise ValueError('DES Piff fitting failed too much!')
        if np.any(~msk):
            g1[~msk] = np.mean(g1[msk])
        self._g1int = CloughTocher2DInterpolator(
            pos, g1, fill_value=np.mean(g1[msk]))

        g2 = pars[:, :, 1].ravel()
        msk = np.isfinite(g2)
        if len(msk) < 10:
            raise ValueError('DES Piff fitting failed too much!')
        if np.any(~msk):
            g2[~msk] = np.mean(g2[msk])
        self._g2int = CloughTocher2DInterpolator(
            pos, g2, fill_value=np.mean(g2[msk]))

        T = pars[:, :, 2].ravel()
        msk = np.isfinite(T)
        if len(msk) < 10:
            raise ValueError('DES Piff fitting failed too much!')
        if np.any(~msk):
            T[~msk] = np.mean(T[msk])
        self._Tint = CloughTocher2DInterpolator(
            pos, T, fill_value=np.mean(T[msk]))

        self._did_fit = True

    def _draw(self, image_pos, wcs=None, n_pix=None,
              x_interpolant='lanczos15', gsparams=None):
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
        x_interpolant : str, optional
            The interpolant to use.
        gsparams : galsim.GSParams, optional
            Ootional galsim configuration data to pass along.

        Returns
        -------
        psf : galsim.InterpolatedImage
            The PSF at the image position.
        """
        if wcs is not None:
            if n_pix is not None:
                n_pix = n_pix
            else:
                n_pix = 53
            pixel_wcs = wcs.local(image_pos)
        else:
            n_pix = 19
            pixel_wcs = galsim.PixelScale(PIFF_SCALE)

        # nice and big image size here cause this has been a problem
        image = galsim.ImageD(ncol=n_pix, nrow=n_pix, wcs=pixel_wcs)

        psf = self.getPiff().draw(
            image_pos.x,
            image_pos.y,
            image=image,
            center=True,
            chipnum=self._chipnum,
            **self.psf_kwargs,
        )

        psf = galsim.InterpolatedImage(
            galsim.ImageD(psf.array),  # make sure galsim is not keeping state
            wcs=pixel_wcs,
            gsparams=gsparams,
            x_interpolant=x_interpolant
        ).withFlux(
            1.0
        )

        return psf

    def getPiff(self):
        return self._piff

    def getPSF(
        self, image_pos, wcs=None,
        smooth=False, n_pix=None, **kwargs
    ):
        """Get an image of the PSF at the given location.

        Parameters
        ----------
        image_pos : galsim.Position
            The image position for the PSF.
        wcs : galsim.BaseWCS or subclass, optional
            The WCS to use to draw the PSF. Currently used only when smoothing
            is turned off.
        smooth : bool, optional
            If True, then smooth the Piff PSFs. Default of False.
        n_pix : int, optional
            The image size to use when drawing without smoothing.
        **kargs : extra keyword arguments
            These are all ignored.

        Returns
        -------
        psf : galsim.GSObject
            The PSF at the image position.
        """
        if smooth or self.smooth:
            if not self._did_fit:
                self._fit_smooth_model()

            arr = np.array([
                np.clip(image_pos.x, 1, 2048),
                np.clip(image_pos.y, 1, 4096)])

            _g1 = self._g1int(arr)[0]
            _g2 = self._g2int(arr)[0]
            _T = self._Tint(arr)[0]
            if np.any(np.isnan(np.array([_g1, _g2, _T]))):
                logger.debug("Piff smooth fit params are NaN: %s %s %s %s", image_pos, _g1, _g2, _T)
                raise RuntimeError("NaN smooth Piff params at %s!" % image_pos)
            pars = np.array([0, 0, _g1, _g2, _T, 1])
            obj = ngmix.gmix.make_gmix_model(pars, 'turb').make_galsim_object()
            return obj.withFlux(1)
        else:
            return self._draw(image_pos, wcs=wcs, n_pix=n_pix)


class PiffLoader(galsim.config.InputLoader):
    def getKwargs(self, config, base, logger):
        req = {'file_name': str}
        opt = {}
        kwargs, safe = galsim.config.GetAllParams(
            config, base, req=req, opt=opt)

        return kwargs, safe


# add a config input section
galsim.config.RegisterInputType('des_piff', PiffLoader(DES_Piff))


# and a builder
def BuildDES_Piff(config, base, ignore, gsparams, logger):
    des_piff = galsim.config.GetInputObj('des_piff', config, base, 'DES_Piff')

    opt = {'flux': float,
           'num': int,
           'image_pos': galsim.PositionD,
           'x_interpolant': str,
           'smooth': bool}
    params, safe = galsim.config.GetAllParams(
        config, base, opt=opt, ignore=ignore)

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
        wcs,
        smooth=params.get('smooth', False),
        gsparams=gsparams)

    if 'flux' in params:
        psf = psf.withFlux(params['flux'])

    # we make sure to declare the returned object as not safe for reuse
    can_be_reused = False
    return psf, can_be_reused


def BuildDES_Piff_with_substitute(config, base, ignore, gsparams, logger):
    # This builder usually just calls BuildDES_Piff, but can also
    # be passed use_substitute = True, in which case it builds some
    # other PSF. We use this for rejectlisted Piff files.
    if "use_substitute" in config:
        use_substitute = galsim.config.ParseValue(config, "use_substitute",
                                                  base, bool)[0]
    else:
        use_substitute = False

    if use_substitute:
        return (galsim.config.BuildGSObject(
            config, "substitute_psf", base=base,
            gsparams=gsparams, logger=logger))
    else:
        ignore += ["use_substitute", "substitute_psf"]
        return BuildDES_Piff(config, base, ignore, gsparams, logger)


galsim.config.RegisterObjectType(
    'DES_Piff', BuildDES_Piff_with_substitute, input_type='des_piff')
