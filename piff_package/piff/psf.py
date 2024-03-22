# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: psf
"""

import numpy as np
import fitsio
import galsim
import sys

from .star import Star, StarData
from .util import write_kwargs, read_kwargs

from ngmix.defaults import LOWVAL
from ngmix.fitting import FitModel
from ngmix.gexceptions import GMixRangeError
from ngmix import (
    observation,
    MultiBandObsList,
    ObsList,
    Observation,
)
from ngmix.fitting.galsim_fitters import GalsimFitter


def _ap_kern_kern(x, m, h):
    # cumulative triweight kernel
    y = (x - m) / h + 3
    apval = np.zeros_like(m)
    msk = y > 3
    apval[msk] = 1
    msk = (y > -3) & (~msk)
    apval[msk] = (
        -5 * y[msk] ** 7 / 69984
        + 7 * y[msk] ** 5 / 2592
        - 35 * y[msk] ** 3 / 864
        + 35 * y[msk] / 96
        + 1 / 2
    )
    return apval


MODEL_FIT = "TURB"
GMIX_MODELS = ["GAUSS", "TURB", "EXP", "DEV"]
NPIX_FIT = 19
USE_CEN = False
DXY = 256
DCOLOR = 8

# def _get_moffat_prior(rng, flux):
#     """
#     get a prior for a moffat fit
#     """
#     import ngmix

#     cen_prior = ngmix.priors.CenPrior(
#         cen1=0, cen2=0, sigma1=0.263, sigma2=0.263,
#         rng=rng,
#     )
#     g_prior = ngmix.priors.GPriorBA(0.5, rng=rng)
#     hlr_prior = ngmix.priors.LMBounds(0.2, 5.0, rng=rng)
#     beta_prior = ngmix.priors.LMBounds(2.0, 5.0, rng=rng)
#     flux_prior = ngmix.priors.LMBounds(0.1 * flux, 10 * flux, rng=rng)

#     # it is called fracdev in this prior, but it represents beta
#     return ngmix.joint_prior.PriorBDFSep(
#         cen_prior=cen_prior,
#         g_prior=g_prior,
#         T_prior=hlr_prior,
#         fracdev_prior=beta_prior,
#         F_prior=flux_prior,
#     )


class MoffatGuesser(object):
    """
    Make Moffat guesses from the input T, flux and prior

    parameters
    ----------
    T: float
        Center for T guesses
    flux: float or sequences
        Center for flux guesses
    prior:
        cen, g drawn from this prior
    """

    def __init__(self, hlr, flux, rng):
        self.hlr = hlr
        self.flux = flux
        self.rng = rng

    def __call__(self):
        """
        Generate a guess
        """
        import numpy as np

        rng = self.rng

        if MODEL_FIT == "MOFFAT":
            guess = np.zeros(7)

            # 0.6 is roughly fwhm 0.9
            hlr_mid = self.hlr
            flux_mid = self.flux

            guess[0] = rng.uniform(low=-0.1, high=0.1)
            guess[1] = rng.uniform(low=-0.1, high=0.1)
            guess[2] = rng.uniform(low=-0.1, high=0.1)
            guess[3] = rng.uniform(low=-0.1, high=0.1)
            guess[4] = rng.uniform(low=0.9 * hlr_mid, high=1.1 * hlr_mid)
            guess[5] = rng.uniform(low=1.5, high=3)
            guess[6] = rng.uniform(low=0.9 * flux_mid, high=1.1 * flux_mid)
        elif MODEL_FIT in GMIX_MODELS:
            guess = np.zeros(6)

            # 0.6 is roughly fwhm 0.9
            hlr_mid = self.hlr
            flux_mid = self.flux

            guess[0] = rng.uniform(low=-0.1, high=0.1)
            guess[1] = rng.uniform(low=-0.1, high=0.1)
            guess[2] = rng.uniform(low=-0.1, high=0.1)
            guess[3] = rng.uniform(low=-0.1, high=0.1)
            guess[4] = rng.uniform(low=0.9 * hlr_mid, high=1.1 * hlr_mid)
            # guess[5] = rng.uniform(low=1.5, high=3)
            guess[5] = rng.uniform(low=0.9 * flux_mid, high=1.1 * flux_mid)

        if USE_CEN:
            return guess[2:]
        else:
            return guess


class CenGalsimMoffatFitModel(FitModel):
    """
    Represent a fitting model for fitting 6 parameter models with galsim,
    as well as generate images and mixtures for the best fit model

    Parameters
    ----------
    obs: observation(s)
        Observation, ObsList, or MultiBandObsList
    model: string
        e.g. 'exp', 'spergel'
    guess: array-like
        starting parameters for the lm fitter
    prior: ngmix prior, optional
        For example ngmix.priors.PriorSimpleSep can
        be used as a separable prior on center, g, size, flux.
    """

    def __init__(self, obs, model, guess, prior=None):
        self.model = model
        self['model'] = model
        self._set_model_class()
        self._set_prior(prior=prior)
        self._set_bounds()

        self._set_kobs(obs)
        self._set_n_prior_pars()
        self._set_totpix()
        self._set_fdiff_size()
        self._init_model_images()
        self._set_band_pars()

        guess = self._get_guess(guess)

    def _set_g(self):
        self["g"] = self["pars"][0:2].copy()
        self["g_cov"] = self["pars_cov"][0:2, 0:2].copy()
        self["g_err"] = self["pars_err"][0:2].copy()

    def _set_T(self):
        self["T"] = self["pars"][4-2]
        self["T_err"] = np.sqrt(self["pars_cov"][4-2, 4-2])

    def _set_flux(self):
        start = 4
        if self.nband == 1:
            self["flux"] = self["pars"][start]
            self["flux_err"] = np.sqrt(self["pars_cov"][start, start])
        else:
            self["flux"] = self["pars"][start:]
            self["flux_err"] = np.sqrt(self["pars_cov"][start:, start:])

    def calc_fdiff(self, pars):
        """

        vector with (model-data)/error.

        The npars elements contain -ln(prior)
        """

        # we cannot keep sending existing array into leastsq, don't know why
        fdiff = np.zeros(self.fdiff_size)

        try:

            self._fill_models(pars)

            start = self._fill_priors(pars, fdiff)

            for band in range(self.nband):

                kobs_list = self.mb_kobs[band]
                for kobs in kobs_list:

                    meta = kobs.meta
                    kmodel = meta["kmodel"]
                    ierr = meta["ierr"]

                    scratch = meta["scratch"]

                    # model-data
                    scratch.array[:, :] = kmodel.array[:, :]
                    scratch -= kobs.kimage

                    # (model-data)/err
                    scratch.array.real[:, :] *= ierr.array[:, :]
                    scratch.array.imag[:, :] *= ierr.array[:, :]

                    # now copy into the full fdiff array
                    imsize = scratch.array.size

                    fdiff[start:start + imsize] = scratch.array.real.ravel()

                    start += imsize

                    fdiff[start:start + imsize] = scratch.array.imag.ravel()

                    start += imsize

        except GMixRangeError:
            fdiff[:] = LOWVAL

        return fdiff

    def _fill_models(self, pars):
        """
        input pars are in linear space

        Fill the list of lists of gmix objects for the given parameters
        """
        try:
            for band, kobs_list in enumerate(self.mb_kobs):
                # pars for this band, in linear space
                band_pars = self.get_band_pars(pars, band)

                for i, kobs in enumerate(kobs_list):

                    gal = self.make_model(band_pars)

                    meta = kobs.meta

                    kmodel = meta["kmodel"]

                    gal._drawKImage(kmodel)

                    if kobs.has_psf():
                        kmodel *= kobs.psf.kimage
        except RuntimeError as err:
            raise GMixRangeError(str(err))

    def make_model(self, pars):
        """
        make the galsim model
        """

        model = self.make_round_model(pars)

        # shift = pars[0:0+2]
        g1 = pars[2-2]
        g2 = pars[3-2]

        # argh another generic error
        try:
            model = model.shear(g1=g1, g2=g2)
        except ValueError as err:
            raise GMixRangeError(str(err))

        # model = model.shift(shift)
        return model

    def make_round_model(self, pars):
        """
        make the galsim Moffat model
        """
        import galsim

        r50 = pars[4-2]
        beta = pars[5-2]
        flux = pars[6-2]

        # generic RuntimeError thrown
        try:
            gal = galsim.Moffat(beta, half_light_radius=r50, flux=flux,)
        except RuntimeError as err:
            raise GMixRangeError(str(err))

        return gal

    def _set_model_class(self):
        import galsim

        self._model_class = galsim.Moffat

    def get_band_pars(self, pars_in, band):
        """
        Get linear pars for the specified band

        input pars are [c1, c2, e1, e2, r50, beta, flux1, flux2, ....]
        """

        pars = self._band_pars

        pars[0:6-2] = pars_in[0:6-2]
        pars[6-2] = pars_in[6-2 + band]
        return pars

    def _set_prior(self, prior=None):
        self.prior = prior

    def _set_n_prior_pars(self):
        if self.prior is None:
            self.n_prior_pars = 0
        else:
            #                 c1  c2  e1e2  r50  beta   fluxes
            self.n_prior_pars = 1 + 1 + 1 + 1 + 1 + self.nband - 2

    def _set_totpix(self):
        """
        Make sure the data are consistent.
        """

        totpix = 0
        for kobs_list in self.mb_kobs:
            for kobs in kobs_list:
                totpix += kobs.kimage.array.size

        self.totpix = totpix

    def _convert2kobs(self, obs):
        kobs = observation.make_kobs(obs)

        return kobs

    def _set_kobs(self, obs_in, **keys):
        """
        Input should be an Observation, ObsList, or MultiBandObsList
        """

        if isinstance(obs_in, (Observation, ObsList, MultiBandObsList)):
            kobs = self._convert2kobs(obs_in)
        else:
            kobs = observation.get_kmb_obs(obs_in)

        self.mb_kobs = kobs
        self.nband = len(kobs)

    def _set_fdiff_size(self):
        # we have 2*totpix, since we use both real and imaginary
        # parts
        self.fdiff_size = self.n_prior_pars + 2 * self.totpix

    def _create_models_in_kobs(self, kobs):
        ex = kobs.kimage

        meta = kobs.meta
        meta["kmodel"] = ex.copy()
        meta["scratch"] = ex.copy()

    def _init_model_images(self):
        """
        add model image entries to the metadata for
        each observation

        these will get filled in
        """

        for kobs_list in self.mb_kobs:
            for kobs in kobs_list:
                meta = kobs.meta

                weight = kobs.weight
                ierr = weight.copy()
                ierr.setZero()

                w = np.where(weight.array > 0)
                if w[0].size > 0:
                    ierr.array[w] = np.sqrt(weight.array[w])

                meta["ierr"] = ierr
                self._create_models_in_kobs(kobs)

    def _check_guess(self, guess):
        """
        check the guess by making a model and checking for an
        exception
        """

        guess = np.array(guess, dtype="f8", copy=False)
        if guess.size != self.npars:
            raise ValueError(
                "expected %d entries in the "
                "guess, but got %d" % (self.npars, guess.size)
            )

        for band in range(self.nband):
            band_pars = self.get_band_pars(guess, band)
            # just doing this to see if an exception is raised. This
            # will bother flake8
            gal = self.make_model(band_pars)  # noqa

        return guess

    def _get_guess(self, guess):
        """
        make sure the guess has the right size and meets the model
        restrictions
        """

        guess = self._check_guess(guess)
        return guess

    def _set_npars(self):
        """
        nband should be set in set_lists, called before this
        """
        from ngmix.fitting.galsim_results import get_galsim_npars
        self.npars = get_galsim_npars(self.model, self.nband) - 2

    def _set_band_pars(self):
        """
        this is the array we fill with pars for a specific band
        """
        self._set_npars()

        npars_band = self.npars - self.nband + 1
        self._band_pars = np.zeros(npars_band)

    def set_fit_result(self, result):
        """
        Get some fit statistics for the input pars.
        """

        self.update(result)
        if self['flags'] == 0:
            self["s2n_r"] = self.calc_s2n_r(self['pars'])
            self._set_g()
            self._set_flux()

    def calc_s2n_r(self, pars):
        """
        we already have the round r50, so just create the
        models and don't shear them
        """

        s2n_sum = 0.0
        for band, kobs_list in enumerate(self.mb_kobs):
            # pars for this band, in linear space
            band_pars = self.get_band_pars(pars, band)

            for i, kobs in enumerate(kobs_list):
                meta = kobs.meta
                weight = kobs.weight

                round_pars = band_pars.copy()
                # round_pars[2:2+2] = 0.0
                gal = self.make_model(round_pars)

                kmodel = meta["kmodel"]

                gal.drawKImage(image=kmodel)

                if kobs.has_psf():
                    kmodel *= kobs.psf.kimage
                kmodel.real.array[:, :] *= kmodel.real.array[:, :]
                kmodel.imag.array[:, :] *= kmodel.imag.array[:, :]

                kmodel.real.array[:, :] *= weight.array[:, :]
                kmodel.imag.array[:, :] *= weight.array[:, :]

                s2n_sum += kmodel.real.array.sum()
                s2n_sum += kmodel.imag.array.sum()

        if s2n_sum > 0.0:
            s2n = np.sqrt(s2n_sum)
        else:
            s2n = 0.0

        return s2n


class CenGalsimMoffatFitter(GalsimFitter):
    """
    Fit a moffat model using galsim

    Parameters
    ----------
    model: string
        e.g. 'exp', 'spergel'
    prior: ngmix prior, optional
        For example ngmix.priors.PriorSimpleSep can
        be used as a separable prior on center, g, size, flux.
    fit_pars: dict, optional
        parameters for the lm fitter, e.g. maxfev, ftol, xtol
    """

    def __init__(self, prior=None, fit_pars=None):
        super().__init__(model="moffat", prior=prior, fit_pars=fit_pars)

    def _make_fit_model(self, obs, guess):
        return CenGalsimMoffatFitModel(
            model="moffat", obs=obs, guess=guess, prior=self.prior,
        )


def _do_ngmix_fit(*, img, cen_xy, scale, ntry=10):
    import ngmix
    from hashlib import sha1

    data = img.tobytes()
    _hash = sha1(data)
    seed = np.frombuffer(_hash.digest(), dtype='uint32')
    rng = np.random.RandomState(seed)

    obs = ngmix.Observation(
        img,
        weight=np.abs(np.ones_like(img)),
        jacobian=ngmix.DiagonalJacobian(
            row=cen_xy[1] - 1,  # ngmix is zero offset
            col=cen_xy[0] - 1,  # ngmix is zero offset
            scale=scale,
        )
    )

    flux = obs.image.sum()
    # prior = _get_moffat_prior(
    #     rng=rng,
    #     flux=obs.image.sum(),
    # )
    # # hlr = 0.5 * (prior.T_prior.minval + prior.T_prior.maxval)
    # hlr = 0.6
    prior = None
    hlr = 0.6

    if MODEL_FIT == "MOFFAT":
        guesser = MoffatGuesser(
            hlr=hlr,
            flux=flux,
            rng=rng,
        )
        if USE_CEN:
            fitter = CenGalsimMoffatFitter(
                prior=prior,
            )
        else:
            fitter = ngmix.fitting.GalsimMoffatFitter(
                prior=prior,
            )
    elif MODEL_FIT in GMIX_MODELS:
        guesser = MoffatGuesser(
            hlr=hlr,
            flux=flux,
            rng=rng,
        )
        fitter = ngmix.fitting.Fitter(MODEL_FIT.lower())

    for i in range(ntry):
        guess = guesser()
        res = fitter.go(obs=obs, guess=guess)
        if res['flags'] == 0:
            break

    if MODEL_FIT == "MOFFAT":
        if USE_CEN:
            beta_ind = 5-2
        else:
            beta_ind = 5
        if (
            res["flags"] == 0
            and res["pars"][beta_ind] >= 2
            and res["pars_err"][beta_ind] <= 1
        ):
            return res["pars"], res
        else:
            return None, res
    elif MODEL_FIT in GMIX_MODELS:
        _g1, _g2, _T = res["pars"][2:5]
        try:
            irr, irc, icc = ngmix.moments.g2mom(_g1, _g2, _T)
            # this is a fudge factor that gets the overall PSF FWHM
            # correct
            # the naive correction for the pixel size is
            # a bit too small
            pixel_var = 0.263 * 0.263 / 12 * 1.73
            irr -= pixel_var
            icc -= pixel_var
            _g1, _g2, _T = ngmix.moments.mom2g(irr, irc, icc)
        except Exception:
            _g1 = np.nan
            _g2 = np.nan
            _T = np.nan
            res["flags"] |= 2**1

        res["pars"][2] = _g1
        res["pars"][3] = _g2
        res["pars"][4] = _T

        if res["flags"] == 0:
            return res["pars"], res
        else:
            return None, res


class PSF(object):
    """The base class for describing a PSF model across a field of view.

    The usual way to create a PSF is through one of the two factory functions::

        >>> psf = piff.PSF.process(config, logger)
        >>> psf = piff.PSF.read(file_name, logger)

    The first is used to build a PSF model from the data according to a config dict.
    The second is used to read in a PSF model from disk.
    """
    @classmethod
    def process(cls, config_psf, logger=None):
        """Process the config dict and return a PSF instance.

        As the PSF class is an abstract base class, the returned type will in fact be some
        subclass of PSF according to the contents of the config dict.

        The provided config dict is typically the 'psf' field in the base config dict in
        a YAML file, although for compound PSF types, it may be the field for just one of
        several components.

        This function merely creates a "blank" PSF object.  It does not actually do any
        part of the solution yet.  Typically this will be followed by fit:

            >>> psf = piff.PSF.process(config['psf'])
            >>> stars, wcs, pointing = piff.Input.process(config['input'])
            >>> psf.fit(stars, wcs, pointing)

        at which point, the ``psf`` instance would have a solution to the PSF model.

        :param config_psf:  A dict specifying what type of PSF to build along with the
                            appropriate kwargs for building it.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance of the appropriate type.
        """
        import piff
        import yaml

        logger = galsim.config.LoggerWrapper(logger)
        logger.debug("Parsing PSF based on config dict:")
        logger.debug(yaml.dump(config_psf, default_flow_style=False))

        # Get the class to use for the PSF
        psf_type = config_psf.get('type', 'Simple') + 'PSF'
        logger.debug("PSF type is %s",psf_type)
        cls = getattr(piff, psf_type)

        # Read any other kwargs in the psf field
        kwargs = cls.parseKwargs(config_psf, logger)

        # Build PSF object
        logger.info("Building %s",psf_type)
        psf = cls(**kwargs)
        logger.debug("Done building PSF")

        return psf

    @classmethod
    def parseKwargs(cls, config_psf, logger=None):
        """Parse the psf field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_psf:      The psf field of the configuration dict, config['psf']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        raise NotImplementedError("Derived classes must define the parseKwargs function")

    # for apoinzation sims in the first round, default was apodize=(1.0, 4.25)
    def draw(self, x, y, chipnum=None, flux=1.0, center=None, offset=None, stamp_size=48,
             image=None, logger=None, apodize=None, use_smooth_model=True, **kwargs):
        r"""Draws an image of the PSF at a given location.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, stamp_size=48)

        However, if the PSF interpolation used extra properties for the interpolation
        (cf. psf.interp_property_names), you need to provide them as additional kwargs.

            >>> print(psf.interp_property_names)
            ('u','v','ri_color')
            >>> image = psf.draw(chipnum=4, x=103.3, y=592.0, ri_color=0.23, stamp_size=48)

        Normally, the image is constructed automatically based on stamp_size, in which case
        the WCS will be taken to be the local Jacobian at this location on the original image.
        However, if you provide your own image using the :image: argument, then whatever WCS
        is present in that image will be respected.  E.g. if you want an image of the PSF in
        sky coordinates rather than image coordinates, you can provide an image with just a
        pixel scale for the WCS.

        When drawing the PSF, there are a few options regarding how the profile will be
        centered on the image.

        1. The default behavior (``center==None``) is to draw the profile centered at the same
           (x,y) as you requested for the location of the PSF in the original image coordinates.
           The returned image will not (normally) be as large as the full image -- it will just be
           a postage stamp centered roughly around (x,y).  The image.bounds give the bounding box
           of this stamp, and within this, the PSF will be centered at position (x,y).
        2. If you want to center the profile at some other arbitrary position, you may provide
           a ``center`` parameter, which should be a tuple (xc,yc) giving the location at which
           you want the PSF to be centered.  The bounding box will still be around the nominal
           (x,y) position, so this should only be used for small adjustments to the (x,y) value
           if you want it centered at a slightly different location.
        3. If you provide your own image with the ``image`` parameter, then you may set the
           ``center`` to any location in this box (or technically off it -- it doesn't check that
           the center is actually inside the bounding box).  This may be useful if you want to draw
           on an image with origin at (0,0) or (1,1) and just put the PSF at the location you want.
        4. If you want the PSf centered exactly in the center of the image, then you can use
           ``center=True``.  This will work for either an automatically built image or one
           that you provide.
        5. With any of the above options you may additionally supply an ``offset`` parameter, which
           will apply a slight offset to the calculated center.  This is probably only useful in
           conjunction with the default ``center=None`` or ``center=True``.

        :param x:           The x position of the desired PSF in the original image coordinates.
        :param y:           The y position of the desired PSF in the original image coordinates.
        :param chipnum:     Which chip to use for WCS information. [required if the psf model
                            covers more than a single chip]
        :param flux:        Flux of PSF to be drawn [default: 1.0]
        :param center:      (xc,yc) tuple giving the location on the image where you want the
                            nominal center of the profile to be drawn.  Also allowed is the
                            value center=True to place in the center of the image.
                            [default: None, which means draw at the position (x,y) of the PSF.]
        :param offset:      Optional (dx,dy) tuple giving an additional offset relative to the
                            center. [default: None]
        :param stamp_size:  The size of the image to construct if no image is provided.
                            [default: 48]
        :param image:       An existing image on which to draw, if desired. [default: None]
        :param logger:      A logger object for logging debug info. [default: None]
        :param apodize:     Optional parameter to set apodizatoon. If a float/int, gives the
                            number of half light radii after which the profile is smoothy apodized
                            to zero a width of ~2.55 half light radii. If a tuple/list, gives
                            the apodization width and the apodization radius in pixels.
                            [default: None, which means no apodization.]
        :param \**kwargs:   Any additional properties required for the interpolation.

        :returns:           A GalSim Image of the PSF
        """
        logger = galsim.config.LoggerWrapper(logger)

        chipnum = self._check_chipnum(chipnum)

        prof, method = self.get_profile(
            x, y, chipnum=chipnum, flux=flux,
            logger=logger, use_smooth_model=use_smooth_model, **kwargs
        )

        logger.debug("Drawing star at (%s,%s) on chip %s", x, y, chipnum)

        # Make the image if necessary
        if image is None:
            image = galsim.Image(stamp_size, stamp_size, dtype=float)
            # Make the center of the image (close to) the image_pos
            xcen = int(np.ceil(x - (0.5 if image.array.shape[1] % 2 == 1 else 0)))
            ycen = int(np.ceil(y - (0.5 if image.array.shape[0] % 2 == 1 else 0)))
            image.setCenter(xcen, ycen)

        # If no wcs is given, use the original wcs
        if image.wcs is None:
            image.wcs = self.wcs[chipnum]

        # Handle the input center
        if center is None:
            center = (x, y)
        elif center is True:
            center = image.true_center
            center = (center.x, center.y)
        elif not isinstance(center, tuple):
            raise ValueError("Invalid center parameter: %r. Must be tuple or None or True"%(
                             center))

        # Handle offset if given
        if offset is not None:
            center = (center[0] + offset[0], center[1] + offset[1])

        prof.drawImage(image, method=method, center=center)

        if apodize:
            xpix, ypix = image.get_pixel_centers()
            dx = xpix - center[0]
            dy = ypix - center[1]
            r2 = dx**2 + dy**2

            if isinstance(apodize, (tuple, list)):
                apwidth, aprad = apodize
            else:
                wcs = image.wcs
                try:
                    image.wcs = None
                    image.scale = 1.0
                    hlr = image.calculateHLR(center=galsim.PositionD(center))
                finally:
                    image.wcs = wcs
                aprad = apodize * hlr
                msk_nonzero = image.array != 0
                max_r = min(
                    np.abs(dx[(dx < 0) & msk_nonzero].min()),
                    np.abs(dx[(dx > 0) & msk_nonzero].max()),
                    np.abs(dy[(dy < 0) & msk_nonzero].min()),
                    np.abs(dy[(dy > 0) & msk_nonzero].max()),
                )
                apwidth = np.abs(hlr) / 2.355
                apwidth = min(max(apwidth, 0.5), 5.0)
                aprad = max(min(aprad, max_r - 6 * apwidth - 1), 2 * apwidth)

            apim = image._array * _ap_kern_kern(aprad, np.sqrt(r2), apwidth)
            image._array = apim / np.sum(apim) * np.sum(image._array)

        return image

    def get_profile(
        self, x, y, chipnum=None, flux=1.0, logger=None,
        symmetrize_90=False, use_smooth_model=True, **kwargs
    ):
        r"""Get the PSF profile at the given position as a GalSim GSObject.

        The normal usage would be to specify (chipnum, x, y), in which case Piff will use the
        stored wcs information for that chip to interpolate to the given position and draw
        an image of the PSF:

            >>> prof, method = psf.get_profile(chipnum=4, x=103.3, y=592.0)

        The first return value, prof, is the GSObject describing the PSF profile.
        The second one, method, is the method parameter that should be used when drawing the
        profile using ``prof.drawImage(..., method=method)``.  This may be either 'no_pixel'
        or 'auto' depending on whether the PSF model already includes the pixel response or not.
        Some underlying models includ the pixel response, and some don't, so this difference needs
        to be accounted for properly when drawing.  This method is also appropriate if you first
        convolve the PSF by some other (e.g. galaxy) profile and then draw that.

        If the PSF interpolation used extra properties for the interpolation (cf.
        psf.extra_interp_properties), you need to provide them as additional kwargs.

            >>> print(psf.extra_interp_properties)
            ('ri_color',)
            >>> prof, method = psf.get_profile(chipnum=4, x=103.3, y=592.0, ri_color=0.23)

        :param x:           The x position of the desired PSF in the original image coordinates.
        :param y:           The y position of the desired PSF in the original image coordinates.
        :param chipnum:     Which chip to use for WCS information. [required if the psf model
                            covers more than a single chip]
        :param flux:        Flux of PSF model [default: 1.0]
        :param symmetrize_90: Symmetrize the profile by adding a 90 degree rotated version to itself.
        :param \**kwargs:   Any additional properties required for the interpolation.

        :returns:           (profile, method)
                            profile = A GalSim GSObject of the PSF
                            method = either 'no_pixel' or 'auto' indicating which method to use
                            when drawing the profile on an image.
        """
        logger = galsim.config.LoggerWrapper(logger)

        chipnum = self._check_chipnum(chipnum)

        properties = {'chipnum' : chipnum}
        for key in self.interp_property_names:
            if key in ['x','y','u','v']: continue
            if key not in kwargs:
                raise TypeError("Extra interpolation property %r is required"%key)
            properties[key] = kwargs.pop(key)
        if len(kwargs) != 0:
            raise TypeError("Unexpected keyword argument(s) %r"%list(kwargs.keys())[0])

        if use_smooth_model:
            if not hasattr(self, "_smooth_model"):
                self._compute_smooth_model(
                    wcs=self.wcs[chipnum],
                    properties=list(properties.keys()),
                    chipnum=chipnum,
                )
            prof = self._eval_smooth_model(
                x=x, y=y, **properties,
            ).withFlux(flux)
            method = "auto"
        else:
            image_pos = galsim.PositionD(x,y)
            wcs = self.wcs[chipnum]
            field_pos = StarData.calculateFieldPos(image_pos, wcs, self.pointing, properties)
            u,v = field_pos.x, field_pos.y

            star = Star.makeTarget(x=x, y=y, u=u, v=v, wcs=wcs, properties=properties,
                                pointing=self.pointing)
            logger.debug("Getting PSF profile at (%s,%s) on chip %s", x, y, chipnum)

            # Interpolate and adjust the flux of the star.
            star = self.interpolateStar(star).withFlux(flux)

            # The last step is implementd in the derived classes.
            prof, method = self._getProfile(star)

        if symmetrize_90:
            prof = (prof + prof.rotate(90 * galsim.degrees)) / 2.0

        return prof, method

    def _compute_smooth_model(self, *, wcs, properties, chipnum):
        from scipy.interpolate import RegularGridInterpolator

        self._smooth_model_failed = False
        self._smooth_model = None
        self._smooth_model_properties = sorted(
            (k for k in properties if k not in ["x", "y", "u", "v", "chipnum"])
        )
        self._smooth_model_fitter = None

        x_space = np.linspace(0.5, 2048+0.5, 2048 // DXY + 1) / 2049
        y_space = np.linspace(0.5, 4096+0.5, 4096 // DXY + 1) / 4097
        assert len(self._smooth_model_properties) == 1
        if self._smooth_model_properties[0].lower() == "gi_color":
            color_space = np.linspace(0, 3.5, DCOLOR)
        elif self._smooth_model_properties[0].lower() == "iz_color":
            color_space = np.linspace(0, 0.65, DCOLOR)
        else:
            raise ValueError(
                "Unknown smooth model property "
                f"{self._smooth_model_properties}"
            )

        from hashlib import sha1
        import os

        assert self.file_name is not None
        data = os.path.basename(self.file_name).encode()
        _hash = sha1(data)
        seed = np.frombuffer(_hash.digest(), dtype='uint32')
        rng = np.random.RandomState(seed)

        pars = None

        for j, _y in enumerate(y_space):
            for i, _x in enumerate(x_space):
                for k, color in enumerate(color_space):
                    x = _x + rng.uniform(low=-0.5, high=0.5) / 2049
                    y = _y + rng.uniform(low=-0.5, high=0.5) / 4097

                    x *= 2049
                    y *= 4097

                    nxy_2 = (NPIX_FIT + 1) / 2
                    cen = (
                        nxy_2 + x - int(x+0.5),
                        nxy_2 + y - int(y+0.5),
                    )
                    image = galsim.Image(NPIX_FIT, NPIX_FIT, scale=0.263)
                    prop = {self._smooth_model_properties[0]: color}
                    image = self.draw(
                        x, y,
                        chipnum=chipnum,
                        use_smooth_model=False,
                        image=image,
                        center=cen,
                        **prop,
                    )
                    _pars, _fitr = _do_ngmix_fit(
                        img=image.array, cen_xy=cen, scale=0.263,
                    )
                    if _pars is None:
                        self._smooth_model_failed = True
                        return

                    if pars is None:
                        pars = np.zeros((
                            len(y_space),
                            len(x_space),
                            len(color_space),
                            len(_pars)-1)
                        )
                    pars[j, i, k] = _pars[:-1]

        self._smooth_model = {
            chipnum: RegularGridInterpolator(
                (y_space, x_space, color_space),
                pars,
                bounds_error=False,
                fill_value=None,
                method="pchip",
            )
        }
        self._smooth_model_fitter = _fitr

    def _eval_smooth_model(self, *, x, y, chipnum, **properties):
        if self._smooth_model_failed:
            return galsim.Moffat(beta=2.5, fwhm=1.0, flux=1.0)
        else:
            _props = [y / 4097, x / 2049] + [
                properties[k]
                for k in self._smooth_model_properties
            ]
            pars = self._smooth_model[chipnum](_props)[0, :]
            pars = np.concatenate((pars, [1.0]), axis=0)
            if MODEL_FIT == "MOFFAT":
                return self._smooth_model_fitter.make_model(pars)
            elif MODEL_FIT in GMIX_MODELS:
                import ngmix
                return ngmix.gmix.make_gmix_model(
                    pars, MODEL_FIT.lower()
                ).make_galsim_object().withFlux(1.0)

    def _check_chipnum(self, chipnum):
        chipnums = list(self.wcs.keys())
        if chipnum is None:
            if len(chipnums) == 1:
                chipnum = chipnums[0]
            else:
                raise ValueError("chipnum is required.  Must be one of %s", str(chipnums))
        elif chipnum not in chipnums:
            raise ValueError("Invalid chipnum.  Must be one of %s", str(chipnums))
        return chipnum

    def interpolateStarList(self, stars):
        """Update the stars to have the current interpolated fit parameters according to the
        current PSF model.

        :param stars:       List of Star instances to update.

        :returns:           List of Star instances with their fit parameters updated.
        """
        return [self.interpolateStar(star) for star in stars]

    def interpolateStar(self, star):
        """Update the star to have the current interpolated fit parameters according to the
        current PSF model.

        :param star:        Star instance to update.

        :returns:           Star instance with its fit parameters updated.
        """
        raise NotImplementedError("Derived classes must define the interpolateStar function")

    def drawStarList(self, stars, copy_image=True):
        """Generate PSF images for given stars. Takes advantage of
        interpolateList for significant speedup with some interpolators.

        .. note::

            If the stars already have the fit parameters calculated, then this will trust
            those values and not redo the interpolation.  If this might be a concern, you can
            force the interpolation to be redone by running

                >>> stars = psf.interpolateList(stars)

            before running `drawStarList`.

        :param stars:       List of Star instances holding information needed
                            for interpolation as well as an image/WCS into
                            which PSF will be rendered.
        :param copy_image:  If False, will use the same image object.
                            If True, will copy the image and then overwrite it.
                            [default: True]

        :returns:           List of Star instances with its image filled with
                            rendered PSF
        """
        if any(star.fit is None or star.fit.params is None for star in stars):
            stars = self.interpolateStarList(stars)
        return [self._drawStar(star, copy_image=copy_image) for star in stars]

    def drawStar(self, star, copy_image=True, center=None):
        """Generate PSF image for a given star.

        .. note::

            If the star already has the fit parameters calculated, then this will trust
            those values and not redo the interpolation.  If this might be a concern, you can
            force the interpolation to be redone by running

                >>> star = psf.interpolateList(star)

            before running `drawStar`.

        :param star:        Star instance holding information needed for interpolation as
                            well as an image/WCS into which PSF will be rendered.
        :param copy_image:  If False, will use the same image object.
                            If True, will copy the image and then overwrite it.
                            [default: True]
        :param center:      An optional tuple (x,y) location for where to center the drawn profile
                            in the image. [default: None, which draws at the star's location.]

        :returns:           Star instance with its image filled with rendered PSF
        """
        # Interpolate parameters to this position/properties (if not already done):
        if star.fit is None or star.fit.params is None:
            star = self.interpolateStar(star)
        # Render the image
        return self._drawStar(star, copy_image=copy_image, center=center)

    def _drawStar(self, star, copy_image=True, center=None):
        # Derived classes may choose to override any of the above functions
        # But they have to at least override this one and interpolateStar to implement
        # their actual PSF model.
        raise NotImplementedError("Derived classes must define the _drawStar function")

    def _getProfile(self, star):
        raise NotImplementedError("Derived classes must define the _getProfile function")

    def write(self, file_name, logger=None):
        """Write a PSF object to a file.

        :param file_name:   The name of the file to write to.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.warning("Writing PSF to file %s",file_name)

        with fitsio.FITS(file_name,'rw',clobber=True) as f:
            self._write(f, 'psf', logger)

    def _write(self, fits, extname, logger=None):
        """This is the function that actually does the work for the write function.
        Composite PSF classes that need to iterate can call this multiple times as needed.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the psf information.
        :param logger:      A logger object for logging debug info.
        """
        from . import __version__ as piff_version
        if len(fits) == 1:
            header = {'piff_version': piff_version}
            fits.write(data=None, header=header)
        psf_type = self.__class__.__name__
        write_kwargs(fits, extname, dict(self.kwargs, type=psf_type, piff_version=piff_version))
        logger.info("Wrote the basic PSF information to extname %s", extname)
        Star.write(self.stars, fits, extname=extname + '_stars')
        logger.info("Wrote the PSF stars to extname %s", extname + '_stars')
        self.writeWCS(fits, extname=extname + '_wcs', logger=logger)
        logger.info("Wrote the PSF WCS to extname %s", extname + '_wcs')
        self._finish_write(fits, extname=extname, logger=logger)

    @classmethod
    def read(cls, file_name, logger=None):
        """Read a PSF object from a file.

        :param file_name:   The name of the file to read.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a PSF instance
        """
        logger = galsim.config.LoggerWrapper(logger)
        logger.warning("Reading PSF from file %s",file_name)

        with fitsio.FITS(file_name,'r') as f:
            logger.debug('opened FITS file')
            return cls._read(f, 'psf', logger, file_name=file_name)

    @classmethod
    def _read(cls, fits, extname, logger, file_name=None):
        """This is the function that actually does the work for the read function.
        Composite PSF classes that need to iterate can call this multiple times as needed.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the psf information.
        :param logger:      A logger object for logging debug info.
        """
        import piff

        # Read the type and kwargs from the base extension
        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        kwargs = read_kwargs(fits, extname)
        psf_type = kwargs.pop('type')

        # If piff_version is not in the file, then it was written prior to version 1.3.
        # Since we don't know what version it was, we just use None.
        piff_version = kwargs.pop('piff_version',None)

        # Check that this is a valid PSF type
        psf_classes = piff.util.get_all_subclasses(piff.PSF)
        valid_psf_types = dict([ (c.__name__, c) for c in psf_classes ])
        if psf_type not in valid_psf_types:
            raise ValueError("psf type %s is not a valid Piff PSF"%psf_type)
        psf_cls = valid_psf_types[psf_type]

        # Read the stars, wcs, pointing values
        stars = Star.read(fits, extname + '_stars')
        logger.debug("stars = %s",stars)
        wcs, pointing = cls.readWCS(fits, extname + '_wcs', logger=logger)
        logger.debug("wcs = %s, pointing = %s",wcs,pointing)

        # Make the PSF instance
        psf = psf_cls(**kwargs)
        psf.stars = stars
        psf.wcs = wcs
        psf.pointing = pointing

        # Just in case the class needs to do something else at the end.
        psf._finish_read(fits, extname, logger)

        # Save the piff version as an attibute.
        psf.piff_version = piff_version

        psf.file_name = file_name

        return psf

    def writeWCS(self, fits, extname, logger):
        """Write the WCS information to a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write to
        :param logger:      A logger object for logging debug info.
        """
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        logger = galsim.config.LoggerWrapper(logger)

        # Start with the chipnums
        chipnums = list(self.wcs.keys())
        cols = [ chipnums ]
        dtypes = [ ('chipnums', int) ]

        # GalSim WCS objects can be serialized via pickle
        wcs_str = [ base64.b64encode(pickle.dumps(w)) for w in self.wcs.values() ]
        max_len = np.max([ len(s) for s in wcs_str ])
        # Some GalSim WCS serializations are rather long.  In particular, the Pixmappy one
        # is longer than the maximum length allowed for a column in a fits table (28799).
        # So split it into chunks of size 2**14 (mildly less than this maximum).
        chunk_size = 2**14
        nchunks = max_len // chunk_size + 1
        cols.append( [nchunks]*len(chipnums) )
        dtypes.append( ('nchunks', int) )

        # Update to size of chunk we actually need.
        chunk_size = (max_len + nchunks - 1) // nchunks

        chunks = [ [ s[i:i+chunk_size] for i in range(0, max_len, chunk_size) ] for s in wcs_str ]
        cols.extend(zip(*chunks))
        dtypes.extend( ('wcs_str_%04d'%i, bytes, chunk_size) for i in range(nchunks) )

        if self.pointing is not None:
            # Currently, there is only one pointing for all the chips, but write it out
            # for each row anyway.
            dtypes.extend( (('ra', float), ('dec', float)) )
            ra = [self.pointing.ra / galsim.hours] * len(chipnums)
            dec = [self.pointing.dec / galsim.degrees] * len(chipnums)
            cols.extend( (ra, dec) )

        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname)

    @classmethod
    def readWCS(cls, fits, extname, logger):
        """Read the WCS information from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to read from
        :param logger:      A logger object for logging debug info.

        :returns: wcs, pointing where wcs is a dict of galsim.BaseWCS instances and
                                      pointing is a galsim.CelestialCoord instance
        """
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        assert extname in fits
        assert 'chipnums' in fits[extname].get_colnames()
        assert 'nchunks' in fits[extname].get_colnames()

        data = fits[extname].read()

        chipnums = data['chipnums']
        nchunks = data['nchunks']
        nchunks = nchunks[0]  # These are all equal, so just take first one.

        wcs_keys = [ 'wcs_str_%04d'%i for i in range(nchunks) ]
        wcs_str = [ data[key] for key in wcs_keys ] # Get all wcs_str columns
        try:
            wcs_str = [ b''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each
        except TypeError:  # pragma: no cover
            # fitsio 1.0 returns strings
            wcs_str = [ ''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each

        wcs_str = [ base64.b64decode(s) for s in wcs_str ] # Convert back from b64 encoding
        # Convert back into wcs objects
        try:
            wcs_list = [ pickle.loads(s, encoding='bytes') for s in wcs_str ]
        except Exception:
            # If the file was written by py2, the bytes encoding might raise here,
            # or it might not until we try to use it.
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]

        wcs = dict(zip(chipnums, wcs_list))

        try:
            # If this doesn't work, then the file was probably written by py2, not py3
            repr(wcs)
        except Exception:
            logger.info('Failed to decode wcs with bytes encoding.')
            logger.info('Retry with encoding="latin1" in case file written with python 2.')
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]
            wcs = dict(zip(chipnums, wcs_list))
            repr(wcs)

        # Work-around for a change in the GalSim API with 2.0
        # If the piff file was written with pre-2.0 GalSim, this fixes it.
        for key in wcs:
            w = wcs[key]
            if hasattr(w, '_origin') and  isinstance(w._origin, galsim._galsim.PositionD):
                w._origin = galsim.PositionD(w._origin)

        if 'ra' in fits[extname].get_colnames():
            ra = data['ra']
            dec = data['dec']
            pointing = galsim.CelestialCoord(ra[0] * galsim.hours, dec[0] * galsim.degrees)
        else:
            pointing = None

        return wcs, pointing

# Make a global function, piff.read, as an alias for piff.PSF.read, since that's the main thing
# users will want to do as their starting point for using a piff file.
def read(file_name, logger=None):
    """Read a Piff PSF object from a file.

    .. note::

        The returned PSF instance will have an attribute piff_version, which
        indicates the version of Piff that was used to create the file.  (If it was written with
        Piff version >= 1.3.0.)

    :param file_name:   The name of the file to read.
    :param logger:      A logger object for logging debug info. [default: None]

    :returns: a piff.PSF instance
    """
    return PSF.read(file_name, logger=logger)
