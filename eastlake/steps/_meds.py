from __future__ import print_function, absolute_import
import os
import json
from functools import reduce

import yaml
from datetime import timedelta
from timeit import default_timer as timer

import fitsio
import numpy as np
import galsim
import galsim.des

import meds
import psfex
import desmeds
from desmeds.files import StagedOutFile

from ..utils import safe_mkdir, pushd, copy_ifnotexists, safe_rm
from ..des_piff import DES_Piff, PSF_KWARGS
from .meds_psf_interface import PSFForMeds
from ..step import Step, run_and_check
from ..stash import Stash
from ..des_files import MAGZP_REF
from .swarp import FITSEXTMAP

# This is for MEDS boxsize calculation.
FWHM_FAC = 2*np.sqrt(2*np.log(2))


def _remap_fitsext(ext):
    if not isinstance(ext, int):
        ext = FITSEXTMAP[ext]
    return ext


# Choose the boxsize - this is the same method as used in desmeds
# Pasted in these functions from desmeds.
def _get_box_sizes(cat, config):
    """
    get box sizes that are wither 2**N or 3*2**N, within
    the limits set by the user
    """
    sigma_size = _get_sigma_size(cat, config)

    # now do row and col sizes
    row_size = cat['ymax_image'] - cat['ymin_image'] + 1
    col_size = cat['xmax_image'] - cat['xmin_image'] + 1

    # get max of all three
    box_size = np.vstack(
        (col_size, row_size, sigma_size)).max(axis=0)

    # clip to range
    box_size = box_size.clip(config['min_box_size'], config['max_box_size'])

    # now put in fft sizes
    bins = [0]
    bins.extend([sze for sze in config['allowed_box_sizes']
                 if sze >= config['min_box_size']
                 and sze <= config['max_box_size']])

    if bins[-1] != config['max_box_size']:
        bins.append(config['max_box_size'])

    bin_inds = np.digitize(box_size, bins, right=True)
    bins = np.array(bins)

    return bins[bin_inds]


def _get_sigma_size(cat, config):
    """
    "sigma" size, based on flux radius and ellipticity
    """
    ellipticity = 1.0 - cat['b_world']/cat['a_world']
    sigma = cat['flux_radius']*2.0/FWHM_FAC
    drad = sigma*config['sigma_fac']
    drad = drad*(1.0 + ellipticity)
    drad = np.ceil(drad)
    # sigma size is twice the radius
    sigma_size = 2*drad.astype('i4')

    return sigma_size


class MEDSRunner(Step):
    """
    Pipeline step for generating MEDS files
    """
    def __init__(self, config, base_dir, name="meds", logger=None,
                 verbosity=0, log_file=None):
        super(MEDSRunner, self).__init__(
            config, base_dir, name=name,
            logger=logger, verbosity=verbosity, log_file=log_file)

        self.config["meds_dir"] = self.base_dir
        os.environ["MEDS_DIR"] = self.config["meds_dir"]

        # And set some defaults in the config
        if "allowed_box_sizes" not in self.config:
            self.config["allowed_box_sizes"] = [
                2, 3, 4, 6, 8, 12, 16, 24, 32, 48,
                64, 96, 128, 192, 256,
                384, 512, 768, 1024, 1536,
                2048, 3072, 4096, 6144
            ]
        if "min_box_size" not in self.config:
            self.config["min_box_size"] = 32
        if "max_box_size" not in self.config:
            self.config["max_box_size"] = 256
        if "sigma_fac" not in self.config:
            self.config["sigma_fac"] = 5.0
        if "refband" not in self.config:
            self.config["refband"] = "r"
        if "stage_output" not in self.config:
            self.config["stage_output"] = False
        self.config["use_nwgint"] = self.config.get("use_nwgint", False)

        if "fpack_pars" not in self.config:
            self.config["fpack_pars"] = {
                "FZQVALUE": 4,
                "FZTILE": "(10240,1)",
                "FZALGOR": "RICE_1",
                # preserve zeros, don't dither them
                "FZQMETHD": "SUBTRACTIVE_DITHER_2",
            }

    def clear_stash(self, stash):
        # If we continued the pipeline from a previous job record file,
        # mof_file entries can mess things up, so clear them
        if "tile_info" in stash:
            for tilename, tile_file_info in stash["tile_info"].items():
                tile_file_info.pop("meds_files", None)

    def execute(self, stash, new_params=None, boxsize=64):

        self.clear_stash(stash)

        if self.config.get("profile", False):
            import cProfile
            pr = cProfile.Profile()
            pr.enable()
        else:
            pr = None

        # And add meds_run and MEDS_DIR to stash for later
        self.config["meds_run"] = stash["desrun"]
        stash["meds_run"] = self.config["meds_run"]
        stash["env"].append(("MEDS_DIR", os.environ["MEDS_DIR"]))

        # https://github.com/esheldon/meds/wiki/Creating-MEDS-Files-in-Python
        # Need to generate
        #  i) object_data structure
        #  ii) image_info structure

        # Loop through tiles
        tilenames = stash["tilenames"]

        if "use_srcex_from" in self.config:
            other_stash = Stash.load(
                self.config["use_srcex_from"],
                os.path.dirname(self.config["use_srcex_from"]), [])

        for tilename in tilenames:

            t0 = timer()

            # meds files
            meds_files = []

            # object_data comes from SExtractor - assume either src extractor
            # catalog filename is in the stash
            # or the data is in the stash as a recarray
            # Use SExtractor catalog for refband

            # Hack to use src extractor catalog and segmentation map from
            # different run
            refband = self.config["refband"]
            if "use_srcex_from" in self.config:
                srcex_cat = other_stash.get_filepaths(
                    "srcex_cat", tilename, band=refband)

                # the seg map may not exist if we are doing true detection
                try:
                    seg_file, seg_ext = other_stash.get_filepaths(
                        "seg_file", tilename, band=refband, with_fits_ext=True,
                    )
                    if len(os.path.split(seg_file)[1]) == 0:
                        raise KeyError("no seg file")
                except KeyError:
                    seg_file = ''
                    seg_ext = -1
            # Normal mode:
            else:
                srcex_cat = stash.get_filepaths(
                    "srcex_cat", tilename, band=refband)

                # the seg map may not exist if we are doing true detection
                try:
                    seg_file, seg_ext = stash.get_filepaths(
                        "seg_file", tilename, band=refband, with_fits_ext=True,
                    )
                    if len(os.path.split(seg_file)[1]) == 0:
                        raise KeyError("no seg file")
                except KeyError:
                    seg_file = ''
                    seg_ext = -1

            seg_ext = _remap_fitsext(seg_ext)

            stash.set_filepaths("srcex_cat", srcex_cat, tilename)

            try:
                srcex_data = fitsio.read(srcex_cat, lower=True)
            except IOError:
                # you can get an IOError here if there's no detected
                # objects...in this case we want to quit the
                # pipeline gracefully
                if os.path.isfile(srcex_cat):
                    self.logger.error(
                        "IOError when trying to read SrcExtractor catalog but "
                        "file exists - maybe no objects were detected?")
                else:
                    self.logger.error(
                        "IOError when trying to read SrcExtractor catalog "
                        "and file does not exist")
                return 1, stash

            extra_obj_data_fields = [
                ('number', 'i8'),
            ]

            # This is the same for each band
            obj_data = meds.util.get_meds_input_struct(
                len(srcex_data), extra_fields=extra_obj_data_fields)
            obj_data["id"] = srcex_data["number"]
            obj_data["number"] = srcex_data["number"]
            obj_data["ra"] = srcex_data["alpha_j2000"]
            obj_data["dec"] = srcex_data["delta_j2000"]

            # Get boxsizes
            obj_data["box_size"] = _get_box_sizes(srcex_data, self.config)

            t1 = timer()
            self.logger.error(
                "Time to set up obj_data for tile %s: %s" % (
                    tilename, str(timedelta(seconds=t1-t0))))

            # image data
            for band in stash["bands"]:
                t0 = timer()

                self._copy_inputs(stash, tilename, band)

                # coadd stuff
                coadd_file, coadd_ext = stash.get_filepaths(
                    "coadd_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                coadd_ext = _remap_fitsext(coadd_ext)
                coadd_weight_file, coadd_weight_ext = stash.get_filepaths(
                    "coadd_weight_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                coadd_weight_ext = _remap_fitsext(coadd_weight_ext)
                coadd_bmask_file, coadd_bmask_ext = stash.get_filepaths(
                    "coadd_mask_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                coadd_bmask_ext = _remap_fitsext(coadd_bmask_ext)

                # headers and WCS
                coadd_header = fitsio.read_header(
                    coadd_file, coadd_ext)
                # delete any None entries
                coadd_header.delete(None)
                coadd_header = desmeds.util.fitsio_header_to_dict(
                    coadd_header)

                coadd_wcs, _ = galsim.wcs.readFromFitsHeader(
                    galsim.FitsHeader(file_name=coadd_file, hdu=coadd_ext)
                )

                if (
                    stash.has_tile_info_quantity("coadd_bkg_file", tilename, band=band)
                    and self.config["sub_coadd_bkg"]
                ):
                    raise ValueError("Backgrounds for coadds are not supported!")

                slen = max(
                    len(coadd_file), len(coadd_weight_file), len(seg_file))

                # get wcs json thing
                wcs_json = []
                wcs_json.append(json.dumps(coadd_header))

                # single-epoch stuff
                if (
                    stash.has_tile_info_quantity("img_files", tilename, band=band)
                    or
                    stash.has_tile_info_quantity("coadd_nwgint_img_files", tilename, band=band)
                ):
                    if self.config["use_nwgint"]:
                        pre = "coadd_nwgint_"
                    else:
                        pre = ""
                    img_files, img_ext = stash.get_filepaths(
                        pre+"img_files", tilename, band=band, with_fits_ext=True,
                    )
                    img_ext = _remap_fitsext(img_ext)
                    head_files = stash.get_filepaths(
                        "head_files", tilename, band=band,
                    )

                    # Are we rejectlisting?
                    is_rejectlisted = [False] * len(img_files)

                    wgt_files, wgt_ext = stash.get_filepaths(
                        pre+"wgt_files", tilename, band=band, with_fits_ext=True,
                    )
                    wgt_ext = _remap_fitsext(wgt_ext)

                    msk_files, msk_ext = stash.get_filepaths(
                        pre+"msk_files", tilename, band=band, with_fits_ext=True,
                    )
                    msk_ext = _remap_fitsext(msk_ext)

                    mag_zps = stash.get_tile_info_quantity("mag_zps", tilename, band=band)

                    bkg_files, bkg_ext = stash.get_filepaths(
                        "bkg_files", tilename, band=band, with_fits_ext=True,
                    )
                    if not isinstance(bkg_ext, int):
                        if bkg_files[0].endswith(".fits.fz"):
                            bkg_ext = 1
                        else:
                            bkg_ext = 0

                    img_files = [f for (i, f) in enumerate(img_files) if not is_rejectlisted[i]]
                    head_files = [f for (i, f) in enumerate(head_files) if not is_rejectlisted[i]]
                    wgt_files = [f for (i, f) in enumerate(wgt_files) if not is_rejectlisted[i]]
                    msk_files = [f for (i, f) in enumerate(msk_files) if not is_rejectlisted[i]]
                    bkg_files = [f for (i, f) in enumerate(bkg_files) if not is_rejectlisted[i]]
                    mag_zps = [m for (i, m) in enumerate(mag_zps) if not is_rejectlisted[i]]

                    n_images = len(img_files)
                    for i in range(n_images):
                        slen = max(
                            slen,
                            max(len(img_files[i]), len(bkg_files[i])),
                        )

                    for img_file, head_file in zip(img_files, head_files):
                        im_h = fitsio.read_header(img_file, img_ext)
                        h = fitsio.read_scamp_head(head_file)
                        for naxis_key in ["naxis1", "naxis2", "znaxis1", "znaxis2"]:
                            if naxis_key in im_h:
                                h[naxis_key] = im_h[naxis_key]
                        h.delete(None)
                        wcs_json.append(
                            json.dumps(desmeds.util.fitsio_header_to_dict(h)))
                else:
                    n_images = 0
                    is_rejectlisted = []
                    slen = len(coadd_file)

                wcs_len = reduce(lambda x, y: max(x, len(y)), wcs_json, 0)
                image_info = meds.util.get_image_info_struct(
                    n_images+1, slen, wcs_len=wcs_len)

                # assuming src extractor positions
                image_info["position_offset"] = 1.

                # fill coadd quantities
                image_info["image_path"][0] = coadd_file
                image_info["image_ext"][0] = coadd_ext
                image_info["weight_path"][0] = coadd_weight_file
                image_info["weight_ext"][0] = coadd_weight_ext
                image_info["bmask_path"][0] = coadd_bmask_file
                image_info["bmask_ext"][0] = coadd_bmask_ext
                image_info["bkg_path"][0] = ""  # No bkg file for coadd
                image_info["bkg_ext"][0] = -1
                image_info["seg_path"][0] = seg_file
                image_info["seg_ext"][0] = seg_ext
                image_info["wcs"][0] = wcs_json[0]
                image_info["magzp"][0] = MAGZP_REF
                image_info["scale"][0] = 1.

                for i in range(n_images):
                    ind = i+1
                    image_info["image_path"][ind] = img_files[i]
                    image_info["image_ext"][ind] = img_ext
                    image_info["weight_path"][ind] = wgt_files[i]
                    image_info["weight_ext"][ind] = wgt_ext
                    image_info["bmask_path"][ind] = msk_files[i]
                    image_info["bmask_ext"][ind] = msk_ext

                    image_info["bkg_path"][ind] = bkg_files[i]
                    image_info["bkg_ext"][ind] = bkg_ext
                    image_info["seg_path"][ind] = ''
                    image_info["seg_ext"][ind] = -1

                    image_info["wcs"][ind] = wcs_json[ind]
                    image_info["magzp"][ind] = mag_zps[i]
                    image_info["scale"][ind] = 10**(0.4*(MAGZP_REF-mag_zps[i]))

                # not sure if we need to set image_id - but do so anyway
                image_info["image_id"] = np.arange(n_images+1)

                # metadata
                meta_data = np.zeros(1, dtype=[("magzp_ref", np.float64)])
                meta_data["magzp_ref"] = MAGZP_REF

                t1 = timer()
                self.logger.error(
                    "Time to set up image_info for tile %s, band %s: %s" % (
                        tilename, band, str(timedelta(seconds=t1-t0))))

                t0 = timer()
                psf_data = self._make_psf_data(
                    stash, coadd_wcs, tilename, band,
                    is_rejectlisted, img_files, img_ext, head_files,
                )
                t1 = timer()
                self.logger.error(
                    "Time to set generate psf data for tile %s, "
                    "band %s: %s" % (
                        tilename, band, str(timedelta(seconds=t1-t0))))

                # call meds maker
                t0 = timer()
                maker = meds.MEDSMaker(
                    obj_data, image_info,
                    psf_data=psf_data,
                    config=self.config,
                    meta_data=meta_data)
                t1 = timer()
                self.logger.error(
                    "Time to set up MEDS file for tile %s, band %s: %s" % (
                        tilename, band, str(timedelta(seconds=t1-t0))))

                # MEDS file should have format
                #  $MEDS_DIR/{meds_run}/{tilename}/{tilename}_{band}_meds-{meds_run}.fits.fz  # noqa
                # The use of an environment variable seems dangerous here...but
                # I guess it shouldn't be a problem as long
                # as we're not running multiple sims on the same machine....
                t0 = timer()
                meds_file = os.path.join(
                    self.config["meds_dir"], self.config["meds_run"], tilename,
                    "%s_%s_meds-%s.fits.fz" % (tilename, band, self.config["meds_run"]))
                meds_files.append(meds_file)

                d = os.path.dirname(os.path.normpath(meds_file))
                safe_mkdir(d)

                # If requested, we stage the file at $TMPDIR and then
                # copy over when finished writing. This is a good idea
                # on many systems, but maybe not all.
                if "fpack_pars" in self.config:
                    if self.config["stage_output"]:
                        tmpdir = os.environ.get("TMPDIR", None)
                        with StagedOutFile(meds_file, tmpdir=tmpdir) as sf:
                            maker.write(sf.path[:-len(".fz")])
                            with pushd(os.path.dirname(sf.path)):
                                run_and_check(
                                    ["fpack", os.path.basename(sf.path[:-len(".fz")])],
                                    "fpack meds",
                                    logger=self.logger,
                                )
                    else:
                        maker.write(meds_file[:-len(".fz")])
                        with pushd(os.path.dirname(meds_file)):
                            run_and_check(
                                ["fpack", os.path.basename(meds_file[:-len(".fz")])],
                                "fpack meds",
                                logger=self.logger,
                            )
                    safe_rm(meds_file[:-len(".fz")])
                else:
                    if self.config["stage_output"]:
                        tmpdir = os.environ.get("TMPDIR", None)
                        with StagedOutFile(meds_file, tmpdir=tmpdir) as sf:
                            maker.write(sf.path)
                    else:
                        maker.write(meds_file)
                t1 = timer()
                self.logger.error(
                    "Time to write meds file for tile %s, band %s: %s" % (
                        tilename, band, str(timedelta(seconds=t1-t0))))

            stash.set_filepaths("meds_files", meds_files, tilename)

        if pr is not None:
            pr.print_stats(sort='time')
        return 0, stash

    def _copy_inputs(self, stash, tilename, band):
        # copy input files
        in_pyml = stash.get_input_pizza_cutter_yaml(tilename, band)
        pyml = stash.get_output_pizza_cutter_yaml(tilename, band)
        for i in range(len(pyml["src_info"])):
            # we don't overwrite these since we could have estimated them
            copy_ifnotexists(
                in_pyml["src_info"][i]["head_path"],
                pyml["src_info"][i]["head_path"],
            )
            copy_ifnotexists(
                in_pyml["src_info"][i]["piff_path"],
                pyml["src_info"][i]["piff_path"],
            )
            copy_ifnotexists(
                in_pyml["src_info"][i]["psf_path"],
                pyml["src_info"][i]["psf_path"],
            )

    def _make_psf_data(self, stash, coadd_wcs, tilename, band, is_rejectlisted, img_files, img_ext, head_files):
        if stash["psf_config"]["type"] == "DES_Piff":
            self.logger.error("Adding Piff info to meds file")
            return self._make_psf_data_piff(
                stash, coadd_wcs, tilename, band, is_rejectlisted, img_files, img_ext, head_files)
        elif stash["psf_config"]["type"] in [
                "DES_PSFEx", "DES_PSFEx_perturbed"
        ]:
            self.logger.error("Adding psfex info to meds file")
            return self._make_psf_data_psfex(
                stash, coadd_wcs, tilename, band, is_rejectlisted, img_files, img_ext, head_files)
        elif stash["psf_config"]["type"] == "Gaussian":
            self.logger.error("Adding Gauss PSF info to meds file")
            return self._make_psf_data_gauss(
                stash, coadd_wcs, tilename, band, is_rejectlisted, img_files, img_ext, head_files)
        else:
            raise RuntimeError(
                "PSF config type '%s' not recognized!" % stash["psf_config"]["type"]
            )

    def _make_psf_data_piff(
        self, stash, coadd_wcs, tilename, band, is_rejectlisted, img_files, img_ext, head_files,
    ):
        smooth = stash["psf_config"].get("smooth", False)
        if smooth:
            assert stash["draw_method"] == "auto"
        else:
            assert stash["draw_method"] == "no_pixel"

        # this is the type for the meds maker - sets the layout etc
        # we are using the same API so our type is psfex even
        # though we will put piff data in it
        self.config["psf_type"] = "psfex"  # it's ok

        psf_data = []

        # there is no piff coadd PSF so we fake it
        psf_data.append(
            PSFForMeds(
                galsim.Gaussian(fwhm=0.9),
                coadd_wcs,
                "auto")
        )

        if (
            stash.has_tile_info_quantity("img_files", tilename, band=band)
            or stash.has_tile_info_quantity("coadd_nwgint_img_files", tilename, band=band)
        ):
            piff_files = stash.get_filepaths("piff_files", tilename, band=band)
            piff_files = [f for (i, f) in enumerate(piff_files) if not is_rejectlisted[i]]

            self.logger.error("se piff files: %s" % piff_files)
            for piff_file, img_file, head_file in zip(piff_files, img_files, head_files):
                psf = DES_Piff(
                    piff_file,
                    smooth=smooth,
                    psf_kwargs=PSF_KWARGS[band],
                )
                img_wcs = galsim.FitsWCS(
                    file_name=head_file,
                    text_file=True,
                )
                psf_data.append(PSFForMeds(psf, img_wcs, stash["draw_method"]))

        return psf_data

    def _make_psf_data_psfex(
        self, stash, coadd_wcs, tilename, band, is_rejectlisted, img_files, img_ext, head_files,
    ):
        self.config["psf_type"] = "psfex"

        psf_data = []

        # first look for coadd psfex file.
        pyml = stash.get_output_pizza_cutter_yaml(tilename, band)
        coadd_psfex_path = pyml["psf_path"]
        coadd_file = pyml["image_path"]
        self.logger.error("coadd psfex file: %s" % (coadd_psfex_path))

        if self.config.get("use_galsim_psfex", True):
            self.logger.error(
                "using galsim to reconstruct psfex for "
                "meds file")
            psf = galsim.des.DES_PSFEx(coadd_psfex_path, image_file_name=coadd_file)
            psf_data.append(PSFForMeds(psf, coadd_wcs, "no_pixel"))
        else:
            psf_data.append(psfex.PSFEx(coadd_psfex_path))

        if (
            stash.has_tile_info_quantity("img_files", tilename, band=band)
            or stash.has_tile_info_quantity("coadd_nwgint_img_files", tilename, band=band)
        ):
            psfex_files = stash.get_filepaths("psfex_files", tilename, band=band)
            psfex_files = [f for (i, f) in enumerate(psfex_files) if not is_rejectlisted[i]]

            self.logger.error("se psfex files: %s" % psfex_files)
            for psfex_file, img_file, head_file in zip(psfex_files, img_files, head_files):
                if self.config.get("use_galsim_psfex", True):
                    self.logger.error(
                        "using galsim to reconstruct psfex "
                        "for meds file")
                    psf = galsim.des.DES_PSFEx(
                        psfex_file, image_file_name=img_file)
                    img_wcs = galsim.FitsWCS(
                        file_name=head_file,
                        text_file=True,
                    )
                    psf_data.append(PSFForMeds(psf, img_wcs, "no_pixel"))
                else:
                    psf_data.append(psfex.PSFEx(psfex_file))

        return psf_data

    def _make_psf_data_gauss(
        self, stash, coadd_wcs, tilename, band, is_rejectlisted, img_files, img_ext, head_files
    ):
        self.config["psf_type"] = "psfex"  # always use this for saving PSFs
        size_key = None
        for size_key in ["half_light_radius", "sigma", "fwhm"]:
            if size_key in stash["psf_config"]:
                try:
                    float(stash["psf_config"][size_key])
                except ValueError as e:
                    self.logger.error(
                        "couldn't interpret psf %s "
                        "as float" % (size_key))
                    raise e
                break
        psf = galsim.Gaussian(**{size_key: stash["psf_config"][size_key]})

        # now generate psf data
        psf_data = []
        psf_data.append(
            PSFForMeds(
                psf,
                coadd_wcs,
                stash.get("draw_method", "auto")))

        if (
            stash.has_tile_info_quantity("img_files", tilename, band=band)
            or stash.has_tile_info_quantity("coadd_nwgint_img_files", tilename, band=band)
        ):
            for head_file in head_files:
                wcs = galsim.FitsWCS(
                    file_name=head_file,
                    text_file=True,
                )
                psf_data.append(
                    PSFForMeds(
                        psf,
                        wcs,
                        stash.get("draw_method", "auto")
                    )
                )

        return psf_data

    @classmethod
    def from_config_file(cls, config_file, base_dir=None, logger=None,
                         name="meds"):
        with open(config_file, "rb") as f:
            config = yaml.load(f)
        return cls(config, base_dir=base_dir, logger=logger, name=name)
