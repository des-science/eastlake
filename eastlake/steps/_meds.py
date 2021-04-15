from __future__ import print_function, absolute_import
import os
import json
from functools import reduce
import shutil

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

from ..utils import safe_mkdir
from ..des_piff import DES_Piff
from .meds_psf_interface import PSFForMeds
from ..step import Step
from ..stash import Stash
from ..des_files import (
    get_orig_coadd_file, get_psfex_path, get_psfex_path_coadd, get_bkg_path,
    get_piff_path)
from ..rejectlist import RejectList

# This is for MEDS boxsize calculation.
FWHM_FAC = 2*np.sqrt(2*np.log(2))


class MEDSRunner(Step):
    """
    Pipeline step for generating MEDS files
    """
    def __init__(self, config, base_dir, name="meds", logger=None,
                 verbosity=0, log_file=None):
        super(MEDSRunner, self).__init__(
            config, base_dir, name=name,
            logger=logger, verbosity=verbosity, log_file=log_file)
        # setup meds_dir environment variable
        if not os.path.isabs(self.config["meds_dir"]):
            self.config["meds_dir"] = os.path.join(
                self.base_dir, self.config["meds_dir"])
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
        if "sub_bkg" not in self.config:
            self.config["sub_bkg"] = False
        if "magzp_ref" not in self.config:
            self.config["magzp_ref"] = 30.
        if "stage_output" not in self.config:
            self.config["stage_output"] = False
        if "use_rejectlist" not in self.config:
            # We can rejectlist some of the images
            self.config["use_rejectlist"] = True

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
        stash["meds_run"] = self.config["meds_run"]
        stash["env"].append(("MEDS_DIR", os.environ["MEDS_DIR"]))

        # If we used DES_Piff psfs, we should have also done some rejectlisting.
        # Make sure then that there is a rejectlist in stash["rejectlist"],
        # and that self.config["use_rejectlist"] is True
        if stash["psf_config"]["type"] == "DES_Piff":
            try:
                assert self.config["use_rejectlist"] is True
            except AssertionError as e:
                self.logger.error("""Found the psf type DES_Piff,
                but use_rejectlist is set to False. This will not stand""")
                raise(e)
        if self.config["use_rejectlist"]:
            try:
                assert "rejectlist" in stash
            except AssertionError as e:
                self.logger.error("""use_rejectlist is True, but no 'rejectlist'
                entry found in stash""")
                raise(e)

        # https://github.com/esheldon/meds/wiki/Creating-MEDS-Files-in-Python
        # Need to generate
        #  i) object_data structure
        #  ii) image_info structure

        # Loop through tiles
        tilenames = stash["tilenames"]

        if "use_sex_from" in self.config:
            other_stash = Stash.load(
                self.config["use_sex_from"],
                os.path.dirname(self.config["use_sex_from"]), [])

        for tilename in tilenames:

            t0 = timer()

            tile_file_info = stash["tile_info"][tilename]

            # meds files
            meds_files = []

            # object_data comes from SExtractor - assume either sextractor
            # catalog filename is in the stash
            # or the data is in the stash as a recarray
            # Use SExtractor catalog for refband

            # Hack to use sextractor catalog and segmentation map from
            # different run
            refband = self.config["refband"]
            if "use_sex_from" in self.config:
                sex_cat = other_stash.get_filepaths(
                    "sex_cat", tilename, band=refband)
                tile_file_info["sex_cat"] = sex_cat

                # the seg map may not exist if we are doing true detection
                try:
                    seg_file = other_stash.get_filepaths(
                        "seg_file", tilename, band=refband)
                    seg_ext = 0
                except KeyError:
                    seg_file = ''
                    seg_ext = -1
            # Normal mode:
            else:
                sex_cat = stash.get_filepaths(
                    "sex_cat", tilename, band=refband)
                stash.set_filepaths("sex_cat", sex_cat, tilename)

                # the seg map may not exist if we are doing true detection
                try:
                    seg_file, seg_ext = stash.get_filepaths(
                        "seg_file", tilename, band=refband), 0
                except KeyError:
                    seg_file = ''
                    seg_ext = -1

            try:
                sex_data = fitsio.read(sex_cat, lower=True)
            except IOError:
                # you can get an IOError here if there's no detected
                # objects...in this case we want to quit the
                # pipeline gracefully
                if os.path.isfile(sex_cat):
                    self.logger.error(
                        "IOError when trying to read SExtractor catalog but "
                        "file exists - maybe no objects were detected?")
                else:
                    self.logger.error(
                        "IOError when trying to read SExtractor catalog "
                        "and file does not exist")
                return 1, stash

            extra_obj_data_fields = [
                ('number', 'i8'),
            ]

            # This is the same for each band
            obj_data = meds.util.get_meds_input_struct(
                len(sex_data), extra_fields=extra_obj_data_fields)
            obj_data["id"] = sex_data["number"]
            obj_data["number"] = sex_data["number"]
            obj_data["ra"] = sex_data["alpha_j2000"]
            obj_data["dec"] = sex_data["delta_j2000"]

            # Choose the boxsize - this is the same method as used in desmeds
            # Pasted in these functions from desmeds.
            def get_box_sizes(cat):
                """
                get box sizes that are wither 2**N or 3*2**N, within
                the limits set by the user
                """
                sigma_size = get_sigma_size(cat)

                # now do row and col sizes
                row_size = cat['ymax_image'] - cat['ymin_image'] + 1
                col_size = cat['xmax_image'] - cat['xmin_image'] + 1

                # get max of all three
                box_size = np.vstack(
                    (col_size, row_size, sigma_size)).max(axis=0)

                # clip to range
                box_size = box_size.clip(
                    self.config['min_box_size'], self.config['max_box_size'])

                # now put in fft sizes
                bins = [0]
                bins.extend([sze for sze in self.config['allowed_box_sizes']
                             if sze >= self.config['min_box_size']
                             and sze <= self.config['max_box_size']])

                if bins[-1] != self.config['max_box_size']:
                    bins.append(self.config['max_box_size'])

                bin_inds = np.digitize(box_size, bins, right=True)
                bins = np.array(bins)

                return bins[bin_inds]

            def get_sigma_size(cat):
                """
                "sigma" size, based on flux radius and ellipticity
                """
                ellipticity = 1.0 - cat['b_world']/cat['a_world']
                sigma = cat['flux_radius']*2.0/FWHM_FAC
                drad = sigma*self.config['sigma_fac']
                drad = drad*(1.0 + ellipticity)
                drad = np.ceil(drad)
                # sigma size is twice the radius
                sigma_size = 2*drad.astype('i4')

                return sigma_size

            # Get boxsizes
            obj_data["box_size"] = get_box_sizes(sex_data)

            t1 = timer()
            self.logger.error(
                "Time to set up obj_data for tile %s: %s" % (
                    tilename, str(timedelta(seconds=t1-t0))))

            # image data
            for band in stash["bands"]:
                t0 = timer()
                img_data = tile_file_info[band]

                # coadd stuff
                coadd_file, coadd_ext = (
                    stash.get_filepaths("coadd_file", tilename, band=band),
                    img_data["coadd_ext"])
                coadd_weight_file, coadd_weight_ext = (
                    stash.get_filepaths(
                        "coadd_weight_file", tilename, band=band),
                    img_data["coadd_weight_ext"])
                coadd_bmask_file, coadd_bmask_ext = (
                    stash.get_filepaths(
                        "coadd_mask_file", tilename, band=band),
                    img_data["coadd_mask_ext"])

                # headers and WCS
                coadd_header = fitsio.read_header(
                    coadd_file, coadd_ext)
                # delete any None entries
                coadd_header.delete(None)
                coadd_header = desmeds.util.fitsio_header_to_dict(
                    coadd_header)

                coadd_wcs, _ = galsim.wcs.readFromFitsHeader(
                    galsim.FitsHeader(coadd_file, coadd_ext))

                if (("coadd_bkg_file" in img_data) and
                        self.config["sub_coadd_bkg"]):
                    raise ValueError(
                        "Backgrounds for coadds are not supported!")

                slen = max(
                    len(coadd_file), len(coadd_weight_file), len(seg_file))

                # get wcs json thing
                wcs_json = []
                wcs_json.append(json.dumps(coadd_header))

                # single-epoch stuff
                if "img_files" in img_data:
                    img_files = stash.get_filepaths(
                        "img_files", tilename, band=band)

                    # Are we rejectlisting?
                    if self.config["use_rejectlist"]:
                        rejectlist = Rejectlist(stash["rejectlist"])
                        # keep only non-rejectlisted files
                        is_rejectlisted = [rejectlist.img_file_is_rejectlisted(f) for f in img_files]
                        img_files = [f for (i, f) in enumerate(img_files) if not is_rejectlisted[i]]
                        wgt_files = [f for (i, f) in enumerate(img_data['wgt_files']) if not is_rejectlisted[i]]
                        msk_files = [f for (i, f) in enumerate(img_data['msk_files']) if not is_rejectlisted[i]]
                        mag_zps = [m for (i, m) in enumerate(img_data['mag_zps']) if not is_rejectlisted[i]]
                    else:
                        wgt_files, msk_files = img_data['wgt_files'], img_data['msk_files']

                    if self.config.get("sub_bkg", True):
                        bkg_filenames = [get_bkg_path(f) for f in img_files]
                        if bkg_filenames[0].endswith(".fits.fz"):
                            bkg_ext = 1
                        else:
                            bkg_ext = 0
                    else:
                        bkg_filenames = [""]*len(img_files)
                        bkg_ext = -1

                    n_images = len(img_files)
                    for i in range(n_images):
                        slen = max(
                            slen,
                            max(len(img_files[i]), len(bkg_filenames[i])))

                    for img_file in img_files:
                        h = fitsio.read_header(img_file, img_data["img_ext"])
                        h.delete(None)
                        wcs_json.append(
                            json.dumps(desmeds.util.fitsio_header_to_dict(h)))
                else:
                    n_images = 0

                wcs_len = reduce(lambda x, y: max(x, len(y)), wcs_json, 0)
                image_info = meds.util.get_image_info_struct(
                    n_images+1, slen, wcs_len=wcs_len)

                # assuming sextractor positions
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
                image_info["magzp"][0] = 30.
                image_info["scale"][0] = 1.

                for i in range(n_images):
                    ind = i+1
                    image_info["image_path"][ind] = img_files[i]
                    image_info["image_ext"][ind] = img_data["img_ext"]
                    image_info["weight_path"][ind] = wgt_files[i]
                    image_info["weight_ext"][ind] = img_data["wgt_ext"]
                    image_info["bmask_path"][ind] = msk_files[i]
                    image_info["bmask_ext"][ind] = img_data["msk_ext"]

                    image_info["bkg_path"][ind] = bkg_filenames[i]
                    image_info["bkg_ext"][ind] = bkg_ext
                    image_info["seg_path"][ind] = None
                    image_info["seg_ext"][ind] = -1

                    image_info["wcs"][ind] = wcs_json[ind]
                    image_info["magzp"][ind] = mag_zps[i]
                    image_info["scale"][ind] = 10**(
                        0.4*(self.config["magzp_ref"]-mag_zps[i]))

                # not sure if we need to set image_id - but do so anyway
                image_info["image_id"] = np.arange(n_images+1)

                # metadata
                meta_data = np.zeros(1, dtype=[("magzp_ref", np.float64)])
                meta_data["magzp_ref"] = self.config["magzp_ref"]

                t1 = timer()
                self.logger.error(
                    "Time to set up image_info for tile %s, band %s: %s" % (
                        tilename, band, str(timedelta(seconds=t1-t0))))

                # PSFEx shit...
                # We may want to add psf data to the meds file e.g. from psfex
                # files. This could be the psfex files on which the true psf
                # model for the sim is based, or they could be psfex files
                # estimated from the sim. Either way, they should be located at
                # the path returned by the get_psfex_path and
                # get_psfex_path_coadd functions.
                # If there's no psfex file present for the coadd, because
                # we haven't re-run
                # psfex for the coadd, use that for the original real data
                # coadd.
                # t0 = timer()

                if self.config.get("add_psf_data", True):
                    if stash["psf_config"]["type"] == "DES_Piff":
                        if stash["psf_config"].get("no_smooth", False):
                            assert stash["draw_method"] == "no_pixel"
                        else:
                            assert stash["draw_method"] == "auto"
                        self.logger.error("Adding Piff info to meds file")
                        # this is the type for the meds maker - sets the layout
                        # etc
                        # we are using the same API so our type is psfex even
                        # though we will put piff data in it
                        self.config["psf_type"] = "psfex"  # it's ok

                        psf_data = []

                        # there is no piff coadd PSF so we fake it
                        psf_data.append(
                            PSFForMeds(
                                galsim.Gaussian(fwhm=0.9),
                                coadd_wcs,
                                "auto"))

                        if "img_files" in img_data:
                            self.logger.error(
                                "se piff files: %s" % [
                                    get_piff_path(f)
                                    for f in img_files])
                            for img_file in img_files:
                                piff_path = get_piff_path(img_file)
                                psf = DES_Piff(piff_path)
                                img_wcs, img_origin \
                                    = galsim.wcs.readFromFitsHeader(
                                        galsim.FitsHeader(
                                            img_file, img_data["img_ext"]))
                                psf_data.append(
                                    PSFForMeds(psf, img_wcs, "auto"))

                    if stash["psf_config"]["type"] in [
                            "DES_PSFEx", "DES_PSFEx_perturbed"]:
                        self.logger.error("Adding psfex info to meds file")
                        self.config["psf_type"] = "psfex"

                        psf_data = []

                        # first look for coadd psfex file.
                        coadd_psfex_path = get_psfex_path_coadd(coadd_file)
                        if not os.path.isfile(coadd_psfex_path):
                            orig_coadd_path = get_orig_coadd_file(
                                stash["desdata"], stash["desrun"],
                                tilename, band)
                            coadd_psfex_path = get_psfex_path_coadd(
                                orig_coadd_path)
                        self.logger.error(
                            "coadd psfex file: %s" % (coadd_psfex_path))

                        if self.config.get("use_galsim_psfex", True):
                            self.logger.error(
                                "using galsim to reconstruct psfex for "
                                "meds file")
                            psf = galsim.des.DES_PSFEx(
                                coadd_psfex_path, image_file_name=coadd_file)
                            psf_data.append(
                                PSFForMeds(psf, coadd_wcs, "no_pixel"))
                        else:
                            psf_data.append(
                                psfex.PSFEx(coadd_psfex_path))

                        if "img_files" in img_data:
                            self.logger.error(
                                "se psfex files: %s" % [
                                    get_psfex_path(f)
                                    for f in img_data["img_files"]])
                            for img_file in img_data["img_files"]:
                                psfex_path = get_psfex_path(img_file)
                                if self.config.get("use_galsim_psfex", True):
                                    self.logger.error(
                                        "using galsim to reconstruct psfex "
                                        "for meds file")
                                    psf = galsim.des.DES_PSFEx(
                                        psfex_path, image_file_name=img_file)
                                    img_wcs, img_origin \
                                        = galsim.wcs.readFromFitsHeader(
                                            galsim.FitsHeader(
                                                img_file, img_data["img_ext"]))
                                    psf_data.append(
                                        PSFForMeds(psf, img_wcs, "no_pixel"))
                                else:
                                    psf_data.append(psfex.PSFEx(psfex_path))

                    if (stash["psf_config"]["type"] == "Gaussian"):
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
                                    raise(e)
                                break
                        psf = galsim.Gaussian(
                            **{size_key: stash["psf_config"][size_key]})

                        # now generate psf data
                        psf_data = []
                        psf_data.append(
                            PSFForMeds(
                                psf,
                                coadd_wcs,
                                stash.get("draw_method", "auto")))

                        if "img_files" in img_data:
                            for img_file in img_files:
                                wcs, origin = galsim.wcs.readFromFitsHeader(
                                    galsim.FitsHeader(
                                        img_file, hdu=img_data["img_ext"]))
                                psf_data.append(
                                    PSFForMeds(
                                        psf,
                                        wcs,
                                        stash.get("draw_method", "auto")))
                else:
                    psf_data = None

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
                meds_run = self.config["meds_run"]
                meds_dir = os.environ.get("MEDS_DIR")
                meds_dir_this_tile = os.path.join(meds_dir, meds_run, tilename)

                # psf map file stuff.
                # This file is used by ngmixer to get the psf file for a given
                # cutout
                if ((stash["psf_config"]["type"] in ["DES_PSFEx", "DES_PSFEx_perturbed"]) and  # noqa
                        (not self.config.get("add_psf_data", True))):
                    t0 = timer()
                    assert stash["draw_method"] == "no_pixel"
                    orig_psfmap_file = os.path.join(
                        stash["desdata"], stash["desrun"], tilename,
                        "%s_%s_psfmap-%s.dat" % (
                            tilename, band, stash["desrun"]))
                    psfmap_file_basename = os.path.basename(
                        orig_psfmap_file).replace(
                            stash["desrun"], self.config["meds_run"])
                    # copy this file to meds directory
                    if not os.path.isdir(meds_dir_this_tile):
                        safe_mkdir(meds_dir_this_tile)
                    shutil.copy(
                        orig_psfmap_file,
                        os.path.join(
                            meds_dir_this_tile,
                            psfmap_file_basename))

                    # while we're at it, copy over the psfs directory too
                    # delete if already exists
                    orig_psfs_dir = os.path.join(
                        stash["desdata"], stash["desrun"], tilename, "psfs")
                    psfs_dir = os.path.join(meds_dir_this_tile, "psfs")
                    if os.path.isdir(psfs_dir):
                        shutil.rmtree(psfs_dir)
                    shutil.copytree(orig_psfs_dir, psfs_dir)
                    t1 = timer()
                    self.logger.error(
                        "Time to copy psfex files for tile %s, band %s: %s" % (
                            tilename, band, str(timedelta(seconds=t1-t0))))

                t0 = timer()
                meds_file = os.path.join(
                    os.environ.get("MEDS_DIR"), meds_run, tilename,
                    "%s_%s_meds-%s.fits.fz" % (tilename, band, meds_run))
                meds_files.append(meds_file)

                d = os.path.dirname(os.path.normpath(meds_file))
                safe_mkdir(d)

                # If requested, we stage the file at $TMPDIR and then
                # copy over when finished writing. This is a good idea
                # on many systems, but maybe not all.
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

    @classmethod
    def from_config_file(cls, config_file, base_dir=None, logger=None,
                         name="meds"):
        with open(config_file, "rb") as f:
            config = yaml.load(f)
        return cls(config, base_dir=base_dir, logger=logger, name=name)
