from __future__ import print_function, absolute_import
import os
import copy
import pkg_resources

import fitsio
import numpy as np

from ..step import Step, run_and_check
from ..stash import Stash
from ..utils import get_relpath, pushd, unpack_fits_file_if_needed


def _get_default(nm):
    return pkg_resources.resource_filename("eastlake", "astromatic/%s" % nm)


class SrcExtractorRunner(Step):
    """
    Pipeline step for running SrcExtractor on detection image
    """
    def __init__(self, config, base_dir, name="src_extractor",
                 logger=None, verbosity=0, log_file=None):

        # name for this step
        super(SrcExtractorRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        self.srcex_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get("config_file", _get_default("Y6A1_v1_srcex.config"))
                )
            )
        )
        self.params_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config("params_file", _get_default("Y6A1_v1_srcex.param_diskonly"))
                )
            )
        )
        self.filter_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get("filter_file", _get_default("Y6A1_v1_gauss_3.0_7x7.conv"))
                )
            )
        )
        self.star_nnw_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get("star_nnw_file", _get_default("Y6A1_v1_srcex.nnw"))
                )
            )
        )
        # src_extractor command my be an environment variable
        # so use os.path.expandvars
        self.srcex_cmd = os.path.expandvars(
            self.config.get("srcex_cmd", _get_default("eastlake-src-extractor"))
        )

        # make sure SrcExtractor works...
        # run_and_check( [self.srcex_cmd], "SrcExtractor" )
        self.srcex_cmd_root = [
            "%s" % (self.srcex_cmd), "-c", "%s" % (self.srcex_config_file)]
        if "update" in self.config:
            for key, val in self.config["update"].items():
                self.srcex_cmd_root += ["-%s" % key, str(val)]

        # update paths to src_extractor files
        self.srcex_cmd_root += ["-PARAMETERS_NAME", self.params_file]
        self.srcex_cmd_root += ["-FILTER_NAME", self.filter_file]
        self.srcex_cmd_root += ["-STARNNW_NAME", self.star_nnw_file]

    def execute(self, stash, new_params=None):
        # Loop through tiles calling SrcExtractor
        tilenames = stash["tilenames"]

        for tilename in tilenames:
            # Get detection weight file. SrcExtractor doesn't like compressed
            # fits files, so may need to funpack
            if "single_band_det" in self.config:
                det_image_file, det_image_ext = stash.get_filepaths(
                    "coadd_file", tilename,
                    band=self.config["single_band_det"],
                    with_fits_ext=True,
                )
                det_weight_file, det_weight_ext = stash.get_filepaths(
                    "coadd_weight_file", tilename,
                    band=self.config["single_band_det"],
                    with_fits_ext=True,
                )
                det_mask_file, det_mask_ext = stash.get_filepaths(
                    "coadd_mask_file", tilename,
                    band=self.config["single_band_det"],
                    with_fits_ext=True,
                )
            else:
                det_image_file, det_image_ext = stash.get_filepaths(
                    "det_image_file", tilename, with_fits_ext=True
                )
                det_weight_file, det_weight_ext = stash.get_filepaths(
                    "det_weight_file", tilename, with_fits_ext=True
                )
                det_mask_file, det_mask_ext = stash.get_filepaths(
                    "det_mask_file", tilename, with_fits_ext=True
                )

            # Astromatic doesn't like compressed fits files, so we may
            # need to unpack here if the weight file is .fits.fz. In this
            # case we also need to subtract 1 from the weight extension
            det_weight_file, det_weight_ext = unpack_fits_file_if_needed(
                det_weight_file, det_weight_ext,
            )

            # Assoc mode hack
            srcex_cmd_root = copy.copy(self.srcex_cmd_root)
            if "use_assoc_from" in self.config:
                # load other stash with the directory containing the
                # job_record file as the base_dir
                other_stash = Stash.load(
                    self.config["use_assoc_from"],
                    os.path.dirname(self.config["use_assoc_from"]), [])
                refband = self.config.get("refband", "i")

                # Grab the SrcExtractor catalog from this other stash
                srcex_cat = other_stash.get_filepaths(
                    "srcex_cat", tilename, band=refband)

                # Write assoc file
                srcex_data = fitsio.read(srcex_cat)
                assoc_file = os.path.join(
                    self.base_dir, "assoc_file_%s.txt" % tilename)
                assoc_data = np.array([
                    srcex_data["NUMBER"],
                    srcex_data["X_IMAGE"],
                    srcex_data["Y_IMAGE"]]).T
                with open(assoc_file, 'w') as f:
                    np.savetxt(
                        f, assoc_data,
                        fmt=["%d", "%.9f", "%.9f"], delimiter=" ")

                # Add assoc options to command
                srcex_cmd_root += ["-ASSOC_NAME", "%s" % assoc_file]
                srcex_cmd_root += ["-ASSOC_PARAMS", "2,3"]

            # Now loop through bands calling SrcExtractor in dual image mode
            for band in stash["bands"]:
                cmd = copy.copy(srcex_cmd_root)
                weight_file, weight_ext = stash.get_filepaths(
                    "coadd_weight_file", tilename, band=band, with_fits_ext=True,
                )
                # Astromatic doesn't like compressed fits files, so we may
                # need to unpack here if the weight file is .fits.fz. In this
                # case we also need to subtract 1 from the weight extension
                weight_file, weight_ext = unpack_fits_file_if_needed(
                    weight_file, weight_ext,
                )

                # set catalog name
                coadd_file, coadd_ext = stash.get_filepaths(
                    "coadd_file", tilename, band=band, with_fits_ext=True,
                )
                # SrcExtractor is annoying and can vomit a segfault if the paths
                # to the input images are too long. So change to the
                # band coadd directory and use relative paths from there
                coadd_dir = os.path.realpath(os.path.dirname(coadd_file))
                with pushd(coadd_dir):

                    # Add weight file stuff to command string
                    cmd += ["-WEIGHT_IMAGE", "%s[%d],%s[%d]" % (
                        get_relpath(det_weight_file), det_weight_ext,
                        get_relpath(weight_file), weight_ext)]

                    catalog_name = coadd_file.replace(".fits", "_sexcat.fits")

                    cmd += ["-CATALOG_NAME", get_relpath(catalog_name)]
                    cmd += ["-CATALOG_TYPE", "FITS_1.0"]

                    # and seg name
                    seg_name = coadd_file.replace(".fits", "_seg.fits")
                    bkg_name = coadd_file.replace(".fits", "_bkg.fits")
                    bkg_rms_name = coadd_file.replace(".fits", "_bkg-rms.fits")
                    cmd += ["-CHECKIMAGE_NAME", "%s,%s,%s" % (
                        get_relpath(seg_name),
                        get_relpath(bkg_name),
                        get_relpath(bkg_rms_name))]

                    # and mask file
                    mask_file = "%s[%d]" % (get_relpath(det_mask_file),
                                            det_mask_ext)
                    if mask_file is not None:
                        cmd += ["-FLAG_IMAGE", get_relpath(mask_file)]
                    else:
                        self.logger.error("No mask for detection image")

                    # image name should be first argument
                    image_arg = "%s[%d],%s[%d]" % (get_relpath(det_image_file),
                                                   det_image_ext,
                                                   get_relpath(coadd_file),
                                                   coadd_ext)
                    cmd = [cmd[0]] + [image_arg] + cmd[1:]

                    if self.logger is not None:
                        self.logger.error("calling source extractor with command:")
                        self.logger.error(" ".join(cmd))
                    run_and_check(cmd, "SrcExtractor")

                with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                    pyml["cat_path"] = catalog_name
                    pyml["seg_path"] = seg_name
                    pyml["seg_ext"] = 0
                stash.set_filepaths("bkg_file", bkg_name, tilename, band=band)
                stash.set_filepaths(
                    "bkg_rms_file", bkg_rms_name, tilename, band=band)

        return 0, stash
