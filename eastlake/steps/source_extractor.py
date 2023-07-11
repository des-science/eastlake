from __future__ import print_function, absolute_import
import os
import copy
import pkg_resources

import fitsio
import numpy as np

from ..step import Step, run_and_check
from ..stash import Stash
from ..utils import get_relpath, pushd
from ..des_piff import PSF_KWARGS
from .swarp import FITSEXTMAP


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
                    self.config.get("params_file", _get_default("Y6A1_v1_srcex.param_diskonly"))
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
            self.config.get("srcex_cmd", "eastlake-src-extractor")
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

        # add default command flags
        self.srcex_cmd_root += [
            "-CHECKIMAGE_TYPE", "SEGMENTATION",
            "-MAG_ZEROPOINT", "30",
            "-DEBLEND_MINCONT", "0.001",
            "-DEBLEND_NTHRESH", "64",
            "-DETECT_THRESH", "0.8",
            "-ANALYSIS_THRESH", "0.8",
        ]

    def execute(self, stash, new_params=None):
        # Loop through tiles calling SrcExtractor
        tilenames = stash["tilenames"]

        for tilename in tilenames:
            det_coadd_file = stash.get_filepaths(
                "det_coadd_file", tilename,
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
                coadd_file, _ = stash.get_filepaths(
                    "coadd_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )

                # SrcExtractor is annoying and can vomit a segfault if the paths
                # to the input images are too long. So change to the
                # band coadd directory and use relative paths from there
                coadd_dir = os.path.realpath(os.path.dirname(coadd_file))
                with pushd(coadd_dir):
                    # Add weight file stuff to command string
                    cmd += ["-WEIGHT_IMAGE", "%s[%d],%s[%d]" % (
                        get_relpath(det_coadd_file), FITSEXTMAP["wgt"],
                        get_relpath(coadd_file), FITSEXTMAP["wgt"])]

                    # catalog items
                    catalog_name = coadd_file.replace(".fits", "_cat.fits")
                    cmd += ["-CATALOG_NAME", get_relpath(catalog_name)]
                    cmd += ["-CATALOG_TYPE", "FITS_1.0"]

                    # and bkg+seg name
                    seg_name = coadd_file.replace(".fits", "_segmap.fits")
                    bkg_name = coadd_file.replace(".fits", "_bkg.fits")
                    bkg_rms_name = coadd_file.replace(".fits", "_bkg-rms.fits")
                    cmd += ["-CHECKIMAGE_NAME", "%s" % get_relpath(seg_name)]

                    # and mask file
                    mask_file = "%s[%d]" % (
                        get_relpath(det_coadd_file),
                        FITSEXTMAP["msk"],
                    )
                    cmd += ["-FLAG_IMAGE", get_relpath(mask_file)]

                    # image name should be first argument
                    image_arg = "%s[%d],%s[%d]" % (
                        get_relpath(det_coadd_file),
                        FITSEXTMAP["sci"],
                        get_relpath(coadd_file),
                        FITSEXTMAP["sci"]
                    )
                    cmd = [cmd[0]] + [image_arg] + cmd[1:]

                    if self.logger is not None:
                        self.logger.error(
                            "calling source extractor with command:\n\t%s",
                            " ".join(cmd)
                        )
                    run_and_check(cmd, "SrcExtractor")

                    with fitsio.FITS(seg_name, "rw") as fits:
                        fits[0].write_key("EXTNAME", "sci")

                with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                    pyml["cat_path"] = catalog_name
                    pyml["seg_path"] = seg_name
                    pyml["seg_ext"] = "sci"
                stash.set_filepaths("bkg_file", bkg_name, tilename, band=band)
                stash.set_filepaths(
                    "bkg_rms_file", bkg_rms_name, tilename, band=band)

            # make coadd object map
            last_coadd_cat = fitsio.read(catalog_name, lower=True)
            dtype = [
                ('id', 'i8'),
                ('object_number', 'i8'),
                ('gi_color', 'f8'),
                ('iz_color', 'f8'),
            ]
            obj_cat = np.zeros(len(last_coadd_cat), dtype=dtype)
            obj_cat["id"] = last_coadd_cat['number']
            obj_cat["object_number"] = last_coadd_cat['number']
            mags = {}
            for band in stash["bands"]:
                pyml = stash.get_output_pizza_cutter_yaml(tilename, band)
                coadd_cat = fitsio.read(pyml["cat_path"], lower=True)
                mags[band] = coadd_cat["mag_auto"]
                assert len(coadd_cat) == len(obj_cat)

            if "g" in mags and "i" in mags:
                obj_cat["gi_color"] = mags["g"] - mags["i"]
            else:
                obj_cat["gi_color"] = PSF_KWARGS["r"]["GI_COLOR"]

            if "i" in mags and "z" in mags:
                obj_cat["iz_color"] = mags["i"] - mags["z"]
            else:
                obj_cat["iz_color"] = PSF_KWARGS["z"]["IZ_COLOR"]

            for band in stash["bands"]:
                coadd_file, _ = stash.get_filepaths(
                    "coadd_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                obj_cat_name = coadd_file.replace(".fits", "_objmap.fits")
                fitsio.write(obj_cat_name, obj_cat, clobber=True)
                with stash.update_output_pizza_cutter_yaml(tilename, band):
                    stash.set_filepaths(
                        "coadd_object_map", obj_cat, tilename, band=band,
                    )

        return 0, stash
