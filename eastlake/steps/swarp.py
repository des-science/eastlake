from __future__ import print_function, absolute_import
import os
import pkg_resources
import multiprocessing

import fitsio
import astropy.io.fits as fits
import galsim
import numpy as np

from ..utils import safe_mkdir, get_relpath, pushd, safe_rm
from ..step import Step, run_and_check
from ..des_files import MAGZP_REF


DEFAULT_SWARP_CONFIG = pkg_resources.resource_filename("eastlake", "astromatic/Y6A1_v1_swarp.config")


# Swarp is annoying and can vomit a segfault if the paths
# to the input images are too long. This can happen with the
# paths in e.g. im_file_list which are absolute paths which
# generally seems a safe thing to use but anyway. So re-make
# the file lists with paths relative to the current working
# directory
def _write_relpath_file_list(lines, output_file_list):
    output_lines = []
    for ln in lines:
        rel_str = get_relpath(ln.strip())
        if len(ln.strip()) < len(rel_str):
            output_lines.append("%s\n" % ln.strip())
        else:
            output_lines.append("%s\n" % rel_str)
        assert (
            os.path.exists(os.path.abspath(output_lines[-1])[:-4])
        ), os.path.abspath(output_lines[-1][:-4])
    with open(output_file_list, 'w') as f:
        f.writelines(output_lines)


def _get_file_dims_pixscale(pth, ext, world_center):
    h = fitsio.read_header(pth, ext=ext)
    if "ZNAXIS1" in h:
        image_shape = (h["ZNAXIS1"], h["ZNAXIS2"])
    else:
        image_shape = (h["NAXIS1"], h["NAXIS2"])

    coadd_header = galsim.fits.FitsHeader(pth)
    coadd_wcs, _ = galsim.wcs.readFromFitsHeader(coadd_header)
    pixel_scale = np.sqrt(coadd_wcs.pixelArea(
        world_pos=galsim.CelestialCoord(
            world_center[0] * galsim.degrees,
            world_center[1] * galsim.degrees,
        )
    ))
    return image_shape, pixel_scale


def _set_fpack_headers(hdu):
    hdu.header["FZALGOR"] = ('RICE_1', "Compression type")
    hdu.header["FZDTHRSD"] = ('CHECKSUM', 'Dithering seed value')
    hdu.header["FZQVALUE"] = (16, "Compression quantization factor")
    hdu.header["FZQMETHD"] = ('SUBTRACTIVE_DITHER_2', 'Compression quantization method')


class SingleBandSwarpRunner(Step):
    def __init__(self, config, base_dir, name="single_band_swarp", logger=None,
                 verbosity=0, log_file=None):
        super(SingleBandSwarpRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)
        self.swarp_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get("config_file", DEFAULT_SWARP_CONFIG)
                )
            )
        )
        self.swarp_cmd = os.path.expandvars(
            self.config.get("swarp_cmd", "eastlake-swarp")
        )
        self.swarp_cmd_root = [
            "%s" % (self.swarp_cmd), "-c", "%s" % (self.swarp_config_file)]

        if "update" in self.config:
            for key, val in self.config["update"].items():
                self.logger.info(
                    "Updating swarp config with %s = %s" % (key, val))
                self.swarp_cmd_root += ["-%s" % key, str(val)]

    def execute(self, stash, new_params=None):
        tilenames = stash["tilenames"]

        extra_cmd_line_args = [
            "-COMBINE_TYPE", "WEIGHTED",
            "-BLANK_BADPIXELS", "Y",
        ]
        if not any(cv == "-NTHREADS" for cv in self.swarp_cmd_root):
            extra_cmd_line_args += [
                "-NTHREADS", "%d" % multiprocessing.cpu_count(),
            ]

        for tilename in tilenames:
            for band in stash["bands"]:
                self.logger.error(
                    "running swarp for tile %s, band %s" % (
                        tilename, band))
                cmd = self.swarp_cmd_root + extra_cmd_line_args

                coadd_center = stash.get_tile_info_quantity("tile_center", tilename)
                orig_coadd_path = stash.get_input_pizza_cutter_yaml(tilename, band)["image_path"]
                orig_coadd_ext = stash.get_input_pizza_cutter_yaml(tilename, band)["image_ext"]
                image_shape, pixel_scale = _get_file_dims_pixscale(
                    orig_coadd_path,
                    orig_coadd_ext,
                    coadd_center
                )

                self.logger.error(
                    "inferred image size|pixel scale: %s|%s", image_shape, pixel_scale,
                )

                coadd_path_from_imsim_data = get_relpath(
                    orig_coadd_path, stash["imsim_data"])
                output_coadd_path = os.path.join(
                    stash["base_dir"], coadd_path_from_imsim_data)
                if output_coadd_path.endswith("fits.fz"):
                    output_coadd_path = output_coadd_path[:-3]
                output_coadd_dir = os.path.dirname(output_coadd_path)

                output_coadd_sci_file = os.path.join(
                    output_coadd_dir, "%s_%s_sci.fits" % (tilename, band))
                output_coadd_weight_file = os.path.join(
                    output_coadd_dir, "%s_%s_wgt.fits" % (tilename, band))
                output_coadd_mask_file = os.path.join(
                    output_coadd_dir, "%s_%s_msk.fits" % (tilename, band))

                safe_rm(output_coadd_sci_file)
                safe_rm(output_coadd_weight_file)
                safe_rm(output_coadd_mask_file)

                # make the output directory and then move here to run swarp
                # this prevents the intermediate files being fucked up by
                # other processes.
                safe_mkdir(output_coadd_dir)
                with pushd(os.path.realpath(output_coadd_dir)):

                    # Swarp is annoying and can vomit a segfault if the paths
                    # to the input images are too long. This can happen with the
                    # paths in e.g. im_file_list which are absolute paths which
                    # generally seems a safe thing to use but anyway. So re-make
                    # the file lists with paths relative to the current working
                    # directory
                    output_pyml = stash.get_output_pizza_cutter_yaml(tilename, band)

                    im_file_list = "im_file_list.dat"
                    _write_relpath_file_list(
                        [
                            "%s[%d]" % (src["image_path"], src["image_ext"])
                            for src in output_pyml["src_info"]
                        ],
                        im_file_list,
                    )

                    wgt_file_list = "wgt_file_list.dat"
                    _write_relpath_file_list(
                        [
                            "%s[%d]" % (src["weight_path"], src["weight_ext"])
                            for src in output_pyml["src_info"]
                        ],
                        wgt_file_list,
                    )

                    msk_file_list = "msk_file_list.dat"
                    _write_relpath_file_list(
                        [
                            "%s[%d]" % (src["bmask_path"], src["bmask_ext"])
                            for src in output_pyml["src_info"]
                        ],
                        msk_file_list,
                    )

                    cmd = [cmd[0]] + ["@%s" % im_file_list] + cmd[1:]
                    cmd += ["-IMAGE_SIZE", "%d,%d" % image_shape]
                    cmd += ["-PIXEL_SCALE", "%0.16f" % pixel_scale]
                    cmd += ["-WEIGHTOUT_NAME", output_coadd_weight_file]
                    cmd += ["-CENTER", "%s,%s" % (
                        coadd_center[0], coadd_center[1])]
                    cmd += ["-IMAGEOUT_NAME", output_coadd_sci_file]
                    cmd += ["-WEIGHT_IMAGE", "@%s" % wgt_file_list]

                    # We need to scale images to all have common zeropoint
                    mag_zp_list = [src["magzp"] for src in output_pyml["src_info"]]
                    fscale_list = [10**(0.4*(MAGZP_REF-mag_zp)) for mag_zp in mag_zp_list]

                    cmd += [
                        "-FSCALE_DEFAULT", ",".join(
                            ["%f" % _f for _f in fscale_list]
                        )
                    ]

                    # run swarp
                    self.logger.error(
                        "running swarp for tile %s, band %s: %s" % (
                            tilename, band, " ".join(cmd)))
                    run_and_check(cmd, "SWarp", logger=self.logger)

                    # Do the same for the masks
                    # Not sure exactly what I should be doing here...for now try
                    # setting mask files as weight images and use
                    # weightout_image....
                    dummy_mask_coadd = os.path.join(
                        output_coadd_dir, "%s_%s_msk-tmp.fits" % (tilename, band))
                    mask_cmd = self.swarp_cmd_root + extra_cmd_line_args
                    mask_cmd = (
                        [mask_cmd[0]] + ["@%s" % msk_file_list] + mask_cmd[1:])
                    mask_cmd += ["-WEIGHTOUT_NAME", output_coadd_mask_file]
                    mask_cmd += ["-CENTER", "%s,%s" % (
                        coadd_center[0], coadd_center[1])]
                    mask_cmd += ["-IMAGEOUT_NAME", dummy_mask_coadd]
                    mask_cmd += ["-WEIGHT_IMAGE", "@%s" % msk_file_list]
                    # run swarp
                    self.logger.error(
                        "running swarp for tile %s, band %s w/ mask: %s" % (
                            tilename, band, " ".join(mask_cmd)))
                    run_and_check(mask_cmd, "Mask SWarp", logger=self.logger)
                    # remove the dummy mask coadd
                    safe_rm(dummy_mask_coadd)

                    try:
                        # We've done the swarping, now combine image, weight and
                        # mask planes
                        # generate an hdu list
                        im_hdus = fits.open(output_coadd_sci_file)
                        im_hdu = im_hdus[0]
                        _set_fpack_headers(im_hdu)
                        # stupidly, if you ask me, we cannot simply read in the
                        # weight and coadd hdus and add them directly to an HDUList
                        # because they are PrimaryHDUs...
                        wgt_hdus = fits.open(output_coadd_weight_file)
                        wgt_fits = wgt_hdus[0]
                        wgt_hdu = fits.ImageHDU(wgt_fits.data, header=wgt_fits.header)
                        _set_fpack_headers(wgt_hdu)
                        msk_hdus = fits.open(output_coadd_mask_file)
                        msk_fits = msk_hdus[0]
                        msk_hdu = fits.ImageHDU(msk_fits.data, header=msk_fits.header)
                        _set_fpack_headers(msk_hdu)
                        hdus = [im_hdu, msk_hdu, wgt_hdu]
                        hdulist = fits.HDUList(hdus)
                        self.logger.error(
                            "writing assembled coadd for tilename %s, "
                            "band %s to %s" % (
                                tilename, band, output_coadd_path))
                        hdulist.writeto(output_coadd_path, overwrite=True)

                        safe_rm(output_coadd_path + ".fz")
                        run_and_check(
                            ["fpack", os.path.basename(output_coadd_path)],
                            "fpack SWarp"
                        )
                    finally:
                        # close the open hdus
                        im_hdus.close()
                        wgt_hdus.close()
                        msk_hdus.close()

                        # delete intermediate files
                        safe_rm(output_coadd_sci_file)
                        safe_rm(output_coadd_weight_file)
                        safe_rm(output_coadd_mask_file)

                with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                    pyml["image_path"] = output_coadd_path + ".fz"
                    pyml["image_ext"] = 1
                    pyml["bmask_path"] = output_coadd_path + ".fz"
                    pyml["bmask_ext"] = 2
                    pyml["weight_path"] = output_coadd_path + ".fz"
                    pyml["weight_ext"] = 3

            self.logger.error(
                "%s complete for tile %s" % (self.name, tilename))

        return 0, stash


class SWarpRunner(Step):
    """
    Pipeline step for running SWarp
    """
    def __init__(self, config, base_dir, name="swarp", logger=None,
                 verbosity=0, log_file=None):
        # name for this step
        super(SWarpRunner, self).__init__(
            config, base_dir, name=name, logger=logger,
            verbosity=verbosity, log_file=log_file)
        self.swarp_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get("config_file", DEFAULT_SWARP_CONFIG)
                )
            )
        )
        # swarp command my be an environment variable
        # so use os.path.expandvars
        self.swarp_cmd = os.path.expandvars(
            self.config.get("swarp_cmd", "eastlake-swarp")
        )
        self.swarp_cmd_root = [
            "%s" % (self.swarp_cmd), "-c", "%s" % (self.swarp_config_file)]

        if "update" in self.config:
            for key, val in self.config["update"].items():
                self.logger.info(
                    "Updating swarp config with %s = %s" % (key, val))
                self.swarp_cmd_root += ["-%s" % key, str(val)]

    def execute(self, stash, new_params=None):
        # Loop through tiles
        tilenames = stash["tilenames"]

        extra_cmd_line_args = [
            "-RESAMPLE", "Y",
            "-RESAMPLING_TYPE", "NEAREST",
            "-COPY_KEYWORDS", "BUNIT,TILENAME,TILEID",
            "-COMBINE_TYPE", "AVERAGE",
            "-BLANK_BADPIXELS", "Y",
        ]
        if not any(cv == "-NTHREADS" for cv in self.swarp_cmd_root):
            extra_cmd_line_args += [
                "-NTHREADS", "%d" % multiprocessing.cpu_count(),
            ]

        for tilename in tilenames:
            cmd = self.swarp_cmd_root + extra_cmd_line_args
            mask_cmd = self.swarp_cmd_root + extra_cmd_line_args
            img_strings = []
            weight_strings = []
            mask_strings = []

            bands = stash["bands"]
            coadd_bands = []
            # Loop through bands collecting image, weight and mask names
            for band in bands:
                # We may not want to coadd all bands - e.g. DES just does riz
                # for the detection image. So check the coadd_bands entry in
                # the config
                if "coadd_bands" in self.config:
                    if band not in self.config["coadd_bands"]:
                        self.logger.error(
                            "Not including band=%s in coadd" % (band))
                        continue
                coadd_bands.append(band)

                # Get image and weight files
                im, ext = stash.get_filepaths(
                    "coadd_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                img_strings.append("%s[%d]" % (im, ext))

                weight_file, weight_ext = stash.get_filepaths(
                    "coadd_weight_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                weight_strings.append("%s[%d]" % (weight_file, weight_ext))

                # We also need to coadd the masks, so get mask filenames.
                mask_file, mask_ext = stash.get_filepaths(
                    "coadd_mask_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                mask_strings.append("%s[%d]" % (mask_file, mask_ext))

                # on first band, get the center, image_shape and pixel scale
                if len(coadd_bands) == 1:
                    coadd_center = stash.get_tile_info_quantity("tile_center", tilename)
                    image_shape, pixel_scale = _get_file_dims_pixscale(
                        im, ext, coadd_center
                    )

            # Add image and weight info to cmd - image files should be first
            # argument
            cmd = [cmd[0]] + [",".join(img_strings)] + cmd[1:]
            cmd += ["-CENTER", "%s,%s" % (coadd_center[0], coadd_center[1])]
            cmd += ["-IMAGE_SIZE", "%d,%d" % image_shape]
            cmd += ["-PIXEL_SCALE", "%0.16f" % pixel_scale]
            cmd += ["-WEIGHT_IMAGE", ",".join(weight_strings)]

            mask_cmd = [mask_cmd[0]] + [",".join(img_strings)] + mask_cmd[1:]
            mask_cmd += ["-CENTER", "%s,%s" % (coadd_center[0], coadd_center[1])]
            mask_cmd += ["-IMAGE_SIZE", "%d,%d" % image_shape]
            mask_cmd += ["-PIXEL_SCALE", "%0.16f" % pixel_scale]
            mask_cmd += ["-WEIGHT_IMAGE", ",".join(mask_strings)]

            # Set output filenames
            coadd_dir = os.path.join(
                self.base_dir, stash["desrun"], tilename, "coadd")
            if not os.path.isdir(coadd_dir):
                safe_mkdir(coadd_dir)
            coadd_file = os.path.join(
                coadd_dir,
                "%s_coadd_%s.fits" % (tilename, "".join(coadd_bands)))
            weight_file = os.path.join(
                coadd_dir,
                "%s_coadd_weight_%s.fits" % (tilename, "".join(coadd_bands)))
            mask_tmp_file = os.path.join(
                coadd_dir,
                "%s_coadd_%s_tmp.fits" % (tilename, "".join(coadd_bands)))
            mask_file = os.path.join(
                coadd_dir,
                "%s_coadd_%s_msk.fits" % (tilename, "".join(coadd_bands)))
            cmd += ["-IMAGEOUT_NAME", coadd_file]
            cmd += ["-WEIGHTOUT_NAME", weight_file]
            mask_cmd += ["-IMAGEOUT_NAME", mask_tmp_file]
            mask_cmd += ["-WEIGHTOUT_NAME", mask_file]

            # Move to the output directory to run swarp in case of
            # interference when running multiple tiles
            with pushd(os.path.realpath(coadd_dir)):
                self.logger.error(
                    "running swarp for tile %s: %s" % (
                        tilename, " ".join(cmd)
                    )
                )
                run_and_check(cmd, "SWarp", logger=self.logger)
                self.logger.error(
                    "running swarp for tile %s w/ mask: %s" % (
                        tilename, " ".join(mask_cmd)
                    )
                )
                run_and_check(mask_cmd, "SWarp", logger=self.logger)

            # remove tmp files
            safe_rm(mask_tmp_file)

            stash.set_filepaths("det_image_file", coadd_file, tilename, ext=0)
            stash.set_filepaths("det_weight_file", weight_file, tilename, ext=0)
            stash.set_filepaths("det_mask_file", mask_file, tilename, ext=0)

            self.logger.error("swarp complete for tile %s" % tilename)

        return 0, stash
