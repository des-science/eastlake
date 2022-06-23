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
from ..des_files import MAGZP_REF, get_tile_center


DEFAULT_SWARP_CONFIG = pkg_resources.resource_filename("eastlake", "astromatic/Y6A1_v1_swarp.config")

FITSEXTMAP = {
    "sci": 0,
    "msk": 1,
    "wgt": 2,
}


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
    """Notes from Robert on what should be done here:

    # this breakdown is based on a Y6A2 coadd tile (it was DES0130-4623) but I have stripped out lots of
    obscurring details from the file names.

    swarp @list/wgt-sci.list \
        -c config/Y6A1_v1_swarp.config \
        -WEIGHTOUT_NAME wgt.fits \
        -CENTER 22.632611,-46.386111  -PIXEL_SCALE 0.263 \
        -FSCALE_DEFAULT @list/wgt-flx.list \
        -IMAGE_SIZE 10000,10000 \
        -IMAGEOUT_NAME sci.fits \
        -COMBINE_TYPE WEIGHTED \
        -WEIGHT_IMAGE @list/wgt-wgt.list \
        -NTHREADS 8  -BLANK_BADPIXELS Y

    swarp @list/msk-sci.list \
        -c config/Y6A1_v1_swarp.config \
        -WEIGHTOUT_NAME msk.fits  \
        -CENTER 22.632611,-46.386111  -PIXEL_SCALE 0.263  \
        -FSCALE_DEFAULT @lists/msk-flx.list \
        -IMAGE_SIZE 10000,10000 \
        -IMAGEOUT_NAME  tmp-sci.fits \
        -COMBINE_TYPE WEIGHTED \
        -WEIGHT_IMAGE @list/msk-wgt.list \
        -NTHREADS 8  -BLANK_BADPIXELS Y

    coadd_assemble  --sci_file sci.fits
        --wgt_file wgt.fits
        --msk_file msk.fits
        --outname  coadd.fits
        --xblock 10  --yblock 3  --maxcols 100  --mincols 1  --no-keep_sci_zeros \
        --magzero 30  --tilename DES0130-4623  --tileid 119590  --interp_image MSK  --ydilate 3


    So the first SWarp execution:
        inputs:
            wgt-sci.list:   comprised of  *nwgint.fits[0]  <-- so reading the SCI hdu
            wgt-flx.list:   comprised of  fluxscale values   <-- 10. ** (0.4 * (30.0 - ZP))
            wgt-wgt.list:   comprised of  *nwgint.fits[2]  <-- so reading the WGT_ME hdu
        output:
            wgt.fits:  (through the WEIGHTOUT side of SWarp)     feeds the --wgt_file in coadd_assemble
            sci.fits:  (through the IMAGEOUT  side of SWarp) feeds the --sci_file in coadd_assemble


    Now the second SWarp execution:
        inputs:
            msk-sci.list:   comprised of *nwgint.fits[0]  <-- so again reading the SCI hdu
            msk-flx.list:   comprised of fluxscale values   (identical to other call)
            msk-wgt.list:   comprised of *nwgint.fits[1]  <-- so reading the WGT hdu
        output:
            msk.fits:     (through the WEIGHTOUT side of SWarp)  feeds the --msk_file in coadd_assemble
            tmp-sci.fits: (through the IMAGEOUT side of SWarp)  THIS IS BEING THROWN AWAY

    The tricky bits:
        coadd_nwgint --> the WGT_ME plane preserves WGT (rather than set to 0) wherever BADPIX_STAR is set
                         (so for instance when a BADPIX_TRAIL) intersects a star (BADPIX_STAR) the WGT is not set to 0)
                     --> the WGT plane takes all flags (from cmdline) and sets WGT=0

        coadd_assemble --> takes that thing called msk.fits above which used the WGT plane and sets the outtput
                    mask plane = 1 wherever the resulting SWarp WGT == 0
                 (i.e. all input images that overlapped that spot had WGT=0 --> no data -->  MSK=1 set
                  - however, WGT_ME will have values if a STAR was present... so the output WGT!=0
                  - later when detection runs those bright stars don't become odd collections of detections i
                        of the edges that do make it into an image

        it comes out as an int because:
            MSK = numpy.where(mask.fits == 0, 1, 0)
            MSK = MSK.astype(numpy.int32)
    """
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
                if not stash.has_tile_info_quantity("tile_center", tilename, band=band):
                    tile_center = get_tile_center(
                        stash.get_input_pizza_cutter_yaml(tilename, band)["image_path"]
                    )
                    stash.set_tile_info_quantity("tile_center", tile_center, tilename)

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
                            "%s[%d]" % (src["coadd_nwgint_path"], 0)
                            for src in output_pyml["src_info"]
                        ],
                        im_file_list,
                    )

                    wgt_file_list = "wgt_file_list.dat"
                    _write_relpath_file_list(
                        [
                            "%s[%d]" % (src["coadd_nwgint_path"], 1)
                            for src in output_pyml["src_info"]
                        ],
                        wgt_file_list,
                    )

                    wgt_me_file_list = "wgt_me_file_list.dat"
                    _write_relpath_file_list(
                        [
                            "%s[%d]" % (src["coadd_nwgint_path"], 2)
                            for src in output_pyml["src_info"]
                        ],
                        wgt_me_file_list,
                    )

                    msk_file_list = "msk_file_list.dat"
                    _write_relpath_file_list(
                        [
                            "%s[%d]" % (src["coadd_nwgint_path"], 3)
                            for src in output_pyml["src_info"]
                        ],
                        msk_file_list,
                    )

                    # We need to scale images to all have common zeropoint
                    mag_zp_list = [src["magzp"] for src in output_pyml["src_info"]]
                    fscale_list = [10**(0.4*(MAGZP_REF-mag_zp)) for mag_zp in mag_zp_list]

                    # the first call
                    cmd = [cmd[0]] + ["@%s" % im_file_list] + cmd[1:]
                    cmd += ["-IMAGE_SIZE", "%d,%d" % image_shape]
                    cmd += ["-PIXEL_SCALE", "%0.16f" % pixel_scale]
                    cmd += ["-CENTER", "%s,%s" % (
                        coadd_center[0], coadd_center[1])]
                    cmd += [
                        "-FSCALE_DEFAULT", ",".join(
                            ["%f" % _f for _f in fscale_list]
                        )
                    ]
                    cmd += ["-WEIGHT_IMAGE", "@%s" % wgt_me_file_list]
                    cmd += ["-IMAGEOUT_NAME", output_coadd_sci_file]
                    cmd += ["-WEIGHTOUT_NAME", output_coadd_weight_file]
                    self.logger.error(
                        "running swarp for tile %s, band %s:\n\t%s" % (
                            tilename, band, " ".join(cmd)))
                    run_and_check(cmd, "SWarp", logger=self.logger)

                    # the second call
                    dummy_mask_coadd = os.path.join(
                        output_coadd_dir, "%s_%s_msk-tmp.fits" % (tilename, band))
                    mask_cmd = self.swarp_cmd_root + extra_cmd_line_args
                    mask_cmd = (
                        [mask_cmd[0]] + ["@%s" % im_file_list] + mask_cmd[1:])
                    mask_cmd += ["-IMAGE_SIZE", "%d,%d" % image_shape]
                    mask_cmd += ["-PIXEL_SCALE", "%0.16f" % pixel_scale]
                    mask_cmd += ["-CENTER", "%s,%s" % (
                        coadd_center[0], coadd_center[1])]
                    cmd += [
                        "-FSCALE_DEFAULT", ",".join(
                            ["%f" % _f for _f in fscale_list]
                        )
                    ]
                    mask_cmd += ["-WEIGHT_IMAGE", "@%s" % wgt_file_list]
                    mask_cmd += ["-IMAGEOUT_NAME", dummy_mask_coadd]
                    mask_cmd += ["-WEIGHTOUT_NAME", output_coadd_mask_file]
                    self.logger.error(
                        "running swarp for tile %s, band %s w/ mask:\n\t%s" % (
                            tilename, band, " ".join(mask_cmd)))
                    run_and_check(mask_cmd, "Mask SWarp", logger=self.logger)
                    safe_rm(dummy_mask_coadd)

                    # now run coadd_assemble
                    safe_rm(output_coadd_path)
                    asmb_cmd = [
                        "coadd_assemble",
                        "--sci_file", output_coadd_sci_file,
                        "--wgt_file", output_coadd_weight_file,
                        "--msk_file", output_coadd_mask_file,
                        "--outname", output_coadd_path,
                        "--xblock", "10",
                        "--yblock", "3",
                        "--maxcols", "100",
                        "--mincols", "1",
                        "--no-keep_sci_zeros",
                        "--magzero", "30",
                        "--tilename", tilename,
                        "--tileid", "-9999",
                        "--interp_image", "MSK",
                        "--ydilate", "3",
                    ]
                    self.logger.error(
                        "running coadd_assemble for tile %s, band %s:\n\t%s" % (
                            tilename, band, " ".join(mask_cmd)))
                    run_and_check(asmb_cmd, "coadd_assemble", logger=self.logger)

                    # set headers and fpack
                    with fits.open(output_coadd_path, mode="update") as hdus:
                        for hdu in hdus:
                            _set_fpack_headers(hdu)
                            assert "EXTNAME" in hdu.header

                    safe_rm(output_coadd_path + ".fz")
                    self.logger.error(
                        "running fpack for tile %s, band %s:\n\t%s" % (
                            tilename, band, " ".join(mask_cmd)))
                    run_and_check(
                        ["fpack", os.path.basename(output_coadd_path)],
                        "fpack SWarp",
                        logger=self.logger
                    )

                    # delete intermediate files
                    safe_rm(output_coadd_sci_file)
                    safe_rm(output_coadd_weight_file)
                    safe_rm(output_coadd_mask_file)

                with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                    pyml["image_path"] = output_coadd_path + ".fz"
                    pyml["image_ext"] = "sci"
                    pyml["bmask_path"] = output_coadd_path + ".fz"
                    pyml["bmask_ext"] = "msk"
                    pyml["weight_path"] = output_coadd_path + ".fz"
                    pyml["weight_ext"] = "wgt"

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
                if not stash.has_tile_info_quantity("tile_center", tilename, band=band):
                    tile_center = get_tile_center(
                        stash.get_input_pizza_cutter_yaml(tilename, band)["image_path"]
                    )
                    stash.set_tile_info_quantity("tile_center", tile_center, tilename)

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
                img_strings.append("%s[%d]" % (im, FITSEXTMAP[ext]))

                weight_file, weight_ext = stash.get_filepaths(
                    "coadd_weight_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                weight_strings.append("%s[%d]" % (weight_file, FITSEXTMAP[weight_ext]))

                # We also need to coadd the masks, so get mask filenames.
                mask_file, mask_ext = stash.get_filepaths(
                    "coadd_mask_file", tilename, band=band, with_fits_ext=True, funpack=True,
                )
                mask_strings.append("%s[%d]" % (mask_file, FITSEXTMAP[mask_ext]))

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
