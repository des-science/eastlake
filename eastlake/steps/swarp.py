from __future__ import print_function, absolute_import
import os
import subprocess

import fitsio
import astropy.io.fits as fits

from ..utils import safe_mkdir
from ..step import Step, run_and_check, _get_relpath
from ..des_tile import get_orig_coadd_file


class SingleBandSwarpRunner(Step):
    def __init__(self, config, base_dir, name="single_band_swarp", logger=None,
                 verbosity=0, log_file=None):
        super(SingleBandSwarpRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)
        self.swarp_config_file = os.path.abspath(
            os.path.expanduser(os.path.expandvars(self.config["config_file"])))
        self.swarp_cmd = os.path.expandvars(self.config["swarp_cmd"])
        self.swarp_cmd_root = [
            "%s" % (self.swarp_cmd), "-c", "%s" % (self.swarp_config_file)]
        if "mag_zp_ref" not in config:
            config["mag_zp_ref"] = 30.

        if "update" in self.config:
            for key, val in self.config["update"].items():
                self.logger.info(
                    "Updating swarp config with %s = %s" % (key, val))
                self.swarp_cmd_root += ["-%s" % key, str(val)]

    def execute(self, stash, new_params=None):
        tilenames = stash["tilenames"]

        orig_working_dir = os.getcwd()
        for tilename in tilenames:
            tile_info = stash["tile_info"][tilename]
            for band in stash["bands"]:
                self.logger.error(
                    "running swarp for tile %s, band %s" % (
                        tilename, band))
                cmd = self.swarp_cmd_root
                im_file_list, wgt_file_list, msk_file_list, magzp_list_file \
                    = tile_info[band]["source_list_files"]

                coadd_center = tile_info["tile_center"]

                orig_coadd_path = get_orig_coadd_file(
                    stash["desdata"], stash["desrun"], tilename, band)
                coadd_path_from_desdata = _get_relpath(
                    orig_coadd_path, stash["desdata"])
                output_coadd_path = os.path.join(
                    stash["base_dir"], coadd_path_from_desdata)
                if output_coadd_path.endswith("fits.fz"):
                    output_coadd_path = output_coadd_path[:-3]
                output_coadd_dir = os.path.dirname(output_coadd_path)

                output_coadd_sci_file = os.path.join(
                    output_coadd_dir, "%s_%s_sci.fits" % (tilename, band))
                output_coadd_weight_file = os.path.join(
                    output_coadd_dir, "%s_%s_wgt.fits" % (tilename, band))

                # make the output directory and then move here to run swarp
                # this prevents the intermediate files being fucked up by
                # other processes.
                safe_mkdir(output_coadd_dir)
                os.chdir(os.path.realpath(output_coadd_dir))

                # Swarp is annoying and can vomit a segfault if the paths
                # to the input images are too long. This can happen with the
                # paths in e.g. im_file_list which are absolute paths which
                # generally seems a safe thing to use but anyway. So re-make
                # the file lists with paths relative to the current working
                # directory
                def write_relpath_file_list(orig_file_list, output_file_list):
                    with open(orig_file_list, 'r') as f:
                        lines = f.readlines()
                    output_lines = []
                    for ln in lines:
                        rel_str = _get_relpath(ln.strip())
                        if len(ln.strip()) < len(rel_str):
                            output_lines.append("%s\n" % ln.strip())
                        else:
                            output_lines.append("%s\n" % rel_str)
                        assert (
                            os.path.exists(os.path.abspath(output_lines[-1])[:-4])
                        ), os.path.abspath(output_lines[-1][:-4])
                    with open(output_file_list, 'w') as f:
                        f.writelines(output_lines)

                rel_im_file_list = "im_file_list.dat"
                rel_wgt_file_list = "wgt_file_list.dat"
                rel_msk_file_list = "msk_file_list.dat"
                write_relpath_file_list(im_file_list, rel_im_file_list)
                im_file_list = rel_im_file_list
                write_relpath_file_list(wgt_file_list, rel_wgt_file_list)
                wgt_file_list = rel_wgt_file_list
                write_relpath_file_list(msk_file_list, rel_msk_file_list)
                msk_file_list = rel_msk_file_list

                cmd = [cmd[0]] + ["@%s" % im_file_list] + cmd[1:]
                cmd += ["-WEIGHTOUT_NAME", output_coadd_weight_file]
                cmd += ["-CENTER", "%s,%s" % (
                    coadd_center[0], coadd_center[1])]
                cmd += ["-IMAGEOUT_NAME", output_coadd_sci_file]
                cmd += ["-WEIGHT_IMAGE", "@%s" % wgt_file_list]

                # Zeropoints...
                with open(magzp_list_file, 'r') as f:
                    lines = f.readlines()
                mag_zp_list = [float(ln) for ln in lines]
                # We need to scale images to all have common zeropoint
                fscale_list = [
                    10**(0.4*(self.config["mag_zp_ref"]-mag_zp))
                    for mag_zp in mag_zp_list]

                cmd += ["-FSCALE_DEFAULT", ",".join(
                    ["%f" % _f for _f in fscale_list])]

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
                output_coadd_mask_file = os.path.join(
                    output_coadd_dir, "%s_%s_msk.fits" % (tilename, band))
                stash.set_filepaths(
                    "coadd_mask_file", output_coadd_mask_file, tilename,
                    band=band)
                mask_cmd = self.swarp_cmd_root
                mask_cmd = (
                    [mask_cmd[0]] + ["@%s" % msk_file_list] + mask_cmd[1:])
                mask_cmd += ["-WEIGHTOUT_NAME", output_coadd_mask_file]
                mask_cmd += ["-CENTER", "%s,%s" % (
                    coadd_center[0], coadd_center[1])]
                mask_cmd += ["-IMAGEOUT_NAME", dummy_mask_coadd]
                mask_cmd += ["-WEIGHT_IMAGE", "@%s" % msk_file_list]
                # run swarp
                run_and_check(mask_cmd, "Mask SWarp", logger=self.logger)
                # remove the dummy mask coadd
                os.remove(dummy_mask_coadd)

                # We've done the swarping, now combine image, weight and
                # mask planes
                # generate an hdu list
                im_hdu = fits.open(output_coadd_sci_file)[0]
                # stupidly, if you ask me, we cannot simply read in the
                # weight and coadd hdus and add them directly to an HDUList
                # because they are PrimaryHDUs...
                wgt_fits = fits.open(output_coadd_weight_file)[0]
                wgt_hdu = fits.ImageHDU(wgt_fits.data, header=wgt_fits.header)
                msk_fits = fits.open(output_coadd_mask_file)[0]
                msk_hdu = fits.ImageHDU(msk_fits.data, header=msk_fits.header)
                hdus = [im_hdu, msk_hdu, wgt_hdu]
                hdulist = fits.HDUList(hdus)
                # output file for coadd
                stash.set_filepaths(
                    "coadd_file", output_coadd_path, tilename, band=band)
                tile_info[band]["coadd_ext"] = 0
                self.logger.info(
                    "writing assembled coadd for tilename %s, "
                    "band %s to %s" % (
                        tilename, band, output_coadd_path))
                hdulist.writeto(output_coadd_path, overwrite=True)
                stash.set_filepaths(
                    "coadd_mask_file", output_coadd_path, tilename, band=band)
                tile_info[band]["coadd_mask_ext"] = 1
                stash.set_filepaths(
                    "coadd_weight_file", output_coadd_path, tilename,
                    band=band)
                tile_info[band]["coadd_weight_ext"] = 2

                # delete intermediate files
                os.remove(output_coadd_sci_file)
                os.remove(output_coadd_weight_file)
                os.remove(output_coadd_mask_file)

                # move back to orig_working_dor
                os.chdir(orig_working_dir)

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
            os.path.expanduser(os.path.expandvars(self.config["config_file"])))
        # swarp command my be an environment variable
        # so use os.path.expandvars
        self.swarp_cmd = os.path.expandvars(self.config["swarp_cmd"])
        # make sure Swarp works...
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

        orig_working_dir = os.getcwd()
        for tilename in tilenames:
            tile_file_info = stash["tile_info"][tilename]
            cmd = self.swarp_cmd_root
            mask_cmd = self.swarp_cmd_root
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
                im, ext = (
                    stash.get_filepaths("coadd_file", tilename, band=band),
                    tile_file_info[band]["coadd_ext"])
                img_strings.append("%s[%d]" % (im, ext))

                weight_file, weight_ext = (
                    stash.get_filepaths(
                        "coadd_weight_file", tilename, band=band),
                    tile_file_info[band]["coadd_weight_ext"])

                # Astromatic doesn't like compressed fits files, so we may
                # need to unpack here if the weight file is .fits.fz. In this
                # case we also need to subtract 1 from the weight extension
                if ".fits.fz" in weight_file:
                    wf_funpacked = weight_file.replace(".fits.fz", ".fits")
                    if not os.path.isfile(wf_funpacked):
                        subprocess.check_output(["funpack", weight_file])
                    weight_file = wf_funpacked
                    weight_ext -= 1
                    stash.set_filepaths(
                        "coadd_weight_file", weight_file, tilename, band=band)
                    tile_file_info[band]["coadd_weight_ext"] = weight_ext
                weight_strings.append("%s[%d]" % (weight_file, weight_ext))

                # We also need to coadd the masks, so get mask filenames.
                mask_file, mask_ext = (
                    stash.get_filepaths(
                        "coadd_mask_file", tilename, band=band),
                    tile_file_info[band]["coadd_mask_ext"])
                mask_strings.append("%s[%d]" % (mask_file, mask_ext))

            # Add image and weight info to cmd - image files should be first
            # argument
            cmd = [cmd[0]] + [",".join(img_strings)] + cmd[1:]
            cmd += ["-WEIGHT_IMAGE", ",".join(weight_strings)]

            mask_cmd = [mask_cmd[0]] + [",".join(img_strings)] + mask_cmd[1:]
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

            # Get center from image file by default
            if self.config.get("center_from_header", True):
                coadd_file_band_0 = stash.get_filepaths(
                        "coadd_file", tilename, band=bands[0])
                ext = tile_file_info[bands[0]]["coadd_ext"]
                h = fitsio.read_header(coadd_file_band_0, ext)
                cen_vals = (str(h["CRVAL1"]), str(h["CRVAL2"]))
                self.logger.error(
                    "Setting SWarp center: %s,%s from header of %s" % (
                        cen_vals[0], cen_vals[1],
                        tile_file_info[bands[0]]["coadd_file"]))
                cen = ["-CENTER", "%s,%s" % (cen_vals)]
                cmd += cen
                mask_cmd += cen

            if self.logger is not None:
                self.logger.error("calling swarp:")
                self.logger.error(" ".join(cmd))

            # Move to the output directory to run swarp in case of
            # interference when running multiple tiles
            os.chdir(os.path.realpath(coadd_dir))

            run_and_check(cmd, "SWarp", logger=self.logger)
            run_and_check(mask_cmd, "SWarp", logger=self.logger)

            # remove tmp files
            os.remove(mask_tmp_file)

            stash.set_filepaths("det_image_file", coadd_file, tilename)
            tile_file_info["det_image_ext"] = 0
            stash.set_filepaths("det_weight_file", weight_file, tilename)
            tile_file_info["det_weight_ext"] = 0
            stash.set_filepaths("det_mask_file", mask_file, tilename)
            tile_file_info["det_mask_ext"] = 0

            self.logger.error("swarp complete for tile %s" % tilename)

        os.chdir(orig_working_dir)

        return 0, stash
