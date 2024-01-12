from __future__ import print_function, absolute_import
import os
import re

from ..step import Step
from ..utils import safe_rm, safe_rmdir, get_relpath


class DeleteSources(Step):
    """
    Pipeline for deleteing all of the source data for a tile.
    """
    def __init__(self, config, base_dir, name="delete_sources",
                 logger=None, verbosity=0, log_file=None):

        # name for this step
        super(DeleteSources, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        if "save_tilenames" not in self.config:
            self.config["save_tilenames"] = []

    def execute(self, stash, new_params=None):

        base_dir = stash["base_dir"]
        tilenames = stash["tilenames"]
        for tilename in tilenames:
            if tilename in self.config["save_tilenames"]:
                continue

            self.logger.error("deleting coadd images for tile %s" % tilename)
            # First check for a detection coadd, weight, mask and seg
            for key in ["det_image_file", "det_weight_file", "det_mask_file"]:
                filename = stash.get_filepaths(key, tilename, keyerror=False)
                if filename is not None:
                    if os.path.isfile(filename):
                        self.logger.debug("removing file %s" % filename)
                        safe_rm(filename)
                    else:
                        self.logger.debug("file %s not found" % filename)
                else:
                    self.logger.debug("key %s not present" % key)

            # Now the per-band coadds
            for band in stash["bands"]:

                coadd_file = stash.get_filepaths(
                    "coadd_file", tilename, band=band,
                    keyerror=False,
                )
                if (coadd_file is not None):
                    if os.path.isfile(coadd_file):
                        self.logger.debug("removing file %s" % coadd_file)
                        safe_rm(coadd_file)

                # Also check for seg file
                seg_file = stash.get_filepaths(
                    "seg_file", tilename, band=band,
                    keyerror=False,
                )
                if (seg_file is not None):
                    if os.path.isfile(seg_file):
                        self.logger.debug("removing file %s" % seg_file)
                        safe_rm(seg_file)

                # Also check for bkg and bkg-rms files
                bkg_file = coadd_file.replace(".fits", "bkg.fits")
                if os.path.isfile(bkg_file):
                    self.logger.debug("removing file %s" % bkg_file)
                    safe_rm(bkg_file)
                bkg_rms_file = coadd_file.replace(".fits", "bkg-rms.fits")
                if os.path.isfile(bkg_rms_file):
                    self.logger.debug("removing file %s" % bkg_rms_file)
                    safe_rm(bkg_rms_file)

                # ...and coadd object map
                coadd_object_map_file = stash.get_filepaths(
                    "coadd_object_map", tilename, band=band,
                    keyerror=False,
                )
                if (coadd_object_map_file is not None):
                    if os.path.isfile(coadd_object_map_file):
                        self.logger.debug("removing file %s" % coadd_object_map_file)
                        safe_rm(coadd_object_map_file)

            self.logger.error("deleting swarp files %s" % tilename)
            coadd_bands = []
            for band in stash["bands"]:
                # Clean up any single-band files leftover from swarp
                orig_coadd_path = stash.get_input_pizza_cutter_yaml(tilename, band)["image_path"]
                coadd_path_from_imsim_data = get_relpath(
                    orig_coadd_path, stash["imsim_data"])
                output_coadd_path = os.path.join(
                    base_dir, coadd_path_from_imsim_data)
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

                dummy_mask_coadd = os.path.join(
                    output_coadd_dir, "%s_%s_msk-tmp.fits" % (tilename, band))
                safe_rm(dummy_mask_coadd)

                safe_rm(output_coadd_path + ".fz")

                im_file_list = os.path.join(output_coadd_dir, "im_file_list.dat")
                wgt_file_list = os.path.join(output_coadd_dir, "wgt_file_list.dat")
                wgt_me_file_list = os.path.join(output_coadd_dir, "wgt_me_file_list.dat")
                msk_file_list = os.path.join(output_coadd_dir, "msk_file_list.dat")
                safe_rm(im_file_list)
                safe_rm(wgt_file_list)
                safe_rm(wgt_me_file_list)
                safe_rm(msk_file_list)

            # Remove swarp coadd files
            # Match the coadd bands with regular expressions as the swarp
            # coadds may be made with a different subset of bands
            coadd_dir = os.path.join(
                base_dir, stash["desrun"], tilename, "coadd")

            bands = "".join(stash["bands"])

            det_coadd_file_re = re.compile(f"{tilename}_coadd_det_[{bands}]+.fits")
            coadd_file_re = re.compile(f"{tilename}_coadd_[{bands}]+.fits")
            weight_file_re = re.compile(f"{tilename}_coadd_weight_[{bands}]+.fits")
            mask_tmp_file_re = re.compile(f"{tilename}_coadd_[{bands}]+_tmp.fits")
            mask_file_re = re.compile(f"{tilename}_coadd_[{bands}]+_msk.fits")

            for coadd_file in os.listdir(coadd_dir):
                if (
                    det_coadd_file_re.fullmatch(coadd_file)
                    or coadd_file_re.fullmatch(coadd_file)
                    or weight_file_re.fullmatch(coadd_file)
                    or mask_tmp_file_re.fullmatch(coadd_file)
                    or mask_file_re.fullmatch(coadd_file)
                 ):
                     coadd_file_path = os.path.join(coadd_dir, coadd_file)
                     safe_rm(coadd_file_path)

            self.logger.error("deleting se images for tile %s" % tilename)
            for band in stash["bands"]:
                img_files = stash.get_filepaths(
                    "img_files", tilename, band=band,
                    keyerror=False)
                if (img_files is not None):
                    for f in img_files:
                        if os.path.isfile(f):
                            self.logger.debug("removing file %s" % f)
                            safe_rm(f)

            self.logger.error("deleting se nwgint images for tile %s" % tilename)
            for band in stash["bands"]:
                img_files = stash.get_filepaths(
                    "coadd_nwgint_img_files", tilename, band=band,
                    keyerror=False)
                if (img_files is not None):
                    for f in img_files:
                        if os.path.isfile(f):
                            self.logger.debug("removing file %s" % f)
                            safe_rm(f)

            self.logger.error("deleting as much as we can for tile %s" % tilename)
            for band in stash["bands"]:
                pyml = stash.get_output_pizza_cutter_yaml(tilename, band)
                for k, v in pyml.items():
                    if isinstance(v, str):
                        totry = [
                            v,
                            v.replace(".fits", ".fits.fz"),
                            v.replace(".fits.fz", ".fits"),
                        ]
                        for t in totry:
                            if os.path.isfile(t):
                                self.logger.debug("removing file %s" % t)
                                safe_rm(t)

                    if k == "src_info":
                        for srci in pyml["src_info"]:
                            for _v in srci.values():
                                if isinstance(_v, str):
                                    totry = [
                                        _v,
                                        _v.replace(".fits", ".fits.fz"),
                                        _v.replace(".fits.fz", ".fits"),
                                    ]
                                    for t in totry:
                                        if os.path.isfile(t):
                                            self.logger.debug("removing file %s" % t)
                                            safe_rm(t)

                self.logger.error("deleting %s-band nullwt links for %s" % (band, tilename))

                for sri in pyml["src_info"]:
                    nullwt_link = os.path.join(
                        base_dir, stash["desrun"], tilename, f"nullwt-{band}",
                        os.path.basename(sri["coadd_nwgint_path"])
                    )
                    safe_rm(nullwt_link)

                nullwt_path = os.path.join(
                    base_dir, stash["desrun"], tilename, f"nullwt-{band}",
                )
                safe_rmdir(nullwt_path)

                self.logger.error("deleting %s-band psfs links for %s" % (band, tilename))

                psf_link = os.path.join(
                    base_dir, stash["desrun"], tilename, "psfs",
                    os.path.basename(pyml["psf_path"])
                )
                safe_rm(psf_link)

                for sri in pyml["src_info"]:
                    psf_link = os.path.join(
                        base_dir, stash["desrun"], tilename, "psfs",
                        os.path.basename(sri["psf_path"])
                    )
                    safe_rm(psf_link)

            psf_path = os.path.join(
                base_dir, stash["desrun"], tilename, "psfs",
            )
            safe_rmdir(psf_path)

            self.logger.error("deleting empty dirs")

            tile_path = os.path.join(base_dir, stash["desrun"], tilename)
            for root, dirs, files in os.walk(tile_path, topdown=False):
                for name in dirs:
                    full_dir = os.path.join(root, name)
                    if len(os.listdir(full_dir)) == 0:
                        safe_rmdir(full_dir)

        return 0, stash
