from __future__ import print_function, absolute_import
import os

from ..step import Step
from ..utils import safe_rm


class DeleteImages(Step):
    """
    Pipeline for deleteing images to save disk space
    e.g. after the meds step has run
    """
    def __init__(self, config, base_dir, name="delete_images",
                 logger=None, verbosity=0, log_file=None):

        # name for this step
        super(DeleteImages, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        if "delete_coadd" not in self.config:
            self.config["delete_coadd"] = False
        if "delete_se" not in self.config:
            self.config["delete_se"] = False
        if "delete_se_nwgint" not in self.config:
            self.config["delete_se_nwgint"] = False
        if "save_tilenames" not in self.config:
            self.config["save_tilenames"] = []
        if "delete_seg" not in self.config:
            self.config["delete_seg"] = False

    def execute(self, stash, new_params=None):

        tilenames = stash["tilenames"]
        for tilename in tilenames:
            if tilename in self.config["save_tilenames"]:
                continue

            # Firstly coadd stuff
            if self.config["delete_coadd"]:
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
                    coadd_file = stash.get_filepaths("coadd_file", tilename, band=band,
                                                     keyerror=False)
                    if (coadd_file is not None):
                        if os.path.isfile(coadd_file):
                            self.logger.debug("removing file %s" % coadd_file)
                            safe_rm(coadd_file)

                    # Also check for seg file
                    if self.config["delete_seg"]:
                        seg_file = stash.get_filepaths("seg_file", tilename, band=band,
                                                       keyerror=False)
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

            # Secondly se stuff
            if self.config["delete_se"]:
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

            if self.config["delete_se_nwgint"]:
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

        return 0, stash
