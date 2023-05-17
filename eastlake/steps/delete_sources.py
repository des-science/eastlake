from __future__ import print_function, absolute_import
import os

from ..step import Step


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
                        self.logger.error("removing file %s" % filename)
                        os.remove(filename)
                    else:
                        self.logger.error("file %s not found" % filename)
                else:
                    self.logger.error("key %s not present" % key)

            # Now the per-band coadds
            for band in stash["bands"]:
                coadd_file = stash.get_filepaths(
                    "coadd_file", tilename, band=band,
                    keyerror=False,
                )
                if (coadd_file is not None):
                    if os.path.isfile(coadd_file):
                        os.remove(coadd_file)

                # Also check for seg file
                seg_file = stash.get_filepaths(
                    "seg_file", tilename, band=band,
                    keyerror=False,
                )
                if (seg_file is not None):
                    if os.path.isfile(seg_file):
                        self.logger.error("removing file %s" % seg_file)

                # Also check for bkg and bkg-rms files
                bkg_file = coadd_file.replace(".fits", "bkg.fits")
                if os.path.isfile(bkg_file):
                    os.remove(bkg_file)
                bkg_rms_file = coadd_file.replace(".fits", "bkg-rms.fits")
                if os.path.isfile(bkg_rms_file):
                    os.remove(bkg_rms_file)

            self.logger.error("deleting se images for tile %s" % tilename)
            for band in stash["bands"]:
                img_files = stash.get_filepaths(
                    "img_files", tilename, band=band,
                    keyerror=False)
                if (img_files is not None):
                    for f in img_files:
                        if os.path.isfile(f):
                            os.remove(f)

            self.logger.error("deleting se nwgint images for tile %s" % tilename)
            for band in stash["bands"]:
                img_files = stash.get_filepaths(
                    "coadd_nwgint_img_files", tilename, band=band,
                    keyerror=False)
                if (img_files is not None):
                    for f in img_files:
                        if os.path.isfile(f):
                            os.remove(f)

            self.logger.error("deleting as much as we can for tile %s" % tilename)
            for band in stash["bands"]:
                pyml = stash.get_output_pizza_cutter_yaml(tilename, band)
                for k, v in pyml.items():
                    if isinstance(v, str) and os.path.isfile(v):
                        os.remove(v)

                    if k == "src_info":
                        for srci in pyml["src_info"]:
                            for _v in srci.items():
                                if isinstance(_v, str) and os.path.isfile(_v):
                                    os.remove(_v)

        return 0, stash
