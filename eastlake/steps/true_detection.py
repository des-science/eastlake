from __future__ import print_function, absolute_import
import os
import shutil

import fitsio
import numpy as np

from ..des_files import get_orig_coadd_file
from ..step import Step
from ..utils import safe_mkdir


class TrueDetectionRunner(Step):
    """Pipeline step for "true detection".

    Note this step is only meant for simulations with sources on a grid.

    Config Params
    -------------
    box_size : float
        The size of the box to use to set fields in the fake source extractor
        catalog.
    """
    def __init__(self, config, base_dir, name="true_detection",
                 logger=None, verbosity=0, log_file=None):

        # name for this step
        super(TrueDetectionRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        # this holds the desired box size which is used to set fields in the
        # for the "fake" catalog
        self.box_size = self.config["box_size"]

    def execute(self, stash, new_params=None, comm=None):
        tilenames = stash["tilenames"]
        for tilename in tilenames:
            for band in stash["bands"]:
                self._reformat_catalog(stash, tilename, band)

        return 0, stash

    def _reformat_catalog(self, stash, tilename, band):
        # make a copy of the coadd image from the data and munge the
        # stash so everything looks as if swarp ran
        coadd_file = self._copy_and_munge_coadd_data(stash, tilename, band)

        # the DES_Tile class will write this file if a grid of source
        # positions is used
        # this file has the coadd x,y and ra,dec of the true sources
        # in the image
        tdetfile = stash.get_filepaths("truepositions_file", tilename)
        tdet = fitsio.read(tdetfile)

        # now we reformat to the sectractor output
        # note that we are hacking on the fields to force the
        # MEDS maker to use the right sized stamps
        # to do this we
        # 1. set the flux radius to zero
        # 2. set the x[y]min[max]_image fields to have the
        #   desired box size
        # 3. set b_world to 1 and a_world to 0
        # 4. set the flags to zero
        dtype = [
            ('number', 'i4'),
            ('xmin_image', 'i4'),
            ('ymin_image', 'i4'),
            ('xmax_image', 'i4'),
            ('ymax_image', 'i4'),
            ('x_image', 'f4'),
            ('y_image', 'f4'),
            ('alpha_j2000', 'f8'),
            ('delta_j2000', 'f8'),
            ('a_world', 'f4'),
            ('b_world', 'f4'),
            ('flags', 'i2'),
            ('flux_radius', 'f4')]
        srcext_cat = np.zeros(len(tdet), dtype=dtype)
        srcext_cat['number'] = np.arange(len(tdet)) + 1
        srcext_cat['x_image'] = tdet['x']
        srcext_cat['y_image'] = tdet['y']
        srcext_cat['alpha_j2000'] = tdet['ra']
        srcext_cat['delta_j2000'] = tdet['dec']
        srcext_cat['a_world'] = 1
        srcext_cat['b_world'] = 0
        srcext_cat['flags'] = 0
        srcext_cat['flux_radius'] = 0

        half = int(self.box_size / 2)
        xint = (tdet['x'] + 0.5).astype(np.int32)
        srcext_cat['xmin_image'] = xint - half
        srcext_cat['xmax_image'] = (
            self.box_size - 1 + srcext_cat['xmin_image'])
        yint = (tdet['y'] + 0.5).astype(np.int32)
        srcext_cat['ymin_image'] = yint - half
        srcext_cat['ymax_image'] = (
            self.box_size - 1 + srcext_cat['ymin_image'])

        # now we add the new srcext catalog to the stash and
        # write it to disk
        srcext_cat_name = coadd_file.replace(".fits", "_sexcat.fits")
        stash.set_filepaths(
            "sex_cat", srcext_cat_name, tilename, band=band)
        fitsio.write(srcext_cat_name, srcext_cat, clobber=True)

    def _copy_and_munge_coadd_data(self, stash, tilename, band):
        # we need to set the coadd path and img ext for downstrem code
        orig_coadd_path = get_orig_coadd_file(
            stash["desdata"], stash["desrun"], tilename, band)
        coadd_path_from_desdata = os.path.relpath(
            orig_coadd_path, stash["desdata"])
        coadd_file = os.path.join(
            stash["base_dir"], coadd_path_from_desdata)

        # make a copy, decompress it if needed,
        if coadd_file[-3:] != '.fz':
            dest_coadd_file = coadd_file + '.fz'
        else:
            dest_coadd_file = coadd_file
            coadd_file = coadd_file[:-3]

        # delete them here to make sure things work ok
        try:
            os.remove(coadd_file)
        except Exception:
            pass

        try:
            os.remove(dest_coadd_file)
        except Exception:
            pass

        safe_mkdir(os.path.dirname(dest_coadd_file))

        shutil.copy(orig_coadd_path, dest_coadd_file)

        if orig_coadd_path.endswith('.fz'):
            os.system('funpack %s' % dest_coadd_file)

        # write all zeros in the image
        with fitsio.FITS(coadd_file, mode='rw') as fp:
            fp[0].write(
                np.zeros((10000, 10000)))

        # mock up the info for the file stash
        tile_info = stash["tile_info"][tilename]
        stash.set_filepaths(
            "coadd_file", coadd_file, tilename, band=band)
        tile_info[band]["coadd_ext"] = 0
        stash.set_filepaths(
            "coadd_mask_file", coadd_file, tilename, band=band)
        tile_info[band]["coadd_mask_ext"] = 1
        stash.set_filepaths(
            "coadd_weight_file", coadd_file, tilename, band=band)
        tile_info[band]["coadd_weight_ext"] = 2

        return coadd_file
