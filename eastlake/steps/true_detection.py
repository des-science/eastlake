from __future__ import print_function, absolute_import
import os

import fitsio
import numpy as np

from ..step import Step
from ..utils import safe_copy, safe_rm
from ..des_piff import PSF_KWARGS
from ..des_files import MAGZP_REF


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

        # now we reformat to the src extractor output
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
            ('x2_image', 'f4'),
            ('y2_image', 'f4'),
            ('errx2_image', 'f4'),
            ('erry2_image', 'f4'),
            ('alpha_j2000', 'f8'),
            ('delta_j2000', 'f8'),
            ('a_world', 'f4'),
            ('b_world', 'f4'),
            ('flags', 'i2'),
            ('flux_radius', 'f4'),
            ('isoarea_image', 'f4'),
            ('flux_auto', 'f4'),
            ("mag_auto", 'f4'),
            ('fluxerr_auto', 'f4'),
            ('magerr_auto', 'f4'),
        ]
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
        if f"mag_{band}" in tdet.dtype.names:
            srcext_cat["mag_auto"] = tdet[f"mag_{band}"]
            srcext_cat["flux_auto"] = 10**(
                0.4*(MAGZP_REF - tdet[f"mag_{band}"])
            )

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
        srcext_cat_name = coadd_file.replace(".fits", "_cat.fits")
        with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
            pyml["cat_path"] = srcext_cat_name
        fitsio.write(srcext_cat_name, srcext_cat, clobber=True)

        # make the coadd object map file
        dtype = [
            ('id', 'i8'),
            ('object_number', 'i8'),
            ('gi_color', 'f8'),
            ('iz_color', 'f8'),
        ]
        obj_cat = np.zeros(len(tdet), dtype=dtype)
        obj_cat_name = coadd_file.replace(".fits", "_objmap.fits")
        obj_cat["id"] = srcext_cat['number']
        obj_cat["object_number"] = srcext_cat['number']
        if "mag_g" in tdet.dtype.names and "mag_i" in tdet.dtype.names:
            obj_cat["gi_color"] = tdet["mag_g"] - tdet["mag_i"]
            bad = ~np.isfinite(obj_cat["gi_color"])
            if np.any(bad):
                obj_cat["gi_color"][bad] = PSF_KWARGS["r"]["GI_COLOR"]
        else:
            obj_cat["gi_color"] = PSF_KWARGS["r"]["GI_COLOR"]

        if "mag_i" in tdet.dtype.names and "mag_z" in tdet.dtype.names:
            obj_cat["iz_color"] = tdet["mag_i"] - tdet["mag_z"]
            bad = ~np.isfinite(obj_cat["iz_color"])
            if np.any(bad):
                obj_cat["iz_color"][bad] = PSF_KWARGS["z"]["IZ_COLOR"]
        else:
            obj_cat["iz_color"] = PSF_KWARGS["z"]["IZ_COLOR"]

        fitsio.write(obj_cat_name, obj_cat, clobber=True)
        with stash.update_output_pizza_cutter_yaml(tilename, band):
            stash.set_filepaths(
                "coadd_object_map", obj_cat_name, tilename, band=band,
            )

    def _copy_and_munge_coadd_data(self, stash, tilename, band):
        # we need to set the coadd path and img ext for downstrem code
        orig_coadd_path = stash.get_input_pizza_cutter_yaml(
            tilename, band
        )["image_path"]
        coadd_path_from_imsim_data = os.path.relpath(
            orig_coadd_path, stash["imsim_data"])
        coadd_file = os.path.join(
            stash["base_dir"], coadd_path_from_imsim_data)

        # make a copy, decompress it if needed,
        if coadd_file[-3:] != '.fz':
            dest_coadd_file = coadd_file
        else:
            dest_coadd_file = coadd_file
            coadd_file = coadd_file[:-3]

        # delete them here to make sure things work ok
        safe_rm(coadd_file)
        safe_rm(dest_coadd_file)

        safe_copy(
            orig_coadd_path,
            dest_coadd_file,
        )

        if orig_coadd_path.endswith('.fz'):
            os.system('funpack %s' % dest_coadd_file)
            safe_rm(dest_coadd_file)

        # write all zeros in the image
        with fitsio.FITS(coadd_file, mode='rw') as fp:
            fp[0].write(
                np.zeros((10000, 10000)))

        # update the file paths
        with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
            pyml["image_path"] = coadd_file
            pyml["image_ext"] = 0
            pyml["bmask_path"] = coadd_file
            pyml["bmask_ext"] = 1
            pyml["weight_path"] = coadd_file
            pyml["weight_ext"] = 2
            pyml["seg_path"] = ""
            pyml["seg_ext"] = -1

        # copy the PSF model even though it is not the actual PSF
        orig_psf_path = stash.get_input_pizza_cutter_yaml(
            tilename, band
        )["psf_path"]
        psf_path_from_imsim_data = os.path.relpath(
            orig_psf_path, stash["imsim_data"])
        psf_file = os.path.join(
            stash["base_dir"], psf_path_from_imsim_data)
        safe_copy(
            orig_psf_path,
            psf_file,
        )
        with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
            pyml["psf_path"] = psf_file

        return coadd_file
