"""
Notes from Megan

command to run:

python balrog_injection.py \
  bal_config_TEST.yaml \
  -c ../../Balrog-GalSim/config/ \
  -g ../../inputs/Y3A2_COADDTILE_GEOM_REDDEN.fits \
  -l ../../inputs/TileList.txt \
  -t ../../../MEDS_DIR/des-pizza-slices-y6-v13/ \
  -p ../../../MEDS_DIR/des-pizza-slices-y6-v13/DES0350-6622/psfs/ \
  -o ../../sim_outputs/des-pizza-slices-y6-v13/ \
  -n 1 \
  -v 3


Balrog code makes:

1. Configs - Contains the injections that galsim needs to run

  sim_outputs/des-pizza-slices-y6-v13/configs/bal_config_0_DES0350-6622.yaml

2. null weight images

  sim_outputs/des-pizza-slices-y6-v13/balrog_images/0/des-pizza-slices-y6-v13/DES0350-6622/
  Contains dirs: <griz> which contain the inj.fits files for each chip and band, example:
  sim_outputs/des-pizza-slices-y6-v13/balrog_images/0/des-pizza-slices-y6-v13/DES0350-6622/i/D00809077_i_c35_r4061p02_balrog_inj.fits

"""  # noqa
from __future__ import print_function, absolute_import
import os
import multiprocessing
import logging

from ..step import Step, run_and_check
from ..utils import LOGGING_LEVELS


class BalrogRunner(Step):
    """
    Pipeline step for running balrog
    """
    def __init__(self, config, base_dir, name="balrog",
                 logger=None, verbosity=0, log_file=None):

        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        self.config["n_jobs"] = int(
            self.config.get(
                "n_jobs",
                multiprocessing.cpu_count(),
            )
        )

    def execute(self, stash, new_params=None):
        if self.logger is not None:
            llevel = self.logger.getEffectiveLevel()
        else:
            llevel = logging.WARNING
        llevel = LOGGING_LEVELS[llevel]

        for tilename in stash["tilenames"]:
            tlist = os.path.join(
                self.base_dir,
                stash["desrun"],
                tilename,
                "balrog_tile_list.txt"
            )
            with open(tlist, "w") as fp:
                fp.write(tilename)

            cmd = [
                "python", "balrog_injection.py",
                self.config["config_file"],
                "-g", self.config["coadd_tile_geom_file"],
                "-l", tlist,
                "-t", os.path.join(stash["imsim_data"], stash["desrun"]),
                "-p", os.path.join(stash["imsim_data"], stash["desrun"], tilename, "psfs"),
                "-o", os.path.join(self.base_dir, stash["desrun"]),
                "-n", "%d" % self.config["n_jobs"],
                "-v", "%d" % llevel,
            ]

            run_and_check(cmd, "BalrogRunner", verbose=True)

            for band in stash["bands"]:
                with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                    for i in range(len(pyml["src_info"])):
                        src = pyml["src_info"][i]
                        bname = os.path.basename(src["image_path"])
                        if bname.endswith(".fz"):
                            bname = bname[:-3]
                        if bname.endswith(".fits"):
                            bname = bname[:-len(".fits")]
                        if bname.endswith("_immasked"):
                            bname = bname[:-len("_immasked")]

                        ofile = os.path.join(
                            self.base_dir,
                            stash["desrun"],
                            tilename,
                            "balrog_images",
                            "0",
                            stash["desrun"],
                            tilename,
                            band,
                            bname + "_balrog_inj.fits",
                        )

                    pyml["coadd_nwgint_path"] = ofile

        return 0, stash
