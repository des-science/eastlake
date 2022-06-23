from __future__ import print_function, absolute_import
import os
import pkg_resources
import multiprocessing

import numpy as np

from ..step import Step, run_and_check
from ..utils import safe_mkdir
from ..des_files import get_pizza_cutter_yaml_path


def _get_default_config(nm):
    return pkg_resources.resource_filename("eastlake", "config/%s" % nm)


class PizzaCutterRunner(Step):
    """
    Pipeline step for making pizza cutter coadd images
    """
    def __init__(self, config, base_dir, name="pizza_cutter",
                 logger=None, verbosity=0, log_file=None):

        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        self.pizza_cutter_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "config_file",
                        _get_default_config("des-pizza-slices-y6-v14.yaml"),
                    )
                )
            )
        )
        self.config["n_jobs"] = int(
            self.config.get(
                "n_jobs",
                multiprocessing.cpu_count()//2,
            )
        )
        self.config["n_chunks"] = 2*self.config["n_jobs"]

    def execute(self, stash, new_params=None):
        rng = np.random.RandomState(seed=stash["step_primary_seed"])
        for tilename in stash["tilenames"]:
            pz_meds = []
            for band in stash["bands"]:

                info_file = get_pizza_cutter_yaml_path(
                    self.base_dir,
                    stash["desrun"],
                    tilename,
                    band,
                )

                tmpdir = os.environ.get("TMPDIR", None)
                odir = os.path.join(
                    self.base_dir, stash["desrun"], tilename
                )
                ofile = os.path.join(
                    odir,
                    "%s_%s_%s_meds-pizza-slices.fits.fz" % (
                        tilename,
                        band,
                        os.path.basename(self.pizza_cutter_config_file.replace(".yaml", ""))
                    ),
                )
                safe_mkdir(odir)

                cmd = [
                    "des-pizza-cutter",
                    "--config", self.pizza_cutter_config_file,
                    "--info=%s" % info_file,
                    "--output-path=%s" % ofile,
                    "--use-tmpdir",
                    "--n-jobs=%d" % self.config["n_jobs"],
                    "--n-chunks=%d" % self.config["n_chunks"],
                    "--seed=%d" % rng.randint(1, 2**31),
                    "--log-level=%s" % self.config.get("log_level", "INFO").upper(),
                ]
                if tmpdir is not None:
                    cmd += ["--tmpdir=%s" % tmpdir]

                run_and_check(cmd, "PizzaCutterRunner")
                pz_meds.append(ofile)

            stash.set_filepaths("pizza_cutter_meds_files", pz_meds, tilename)

        return 0, stash
