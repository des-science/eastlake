from __future__ import print_function, absolute_import
import os
import pkg_resources
import multiprocessing
import logging
import glob

import numpy as np

from ..step import Step, run_and_check
from ..utils import safe_mkdir


def _get_default_config(nm):
    return pkg_resources.resource_filename("eastlake", "config/%s" % nm)


class MetadetectRunner(Step):
    """
    Pipeline step for running metadetect
    """
    def __init__(self, config, base_dir, name="metadetect",
                 logger=None, verbosity=0, log_file=None):

        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        self.metadetect_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "config_file",
                        _get_default_config("metadetect-v5.yaml"),
                    )
                )
            )
        )
        self.config["n_jobs"] = int(
            self.config.get(
                "n_jobs",
                multiprocessing.cpu_count(),
            )
        )
        self.config["bands"] = self.config.get("bands", ["g", "r", "i", "z"])

    def execute(self, stash, new_params=None):
        rng = np.random.RandomState(seed=stash["step_primary_seed"])
        if self.logger is not None:
            llevel = self.logger.getEffectiveLevel()
            if llevel > 20:
                llevel = 20
            llevel = logging.getLevelName(llevel)
        else:
            llevel = "INFO"

        tmpdir = os.environ.get("TMPDIR", None)

        for tilename in stash["tilenames"]:
            seed = rng.randint(1, 2**31)
            odir = os.path.join(
                self.base_dir, stash["desrun"], tilename, "metadetect",
            )
            safe_mkdir(odir)

            in_mfiles = stash.get_filepaths("pizza_cutter_meds_files", tilename)
            in_bands = []
            for mf in in_mfiles:
                _mf = os.path.basename(mf)
                in_bands.append(_mf.split("_")[1])

            mfiles = []
            for band in self.bands:
                for i in range(len(in_bands)):
                    found = None
                    if band == in_bands[i]:
                        found = i
                        break

                if found is None:
                    raise RuntimeError(
                        "band %s not found for tile %s in metadetect!" % (
                            band, tilename
                        )
                    )
                mfiles.append(in_mfiles[found])

            bn = "".join(self.bands)
            cmd = [
                "run-metadetect-on-slices",
                "--config=%s" % self.metadetect_config_file,
                "--seed=%d" % seed,
                "--n-jobs=%d" % self.n_jobs,
                "--log-level=%s" % llevel,
                "--use-tmpdir",
                "--output-path=%s" % odir,
                "--band-names=%s" % bn,
            ]
            if tmpdir is not None:
                cmd += ["--tmpdir=%s" % tmpdir]
            cmd += mfiles

            run_and_check(cmd, "PizzaCutterRunner", verbose=True)

            mdetfiles = glob.glob("%s/*.fits.fz" % odir)
            stash.set_filepaths("metadetect_files", mdetfiles, tilename)

            maskfiles = glob.glob("%s/*.hs" % odir)
            stash.set_filepaths("metadetect_mask_files", maskfiles, tilename)

        return 0, stash
