from __future__ import print_function, absolute_import
import os
import pkg_resources
import multiprocessing
import logging
import glob

import yaml
import numpy as np

from ..step import Step, run_and_check
from ..utils import safe_mkdir, safe_copy


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
        self.config["bands"] = self.config.get("bands", None)

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

            mdet_bands, mdet_conf = self._prep_config(in_bands, odir)

            mfiles = []
            for band in mdet_bands:
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

            bn = "".join(mdet_bands)
            cmd = [
                "run-metadetect-on-slices",
                "--config=%s" % mdet_conf,
                "--seed=%d" % seed,
                "--n-jobs=%d" % self.config["n_jobs"],
                "--log-level=%s" % llevel,
                "--use-tmpdir",
                "--output-path=%s" % odir,
                "--band-names=%s" % bn,
            ]
            if tmpdir is not None:
                cmd += ["--tmpdir=%s" % tmpdir]
            cmd += mfiles

            run_and_check(cmd, "MetadetectRunner", verbose=True)

            mdetfiles = glob.glob("%s/*_mdetcat_*.fits.fz" % odir)
            stash.set_filepaths("metadetect_files", mdetfiles, tilename)

            maskfiles = glob.glob("%s/*.hs" % odir)
            stash.set_filepaths("metadetect_mask_files", maskfiles, tilename)

        return 0, stash

    def _prep_config(self, in_bands, odir):
        if self.config["bands"] is not None:
            bands = self.config["bands"]
            det_bands = [list(range(len(bands)))]
            shear_bands = [list(range(len(bands)))]
        else:
            bands = in_bands
            if set(in_bands) == set(["g", "r", "i", "z"]):
                det_bands = [[1, 2, 3]]
                shear_bands = [[1, 2, 3]]
            else:
                det_bands = [list(range(len(bands)))]
                shear_bands = [list(range(len(bands)))]

        mdet_pth = os.path.join(odir, "metadetect-config.yaml")
        safe_copy(
            self.metadetect_config_file,
            mdet_pth,
        )
        with open(mdet_pth, "r") as fp:
            mdet_cfg = yaml.safe_load(fp.read())

        mdet_cfg["shear_band_combs"] = shear_bands
        mdet_cfg["det_band_combs"] = det_bands

        with open(mdet_pth, "w") as fp:
            yaml.dump(mdet_cfg, fp)

        return bands, mdet_pth
