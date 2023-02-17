from __future__ import print_function, absolute_import
import os
import copy
import logging
import pkg_resources

import joblib

from ..step import Step, run_and_check
from ..utils import safe_mkdir, copy_ifnotexists, safe_rm


def _get_default_data(nm):
    return pkg_resources.resource_filename("eastlake", "data/%s" % nm)


def _get_default_config(nm):
    return pkg_resources.resource_filename("eastlake", "config/%s" % nm)


class CoaddNwgintRunner(Step):
    """
    Pipeline step for making coadd null weight images
    """
    def __init__(self, config, base_dir, name="coadd_nwgint",
                 logger=None, verbosity=0, log_file=None):

        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        self.streaks_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "streaks_file",
                        _get_default_data("Y3A2_v11_streaks_update-Y1234_FINALCUT_v1.fits"),
                    )
                )
            )
        )
        self.coadd_nwgint_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "config_file",
                        _get_default_config("Y6A1_v1_coadd_nwgint.config"),
                    )
                )
            )
        )

        self.coadd_nwgint_cmd = [
            "coadd_nwgint",
            "--hdupcfg", self.coadd_nwgint_config_file,
            "--streak_file", self.streaks_file,
            "--max_cols", "50",
            "-v",
            "--interp_mask", "TRAIL,BPM",
            "--invalid_mask", "EDGE",
            "--null_mask", "BPM,BADAMP,EDGEBLEED,EDGE,CRAY,SSXTALK,STREAK,TRAIL",
            "--block_size", "5",
            "--me_wgt_keepmask", "STAR",
        ]

        self.config["run_on_inputs"] = self.config.get("run_on_inputs", False)

    def execute(self, stash, new_params=None):
        for tilename in stash["tilenames"]:
            for band in stash["bands"]:
                in_pyml = stash.get_input_pizza_cutter_yaml(tilename, band)

                if self.config["run_on_inputs"]:
                    self._run(tilename, band, in_pyml, in_pyml, check_exists=True)
                    stash.write_input_pizza_cutter_yaml(in_pyml, tilename, band)
                else:
                    with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                        self._run(tilename, band, in_pyml, pyml)

        return 0, stash

    def _run(self, tilename, band, in_pyml, pyml, check_exists=False):
        jobs = []
        for i in range(len(pyml["src_info"])):
            src_info = pyml["src_info"][i]

            # get output path
            ofile = copy.deepcopy(src_info["image_path"])
            if ofile.endswith(".fz"):
                ofile = ofile[:-3]
            if ofile.endswith(".fits"):
                ofile = ofile[:-5]
            ofile = ofile + "_nullwt.fits"

            pyml["src_info"][i]["coadd_nwgint_path"] = ofile

            if check_exists and os.path.exists(ofile):
                if self.logger is not None:
                    self.logger.info("skipping existing coadd null weight filename: %s", ofile)
                continue

            if self.logger is not None:
                self.logger.info("making coadd null weight filename: %s", ofile)

            safe_mkdir(os.path.dirname(ofile))
            safe_rm(ofile)

            # copy scamp header
            copy_ifnotexists(
                in_pyml["src_info"][i]["head_path"],
                pyml["src_info"][i]["head_path"],
            )

            cmd = copy.deepcopy(self.coadd_nwgint_cmd)
            cmd += [
                "--tilename",
                tilename,
            ]
            cmd += [
                "--tileid",
                "-9999",
            ]
            cmd += [
                "-i",
                src_info["image_path"],
            ]
            cmd += [
                "--headfile",
                src_info["head_path"],
            ]
            cmd += [
                "-o",
                ofile,
            ]

            if self.logger is not None and self.logger.isEnabledFor(logging.DEBUG):
                verbose = True
            else:
                verbose = False

            jobs.append(joblib.delayed(run_and_check)(cmd, "CoaddNwgint", verbose=verbose))

        if len(jobs) > 0:
            if "n_jobs" in self.config.keys():
                n_jobs = self.config["n_jobs"]
            else:
                n_jobs = -1


            if self.logger is not None:
                self.logger.warning(
                    "making null weight images for tile %s band %s",
                    tilename, band,
                )
            else:
                print(
                    "making null weight images for tile %s band %s" % (
                        tilename, band,
                    )
                )

            with joblib.Parallel(n_jobs=n_jobs, backend="multiprocessing", verbose=100) as par:
                par(jobs)
