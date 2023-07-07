from __future__ import print_function, absolute_import
import os
import multiprocessing

import pkg_resources
import yaml
from datetime import timedelta
from timeit import default_timer as timer


from ..utils import safe_mkdir, safe_rm
from ..step import Step, run_and_check


def _get_default_config(nm):
    return pkg_resources.resource_filename("eastlake", "config/%s" % nm)


class DESDMMEDSRunner(Step):
    """
    Pipeline step for generating MEDS files as DESDM does it
    """
    def __init__(self, config, base_dir, name="desdm_meds", logger=None,
                 verbosity=0, log_file=None):
        super().__init__(
            config, base_dir, name=name,
            logger=logger, verbosity=verbosity, log_file=log_file)

        self.pizza_cutter_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "config_file",
                        _get_default_config("Y6A1_v1_meds-desdm-Y6A1v11.yaml"),
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
        self.config["use_nwgint"] = self.config.get("use_nwgint", False)

    def clear_stash(self, stash):
        # If we continued the pipeline from a previous job record file,
        # mof_file entries can mess things up, so clear them
        if "tile_info" in stash:
            for tilename, tile_file_info in stash["tile_info"].items():
                tile_file_info.pop("meds_files", None)

    def execute(self, stash, new_params=None, boxsize=64):

        self.clear_stash(stash)

        os.environ["MEDS_DIR"] = self.base_dir

        # Loop through tiles
        tilenames = stash["tilenames"]

        for tilename in tilenames:
            # meds files
            meds_files = []

            # image data
            for band in stash["bands"]:
                t0 = timer()

                if self.config["use_nwgint"]:
                    self._setup_nwgint()

                meds_run = os.path.basename(
                    self.config["config_file"]
                ).rsplit(".", 1)[0]
                meds_file = os.path.join(
                    self.base_dir, meds_run, tilename,
                    "%s_%s_meds-%s.fits.fz" % (tilename, band, meds_run))
                meds_files.append(meds_file)

                d = os.path.dirname(os.path.normpath(meds_file))
                safe_mkdir(d)
                safe_rm(meds_file[:-len(".fz")])
                safe_rm(meds_file)

                tmpdir = os.environ.get("TMPDIR", None)
                fileconf = stash.get_filepaths(
                    "desdm-fileconf", tilename, band=band
                )
                cfg = self.config["config_file"]
                cmd = (
                    f"desmeds-make-meds-desdm "
                    f"{cfg}"
                    f"{fileconf} "
                    f"--tmpdir=${tmpdir}"
                )
                run_and_check(
                    cmd, "desmeds-make-meds-desdm", logger=self.logger
                )

                t1 = timer()
                self.logger.error(
                    "Time to write meds file for tile %s, band %s: %s" % (
                        tilename, band, str(timedelta(seconds=t1-t0))))

            stash.set_filepaths("meds_files", meds_files, tilename)

        return 0, stash

    def _setup_nwgint(self):
        pass

    @classmethod
    def from_config_file(cls, config_file, base_dir=None, logger=None,
                         name="meds"):
        with open(config_file, "rb") as f:
            config = yaml.safe_load(f)
        return cls(config, base_dir=base_dir, logger=logger, name=name)
