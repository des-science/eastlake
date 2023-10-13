from __future__ import print_function, absolute_import
import fitsio
import numpy as np
import os
import multiprocessing
import shutil
import subprocess

import joblib
import pkg_resources
import yaml
from datetime import timedelta
from timeit import default_timer as timer


from ..utils import safe_mkdir, safe_rm, copy_ifnotexists, safe_copy
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

        self.meds_config = os.path.abspath(
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
        self.config["use_nwgint"] = self.config.get("use_nwgint", True)
        self.fpack_seeds_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "fpack_seeds_file",
                        _get_default_config("Y6A1_v1_meds-desdm-Y6A1v11-fpack-seeds.yaml"),
                    )
                )
            )
        )

    def clear_stash(self, stash):
        # If we continued the pipeline from a previous job record file,
        # mof_file entries can mess things up, so clear them
        if "tile_info" in stash:
            for tilename, tile_file_info in stash["tile_info"].items():
                tile_file_info.pop("meds_files", None)

    def execute(self, stash, new_params=None, boxsize=64):

        self.clear_stash(stash)

        os.environ["MEDS_DIR"] = self.base_dir
        meds_run = stash["desrun"]

        # Loop through tiles
        tilenames = stash["tilenames"]

        for tilename in tilenames:
            # meds files
            meds_files = []

            # image data
            for band in stash["bands"]:
                t0 = timer()

                self._copy_inputs(stash, tilename, band)
                meds_config, fileconf = self._prep_config_and_flist(
                    stash, tilename, band, meds_run,
                )

                meds_file = os.path.join(
                    self.base_dir, meds_run, tilename,
                    "%s_%s_meds-%s.fits.fz" % (tilename, band, meds_run))
                final_meds_file = os.path.join(
                    self.base_dir, stash["desrun"], tilename,
                    "%s_%s_meds-%s.fits.fz" % (tilename, band, meds_run))
                meds_files.append(final_meds_file)
                for _mpth in [meds_file, final_meds_file]:
                    safe_mkdir(os.path.dirname(os.path.normpath(_mpth)))
                    safe_rm(_mpth[:-len(".fz")])
                    safe_rm(_mpth)

                tmpdir = os.environ.get("TMPDIR", None)
                cmd = [
                    "desmeds-make-meds-desdm",
                    f"{meds_config}",
                    f"{fileconf}",
                    f"--tmpdir={tmpdir}",
                ]
                run_and_check(
                    cmd,
                    "desmeds-make-meds-desdm",
                    logger=self.logger,
                    verbose=True,
                )

                # move to final spot
                shutil.move(meds_file, final_meds_file)

                t1 = timer()
                self.logger.error(
                    "Time to write meds file for tile %s, band %s: %s" % (
                        tilename, band, str(timedelta(seconds=t1-t0))))

            stash.set_filepaths("meds_files", meds_files, tilename)

        return 0, stash

    def _copy_inputs(self, stash, tilename, band):
        # copy input files
        in_pyml = stash.get_input_pizza_cutter_yaml(tilename, band)
        pyml = stash.get_output_pizza_cutter_yaml(tilename, band)

        def _process_entry(i):
            # we don't overwrite these since we could have estimated them
            copy_ifnotexists(
                in_pyml["src_info"][i]["head_path"],
                pyml["src_info"][i]["head_path"],
            )
            copy_ifnotexists(
                in_pyml["src_info"][i]["piff_path"],
                pyml["src_info"][i]["piff_path"],
            )
            copy_ifnotexists(
                in_pyml["src_info"][i]["psf_path"],
                pyml["src_info"][i]["psf_path"],
            )
            copy_ifnotexists(
                in_pyml["src_info"][i]["seg_path"],
                pyml["src_info"][i]["seg_path"],
            )

            seg_pth = pyml["src_info"][i]["seg_path"]
            if seg_pth.endswith(".fz"):
                nofz_seg_pth = seg_pth[:-len(".fz")]
                safe_rm(nofz_seg_pth)
                subprocess.run(
                    f"funpack {seg_pth}",
                    shell=True,
                    check=True,
                )
            else:
                nofz_seg_pth = seg_pth

            with fitsio.FITS(nofz_seg_pth, mode='rw') as fp:
                fp[0].write(np.zeros(tuple(fp[0].get_dims())))

            if seg_pth.endswith(".fz"):
                safe_rm(seg_pth)
                subprocess.run(
                    f"fpack {nofz_seg_pth}",
                    shell=True,
                    check=True,
                )
                safe_rm(nofz_seg_pth)

        print("prepping data for MEDS making", flush=True)
        jobs = [
            joblib.delayed(_process_entry)(i)
            for i in range(len(pyml["src_info"]))
        ]
        with joblib.Parallel(
            n_jobs=self.config["n_jobs"],
            backend="loky",
            verbose=100,
        ) as par:
            par(jobs)

    def _prep_config_and_flist(self, stash, tilename, band, meds_run):
        # set # of workers in config
        meds_yml_pth = os.path.join(
            self.base_dir,
            stash["desrun"],
            tilename,
            "meds-config.yaml",
        )
        safe_copy(
            self.meds_config,
            meds_yml_pth,
        )
        with open(meds_yml_pth, "r") as fp:
            meds_yml = yaml.safe_load(fp.read())

        # read the fpack seeds and set them
        if os.path.exists(self.fpack_seeds_file):
            with open(self.fpack_seeds_file, "r") as fp:
                fpack_seeds = yaml.safe_load(fp.read())

            if fpack_seeds.get(tilename, {}).get(band, None) is not None:
                meds_yml["fpack_seeds"] = fpack_seeds[tilename][band]
            else:
                raise RuntimeError(
                    f"Could not find fpack seeds for {tilename} {band}!"
                )

        if "joblib" in meds_yml:
            meds_yml["joblib"]["max_workers"] = self.config["n_jobs"]

        with open(meds_yml_pth, "w") as fp:
            yaml.dump(meds_yml, fp)

        # copy in null weight images if needed
        orig_fileconf = stash.get_filepaths(
            "desdm-fileconf", tilename, band=band
        )
        if self.config["use_nwgint"]:
            fileconf = os.path.join(
                os.path.dirname(orig_fileconf),
                f"{tilename}_{band}_nullwtfileconf-{meds_run}.yaml",
            )
            safe_copy(
                orig_fileconf,
                fileconf,
            )
            with open(fileconf, "r") as fp:
                fconf = yaml.safe_load(fp.read())
            fconf["finalcut_flist"] = fconf["nullwt_flist"]
            with open(fileconf, "w") as fp:
                yaml.dump(fconf, fp)
        else:
            fileconf = orig_fileconf

        return meds_yml_pth, fileconf

    @classmethod
    def from_config_file(cls, config_file, base_dir=None, logger=None,
                         name="meds"):
        with open(config_file, "rb") as f:
            config = yaml.safe_load(f)
        return cls(config, base_dir=base_dir, logger=logger, name=name)
