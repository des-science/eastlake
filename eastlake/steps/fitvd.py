from __future__ import print_function, absolute_import
import os
import pkg_resources

import fitsio
import joblib
import numpy as np

from ..step import Step, run_and_check
from ..utils import safe_mkdir, safe_rm


def _get_default_config(nm):
    return pkg_resources.resource_filename("eastlake", "config/%s" % nm)


def get_fofs_chunks(*, n_chunks=None, fofs_path=None):

    # Chunk up the fofs groups:
    with fitsio.FITS(fofs_path) as fits:
        fofs = fits[1].read()

    n_fofs = len(np.unique(fofs['fof_id']))

    print("fof groups: ", n_fofs)

    n_objs_chunks = n_fofs // n_chunks

    print("n_chunks : ", n_chunks)
    print("n_objs_chunks : ", n_objs_chunks)

    starts_ends = []
    for i in range(n_chunks):
        start = i * n_objs_chunks
        end = i * n_objs_chunks + n_objs_chunks - 1
        if i == n_chunks - 1:
            end = n_fofs - 1
        starts_ends.append((start, end))

    return starts_ends

def get_fitvd_chunks(*, n_chunks=None, shredx_fits_path=None):

    with fitsio.FITS(shredx_fits_path) as fits:
        shredx = fits[1].read()

    n_objs = len(shredx)

    print("fitvd objs: ", n_objs)
    n_objs_chunks = n_objs // n_chunks

    print("n_chunks : ", n_chunks)
    print("n_objs_chunks : ", n_objs_chunks)

    starts_ends = []
    for i in range(n_chunks):
        start = i * n_objs_chunks
        end = i * n_objs_chunks + n_objs_chunks - 1
        if i == n_chunks - 1:
            end = n_objs - 1
        starts_ends.append((start, end))

    return starts_ends


class FitvdRunner(Step):
    """
    Pipeline step for running shredx and fitvd
    """
    def __init__(self, config, base_dir, name="fitvd",
                 logger=None, verbosity=0, log_file=None):

        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        self.shredx_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "shredx_config",
                        _get_default_config("Y6A1_v1_shredx.yaml"),
                    )
                )
            )
        )
        self.fitvd_config_file = os.path.abspath(
            os.path.expanduser(
                os.path.expandvars(
                    self.config.get(
                        "fitvd_config",
                        _get_default_config("Y6A1_v1_fitvd.yaml"),
                    )
                )
            )
        )

        # fitvd relies on an older version of ngmix; here, we use a separate conda
        # environment to run these codes.
        if (CONDA_PREFIX := self.config.get("conda_prefix", None)) is not None:
            CMD_PREFIX = [
                "conda", "run",
                "--prefix", CONDA_PREFIX,
            ]
        else:
            CMD_PREFIX = []
        self.CMD_PREFIX = CMD_PREFIX

    def execute(self, stash, new_params=None):
        rng = np.random.RandomState(seed=stash["step_primary_seed"])
        for tilename in stash["tilenames"]:
            shredx_seed = rng.randint(1, 2**31)
            fitvd_seed = rng.randint(1, 2**31)

            fitvd_files = []

            output_dir = os.path.join(
                self.base_dir, stash["desrun"], tilename, "fitvd",
            )
            shredx_dir = os.path.join(
                output_dir,
                "shredx",
            )
            sof_dir = os.path.join(
                output_dir,
                "sof",
            )
            safe_mkdir(output_dir)
            safe_mkdir(shredx_dir)
            safe_mkdir(sof_dir)

            meds_dict = dict(
                zip(
                    stash.get("bands"),
                    stash.get_filepaths("meds_files", tilename)
                )
            )
            pyml_dict = {
                band: stash.get_output_pizza_cutter_yaml(
                    tilename,
                    band,
                )
                for band in self.config.get("bands")
            }

            bands = self.config.get("bands")
            det_band = self.config.get("det_band")

            image_paths = {}
            segmap_paths = {}
            catalog_paths = {}
            psf_paths = {}
            meds_paths = {}

            for band in bands:
                _image_path = pyml_dict[band].get("image_path")
                _segmap_path = pyml_dict[band].get("seg_path")
                _catalog_path = pyml_dict[band].get("cat_path")
                _psf_path = pyml_dict[band].get("psf_path")
                _meds_path = meds_dict[band]

                image_paths[band] = _image_path
                segmap_paths[band] = _segmap_path
                catalog_paths[band] = _catalog_path
                psf_paths[band] = _psf_path
                meds_paths[band] = _meds_path

            # Preserve this in case needed later
            req_num = os.path.basename(segmap_paths["r"])[13:21]

            # 1. Create shredx fofs
            shredx_fofslist = os.path.join(
                output_dir,
                f"{tilename}_{req_num}_shredx-fofslist.fits",
            )
            cmd = [
                "shredx-make-fofs",
                "--seg", segmap_paths[det_band],
                "--output", shredx_fofslist,
            ]
            cmd = self.CMD_PREFIX + cmd
            run_and_check(
                cmd,
                "shredx-make-fofs",
            )

            # Prepare fof chunks
            fofs_chunks = get_fofs_chunks(
                n_chunks=self.config.get("n_jobs"),
                fofs_path=shredx_fofslist,
            )

            shredx_chunklist = [
                os.path.join(
                    shredx_dir,
                    f"{tilename}_shredx-chunk-{start}-{end}.fits",
                )
                for start, end in fofs_chunks
            ]

            # 2. Run shredx
            cmd_list = [
                [
                    "shredx",
                    "--start", str(start),
                    "--end", str(end),
                    "--seed", str(shredx_seed),
                    "--images", *[image_paths[band] for band in bands],
                    "--psf", *[psf_paths[band] for band in bands],
                    "--cat", catalog_paths[det_band],
                    "--seg", segmap_paths[det_band],
                    "--config", self.shredx_config_file,
                    "--fofs", shredx_fofslist,
                    "--outfile", shredx_chunklist[i],
                ]
                for i, (start, end) in enumerate(fofs_chunks)
            ]

            jobs = []
            for cmd in cmd_list:
                cmd = self.CMD_PREFIX + cmd
                jobs.append(joblib.delayed(
                    run_and_check
                )(cmd, "shredx"))

            with joblib.Parallel(
                n_jobs=self.config.get("n_jobs", -1),
                backend="loky",
                verbose=100,
            ) as par:
                par(jobs)

            shredx_flist = os.path.join(
                output_dir,
                "shredx_flist.txt",
            )
            with open(shredx_flist, "w") as fobj:
                for shredx_chunk in shredx_chunklist:
                    fobj.write(f"{shredx_chunk}\n")

            # 3. Collate shredx output
            shredx_fits = os.path.join(
                output_dir,
                f"{tilename}_{req_num}_shredx.fits",
            )
            cmd = [
                "shredx-collate",
                "--tilename", tilename,
                "--output", shredx_fits,
                "--flist", shredx_flist,
            ]
            cmd = self.CMD_PREFIX + cmd
            run_and_check(
                cmd,
                "shredx-collate"
            )

            safe_rm(shredx_flist)

            fitvd_files.append(shredx_fits)

            # Prepare fitvd chunks
            fitvd_chunks = get_fitvd_chunks(
                n_chunks=self.config.get("n_jobs"),
                shredx_fits_path=shredx_fits,
            )
            sof_output = [
                os.path.join(
                    sof_dir,
                    f"{tilename}_sof-chunk-{start}-{end}.fits",
                )
                for start, end in fitvd_chunks
            ]

            # 4. Run fitvd
            cmd_list = [
                [
                    "fitvd",
                    "--start", str(start),
                    "--end", str(end),
                    "--seed", str(fitvd_seed),
                    "--config", self.fitvd_config_file,
                    "--model-pars", shredx_fits,
                    "--output", sof_output[i],
                    *[meds_paths[band] for band in bands],
                ]
                for i, (start, end) in enumerate(fitvd_chunks)
            ]

            jobs = []
            for cmd in cmd_list:
                cmd = self.CMD_PREFIX + cmd
                jobs.append(joblib.delayed(
                    run_and_check
                )(cmd, "fitvd"))

            with joblib.Parallel(
                n_jobs=self.config.get("n_jobs", -1),
                backend="loky",
                verbose=100,
            ) as par:
                par(jobs)

            # 5. Collate fitvd output
            sof_fits = os.path.join(
                output_dir,
                f"{tilename}_{req_num}_sof.fits",
            )
            cmd = [
                "fitvd-collate",
                "--meds", meds_paths[det_band],
                "--output", sof_fits,
                *sof_output,
            ]
            cmd = self.CMD_PREFIX + cmd
            run_and_check(
                cmd,
                "fitvd-collate"
            )
            fitvd_files.append(sof_fits)

            # Cleaning up

            # Remove intermediate outputs
            safe_rm(shredx_fofslist)
            for outfile in shredx_chunklist:
                safe_rm(outfile)
            for outfile in sof_output:
                safe_rm(outfile)

            # Inform the stash of the final outputs
            stash.set_filepaths("fitvd_files", fitvd_files, tilename)

        return 0, stash
