from __future__ import print_function, absolute_import
import os
import pkg_resources
import multiprocessing
import logging

import yaml
import numpy as np

from ..step import Step, run_and_check
from ..utils import safe_mkdir, safe_copy, copy_ifnotexists
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
                        _get_default_config("des-pizza-slices-y6-v15.yaml"),
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
        self.config["use_nwgint"] = self.config.get("use_nwgint", False)

    def execute(self, stash, new_params=None):
        rng = np.random.RandomState(seed=stash["step_primary_seed"])
        for tilename in stash["tilenames"]:
            pz_meds = []
            for band in stash["bands"]:
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

                # copy input files
                in_pyml = stash.get_input_pizza_cutter_yaml(tilename, band)
                pyml = stash.get_output_pizza_cutter_yaml(tilename, band)
                safe_copy(
                    in_pyml["gaia_stars_file"],
                    pyml["gaia_stars_file"],
                )
                for i in range(len(pyml["src_info"])):
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

                pzyml_pth, info_file = self._prep_input_config_and_info_file(
                    tilename, band, stash, odir, stash["bands"],
                )

                if self.logger is not None:
                    llevel = logging.getLevelName(self.logger.getEffectiveLevel())
                else:
                    llevel = "WARNING"

                cmd = [
                    "des-pizza-cutter",
                    "--config", pzyml_pth,
                    "--info=%s" % info_file,
                    "--output-path=%s" % ofile,
                    "--use-tmpdir",
                    "--n-jobs=%d" % self.config["n_jobs"],
                    "--n-chunks=%d" % self.config["n_chunks"],
                    "--seed=%d" % rng.randint(1, 2**31),
                    "--log-level=%s" % llevel,
                ]
                if tmpdir is not None:
                    cmd += ["--tmpdir=%s" % tmpdir]

                run_and_check(cmd, "PizzaCutterRunner", verbose=True)
                pz_meds.append(ofile)

            stash.set_filepaths("pizza_cutter_meds_files", pz_meds, tilename)

        return 0, stash

    def _prep_input_config_and_info_file(self, tilename, band, stash, odir, allbands):
        pzyml_pth = os.path.join(odir, "pizza-cutter-config.yaml")
        safe_copy(
            self.pizza_cutter_config_file,
            pzyml_pth,
        )
        with open(pzyml_pth, "r") as fp:
            pzyml = yaml.safe_load(fp.read())

        if stash["psf_config"]["type"] in ["DES_Piff"]:
            # this is the default and let's make sure
            assert pzyml["single_epoch"]["psf_type"] == "piff"
        else:
            pzyml["single_epoch"]["psf_kwargs"] = {b: {} for b in allbands}

            if stash["psf_config"]["type"] in ["DES_PSFEx", "DES_PSFEx_perturbed"]:
                pzyml["single_epoch"]["psf_type"] = "psfex"
            else:
                pzyml["single_epoch"]["psf_type"] = "galsim"

                # do this check to ensure things are ~constant
                # not perfect but ok
                for size_key in ["half_light_radius", "sigma", "fwhm"]:
                    if size_key in stash["psf_config"]:
                        try:
                            float(stash["psf_config"][size_key])
                        except ValueError as e:
                            self.logger.error(
                                "couldn't interpret psf %s "
                                "as float" % (size_key))
                            raise e
                        break

                # write the config info for later
                with stash.update_output_pizza_cutter_yaml(tilename, band) as pyml:
                    for i in range(len(pyml["src_info"])):
                        pyml["src_info"][i]["galsim_psf_config"] = stash["psf_config"]

        with open(pzyml_pth, "w") as fp:
            yaml.dump(pzyml, fp)

        # make info file
        info_file_pth = os.path.join(
            odir, "%s_%s_pizza_cutter_info_used.yaml")
        safe_copy(
            get_pizza_cutter_yaml_path(
                self.base_dir,
                stash["desrun"],
                tilename,
                band,
            ),
            info_file_pth,
        )
        if self.config["use_nwgint"]:
            with open(info_file_pth, "r") as fp:
                info = yaml.safe_load(fp.read())

            _, img_ext = stash.get_filepaths(
                "coadd_nwgint_img_files", tilename, band=band, with_fits_ext=True
            )
            _, wgt_ext = stash.get_filepaths(
                "coadd_nwgint_wgt_files", tilename, band=band, with_fits_ext=True
            )
            _, msk_ext = stash.get_filepaths(
                "coadd_nwgint_msk_files", tilename, band=band, with_fits_ext=True
            )
            for i in range(len(info["src_info"])):
                info["src_info"][i]["image_path"] = info["src_info"][i]["coadd_nwgint_path"]
                info["src_info"][i]["image_ext"] = img_ext
                info["src_info"][i]["weight_path"] = info["src_info"][i]["coadd_nwgint_path"]
                info["src_info"][i]["weight_ext"] = wgt_ext
                info["src_info"][i]["bmask_path"] = info["src_info"][i]["coadd_nwgint_path"]
                info["src_info"][i]["bmask_ext"] = msk_ext

            with open(info_file_pth, "w") as fp:
                yaml.dump(info, fp)

        return pzyml_pth, info_file_pth
