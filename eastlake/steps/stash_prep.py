from __future__ import print_function, absolute_import
import os

from ..step import Step
from ..des_files import read_pizza_cutter_yaml


class StashPrep(Step):
    """
    Pipeline step which prepares the stash with imsidata, desrun and tilename
    """

    def __init__(
        self, config, base_dir, name="stash_prep", logger=None, verbosity=0, log_file=None
    ):
        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

    def execute(self, stash, new_params=None, except_abort=False, verbosity=1.,
                log_file=None, comm=None):

        stash["tilenames"] = [self.config["tilename"]]
        stash["desrun"] = self.config["desrun"]
        try:
            stash["imsim_data"] = self.config["imsim_data"]
        except KeyError:
            stash["imsim_data"] = os.environ["IMSIM_DATA"]
        stash["bands"] = self.config.get("bands", ["g", "r", "i", "z"])

        for tilename in stash["tilenames"]:
            for band in stash["bands"]:
                stash.set_input_pizza_cutter_yaml(
                    read_pizza_cutter_yaml(
                        stash["imsim_data"],
                        stash["desrun"],
                        tilename,
                        band,
                        n_se_test=self.config.get("n_se_test", None),
                    ),
                    tilename,
                    band,
                )

        # Return status and stash
        return 0, stash
