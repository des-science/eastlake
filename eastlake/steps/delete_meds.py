from __future__ import print_function, absolute_import
import os

from ..step import Step
from ..utils import safe_rm


class DeleteMeds(Step):
    """
    Pipeline for deleteing meds files to save disk space
    e.g. after the mcal step has run
    """
    def __init__(self, config, base_dir, name="delete_meds",
                 logger=None, verbosity=0, log_file=None):

        # name for this step
        super(DeleteMeds, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        if "save_tilenames" not in self.config:
            self.config["save_tilenames"] = []

    def execute(self, stash, new_params=None):
        tilenames = stash["tilenames"]
        for tilename in tilenames:
            if tilename in self.config["save_tilenames"]:
                continue

            meds_files = stash.get_filepaths("meds_files", tilename, keyerror=False)
            if meds_files is not None:
                for m in meds_files:
                    if os.path.isfile(m):
                        self.logger.debug("removing meds file %s" % m)
                        safe_rm(m)

            meds_files = stash.get_filepaths("pizza_cutter_meds_files", tilename, keyerror=False)
            if meds_files is not None:
                for m in meds_files:
                    if os.path.isfile(m):
                        self.logger.debug("removing pizza-cutter meds file %s" % m)
                        safe_rm(m)

        return 0, stash
