from __future__ import print_function, absolute_import
import pprint

import galsim
import galsim.config

from ..step import Step


class GalSimRunner(Step):
    """
    Pipeline step which runs galsim

    The config attribute is a little different here, since it is updated when
    running GalSim
    """

    def __init__(self, config, base_dir, name="galsim", logger=None,
                 verbosity=0, log_file=None):
        super().__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)
        self.config['output']['dir'] = base_dir

        self.config_orig = galsim.config.CopyConfig(self.config)

    def execute(self, stash, new_params=None, except_abort=False, verbosity=1.,
                log_file=None, comm=None):

        if comm is not None:
            rank = comm.Get_rank()
        else:
            rank = 0

        if new_params is not None:
            galsim.config.UpdateConfig(self.config, new_params)

        # Make a copy of original config
        config = galsim.config.CopyConfig(self.config)
        if rank == 0:
            self.logger.debug(
                "Process config dict: \n%s", pprint.pformat(config))

        if self.name not in stash:
            stash[self.name] = {}

        galsim.config.Process(config, self.logger, except_abort=except_abort)

        # Return status and stash
        return 0, stash

    @classmethod
    def from_config_file(cls, config_file, base_dir, logger=None):
        all_config = galsim.config.ReadConfig(config_file, None, logger)
        assert len(all_config) == 1
        return cls(all_config[0], base_dir, logger=logger)

    def set_base_dir(self, base_dir):
        self.base_dir = base_dir
        # Update the output directory.
        self.config['output']['dir'] = base_dir
