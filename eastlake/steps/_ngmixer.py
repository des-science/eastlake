from __future__ import print_function, absolute_import
import os
import glob
import shutil

import numpy as np
import fitsio
import ngmixer

from ..megamixer import ImSimMegaMixer
from ..step import Step, run_subprocess
from ..utils import safe_mkdir

class NGMixerRunner(Step):
    """Class for running ngmixer.
    Parallelisation could definitely be made more efficient...in particular the
    setting up scripts part; the master-worker code for running the scrips
    should be efficient.
    """
    def __init__(self, config, base_dir, name="ngmixer", logger=None,
                 verbosity=0, log_file=None):
        super(NGMixerRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)
        # Set environment variables
        os.environ["NGMIXER_OUTPUT_DIR"] = os.path.join(
            self.base_dir, "ngmixer")
        # ngmixer uses $TMPDIR. It may be on some machines you don't want to
        # use the default for this so you can set an alternative in the config
        # file. This should be a relative path w.r.t self.base_dir.
        # We won't actually set the env variable until the execute function as
        # if we set it here that could screw up other pipeline steps.
        self.tmpdir_orig = os.environ.get("TMPDIR", "")
        if "tmpdir" in self.config:
            self.tmpdir_new = os.path.join(
                self.base_dir, self.config["tmpdir"])
            safe_mkdir(self.tmpdir_new)
        else:
            self.tmpdir_new = None

        # modes are mof and mcal
        if "mode" in self.config:
            self.mode = self.config["mode"]
        elif self.name in ["mof", "mcal", "sof"]:
            self.mode = self.name
        else:
            raise ValueError(
                "Either step name should be one of the defaults ('mof' or "
                "'mcal') or a mode should be provided in the config file")
        self.file_key = self.config.get("file_key", "%s_file" % self.mode)
        self.meds_run_from_config = self.config.get("meds_conf", None)
        self.meds_dir_from_config = self.config.get("meds_dir", None)

        # handy to optionally set the chunksize here
        self.set_chunk_size = None
        if "chunksize" in self.config:
            self.set_chunksize = self.config["chunksize"]

        # Might be that e.g. for testing, we've reduced the number of bands
        # simulated, but don't want to mess with the ngmixer config file.
        # You can set use_available_bands to True to just look for bands
        # that have been simulated.
        self.use_available_bands = self.config.get(
            "use_available_bands", False)

        if "delete_chunks" not in self.config:
            self.config["delete_chunks"] = True

    def clear_stash(self, stash):
        pass

    def execute(self, stash, new_params=None):
        """Perform the following steps:
        1) Setup a megamixer for each tile in stash
        2) If running mof, also call the setup_nbrs function for each
            megamixer
        3) If running mof, loop through the megamixers, running the script
            generated by step (ii)
        4) Loop through megamixers calling setup function
        5) Now loop through the megamixers, calling all the ngmixer scripts
            generated by step (iv).
        6) When all ngmixer scripts for a given tile have been called,
            collate that tile.
        """
        self.clear_stash(stash)

        # tmpdir (see discussion in __init__)
        if self.tmpdir_new is not None:
            os.environ["TMPDIR"] = self.tmpdir_new

        # get meds_conf and meds_dir - these are needed for finding the meds
        # files
        try:
            medsconf = stash["meds_run"]
        except KeyError:
            self.logger.info(
                "No meds_run entry in stash, using meds_conf entry "
                "from config")
            try:
                assert (self.meds_run_from_config is not None)
            except AssertionError as e:
                self.logger.error(
                    "No meds_run entry in the stash, or the config")
                raise(e)
            medsconf = self.meds_run_from_config
        conf = ngmixer.files.read_config(self.config["config_file"])
        if self.use_available_bands:
            conf["jobs"]["bands"] = stash["bands"]
        # record what bands we're running in the stash
        stash["%s_bands" % self.mode] = conf["jobs"]["bands"]

        try:
            os.environ['MEDS_DIR']
        except KeyError:
            assert (self.meds_dir_from_config is not None)
            os.environ['MEDS_DIR'] = self.meds_dir_from_config

        # Loop through tiles setting up megamixers
        # The setup_nbrs() function generates lists of scripts for setting up
        # neighbours.
        megamixers = []
        tile_nums = []

        self.logger.error("setting up %d megamixers" % (
                len(stash["tilenames"])))
        for tile_num, tilename in enumerate(stash["tilenames"]):
            tile_file_info = stash["tile_info"][tilename]

            # Check if we're skipping this tile
            if tile_file_info.get("skip", False):
                self.logger.error(
                    "skipping tile %s" % tile_file_info["tilename"])
                continue
            tile_id = tilename
            meds_files = ngmixer.files.get_meds_files(
                medsconf,
                tile_id,
                conf['jobs']['bands'],
            )
            # Check for psfs in file.
            # The config format for this changed in ngmixer.
            # Version v0.9.4b used for Y3 just has
            # conf["psfs_in_file"] = True (or False).
            # In newer versions conf['imageio']['psfs']["type"] = "infile"
            # So need to check for both of these
            use_psf_map_files = True
            if "imageio" in conf:
                if "psfs" in conf['imageio']:
                    psfs = conf['imageio']['psfs']
                    if psfs['type'] == 'infile':
                        use_psf_map_files = False
            elif "psfs_in_file" in conf:
                if conf["psfs_in_file"] is True:
                    use_psf_map_files = False
            if use_psf_map_files:
                if "imageio" in conf:
                    if (("psfs" in conf["imageio"]) and
                            (conf["imageio"]["psfs"]["type"] == "piff")):
                        psfs = conf["imageio"]["psfs"]
                        piff_run = psfs['piff_run']
                        psf_map_files = [
                            ngmixer.files.get_piff_map_file_fromfile(
                                piff_run, f)
                            for f in meds_files]
                    else:
                        psf_map_files = [
                            ngmixer.files.get_psfmap_file_fromfile(f)
                            for f in meds_files]
                else:
                    psf_map_files = [
                        ngmixer.files.get_psfmap_file_fromfile(f)
                        for f in meds_files]
            else:
                psf_map_files = None

            # Make megamixer.
            MMixer = ImSimMegaMixer(
                self.config["config_file"], meds_files,
                psf_map_files=psf_map_files,
                clobber=True,
                seed=self.config.get("seed", None))

            if self.set_chunk_size is not None:
                MMixer["jobs"]["chunksize"] = self.set_chunksize

            # If not clobber, check for collated file and don't add
            # megamixer to list if it's there
            if not self.config.get("clobber", True):
                ngmixer_file = MMixer.get_collated_file()
                if os.path.isfile(ngmixer_file):
                    # Try reading it to make sure it contains data, and
                    # if so, add existing file to stash and continue. If
                    # we get an IOError, pass and we'll re-do this file.
                    try:
                        fitsio.read(ngmixer_file)
                        # The collated output file already exists, so
                        # continue after adding to stash
                        self.updated_tilenames.append(tilename)
                        self.logger.error(
                            "Found existing collated file %s, "
                            "skipping tile %s" % (ngmixer_file, tile_id))
                        stash.set_filepaths(
                            self.file_key, ngmixer_file, tilename)
                        continue
                    except IOError:
                        pass

            # Otherwise add this megamixer to the list
            if self.mode == "mof":
                MMixer.setup_nbrs()
            megamixers.append(MMixer)
            tile_nums.append(tile_num)

        # It maybe all tiles completed in a previous run
        if len(megamixers) == 0:
            return 0, stash

        if self.mode == "mof":
            # Now loop through megamixers, calling neighbour setup scripts
            # Let all processes join in here
            self.logger.error("Setting up neighbors")
            for i, mm in enumerate(megamixers):
                returncode, job_output = run_subprocess(
                    ["bash", mm.nbr_script])
                if returncode != 0:
                    self.logger.error(
                        "rank %d recieved return code %d "
                        "for script %s\n stdout: \n %s " % (
                            job_output.returncode,
                            mm.nbr_script,
                            job_output.stdout))

        # Now loop through again doing setup(). rank 0 does this and again
        # bcasts list of megamixers to other processes so they have access to
        # the list of run scripts written by setup()
        # TODO: run in parallel?
        for mm in megamixers:
            mm.setup()
            if self.config.get("clear_checkpoint_files", True):
                for s in mm.scripts:
                    d = os.path.dirname(s)
                    check_files = glob.glob(
                        os.path.join(d, "*checkpoint.fits"))
                    for f in check_files:
                        self.logger.error("remove checkpoint file %s" % f)
                        os.remove(f)

        if self.config.get('use_joblib', True):
            return self._do_exe_joblib(
                megamixers, stash, tile_nums)
        else:
            for mm_ind, mm in enumerate(megamixers):
                tile_num = tile_nums[mm_ind]
                tilename = stash["tilenames"][tile_num]
                for chunk_script in mm.scripts:
                    self.logger.error("calling %s" % chunk_script)
                    returncode, job_output = run_subprocess(
                        ["bash", chunk_script])

                ngmixer_file = mm.collate()
                self.logger.error(
                    "produced collated file %s" % (
                        ngmixer_file))
                stash.set_filepaths(self.file_key, ngmixer_file, tilename)

        # Unset TMPDIR if we set it
        if self.tmpdir_new is not None:
            os.environ["TMPDIR"] = self.tmpdir_orig

        return 0, stash

    def _do_exe_joblib(
            self, megamixers, stash, tile_nums):
        # going to keep this here to not break Niall's env
        import joblib

        # Now its time to run the actual ngmixer scripts.
        # for this part, each task picks up a single tile and runs it via
        # joblib
        def _run_script(chunk_script, output_file, clobber):
            for _ in range(10):
                if clobber:
                    try:
                        os.remove(output_file)
                    except OSError:
                        assert (not os.path.isfile(output_file))

                returncode, _ = run_subprocess(["bash", chunk_script])

                output_file_exists = os.path.isfile(output_file)
                if returncode == 0 and output_file_exists:
                    break

        for mm_ind, mm in enumerate(megamixers):
            tile_num = tile_nums[mm_ind]
            tilename = stash["tilenames"][tile_num]
            jobs = [
                joblib.delayed(_run_script)(chunk_script, output_file, self.config.get("clobber_chunk", True))
                for chunk_script, output_file in zip(
                    mm.scripts, mm.output_files)]

            print('# of jobs', len(jobs))

            joblib.Parallel(
                n_jobs=self.config.get("n_jobs", -1), backend='loky',
                verbose=100, pre_dispatch='2*n_jobs',
                max_nbytes=None)(jobs)

            ngmixer_file = mm.collate()
            self.logger.error(
                "Produced collated file %s" % (ngmixer_file))
            stash.set_filepaths(self.file_key, ngmixer_file, tilename)

            #If mm.collate didn't fail, it should now be safe to
            #delete all the chunk files, if config["delete_chunks"]
            #is true (to save some space)
            if self.config["delete_chunks"]:
                #delete the output files
                for f in mm.output_files:
                    os.remove(f)

        # Unset TMPDIR if we set it
        if self.tmpdir_new is not None:
            os.environ["TMPDIR"] = self.tmpdir_orig

        return 0, stash


class MOFRunner(NGMixerRunner):
    """
    Convenience class for calling mof. Just sets the mode in the config to mof.
    """
    def __init__(self, config, base_dir, name="mof", logger=None,
                 verbosity=0, log_file=None):
        if "mode" in config:
            try:
                assert config["mode"] == "mof"
            except AssertionError:
                print("the mode option in the config for class "
                      "MOFRunner must be 'mof'")
        else:
            config["mode"] = "mof"
        super(MOFRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)


class McalRunner(NGMixerRunner):
    """
    Convenience class for calling metacal.
    Just sets the mode in the config to 'mcal'.
    """
    def __init__(self, config, base_dir, name="mcal", logger=None, verbosity=0,
                 log_file=None):
        if "mode" in config:
            try:
                assert config["mode"] == "mcal"
            except AssertionError:
                print("the mode option in the config for class "
                      "McalRunner must be 'mcal'")
        else:
            config["mode"] = "mcal"
        super(McalRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)


class SOFRunner(NGMixerRunner):
    """
    Convenience class for calling sof.
    Just sets the mode in the config to 'sof'.
    """
    def __init__(self, config, base_dir, name="sof", logger=None, verbosity=0,
                 log_file=None):
        if "mode" in config:
            try:
                assert config["mode"] == "sof"
            except AssertionError:
                print("the mode option in the config for class "
                      "SOFRunner must be 'sof'")
        else:
            config["mode"] = "sof"
        super(SOFRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)
