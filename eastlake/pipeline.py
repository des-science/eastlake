# Code for running deblend simulation pipelines
# There is a parent class 'Step' for pipeline steps,
# then a 'Pipeline' class for running these steps
from __future__ import print_function
import galsim
import galsim.config.process
import os
import yaml
import pickle
# for metacal
from collections import OrderedDict
from .steps import (
    GalSimRunner,
    SWarpRunner,
    SrcExtractorRunner,
    MEDSRunner,
    SingleBandSwarpRunner,
    TrueDetectionRunner,
    DeleteImages,
    DeleteMeds,
    NewishMetcalRunner,
)
from .utils import get_logger, safe_mkdir
from .stash import Stash

PANIC_STRING = "!!!!!!!!!!!!!!\nWTF*WTF*WTF*WTF*\n"
THINGS_GOING_FINE = "****************\n=>=>=>=>=>=>=>=>\n"


STEP_CLASSES = OrderedDict([
    ('galsim', GalSimRunner),
    ('swarp', SWarpRunner),
    ('src_extractor', SrcExtractorRunner),
    ('meds', MEDSRunner),
    ('single_band_swarp', SingleBandSwarpRunner),
    ('true_detection', TrueDetectionRunner),
    ('delete_images', DeleteImages),
    ('delete_meds', DeleteMeds),
    ('newish_metacal', NewishMetcalRunner),
])

STEP_IS_GALSIM = set(["galsim"])

DEFAULT_STEPS = ["galsim", "swarp", "src_extractor", "meds"]


def register_pipeline_step(step_name, step_class, is_galsim=False):
    """Register a pipeline step w/ eastlake

    Parameters
    ----------
    step_name : str
        The name of the step.
    step_class : class
        The step class.
    is_galsim : bool, optional
        Set to true if the step is running a galsim config.
    """
    global STEP_CLASSES
    global STEP_IS_GALSIM

    if step_name in STEP_CLASSES:
        raise ValueError("A step with the name '%s' already exists!" % step_name)
    STEP_CLASSES[step_name] = step_class

    if is_galsim:
        STEP_IS_GALSIM.add(step_name)


class Pipeline(object):
    def __init__(self, steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=None,
                 record_file="job_record.pkl"):
        """Class for running an image simulation pipeline.
        @param steps    List of Step instances (see step.py)
        @base_dir       Path base output directory for this pipeline
        @logger         Logger instance
        @verbosity      Verbosity to use for new logger if no logger provided
        @log_file       File to use for new logger if no logger provided (stdout if None)
        @name           Name for pipeline
        @config         Dictionary that will be saved to yaml file in base_dir, intended to be config used in
        from_config_file.
        """
        self.name = name
        # First set up logger if not provided
        self.logger = logger
        if self.logger is None:
            self.logger = get_logger(self.name, verbosity, log_file=log_file)

        # Make base_dir if it doesn't exist.
        # If base_dir is None, set to current directory
        if base_dir is not None:
            safe_mkdir(base_dir)
        else:
            base_dir = "."

        # Set base_dir as absolute path
        self.base_dir = os.path.abspath(base_dir)
        self.logger.error("Setting up pipeline with base_dir = %s" % self.base_dir)
        self.steps = steps
        self.step_names = [s.name for s in self.steps]
        self.logger.error("Pipeline set up to run the following steps: %s" %
                          ' '.join([s.name for s in self.steps]))

        if config is not None:
            galsim.config.process.ImportModules(config)
            with open(os.path.join(self.base_dir, "config.yaml"), 'w') as f:
                yaml.dump(config, f)

        self.record_file = record_file
        if record_file is None:
            record_file = "job_record.pkl"
        if not os.path.isabs(record_file):
            record_file = os.path.join(self.base_dir, record_file)
        self.record_file = record_file

        # stash is a dictionary that gets passed to and updated by pipeline steps.
        # It is also what is saved as the job_record, allowing restarting the pipeline
        self.init_stash()

    def init_stash(self):
        self.stash = Stash(self.base_dir, self.step_names)

    @classmethod
    def from_record_file(cls, config_file, job_record_file, base_dir=None, logger=None, verbosity=1,
                         log_file=None, name="pipeline_cont", step_names=None,
                         new_params=None, record_file=None):
        """
        Initialize pipeline from record file from previous simulation run
        """

        if logger is None:
            logger = get_logger(name, verbosity, log_file=log_file)

        if base_dir is None:
            base_dir = os.path.dirname(job_record_file)
        # if record_file=None, assume we're using the same file we're resuming from
        if record_file is None:
            record_file = job_record_file

        pipe = cls.from_config_file(config_file, base_dir, logger=logger, verbosity=verbosity,
                                    log_file=log_file, name=name, step_names=step_names,
                                    new_params=new_params, record_file=record_file)

        # Load stash from previous job
        stash = Stash.load(job_record_file, base_dir, pipe.step_names)

        # and set this as the pipeline's stash
        pipe.stash = stash

        # Reset environment variables that were set in previous pipeline
        for key, val in stash["env"]:
            os.environ[key] = val
        return pipe

    @classmethod
    def from_config_file(cls, config_file, base_dir, logger=None, verbosity=1,
                         log_file=None, name="pipeline", step_names=None,
                         new_params=None, record_file=None):
        """
        Initialize a pipeline from a config file.
        """
        # For now we assume the first step is always GalSim.
        # So we read the yaml config_file into a dict, and pop
        # out the other steps.
        steps = []
        # Use galsim config reader to read config file
        config = galsim.config.ReadConfig(config_file)

        if logger is None:
            logger = get_logger(name, verbosity, log_file=log_file)

        if len(config) != 1:
            print("Multiple documents in config file not supported, sorry.")
            raise RuntimeError("Multiple documents in config file not supported, sorry.")

        config = config[0]

        galsim.config.process.ImportModules(config)

        # Process templates.
        # allow for template config in the same directory as the config file
        # to be specified with just the basename of the path. This helps
        # if the config file is being run from some other directory. It could
        # produce ambiguous behaviour though if a file with the same name as the
        # template is present in the current working directory. In this case
        # throw a warning, but use the version in the same directory as the main
        # config file.
        if "template" in config:
            cwd = os.getcwd()
            config_dirname = os.path.dirname(config_file)
            if not os.path.isabs(config["template"]):
                if (cwd != os.path.dirname) and (os.path.isfile(os.path.join(config_dirname, config["template"]))):
                    # we're running from a different directory to where the main config file is,
                    # and a config file with the name
                    # config["template"] exists in the same directory as the main config file.
                    # use this template file, regardless of whether there is a file of the same
                    # name in cwd
                    template_file_to_use = os.path.join(
                        config_dirname, config["template"])
                    if os.path.isfile(config["template"]):
                        logger.error("Using template config %s, not %s. Make sure this is what you want" % (
                            template_file_to_use, os.path.join(cwd, config["template"])))
                    config["template"] = template_file_to_use

        galsim.config.ProcessAllTemplates(config)

        # update with new_params
        if new_params is not None:
            galsim.config.UpdateConfig(config, new_params)

        if "pipeline" not in config:
            config["pipeline"] = {"ntiles": 1}
        if step_names is None:
            # (config["steps"].strip()).split()
            step_names = config["pipeline"]["steps"]

        steps = []
        # make sure base_dir is an absolute path
        if base_dir is not None:
            base_dir = os.path.abspath(base_dir)

        for step_name in step_names:
            if step_name == "galsim":
                steps.append(GalSimRunner(config, base_dir, logger=logger))
            else:
                try:
                    step_config = config.pop(step_name)
                except KeyError:
                    logger.error(
                        "no entry for step %s found in config file, continuing with empty step_config" % step_name)
                    step_config = {}
                # We can explicitly set a key into STEP_CLASSES as the step_class option in the config file,
                # otherwise the step's name will be used as the key
                if "step_class" in step_config:
                    try:
                        step_class = STEP_CLASSES[step_config["step_class"]]
                    except KeyError as e:
                        print("step_class must be in %s" % (str(STEP_CLASSES.keys())))
                        raise(e)
                else:
                    step_class = STEP_CLASSES[step_name]
                if "verbosity" in step_config:
                    step_verbosity = step_config["verbosity"]
                else:
                    step_verbosity = 1

                if step_name in STEP_IS_GALSIM:
                    steps.append(step_class(config, base_dir, logger=logger,
                                 verbosity=step_verbosity, name=step_name))
                else:
                    steps.append(step_class(step_config, base_dir, logger=logger,
                                 verbosity=step_verbosity, name=step_name))

        return cls(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=config,
                   record_file=record_file)

    def execute(self, new_params_list=None, base_dir=None, no_overwrite_job_record=False,
                skip_completed_steps=False):
        self.logger.error("Executing pipeline")

        # Update base_dir if provided
        if base_dir is not None:
            self.base_dir = base_dir
        if new_params_list is None:
            new_params_list = [None] * len(self.steps)
        elif len(new_params_list) < len(self.steps):
            while len(new_params_list) < len(self.steps):
                new_params_list.append(None)

        self.logger.error("Running steps: %s" % ", ".join([s.name for s in self.steps]))

        # Loop through steps calling execute function. Pass self.stash and any new_params
        # Move to base_dir, first get cwd which we'll return to later
        cwd = os.getcwd()
        try:
            os.chdir(self.base_dir)

            for step, new_params in zip(self.steps, new_params_list):
                if skip_completed_steps:
                    if (step.name, 0) in self.stash["completed_step_names"]:
                        self.logger.error("""Skipping step %s since already completed with status 0,
                        and you have skip_completed_steps=True""" % step.name)
                        continue
                self.logger.error(THINGS_GOING_FINE+"Running step %s\n" %
                                  step.name + THINGS_GOING_FINE)
                status, self.stash = step.execute_step(self.stash, new_params=new_params)
                if status != 0:
                    self.logger.error(
                        PANIC_STRING
                        + "step %s return status %d: quitting pipeline here\n" % (step.name, status)
                        + PANIC_STRING)
                    return status
                else:
                    self.logger.error(THINGS_GOING_FINE + "Completed step %s\n" %
                                      step.name + THINGS_GOING_FINE)
                # record that we've completed this step
                self.stash["completed_step_names"].append((step.name, status))
                # save stash
                self._save_restart(no_overwrite_job_record)
        finally:
            # Return to the previous cwd
            os.chdir(cwd)
        return 0

    def _save_restart(self, no_overwrite_job_record):
        if no_overwrite_job_record is False:
            # Save stash
            self.logger.error("Saving job record with completed steps %s" %
                              (str(self.stash["completed_step_names"])))
            self.save_job_record(self.stash, self.record_file)

    def set_step_config(self, step_name, key, value):
        """
        Set a config entry for the step named step_name.
        For nesting, key can be provided as e.g.
        key1.key2.key3 which will refer to config[key1][key2][key3]
        """
        key_split = key.split(".")
        config = self.steps[self.step_names.index(step_name)].config
        n_nest = 0
        while n_nest < len(key_split)+1:
            d = config
            k = key_split.pop(0)
            config = d[k]
            n_nest += 1
        self.logger.error("Setting key %s in step %s to: %s" %
                          (key, step_name, str(value)))
        d[k] = value

    def save_job_record(self, job_record, filename):
        with open(filename, "wb") as f:
            pickle.dump(job_record, f, protocol=2)
