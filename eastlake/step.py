from __future__ import print_function
import os
import subprocess
import logging
from subprocess import Popen, PIPE, CalledProcessError

from .utils import get_logger, safe_mkdir

from timeit import default_timer as timer
from datetime import timedelta


def run_and_check(command, command_name, logger=None, verbose=None):
    if logger is not None:
        logger.info("running cmd: %s" % (" ".join(command),))
    else:
        print("running cmd: %s" % (" ".join(command),))

    if logger is not None and logger.isEnabledFor(logging.DEBUG):
        rc_verbose = True
    else:
        rc_verbose = False

    if verbose is None and logger is not None:
        verbose = rc_verbose
    elif verbose is None:
        verbose = False

    try:
        with Popen(
            command,
            stdout=PIPE,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            stderr=subprocess.STDOUT,
        ) as p:
            output = ""
            for line in p.stdout:
                if verbose:
                    print(line, end='', flush=True)
                output += line

            output = output.encode("utf-8")

        if p.returncode != 0:
            e = CalledProcessError(p.returncode, p.args)
            e.output = output
            raise e

        return output
    except Exception as e:

        print("Failed calling %s using command %s, output follows:" % (
            command_name, command))
        if hasattr(e, "output"):
            print(e.output.decode("utf-8"))
        else:
            print(repr(e))

        if logger is not None:
            logger.error("Failed calling %s using command %s, output follows:" % (
                command_name, command))
            if hasattr(e, "output"):
                logger.error(e.output.decode("utf-8"))
            else:
                logger.error(repr(e))

        raise e


def run_subprocess(command):
    """Run subprocess and return exit code and stdout. This functions allows us to
    stubbornly use subprocess.run when it is available (i.e. when using python 3)"""
    try:
        c = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return c.returncode, c.stdout
    except AttributeError:  # raised if using python 2
        c = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = c.communicate()[0]
        return c.returncode, stdout


def get_image_and_ext(s, default_ext, image_name, logger):
    if isinstance(s, tuple):
        img_file, img_ext = s
    else:
        logger.error("No extension number provided for %s file - setting to %d" % (
            image_name, default_ext))
        img_file = s
        img_ext = default_ext
    return img_file, img_ext


class Step(object):
    """
    Parent class for pipeline step
    Each step has a name, a config dict, and a logger
    It also has an execute function which is called by the Pipeline instance.
    """

    def __init__(self, config, base_dir, name=None, logger=None,
                 verbosity=None, log_file=None):
        self.name = name
        self.config = config
        if self.config is None:
            self.config = {}
        self.logger = logger
        if self.logger is None:
            if verbosity is None:
                verbosity = 1
            self.logger = get_logger(self.name, verbosity, log_file=log_file)

        # Make base_dir if it doesn't exist.
        # If base_dir is None, set to current directory
        if base_dir is not None:
            safe_mkdir(base_dir)
        else:
            base_dir = "."
        self.base_dir = os.path.abspath(base_dir)

    def execute_step(self, stash, new_params=None, do_timing=True):
        """Call the execute function and do timing"""
        step_start_time = timer()
        status, stash = self.execute(stash, new_params=new_params)
        step_end_time = timer()
        time_for_step = step_end_time-step_start_time
        self.logger.error("\nCompleted step %s in %s updated tile: %s" % (
            self.name, str(timedelta(seconds=time_for_step)),
            stash.get("tilenames", None)))
        return status, stash

    def execute(self, stash, new_params=None, comm=None):
        """
        Execute whatever this pipeline step does.
        stash is a dictionary which contains output from the previous step.
        This step also writes output to stash for the next step.
        """
        pass

    def set_base_dir(self, base_dir):
        safe_mkdir(base_dir)
        self.base_dir = os.path.abspath(base_dir)

    def clear_stash(self, stash):
        return
