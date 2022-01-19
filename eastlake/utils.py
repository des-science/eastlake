# Some misc utility functions for pipeline code
from __future__ import print_function
import sys
import logging
import errno
import os
import subprocess
import contextlib

LOGGING_LEVELS = {
    '0': logging.CRITICAL,
    '1': logging.WARNING,
    '2': logging.INFO,
    '3': logging.DEBUG,
    0: logging.CRITICAL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG}


def get_logger(logger_name, verbosity, log_file=None, filemode="w"):
    print(
        "log file|mode: %s|%s" % (
            log_file or "stdout",
            filemode,
        ),
        flush=True,
    )

    # initialize a logger Galsim-style
    logging_level = LOGGING_LEVELS[verbosity]
    if log_file is None:
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            filemode=filemode,
        )
    else:
        logging.basicConfig(
            format="%(message)s",
            filename=log_file,
            filemode=filemode,
        )
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    return logger


def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise(e)


def get_relpath(pth, start=None):
    if start is not None:
        real_rel = os.path.relpath(os.path.realpath(pth), os.path.realpath(start))
        rel = os.path.relpath(pth, start)
    else:
        real_rel = os.path.relpath(os.path.realpath(pth))
        rel = os.path.relpath(pth)
    if len(real_rel) < len(rel):
        return real_rel
    else:
        return rel


def unpack_fits_file_if_needed(pth, ext):
    if ".fits.fz" in pth:
        pth_funpacked = pth.replace(".fits.fz", ".fits")
        # There may already be a funpacked version there
        if not os.path.isfile(pth_funpacked):
            subprocess.check_output(["funpack", pth])
        pth = pth_funpacked
        # If we've funpacked, we'll also need to reduce the
        # extension number by 1
        if isinstance(ext, int):
            ext -= 1
    return pth, ext


# https://stackoverflow.com/questions/6194499/pushd-through-os-system
@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)
