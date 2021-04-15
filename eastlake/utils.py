# Some misc utility functions for pipeline code
from __future__ import print_function
import sys
import logging
import errno
import os

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
        logging.basicConfig(format="%(message)s", level=logging_level,
                            stream=sys.stdout, filemode=filemode)
    else:
        logging.basicConfig(format="%(message)s", level=logging_level,
                            filename=log_file, filemode=filemode)
    return logging.getLogger(logger_name)


def safe_mkdir(d):
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise(e)
