# Some misc utility functions for pipeline code
from __future__ import print_function
import sys
import logging
import errno
import os
import numpy as np

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
    print("log_file=", log_file)
    print("filemode=", filemode)
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


def hstack2(array1, array2):
    return array1.__array_wrap__(np.hstack([array1, array2]))


def vstack2(array1, array2):
    if array1.dtype.fields is None:
        raise ValueError("`array1' must be a structured numpy array")
    if array2.dtype.fields is None:
        raise ValueError("`array2' must be a structured numpy array")
    if array1.shape != array2.shape:
        raise ValueError("input arrays must have same shape")
    out = np.empty(array1.shape, dtype=array1.dtype.descr + array2.dtype.descr)
    for arr in [array1, array2]:
        for name in arr.dtype.names:
            out[name] = arr[name]
    return out


def add_field(a, descr, arrays):
    """Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise(ValueError, "`A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    for d, c in zip(descr, arrays):
        b[d[0]] = c
    return b


def apply_cut_dict(input_data, cut_dict, inclusive=True, verbose=False, func_dict=None):
    """
    - input_data is a numpy recarray
    - cut_dict is a dictionary where the key is the name of the column in input_data
    you want to cut on, and the value is
    either
    i) a tuple, in which case data between value[0] and value[1] are retained
    ii) a scalar, in which case data equal to value is retained - usually this would be
    an integer e.g. a flag or something.
    """
    # Start with a mask of all True, then iterate through cut_dict, applying cuts.
    use = np.ones(len(input_data), dtype=bool)
    for key, val in cut_dict.items():
        column_data = input_data[key]
        if len(val) == 1:
            mask = column_data == val[0]
            cut_string = "{0} == {1}".format(key, val[0])
        elif len(val) == 2:
            if inclusive:
                mask = (column_data >= val[0])*(column_data <= val[1])
                cut_string = "{0} <= {1} <= {2}".format(val[0], key, val[1])
            else:
                mask = (column_data > val[0])*(column_data < val[1])
                cut_string = "{0} < {1} < {2}".format(val[0], key, val[1])
        else:
            raise ValueError("cut value should be of length 1 or 2")
        use[~mask] = False
        if verbose:
            print('applying {0} leaves fraction {1}'.format(cut_string, float(mask.sum())/len(mask)))
    if verbose:
        print('all cuts leave fraction {0} ({1} out of {2} original objects)'.format(
            float(use.sum())/len(use), use.sum(), len(use)))
    return input_data[use], use
