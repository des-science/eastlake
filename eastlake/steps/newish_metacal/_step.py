from __future__ import print_function, absolute_import
import logging
import os

import numpy as np
import joblib
import esutil as eu
import fitsio
from ngmix import ObsList, MultiBandObsList
from ngmix.gexceptions import GMixRangeError

from .ngmix_compat import NGMixMEDS, MultiBandNGMixMEDS, NGMIX_V1
from .metacal import MetacalFitter
from eastlake.step import Step
from eastlake.utils import safe_mkdir

logger = logging.getLogger(__name__)

# always and forever
MAGZP_REF = 30.0

CONFIG = {
    'metacal': {
        # check for an edge hit
        'bmask_flags': 2**30,

        'metacal_pars': {'psf': 'fitgauss'},

        'model': 'gauss',

        'max_pars': {
            'ntry': 2,
            'pars': {
                'method': 'lm',
                'lm_pars': {
                    'maxfev': 2000,
                    'xtol': 5.0e-5,
                    'ftol': 5.0e-5,
                }
            }
        },

        'priors': {
            'cen': {
                'type': 'normal2d',
                'sigma': 0.263
            },

            'g': {
                'type': 'ba',
                'sigma': 0.2
            },

            'T': {
                'type': 'two-sided-erf',
                'pars': [-1.0, 0.1, 1.0e+06, 1.0e+05]
            },

            'flux': {
                'type': 'two-sided-erf',
                'pars': [-100.0, 1.0, 1.0e+09, 1.0e+08]
            }
        },

        'psf': {
            'model': 'gauss',
            'ntry': 2,
            'lm_pars': {
                'maxfev': 2000,
                'ftol': 1.0e-5,
                'xtol': 1.0e-5
            }
        }
    },
}

if not NGMIX_V1:
    CONFIG['metacal']['metacal_pars'] = {
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        # 'symmetrize_psf': True
    }
else:
    CONFIG['metacal']['metacal_pars'] = {
        'psf': 'fitgauss',
        'types': ['noshear', '1p', '1m', '2p', '2m'],
        # 'use_noise_image': True,
    }


class NewishMetcalRunner(Step):
    """Run a newer metacal.

    Config Params
    -------------
    bands : list of str, optional
        A list of bands to use. Defaults to `["r", "i", "z"]`.
    """
    def __init__(self, config, base_dir, name="newish-metacal", logger=None,
                 verbosity=0, log_file=None):
        super(NewishMetcalRunner, self).__init__(
            config, base_dir, name=name, logger=logger, verbosity=verbosity,
            log_file=log_file)

        # bands to use
        self.bands = self.config.get("bands", ["r", "i", "z"])

    def clear_stash(self, stash):
        pass

    def execute(self, stash, new_params=None):
        self.clear_stash(stash)

        base_output_dir = os.path.join(self.base_dir, "newish-metacal")
        safe_mkdir(base_output_dir)

        for tilename in stash["tilenames"]:

            # meds files
            built_meds_files = stash.get_filepaths("meds_files", tilename)

            meds_files_to_use = []
            for band in self.bands:
                sstr = "%s_%s" % (tilename, band)
                if not any(sstr in f for f in built_meds_files):
                    raise RuntimeError(
                        "could not find meds file for tilename %s band %s" % (
                            tilename, band))
                else:
                    for f in built_meds_files:
                        if sstr in f:
                            meds_files_to_use.append(f)
                            break
            assert len(meds_files_to_use) == 3, (
                "We did not find the right number of meds files!")

            # record what bands we're running in the stash
            stash["newish_metacal_bands"] = self.bands

            try:
                staged_meds_files = []
                tmpdir = os.environ['TMPDIR']
                for fname in meds_files_to_use:
                    os.system("cp %s %s/." % (fname, tmpdir))
                    staged_meds_files.append(
                        os.path.join(tmpdir, os.path.basename(fname))
                    )
                # FIXME hard coded seed
                output = _run_metacal(staged_meds_files, 42)
            finally:
                for fname in staged_meds_files:
                    os.system("rm -f %s" % fname)

            output_dir = os.path.join(base_output_dir, tilename)
            safe_mkdir(output_dir)

            fname = os.path.join(output_dir, "%s_newish_metacal.fits" % tilename)
            fitsio.write(fname, output, clobber=True)

            stash.set_filepaths("newish_metacal_output", fname, tilename)

        return 0, stash


def _run_metacal(meds_files, seed):
    """Run metacal on a tile.

    Parameters
    ----------
    meds_files : list of str
        A list of the meds files to run metacal on.
    seed : int
        The seed for the global RNG.
    """
    with NGMixMEDS(meds_files[0]) as m:
        cat = m.get_cat()
    logger.info(' meds files %s', meds_files)

    n_cpus = joblib.externals.loky.cpu_count()
    n_chunks = max(n_cpus, 60)
    n_obj_per_chunk = int(cat.size / n_chunks)
    if n_obj_per_chunk * n_chunks < cat.size:
        n_obj_per_chunk += 1
    assert n_obj_per_chunk * n_chunks >= cat.size
    logger.info(
        ' running metacal for %d objects in %d chunks', cat.size, n_chunks)

    seeds = np.random.RandomState(seed=seed).randint(1, 2**30, size=n_chunks)

    jobs = []
    for chunk in range(n_chunks):
        start = chunk * n_obj_per_chunk
        end = min(start + n_obj_per_chunk, cat.size)
        jobs.append(joblib.delayed(_run_mcal_one_chunk)(
            meds_files, start, end, seeds[chunk]))

    with joblib.Parallel(
        n_jobs=n_cpus, backend='multiprocessing',
        verbose=100, max_nbytes=None
    ) as p:
        outputs = p(jobs)

    assert not all([o is None for o in outputs]), (
        "All metacal fits failed!")

    output = eu.numpy_util.combine_arrlist(
        [o for o in outputs if o is not None])
    logger.info(' %d of %d metacal fits worked!', output.size, cat.size)

    return output


def _run_mcal_one_chunk(meds_files, start, end, seed):
    """Run metcal for `meds_files` only for objects from `start` to `end`.

    Note that `start` and `end` follow normal python indexing conventions so
    that the list of indices processed is `list(range(start, end))`.

    Parameters
    ----------
    meds_files : list of str
        A list of paths to the MEDS files.
    start : int
        The starting index of objects in the file on which to run metacal.
    end : int
        One plus the last index to process.
    seed : int
        The seed for the RNG.

    Returns
    -------
    output : np.ndarray
        The metacal outputs.
    """
    rng = np.random.RandomState(seed=seed)

    # seed the global RNG to try to make things reproducible
    np.random.seed(seed=rng.randint(low=1, high=2**30))

    output = None
    mfiles = []
    data = []
    try:
        # get the MEDS interface
        for m in meds_files:
            mfiles.append(NGMixMEDS(m))
        mbmeds = MultiBandNGMixMEDS(mfiles)
        cat = mfiles[0].get_cat()

        for ind in range(start, end):
            o = mbmeds.get_mbobs(ind)
            o = _strip_coadd(o)
            o = _strip_zero_flux(o)
            o = _apply_pixel_scale(o)

            skip_me = False
            for ol in o:
                if len(ol) == 0:
                    logger.debug(' not all bands have images - skipping!')
                    skip_me = True
            if skip_me:
                continue

            o.meta['id'] = ind
            o[0].meta['Tsky'] = 1
            o[0].meta['magzp_ref'] = MAGZP_REF
            o[0][0].meta['orig_col'] = cat['orig_col'][ind, 0]
            o[0][0].meta['orig_row'] = cat['orig_row'][ind, 0]

            nband = len(o)
            mcal = MetacalFitter(CONFIG, nband, rng)

            try:
                mcal.go([o])
                res = mcal.result
            except GMixRangeError as e:
                logger.debug(" metacal error: %s", str(e))
                res = None

            if res is not None:
                data.append(res)

        if len(data) > 0:
            output = eu.numpy_util.combine_arrlist(data)
    finally:
        for m in mfiles:
            m.close()

    return output


def _strip_coadd(mbobs):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        for i in range(1, len(ol)):
            _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs


def _strip_zero_flux(mbobs):
    _mbobs = MultiBandObsList()
    _mbobs.update_meta_data(mbobs.meta)
    for ol in mbobs:
        _ol = ObsList()
        _ol.update_meta_data(ol.meta)
        for i in range(len(ol)):
            if np.sum(ol[i].image) > 0:
                _ol.append(ol[i])
        _mbobs.append(_ol)
    return _mbobs


def _apply_pixel_scale(mbobs):
    for ol in mbobs:
        for o in ol:
            scale = o.jacobian.get_scale()
            scale2 = scale * scale
            scale4 = scale2 * scale2
            o.image = o.image / scale2
            o.weight = o.weight * scale4
    return mbobs
