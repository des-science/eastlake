#!/Usr/bin/env python
"""
Process the input meds files in chunks
Possible commands are:
    setup - setup jobs
    setup-nbrs - set up jobs for neighbor finding
    collate - combine job outputs into a single file
    verify - verify that all job outputs are present and OK
    clean - clean all outputs from a run
    archive - run after collate to delete intermediate files and tar logs
    link - make symlinks to all final outputs under {run}/output
    collate-link - collate and link the final outputs
You must have the following environment variables set
    $NGMIXER_OUTPUT_DIR - base output dir
    $TMPDIR - temporary directory for work
TODO: 
    - work on list of tile ids
"""

from __future__ import print_function
try:
    xrange
except:
    xrange=range

import os
import sys
import numpy
import meds
import fitsio
import glob

import ngmixer
from ngmixer import files
from ngmixer.files import read_yaml
from ngmixer.megamixer import NGMegaMixer
from ngmixer.megamixer import SLACNGMegaMixer,SLACArrayNGMegaMixer
from ngmixer.megamixer import CakeNGMegaMixer


from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
parser = ArgumentParser(
    description=__doc__,
    formatter_class=RawTextHelpFormatter,
)

parser.add_argument('ngmix_config',help='config file path')
parser.add_argument('command',help='command to run')
parser.add_argument(
    'meds_files',
    nargs='+',
    help=('the meds files to process, or a yaml file '
          'describing the tile set to use'),
)

parser.add_argument("--walltime",default='48:00',
                  help=("wall time for queue determination (e.g. in lsf)"))
parser.add_argument("--walltime-e",default='4:00',
                  help=("esimated wall time for scheduling (e.g. in lsf)"))
parser.add_argument("--ncores",type=int,default=1,
                  help=("number of cores per job"))

parser.add_argument("--system",default='shell',
                  help=("name of system for jobs"))

parser.add_argument("--queue",default=None,
                  help=("queue to submit to"))

parser.add_argument('--missing',action='store_true',default=False,
                    help="write scripts only for missing output files")

parser.add_argument('--noblind',action='store_true',default=False,
                  help="don't blind the catalog")

parser.add_argument('--clobber',action='store_true',default=False,
                  help="clobber existing catalog, else skip over")

parser.add_argument('--skip-errors',action='store_true',default=False,
                  help="skip over errors")

parser.add_argument('--nocompress',action='store_true',default=False,
                  help="do not compress when archiving logs")

parser.add_argument("--verbosity", default=0, type=int,
                  help=("set verbosity level"))

parser.add_argument("--seed", default=None,type=int,
                  help=("random seed to make seeds for jobs.  "
                        "If not sent, is gotten from the config file"))

parser.add_argument('--desdm',action='store_true',
                  help="the MEDS files, psf maps are DESDM style")

class MegaMixer(dict):
    """
    conf has ngmix config plus
    """
    def __init__(self, ngmix_config, meds_files,
                 psf_map_files=None,
                 walltime='48:00',
                 walltime_e='4:00',
                 ncores=1,
                 missing=False,
                 blind=False,
                 skip_errors=False,
                 clobber=False,
                 seed=None):

        self.ngmix_config = os.path.abspath( os.path.expanduser( os.path.expandvars( ngmix_config ) ) )
        if psf_map_files is not None:
            self.psf_map_files=[os.path.abspath(f) for f in psf_map_files]
        else:
            self.psf_map_files=None
        
        self.missing=missing
        self.walltime=walltime
        self.walltime_e=walltime_e
        self.ncores=ncores

        self.blind=blind
        self.skip_errors=skip_errors
        self.clobber=clobber

        self._load_config()
        self._make_rng(seed=seed)

        self.meds_files=[os.path.abspath(mf) for mf in meds_files]
        meds_list = [meds.MEDS(f) for f in self.meds_files]

        try:
            ngmixer.imageio.medsio.verify_meds(meds_list)
        except AttributeError:
            print("ngmixer.imageio.medsio.verify_meds not found - probably this version is too old")
            pass

        self.meds_string=' '.join(self.meds_files)

        self.info= files.get_meds_info(meds_files[0])

        self._set_files()

    def setup(self):
        """
        make directories, write scripts
        """
        self._set_chunk_ranges()
        self._make_scripts()

    def setup_nbrs(self):
        """
        make directories, write scripts
        """
        self._write_nbrs_script()

    def get_collated_file(self):
        return files.get_collated_file_fromfile(
            self.meds_files[0],
            self.conf['run'],
            blind=self.blind
        )

    def collate(self):
        """
        concatenate all the chunk files, adding in
        some calculated information such as mags
        """

        self._set_chunk_ranges()

        cls = ngmixer.megamixer.concat_io.get_concat_class(
            self.conf['collate']['concat_type'],
        )
        flist = []
        for chunk,rng in enumerate(self.chunk_ranges):
            fname = self._get_chunk_file(chunk,rng,ext='.fits')
            flist.append(fname)

        collated_file=files.get_collated_file_fromfile(
            self.meds_files[0],
            self.conf['run'],
            blind=self.blind,
        )

        concatter = cls(
            self.ngmix_config,
            flist,
            collated_file,
            bands=self.conf['jobs']['bands'],
            blind=self.blind,
            clobber=self.clobber,
            skip_errors=self.skip_errors,
        )

        concatter.concat()
        return collated_file


    def _make_scripts(self):
        """
        write the scripts
        """
        for chunk,rng in enumerate(self.chunk_ranges):
            self._write_script(chunk, rng)

    def _write_script(self, chunk, rng):

        self._make_chunk_dir(chunk, rng)

        fmt=self._get_script_template()
        args = {}

        output_file  = self._get_chunk_file(chunk,rng,ext='.fits')
        logfile      = self._get_chunk_file(chunk,rng,ext='.log')


        args['ngmix_config'] = self.ngmix_config
        if self.psf_map_files is not None:
            psf_maps     = ','.join(self.psf_map_files)
            args['psf_map_arg'] = '--psf-map="%s"' % psf_maps
        else:
            args['psf_map_arg'] = ''

        args['meds_files']   = self.meds_string
        args['output_file']  = output_file
        args['logfile']      = logfile
        args['start']        = rng[0]
        args['stop']         = rng[1]

        args['mof_opt'] = ''
        model_nbrs = self.conf.get('model_nbrs',False)
        if model_nbrs:
            fof_file = self._get_fof_file(ext='.fits')
            args['fof_opt'] = '--fof-file=%s'% fof_file 

            nbrs_file = self._get_nbrs_file(ext='.fits')
            args['nbrs_opt'] = '--nbrs-file=%s'% nbrs_file 

        else:
            args['fof_opt'] = ''
            args['nbrs_opt'] = ''

            if 'correct_meds' in self.conf:
                if self.conf['correct_meds'] is not False:
                    mof_file = self._get_mof_file()
                    args['mof_opt'] = '--mof-file=%s' % mof_file

        if 'forced_photometry' in self.conf:
            models_file = files.get_collated_file_fromfile(
                self.meds_files[0],
                self.conf['forced_photometry']['models_run'],
            )
            args['models_opt'] = '--models="%s"' % models_file

        else:
            args['models_opt'] = ''

        seed = self.rng.randint(low=1,high=1000000000)
        args['seed_opt'] = '--seed=%d' % seed

        scr = fmt.format(**args)

        script_name = self._get_chunk_file(chunk,rng,ext='.sh')

        dowrite=True
        if self.missing:
            if os.path.exists(script_name):
                os.remove(script_name)
            if os.path.exists(output_file):
                dowrite=False

        if dowrite:
            print("writing:",script_name)
            with open(script_name,'w') as fp:
                fp.write(scr)

            os.system('chmod 755 %s' % script_name)

        return dowrite, script_name

    def _write_nbrs_script(self):

        self._make_nbrs_dir()

        fmt=self._get_nbrs_script_template()
        args = {}

        logfile      = self._get_nbrs_file(ext='.log')

        args['ngmix_config'] = self.ngmix_config
        args['meds_file']    = self.meds_files[0]
        args['logfile']      = logfile

        scr = fmt.format(**args)

        script_name = self._get_nbrs_file(ext='.sh')

        print("writing:",script_name)
        with open(script_name,'w') as fp:
            fp.write(scr)

        os.system('chmod 755 %s' % script_name)
        return script_name

    def _get_mof_file(self):
        """
        mof file used for subtracting neighbor light
        """

        mof_file = files.get_collated_file_fromfile(
            self.meds_files[0],
            self.conf['correct_meds']['mof_run'],
        )
        return mof_file

    def _make_chunk_dir(self, chunk, rng):
        f=files.get_chunk_file_fromfile(
            self.meds_files[0],
            self.conf['run'],
            rng,
        )
        files.makedirs_fromfile(f)

    def _get_chunk_file(self, chunk, rng, ext='.fits'):
        fname = files.get_chunk_file_fromfile(
            self.meds_files[0],
            self.conf['run'],
            rng,
            missing=self.missing,
            ext=ext,
        )
        return expand_path(fname)

    def _get_nbrs_file(self, ext='.fits'):
        fname = files.get_nbrs_file_fromfile(
            self.meds_files[0],
            self.conf['run'],
            ext=ext,
        )
        return expand_path(fname)

    def _get_fof_file(self, ext='.fits'):
        fname = files.get_fof_file_fromfile(
            self.meds_files[0],
            self.conf['run'],
            ext=ext,
        )
        return expand_path(fname)


    def _make_nbrs_dir(self):
        f=self._get_nbrs_file()
        files.makedirs_fromfile(f)

    def _get_script_template(self):
        template=r"""#!/bin/bash
if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi
export OMP_NUM_THREADS=1
logfile="{logfile}"
logbase=$(basename $logfile)
tmplog=$tmpdir/$logbase
config="{ngmix_config}"
meds="{meds_files}"
outfile="{output_file}"
start={start}
stop={stop}
ngmixit                                   \
    --fof-range=$start,$stop              \
    --work-dir=$tmpdir                    \
    {psf_map_arg}                         \
    {models_opt}                          \
    {nbrs_opt}                            \
    {fof_opt}                             \
    {mof_opt}                             \
    {seed_opt}                            \
    $config $outfile $meds &> $tmplog
mv -vf $tmplog $logfile
"""

        return template

    def _get_nbrs_script_template(self):
        template=r"""#!/bin/bash
if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    # the condor system creates a scratch directory for us,
    # and cleans up afterward
    tmpdir=$_CONDOR_SCRATCH_DIR
    export TMPDIR=$tmpdir
else
    # otherwise use the TMPDIR
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi
logfile="{logfile}"
logbase=$(basename $logfile)
tmplog=$tmpdir/$logbase
config="{ngmix_config}"
meds="{meds_file}"
ngmixer-meds-make-nbrs-data $config $meds &> $tmplog
mv -vf $tmplog $logfile
"""

        return template



    def _set_files(self):
        files={
            'ngmix_config':self.ngmix_config,
            'meds_files':self.meds_files,
        }

        self.files=files

    def _get_num(self):

        model_nbrs = self.conf.get('model_nbrs',False)
        if model_nbrs:
            fname = self._get_fof_file()
            fofs = fitsio.read(fname)
            num = numpy.unique(fofs['fofid']).size
        else:
            fname=self.meds_files[0]

            if not os.path.exists(fname):
                raise ngmixer.util.MissingDataError("missing meds file: %s" % fname)

            with fitsio.FITS(fname) as fits:
                num = fits['object_data'].get_nrows()

        return num


    def _set_chunk_ranges(self):

        if hasattr(self, 'chunk_ranges'):
            return

        files=self.files

        # could be number of objects or number of
        # fof groups
        nrows = self._get_num()

        chunksize=self.jobs_conf['chunksize']
        nchunks = nrows//chunksize
        if nchunks*chunksize < nrows:
            nchunks += 1

        chunk_ranges = []
        for chunk in xrange(nchunks):
            sr = chunk*chunksize
            sp = sr + chunksize - 1
            if sp >= nrows:
                sp = nrows-1
            chunk_ranges.append([sr,sp])

        self.chunk_ranges = chunk_ranges

    def _load_config(self):
        self.conf = files.read_config(self.ngmix_config)
        self.jobs_conf = self.conf['jobs']

    def _make_rng(self, seed=None):
        if seed is None:
            if 'global_seed' not in self.conf:
                raise RuntimeError("either set 'global_seed' in the"
                                   "config or send --seed=")
            seed=self.conf['global_seed']

        self.rng = numpy.random.RandomState(seed=seed)

def expand_path(path):
    return os.path.abspath(
        os.path.expandvars(
            os.path.expanduser(
                path,
            )
        )
    )

import subprocess
class ImSimMegaMixer(MegaMixer):

    #def setup_mof(self):
    #    self.setup_nbrs()
    #    self.setup()

    def _make_scripts(self):
        self.scripts = []
        self.output_files = []
        for chunk,rng in enumerate(self.chunk_ranges):
            _, script_name = self._write_script(chunk, rng)
            self.scripts.append( script_name )
            output_file  = self._get_chunk_file(chunk,rng,ext='.fits')
            self.output_files.append(output_file)

    def _write_nbrs_script(self):

        self._make_nbrs_dir()

        fmt=self._get_nbrs_script_template()
        args = {}

        logfile      = self._get_nbrs_file(ext='.log')

        args['ngmix_config'] = self.ngmix_config
        args['meds_file']    = self.meds_files[0]
        args['logfile']      = logfile

        scr = fmt.format(**args)

        script_name = self._get_nbrs_file(ext='.sh')

        print("writing:",script_name)
        with open(script_name,'w') as fp:
            fp.write(scr)

        os.system('chmod 755 %s' % script_name)
        self.nbr_script = script_name
