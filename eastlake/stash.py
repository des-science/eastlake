from __future__ import print_function
import pickle
import os

class Stash(dict):
    """
    This class is used to store information about a pipeline, and pass information
    output from pipeline steps to other pipeline steps. It's also saved as a pickle
    file between steps and at the end of the pipeline, and used to restart pipelines

    Parameters
    ----------
    base_dir : string
       The base directory for the pipeline
    step_names: list of strings
        List of step names for the pipeline
    stash: dictionary or subclass
        stash from previous run when resuming

    Methods
    ------
    get_filepath(file_key, tilename, band=None)
        Get path from file_key for a given tilename 
        and if appropriate a band
    set_filepath(file_key, tilename, band=None, ret_abs=True)
        Set path from file_key for a given tilename 
        and if appropriate a band
    
    """
    def __init__(self, base_dir, step_names, stash=None):
        self['base_dir'] = os.path.abspath(base_dir)
        self['step_names'] = step_names
        self['completed_step_names'] = []
        self['env'] = []
        self['orig_base_dirs'] = []

        if stash is not None:
            #this means we're resuming a pipeline
            #and so can update with the info output
            #from the original run. 
            #However, it may be we are using a different
            #base_dir, because e.g. we've moved the output.
            #So pop out the base_dir key before updating
            orig_base_dirs = stash.pop("orig_base_dirs", [])
            orig_base_dir = stash.pop("base_dir")
            if os.path.normpath( orig_base_dir ) != base_dir:
                orig_base_dirs.append( orig_base_dir )
            stash["orig_base_dirs"] = orig_base_dirs
            #pop removed the base_dir and orig_base_dir keys, so we can go ahead and
            #update everything else
            self.update(stash)

    def get_abs_path(self, file_key, tilename, band=None):
        #Paths in stash should be relative paths, with respect to self['base_dir']
        if band is not None:
            relpath = self["tile_info"][tilename][band][file_key]
        else:
            relpath = self["tile_info"][tilename][file_key]
        assert (not os.path.isabs(relpath))
        return os.path.join(self['base_dir'], relpath)

    def set_filepaths(self, file_key, filepaths, tilename, band=None):
        #For simulation output files, we want to save a relative path w.r.t the base_dir
        #filepaths can be a list or a single-path.
        #print("file_key:", file_key)
        #print("filepaths:", filepaths)
        #print("tilename:", tilename)
        #print("band:", band)
        islist=True
        if not isinstance(filepaths, list):
            filepaths = [filepaths]
            islist=False
        filepaths_out = []
        for filepath in filepaths:
            if os.path.isabs(filepath):
                filepaths_out.append( os.path.relpath(filepath, start=self['base_dir']) )
            else:
                filepaths_out.append( filepath )
        if not islist:
            filepaths_out = filepaths_out[0]
                    
        if band is None:
            self['tile_info'][tilename][file_key] = filepaths_out
        else:
            self['tile_info'][tilename][band][file_key] = filepaths_out


    def get_filepaths(self, file_key, tilename, band=None, ret_abs=True,
                      keyerror=True):
        #if keyerror=False, don't raise an error if file_key not found
        #and just return None
        if band is not None:
            try:
                filepaths_in_stash = self["tile_info"][tilename][band][file_key]
            except KeyError as e:
                if not keyerror:
                    return None
                else:
                    raise(e)
        else:
            try:
                filepaths_in_stash = self["tile_info"][tilename][file_key]
            except KeyError as e:
                if not keyerror:
                    return None
                else:
                    raise(e)
        islist=True
        if not isinstance(filepaths_in_stash, list):
            islist=False
            filepaths_in_stash = [filepaths_in_stash]
        filepaths_out = []
        for f in filepaths_in_stash:
            if os.path.isabs(f):
                filepaths_out.append(f)
            elif ret_abs:
                filepaths_out.append(os.path.join(self['base_dir'], f))
            else:
                filepaths_out.append(f)
        if not islist:
            return filepaths_out[0]
        else:
            return filepaths_out

    def save(self, filename, overwrite=True):
        if not overwrite:
            if os.isfile(filename):
                raise IOError("overwrite is False and output file exists")
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=2)

    @classmethod
    def load(cls, filename, base_dir, step_names):
        with open(filename, "rb") as f:
            s = pickle.load(f)
        return cls(base_dir, step_names, s)
