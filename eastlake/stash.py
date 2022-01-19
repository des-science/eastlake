from __future__ import print_function
import copy
import pickle
import os
from .des_files import replace_imsim_data_in_pizza_cutter_yaml


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
    stash: dictionary or subclass, optional
        stash from previous run when resuming
    """
    def __init__(self, base_dir, step_names, stash=None):
        self['base_dir'] = os.path.abspath(base_dir)
        self['step_names'] = step_names
        self['completed_step_names'] = []
        self['env'] = []
        self['orig_base_dirs'] = []

        if stash is not None:
            # this means we're resuming a pipeline
            # and so can update with the info output
            # from the original run.
            # However, it may be we are using a different
            # base_dir, because e.g. we've moved the output.
            # So pop out the base_dir key before updating
            orig_base_dirs = stash.pop("orig_base_dirs", [])
            orig_base_dir = stash.pop("base_dir")
            if os.path.normpath(orig_base_dir) != base_dir:
                orig_base_dirs.append(orig_base_dir)
            stash["orig_base_dirs"] = orig_base_dirs
            # pop removed the base_dir and orig_base_dir keys, so we can go ahead and
            # update everything else
            self.update(stash)

        # replace imsim_data with current value if it exists
        if "IMSIM_DATA" in os.environ:
            self["imsim_data"] = os.environ["IMSIM_DATA"]

    def set_filepaths(self, file_key, filepaths, tilename, band=None, ext=None):
        # For simulation output files, we want to save a relative path w.r.t
        # the base_dir
        # filepaths can be a list or a single-path.
        if ext is not None:
            ext_key = file_key.rsplit("_", 1)[0] + "_ext"

        islist = True
        if not isinstance(filepaths, list):
            filepaths = [filepaths]
            islist = False
        filepaths_out = []
        for filepath in filepaths:
            if os.path.isabs(filepath):
                filepaths_out.append(os.path.relpath(filepath, start=self['base_dir']))
            else:
                filepaths_out.append(filepath)
        if not islist:
            filepaths_out = filepaths_out[0]

        if "tile_info" not in self:
            self["tile_info"] = {}
        if tilename not in self["tile_info"]:
            self["tile_info"][tilename] = {}
        if band is not None and band not in self["tile_info"][tilename]:
            self["tile_info"][tilename][band] = {}

        if band is None:
            assert file_key not in ["g", "r", "i", "z", "y", "Y"]
            self['tile_info'][tilename][file_key] = filepaths_out
            if ext is not None:
                self['tile_info'][tilename][ext_key] = ext
        else:
            self['tile_info'][tilename][band][file_key] = filepaths_out
            if ext is not None:
                self['tile_info'][tilename][band][ext_key] = ext

    def get_filepaths(
        self, file_key, tilename, band=None, ret_abs=True,
        keyerror=True, with_fits_ext=False,
    ):
        if with_fits_ext:
            ext_key = file_key.rsplit("_", 1)[0] + "_ext"

        # if keyerror=False, don't raise an error if file_key not found
        # and just return None
        if band is not None:
            try:
                filepaths_in_stash = self["tile_info"][tilename][band][file_key]
                if with_fits_ext:
                    file_ext = self.get_tile_info_quantity(
                        ext_key, tilename, band=band, keyerror=keyerror,
                    )
            except KeyError as e:
                if not keyerror:
                    return None
                else:
                    raise(e)
        else:
            try:
                filepaths_in_stash = self["tile_info"][tilename][file_key]
                if with_fits_ext:
                    file_ext = self.get_tile_info_quantity(
                        ext_key, tilename, band=band, keyerror=keyerror,
                    )
            except KeyError as e:
                if not keyerror:
                    return None
                else:
                    raise(e)
        islist = True
        if not isinstance(filepaths_in_stash, list):
            islist = False
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
            filepaths_out = filepaths_out[0]

        if with_fits_ext:
            return filepaths_out, file_ext
        else:
            return filepaths_out

    def has_tile_info_quantity(self, key, tilename, band=None):
        if "tile_info" not in self:
            self["tile_info"] = {}
        if tilename not in self["tile_info"]:
            self["tile_info"][tilename] = {}
        if band is not None and band not in self["tile_info"][tilename]:
            self["tile_info"][tilename][band] = {}

        if band is None:
            return key in self['tile_info'][tilename]
        else:
            return key in self['tile_info'][tilename][band]

    def set_tile_info_quantity(self, key, value, tilename, band=None):
        if "tile_info" not in self:
            self["tile_info"] = {}
        if tilename not in self["tile_info"]:
            self["tile_info"][tilename] = {}
        if band is not None and band not in self["tile_info"][tilename]:
            self["tile_info"][tilename][band] = {}

        if band is None:
            assert key not in ["g", "r", "i", "z", "y", "Y"]
            self['tile_info'][tilename][key] = value
        else:
            self['tile_info'][tilename][band][key] = value

    def get_tile_info_quantity(
        self, key, tilename, band=None, keyerror=True,
    ):
        # if keyerror=False, don't raise an error if file_key not found
        # and just return None
        if band is not None:
            try:
                return self["tile_info"][tilename][band][key]
            except KeyError as e:
                if not keyerror:
                    return None
                else:
                    raise(e)
        else:
            try:
                return self["tile_info"][tilename][key]
            except KeyError as e:
                if not keyerror:
                    return None
                else:
                    raise(e)

    def set_output_pizza_cutter_yaml(self, data, tilename, band):
        if "_output_pizza_cutter_yaml" not in self:
            self["_output_pizza_cutter_yaml"] = {}
        if tilename not in self["_output_pizza_cutter_yaml"]:
            self["_output_pizza_cutter_yaml"][tilename] = {}
        self["_output_pizza_cutter_yaml"][tilename][band] = data

        # now we use the output to set the relevant quantities in the tile info
        # these are used by downstream steps

        #######################
        # SE images
        def _set_paths(dest_key, src_key, src_ext_key):
            exts = list(set([src[src_ext_key] for src in data["src_info"]]))
            assert len(exts) == 1
            self.set_filepaths(
                dest_key,
                [src[src_key] for src in data["src_info"]],
                tilename,
                band=band,
                ext=exts[0]
            )

        _set_paths("img_files", "image_path", "image_ext")
        _set_paths("wgt_files", "weight_path", "weight_ext")
        _set_paths("msk_files", "bmask_path", "bmask_ext")
        _set_paths("bkg_files", "bkg_path", "bkg_ext")
        self.set_tile_info_quantity(
            "mag_zps",
            [src["magzp"] for src in data["src_info"]],
            tilename,
            band=band,
        )
        self.set_filepaths(
            "piff_files",
            [src["piff_path"] for src in data["src_info"]],
            tilename,
            band=band,
        )
        self.set_filepaths(
            "psfex_files",
            [src["psfex_path"] for src in data["src_info"]],
            tilename,
            band=band,
        )

        #######################
        # coadd images
        self.set_filepaths(
            "coadd_file", data["image_path"], tilename, band=band, ext=data["image_ext"]
        )
        self.set_filepaths(
            "coadd_mask_file", data["bmask_path"], tilename, band=band, ext=data["bmask_ext"],
        )
        if "weight_path" in data and "weight_ext" in data:
            # sometimes this is not there
            # appears to be bug?
            self.set_filepaths(
                "coadd_weight_file", data["weight_path"], tilename, band=band, ext=data["weight_ext"],
            )

        #######################
        # src extractor
        self.set_filepaths("srcex_cat", data["cat_path"], tilename, band=band)
        self.set_filepaths(
            "seg_file", data["seg_path"], tilename, band=band, ext=data["seg_ext"],
        )

    def has_output_pizza_cutter_yaml(self, tilename, band):
        if (
            tilename in self["_output_pizza_cutter_yaml"]
            and band in self["_output_pizza_cutter_yaml"][tilename]
        ):
            return True
        else:
            return False

    def get_output_pizza_cutter_yaml(self, tilename, band):
        if (
            tilename in self["_output_pizza_cutter_yaml"]
            and band in self["_output_pizza_cutter_yaml"][tilename]
        ):
            pyml = copy.deepcopy(self["_output_pizza_cutter_yaml"][tilename][band])
            replace_imsim_data_in_pizza_cutter_yaml(
                pyml, self["base_dir"]
            )
            return pyml
        else:
            raise RuntimeError(
                f"Could not find output pizza cutter yaml entry for tile|band {tilename}|{band}"
            )

    def update_output_pizza_cutter_yaml(self, tilename, band):
        self._output_pyml_info = (
            self.get_output_pizza_cutter_yaml(tilename, band),
            tilename,
            band,
        )
        return self

    def __enter__(self):
        return self._output_pyml_info[0]

    def __exit__(self, exception_type, exception_value, traceback):
        self.set_output_pizza_cutter_yaml(*self._output_pyml_info)
        self._output_pyml_info = None

    def set_input_pizza_cutter_yaml(self, data, tilename, band):
        if "_input_pizza_cutter_yaml" not in self:
            self["_input_pizza_cutter_yaml"] = {}
        if tilename not in self["_input_pizza_cutter_yaml"]:
            self["_input_pizza_cutter_yaml"][tilename] = {}
        self["_input_pizza_cutter_yaml"][tilename][band] = data

        if not self.has_output_pizza_cutter_yaml(tilename, band):
            pyml = copy.deepcopy(data)
            replace_imsim_data_in_pizza_cutter_yaml(
                pyml, self["base_dir"]
            )
            self.set_output_pizza_cutter_yaml(pyml, tilename, band)

    def get_input_pizza_cutter_yaml(self, tilename, band, imsim_data=None):
        if imsim_data is None:
            imsim_data = self["imsim_data"]

        if (
            tilename in self["_input_pizza_cutter_yaml"]
            and band in self["_input_pizza_cutter_yaml"][tilename]
        ):
            pyml = copy.deepcopy(self["_input_pizza_cutter_yaml"][tilename][band])
            replace_imsim_data_in_pizza_cutter_yaml(
                pyml, imsim_data
            )
            return pyml
        else:
            raise RuntimeError(
                f"Could not find input pizza cutter yaml entry for tile|band={tilename}|{band}"
            )

    def save(self, filename, overwrite=True):
        if not overwrite:
            if os.path.isfile(filename):
                raise IOError("overwrite is False and output file exists")
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=2)

    @classmethod
    def load(cls, filename, base_dir, step_names):
        with open(filename, "rb") as f:
            s = pickle.load(f)
        return cls(base_dir, step_names, s)
