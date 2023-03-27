from __future__ import print_function
import copy
import pickle
import os
import yaml
from .des_files import replace_imsim_data_in_pizza_cutter_yaml, get_pizza_cutter_yaml_path
from .utils import unpack_fits_file_if_needed, pushd, safe_rm


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
        else:
            self["imsim_data"] = None

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
        keyerror=True, with_fits_ext=False, funpack=False,
    ):
        if funpack and not with_fits_ext:
            raise RuntimeError("You must return the FITS extension in order to funpack!")

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
                    raise e
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
                    raise e
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
            if funpack:
                if not islist:
                    return unpack_fits_file_if_needed(filepaths_out, file_ext)
                else:
                    new_filepaths_out = [
                        unpack_fits_file_if_needed(fn, file_ext)
                        for fn in filepaths_out
                    ]
                    return [nn[0] for nn in new_filepaths_out], new_filepaths_out[0][1]
            else:
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
                    raise e
        else:
            try:
                return self["tile_info"][tilename][key]
            except KeyError as e:
                if not keyerror:
                    return None
                else:
                    raise e

    def set_output_pizza_cutter_yaml(self, _data, tilename, band):
        data = copy.deepcopy(_data)
        replace_imsim_data_in_pizza_cutter_yaml(data, self["base_dir"])

        self._make_lists_psfmaps_symlinks(self["base_dir"], tilename, band, data)

        if "_output_pizza_cutter_yaml" not in self:
            self["_output_pizza_cutter_yaml"] = {}
        if tilename not in self["_output_pizza_cutter_yaml"]:
            self["_output_pizza_cutter_yaml"][tilename] = {}
        self["_output_pizza_cutter_yaml"][tilename][band] = data

        # now we use the output to set the relevant quantities in the tile info
        # these are used by downstream steps

        #######################
        # SE images
        def _set_paths(dest_key, src_key, src_ext_key, ext_not_a_key=False):
            if ext_not_a_key:
                exts = [src_ext_key]
            else:
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
        self.set_filepaths(
            "head_files",
            [src["head_path"] for src in data["src_info"]],
            tilename,
            band=band,
        )

        if "coadd_nwgint_path" in data["src_info"][0]:
            _set_paths(
                "coadd_nwgint_img_files",
                "coadd_nwgint_path",
                "sci",
                ext_not_a_key=True,
            )
            _set_paths(
                "coadd_nwgint_wgt_files",
                "coadd_nwgint_path",
                "wgt",
                ext_not_a_key=True,
            )
            _set_paths(
                "coadd_nwgint_msk_files",
                "coadd_nwgint_path",
                "msk",
                ext_not_a_key=True,
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

        #######################
        # push changes to disk
        self.write_output_pizza_cutter_yaml()

    def has_output_pizza_cutter_yaml(self, tilename, band):
        if (
            "_output_pizza_cutter_yaml" in self
            and tilename in self["_output_pizza_cutter_yaml"]
            and band in self["_output_pizza_cutter_yaml"][tilename]
        ):
            return True
        else:
            return False

    def get_output_pizza_cutter_yaml(self, tilename, band):
        if (
            "_output_pizza_cutter_yaml" in self
            and tilename in self["_output_pizza_cutter_yaml"]
            and band in self["_output_pizza_cutter_yaml"][tilename]
        ):
            return copy.deepcopy(self["_output_pizza_cutter_yaml"][tilename][band])
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

    def write_output_pizza_cutter_yaml(self):
        if "_output_pizza_cutter_yaml" in self and "desrun" in self:
            for tilename in self["_output_pizza_cutter_yaml"]:
                for band in self["_output_pizza_cutter_yaml"][tilename]:
                    pth = get_pizza_cutter_yaml_path(
                        self["base_dir"],
                        self["desrun"],
                        tilename,
                        band,
                    )
                    os.makedirs(os.path.dirname(pth), exist_ok=True)
                    with open(pth, "w") as fp:
                        yaml.dump(
                            self["_output_pizza_cutter_yaml"][tilename][band],
                            fp
                        )
        else:
            raise RuntimeError(
                "Could not write pizza cutter yaml due to missing yaml or desrun!"
            )

    def write_input_pizza_cutter_yaml(self, data, tilename, band, skip_existing=True):
        self.set_input_pizza_cutter_yaml(data, tilename, band, skip_existing=skip_existing)

        _data = self.get_input_pizza_cutter_yaml(tilename, band)
        pth = get_pizza_cutter_yaml_path(
            self["imsim_data"],
            self["desrun"],
            tilename,
            band,
        )
        os.makedirs(os.path.dirname(pth), exist_ok=True)
        with open(pth, "w") as fp:
            yaml.dump(_data, fp)

    def set_input_pizza_cutter_yaml(self, _data, tilename, band, skip_existing=True):
        data = copy.deepcopy(_data)
        if "imsim_data" in self and self["imsim_data"] is not None:
            replace_imsim_data_in_pizza_cutter_yaml(data, self["imsim_data"])

        self._make_lists_psfmaps_symlinks(
            self["imsim_data"], tilename, band, data,
            skip_existing=skip_existing,
        )

        if "_input_pizza_cutter_yaml" not in self:
            self["_input_pizza_cutter_yaml"] = {}
        if tilename not in self["_input_pizza_cutter_yaml"]:
            self["_input_pizza_cutter_yaml"][tilename] = {}
        self["_input_pizza_cutter_yaml"][tilename][band] = data

        if not self.has_output_pizza_cutter_yaml(tilename, band):
            self.set_output_pizza_cutter_yaml(data, tilename, band)

    def get_input_pizza_cutter_yaml(self, tilename, band):
        if (
            "_input_pizza_cutter_yaml" in self
            and tilename in self["_input_pizza_cutter_yaml"]
            and band in self["_input_pizza_cutter_yaml"][tilename]
        ):
            return copy.deepcopy(self["_input_pizza_cutter_yaml"][tilename][band])
        else:
            raise RuntimeError(
                f"Could not find input pizza cutter yaml entry for tile|band={tilename}|{band}"
            )

    def _make_lists_psfmaps_symlinks(
        self, base_dir_or_imsim_data, tilename, band, pyml, skip_existing=False,
    ):
        odir = os.path.join(base_dir_or_imsim_data, self["desrun"], tilename)

        ##############################################
        # make bkg and nullwt flist files
        os.makedirs(os.path.join(odir, "lists"), exist_ok=True)
        bkg_file = os.path.join(
            odir, "lists", f"{tilename}_{band}_bkg-flist-{self['desrun']}.dat",
        )
        if not (skip_existing and os.path.exists(bkg_file)):
            with open(bkg_file, "w") as fp_bkg:
                for i in range(len(pyml["src_info"])):
                    fp_bkg.write(pyml["src_info"][i]["bkg_path"] + "\n")

        nw_file = os.path.join(
            odir, "lists", f"{tilename}_{band}_nullwt-flist-{self['desrun']}.dat",
        )
        if (
            (not (skip_existing and os.path.exists(nw_file)))
            and any(
                "coadd_nwgint_path" in pyml["src_info"][i]
                for i in range(len(pyml["src_info"]))
            )
        ):
            with open(nw_file, "w") as fp_nw:
                for i in range(len(pyml["src_info"])):
                    if "coadd_nwgint_path" in pyml["src_info"][i]:
                        fp_nw.write(
                            "%s %r\n" % (
                                pyml["src_info"][i]["coadd_nwgint_path"],
                                pyml["src_info"][i]["magzp"],
                            )
                        )

        # make psf map files
        pmap_file = os.path.join(odir, f"{tilename}_{band}_psfmap-{self['desrun']}.dat")
        if not (skip_existing and os.path.exists(pmap_file)):
            with open(pmap_file, "w") as fp:
                fp.write(
                    "%d %d %s\n" % (
                        -9999,
                        -9999,
                        pyml["psf_path"],
                    )
                )
                for i in range(len(pyml["src_info"])):
                    fn = pyml["src_info"][i]["psfex_path"]
                    en, _, cn = os.path.basename(fn).split("_")[:3]
                    en = en[1:]
                    cn = cn[1:]
                    fp.write("%s %s %s\n" % (en, cn, fn))

        pmap_file = os.path.join(odir, f"{tilename}_all_psfmap.dat")
        if not (skip_existing and os.path.exists(pmap_file)):
            with open(pmap_file, "w") as fp_w:
                for _bn in ["g", "r", "i", "z"]:
                    _bn_fn = os.path.join(odir, f"{tilename}_{_bn}_psfmap-{self['desrun']}.dat")
                    if os.path.exists(_bn_fn):
                        with open(_bn_fn, "r") as fp_r:
                            for line in fp_r.readlines():
                                fp_w.write(line)

        # symlinks
        def _symlink_file_rel_to_cwd(fn):
            ln = os.path.basename(fn)
            if not (skip_existing and os.path.exists(ln)):
                fn_rel = os.path.relpath(fn)
                safe_rm(ln)
                os.symlink(fn_rel, ln)

        # symlink nullwt files
        if any(
            "coadd_nwgint_path" in pyml["src_info"][i]
            for i in range(len(pyml["src_info"]))
        ):
            nodir = os.path.join(odir, f"nullwt-{band}")
            os.makedirs(nodir, exist_ok=True)
            with pushd(nodir):
                for i in range(len(pyml["src_info"])):
                    if "coadd_nwgint_path" in pyml["src_info"][i]:
                        _symlink_file_rel_to_cwd(pyml["src_info"][i]["coadd_nwgint_path"])

        # symlink psf files
        podir = os.path.join(odir, "psfs")
        os.makedirs(podir, exist_ok=True)
        with pushd(podir):
            _symlink_file_rel_to_cwd(pyml["psf_path"])
            for i in range(len(pyml["src_info"])):
                _symlink_file_rel_to_cwd(pyml["src_info"][i]["psfex_path"])

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
