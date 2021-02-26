from __future__ import print_function
import galsim
import numpy as np
import sys
import os
import pprint
import subprocess
import meds
import desmeds
import yaml
import fitsio
import psfex
import pickle
#for metacal
from ngmixer.ngmixing import NGMixer
import ngmixer
from collections import OrderedDict
import copy
import glob
import json
from .megamixer import ImSimMegaMixer
from .utils import get_logger, safe_mkdir
import astropy.io.fits as fits
from shutil import copy, copytree, rmtree

from .step import Step
#import steps
#import inspect
#print(inspect.getmembers(steps))

import matplotlib
import pylab

from esutil.htm import HTM

from scipy.stats import sigmaclip

from .mcal_cat import apply_cut_dict, add_field, vstack2

from .des_tile import get_orig_coadd_file

#for coordinate matching
from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky


def find_index_of_y_in_x(x, y):
    assert np.all(np.in1d(y,x))
    x_sort_inds = np.argsort(x)
    ypos = np.searchsorted(x[x_sort_inds], y)
    return x_sort_inds[ypos]

def read_truth(truth_file, logger, dtype=None):
    if not os.path.isfile(truth_file):
        logger.error("no truth file %s found"%truth_file)
        return None
    try:
        truth_data = fitsio.read(truth_file)
    except IOError:
        try:
            #dtype=None means choose the dtype automatically
            truth_data = np.genfromtxt(truth_file, names=True, dtype=dtype)
        except IndexError: #this throws an IndexError if file is empty...
            return None
    return truth_data

class TruthTest(Step):
    """Test whether truth information makes sense - in particular whether entries in the truth catalog which correspond to the same object,
    have the same e.g. ra/dec.
    params:
    - obj_id (default=id)      object identifier in truth catalog
    - match_band_tile          Quantities that should be the same for a given object
                               for all exposures in the same tile and band (default ra, dec).
    - match_tile               Quantities that should be the same for a given object
                               for all exposures in the same tile (default ra, dec).
    -
    """
    output_dir = "truth_data"
    def __init__(self, config, base_dir, name="truth_test", logger=None, verbosity=0, log_file=None):
        super(TruthTest, self).__init__(config, base_dir, name=name, logger=logger, verbosity=verbosity, log_file=log_file)
        if "id_column" not in self.config:
            self.config["id_column"] = "id"
        if "match_band_tile" not in self.config:
            self.config["match_band_tile"] = ["ra", "dec"]
        if "match_tile" not in self.config:
            self.config["match_tile"] = ["ra", "dec"]
        if "consolidated_columns" not in self.config:
            self.config["consolidated_columns"] = [(self.config["id_column"], "int"),
                                                   ("ra", "float"), ("dec", "float"),
                                                   ("gal_catalog_row", "int"),
                                                   ("star_catalog_row", "int"),
                                                   ("obj_type_index", "int"), ("g1", "float"), 
                                                   ("g2", "float")
                                                   ]
        if "add_flux" not in self.config:
            self.config["add_flux"] = True
        if "do_checks" not in self.config:
            self.config["do_checks"] = True
        if "extra_cols" not in self.config:
            self.config["extra_cols"] = None
        if "rtol" not in self.config:
            self.config["rtol"] = 1.e-7
        if "atol" not in self.config:
            self.config["atol"] = None

    def execute(self, stash, new_params=None, comm=None):
        tilenames = stash["tilenames"]

        output_dir = os.path.join(self.base_dir, self.output_dir)
        safe_mkdir(output_dir)
        truth_files = []
        tilenames = stash["tilenames"]
        bands = stash["bands"]
        if self.config["do_checks"]:
            self.logger.error("checking truth info for tiles: %s, bands: %s"%(str(tilenames), str(bands)))
            self.logger.error("checking that columns %s match for a given tile object for a given tile and band"%(str(self.config["match_band_tile"])))
            self.logger.error("checking that columns %s match for a given tile object for a given tile"%(str(self.config["match_tile"])))

        for tilename in tilenames:
            tile_file_info = stash["tile_info"][tilename]
            truth_data_tile_list = []
            truth_files_tile = []
            self.logger.error("doing truth_test for tile %s"%(tilename))

            truth_file_ind_tile = []

            #There can be issues with the __array_wrap__ below
            #with string columns getting slightly different dtype
            #so force this
            dtype=None
            flux_data = [] #Collect per-band fluxes
            for band in bands:
                band_errors = []
                band_info = tile_file_info[band]
                #Read all truth information for this band 
                truth_data_band_list = []
                truth_files_band = band_info["truth_files"]
                truth_files_tile += truth_files_band
                for i,f in enumerate(truth_files_band):
                    d = read_truth(f, self.logger, dtype=dtype)
                    if d is not None:
                        if dtype is None:
                            dtype=d.dtype
                        d = np.atleast_1d(d) #in case only one row in which case d is a scalar
                        #also add a column with truth file index
                        d = add_field(d, [("truth_file_index",int)], [i*np.ones(len(d), dtype=int),])
                        truth_data_band_list.append(d)
                        
                #concatenate all truth data for this band
                try:
                    truth_data_band = truth_data_band_list[0].__array_wrap__(np.hstack(truth_data_band_list))
                except TypeError as e:
                    dtypes = [t.dtype for t in truth_data_band_list]
                    print("############")
                    print("dtypes for first truth file %s:"%truth_files_band[0])
                    print(dtypes[0])
                    for t,d in zip(truth_files_band[1:], dtypes[1:]):
                        if d!=dtypes[0]:
                            print("found different dtypes for truth file %s:"%t)
                            print(d)
                    raise(e)
                #save consolidated truth information for this band
                truth_data_tile_list.append(truth_data_band)

                all_ids = truth_data_band[self.config["id_column"]].astype(int)
                unq_ids, unq_inds = np.unique(all_ids, return_index=True)
                output_file = os.path.join(output_dir, "truth_%s_%s.fits"%(tilename, band))
                fitsio.write(output_file, truth_data_band[unq_inds], clobber=True)
                band_info["consolidated_truth_file"] = output_file

                band_flux_data = np.zeros(len(unq_ids), dtype=[(self.config["id_column"], int),("flux", float)])
                band_flux_data[self.config["id_column"]] = unq_ids
                band_flux_data["flux"] = truth_data_band[unq_inds]["flux"]
                flux_data.append(band_flux_data)
                
                if self.config["do_checks"]:
                    #assert that all objects with the same id have the same columns in self.config["match_band_tile"] 
                    #find unique object ids, then loop through these asserting that self.config["match_band_tile"] properties do indeed match.
                    for obj_id in unq_ids:
                        use_inds = all_ids==obj_id
                        truth_file_inds = truth_data_band["truth_file_index"][use_inds]
                        for key in self.config["match_band_tile"]:
                            use_vals = truth_data_band[key][use_inds]
                            #check that all items are close to the zeroth one
                            try:
                                np.testing.assert_allclose( use_vals[0], use_vals, rtol=self.config["rtol"], atol=self.config["atol"] )
                            except AssertionError as e:
                                print("mismatch in column %s for object id %d"%(key, obj_id))
                                print("%s values:"%key)
                                print("with tolerance rtol=%.2e, atol=%.2e"%(self.config["rtol"],
                                                                             self.config["atol"])
                                      )
                                print([repr(x) for x in use_vals])
                                print("truth files:")
                                for i in truth_file_inds:
                                    print(truth_files_band[i])
                                band_errors.append(e)
                                print("************")

            truth_data_tile = truth_data_tile_list[0].__array_wrap__(np.hstack(truth_data_tile_list))

            #assert that all objects with the same id have the same columns in self.config["match_tile"] 
            all_ids = truth_data_tile[self.config["id_column"]].astype(int)
            unq_ids, unq_inds = np.unique(all_ids, return_index=True)
            tile_errors = []
            if self.config["do_checks"]:
                for obj_id in unq_ids:
                    use_inds = (all_ids==obj_id)
                    use_bands = truth_data_tile["band"][use_inds]
                    truth_file_inds = truth_data_tile["truth_file_index"][use_inds]
                    for key in self.config["match_tile"]:
                        use_vals = truth_data_tile[key][use_inds]
                        #check that all items are close to the zeroth one
                        try:
                            np.testing.assert_allclose( use_vals[0], use_vals, rtol=self.config["rtol"] )
                        except AssertionError as e:
                            print("mismatch in column %s for object id %d"%(key, obj_id))
                            print("%s values:"%key)
                            print("with tolerance rtol=%.2e, atol=%.2e"%(self.config["rtol"],
                                                                         self.config["atol"])
                            )
                            print([repr(x) for x in use_vals])
                            print("truth files:")

                            for band,truth_file_ind in zip(use_bands, truth_file_inds):
                                print(tile_file_info[band]["truth_files"][truth_file_ind])
                            tile_errors.append(e)
                            print("***************")
            if band_errors:
                raise(band_errors[0])
            if tile_errors:
                raise(tile_errors[0])
            
            #save consolidated truth information
            output_cols = list(self.config["consolidated_columns"])
            if self.config["extra_cols"] is not None:
                output_cols += self.config["extra_cols"]
            print(output_cols)
            output_cols = [(c[0], eval(c[1])) for c in output_cols]
            dtype = [c for c in output_cols if c[0] in truth_data_tile.dtype.names]
            output_data = np.zeros(len(unq_ids), dtype=dtype)
            for (colname,t) in dtype:
                if t==int:
                    output_data[colname] = (truth_data_tile[colname][unq_inds]).astype(int)
                else:
                    output_data[colname] = truth_data_tile[colname][unq_inds]

            if self.config["add_flux"]:
                #Loop through bands adding a flux column
                for band in bands:
                    band_flux_data = flux_data[bands.index(band)]
                    flux = np.zeros(len(output_data))
                    truth_data_tile_inds = find_index_of_y_in_x(output_data["id"], band_flux_data["id"])
                    flux[truth_data_tile_inds] = band_flux_data["flux"]
                    output_data = add_field(output_data, [("flux_%s"%band, float)], [flux])
                    
            truth_file_out = os.path.join(output_dir, "truth_%s.fits"%(tilename))
            self.logger.error("Saving truth data for %d unique objects to %s"%(len(unq_ids), truth_file_out))
            fitsio.write( truth_file_out, output_data, clobber=True )
            tile_file_info["consolidated_truth_file"] = truth_file_out
            
        print("done test truth")
        return 0, stash

class SExtractorTruthTest(Step):
    """Match SExtractor detections to truth catalog and optionally make comparison plots.
    Also save match file"""
    plot_dir = "plots_sex_truth_test"
    cat_dir = "matched_sextractor_truth_catalogs"
    def __init__(self, config, base_dir, name="sex_truth_test", logger=None, verbosity=0, log_file=None):
        super(SExtractorTruthTest, self).__init__(config, base_dir, name=name, logger=logger, verbosity=verbosity, log_file=log_file)
        #Specify in config which quantities to compare as a list of lists of form [<sextractor quantity>, <truth quantity>]
        #e.g. [MAG_AUTO, mag]. If not specified in config, use some defaults:
        if "compare" not in self.config:
            self.config["compare"] = [["ALPHA_J2000", "ra"],
                                      ["DELTA_J2000", "dec"],
                                      ["MAG_AUTO", "mag"]]
        if "id_column" not in self.config:
            self.config["id_column"] = "id"
        if "do_plots" not in self.config:
            self.config["do_plots"] = False
        if "match_dist_arcsec" not in self.config:
            self.config["match_dist_arcsec"] = 2. * 0.263 #use 2 pixel match radius by default
        if "num_mag_bins" not in self.config:
            self.config["num_mag_bins"] = 10
        if "mag_sigma_clip" not in self.config:
            self.config["mag_sigma_clip"] = 0.
        if "max_mag" not in self.config:
            self.config["max_mag"] = 30.

    def execute(self, stash, new_params=None, comm=None):
        tilenames = stash["tilenames"]

        bands = stash["bands"]
        #make plot directory if it doesn't already exist
        if self.config["do_plots"]:
            plot_dir = os.path.join(stash["base_dir"], self.plot_dir)
            safe_mkdir(plot_dir)
        #same for catalog directory
        cat_dir = os.path.join(self.base_dir, self.cat_dir)
        safe_mkdir(cat_dir)

        #First we need to get the truth information. The TruthTest step gets nice consolidated truth information for us.
        print(stash["completed_step_names"])
        if ("truth_test",0) not in stash["completed_step_names"]:
            truth_test_config = {"do_checks":False}
            truth_test_step = TruthTest(truth_test_config, self.base_dir)
            truth_test_status, stash = truth_test_step.execute_step(stash, comm=comm, collect_stashes=True)
            stash["completed_step_names"].append(("truth_test",0))

        for tilename in tilenames:
            tile_file_info = stash["tile_info"][tilename]
            truth_file_tile = tile_file_info["consolidated_truth_file"]
            truth_data_tile = fitsio.read(truth_file_tile)

            #Get the truth flux - use sum over all bands
            truth_fluxes = np.zeros(len(truth_data_tile))
            for band in bands:
                truth_fluxes += truth_data_tile["flux_%s"%band]
            
            #Read in SExtractor data - shouldn't matter which band since we always run detection on riz
            sex_data = fitsio.read( tile_file_info[bands[bands.index("i")]]["sex_cat"] )

            #Now find matches in truth catalog 
            #This may not always be unambiguous, so record the closest 3 matches
            n_matches=3
            match_data = np.zeros(len(sex_data), dtype = [("NUMBER",int),
                                                          ("truth_ind_1",int),
                                                          ("truth_ind_2",int),
                                                          ("truth_ind_3",int),
                                                          ("sep_arcsec_1", float),
                                                          ("sep_arcsec_2", float),
                                                          ("sep_arcsec_3", float),
                                                          ("truth_ind_final", int),
                                                          ("match_type", int)
                                                      ]
                                  )
            sex_coords = SkyCoord(ra = sex_data["ALPHA_J2000"]*u.degree, dec = sex_data["DELTA_J2000"]*u.degree)
            truth_coords = SkyCoord(ra = truth_data_tile["ra"]*u.degree, dec = truth_data_tile["dec"]*u.degree)
            for nthneighbor in range(1,3+1):
                ind_truth, sep, _ = match_coordinates_sky(sex_coords, truth_coords, nthneighbor=nthneighbor)
                match_data["truth_ind_%d"%nthneighbor] = ind_truth
                match_data["sep_arcsec_%d"%nthneighbor] = sep.arcsecond
            match_data["NUMBER"] = sex_data["NUMBER"]

            #Ok, so we got the three closest matches for each object. Now we need to assign the "best" match
            #Define:
            #Type 0: a detected object has a single, closest match within d_max, that is not a close match
            #for any other object.
            #Type 1: a detected object with no close match.
            #Type 2: A match is assigned during the bright-to-faint loop
            #Type 3: No assigned match after the bright-to-faint loop.
            match_data["match_type"] = -99
            match_data["truth_ind_final"] = -99
            has_close_match = match_data["sep_arcsec_1"]<self.config["match_dist_arcsec"]
            has_second_close_match = match_data["sep_arcsec_2"]<self.config["match_dist_arcsec"]

            #Find truth_ind_1 values that are unique and the indices of those 
            _, unique_inds = np.unique(match_data['truth_ind_1'], return_index=True)
            #Any entries in match_data['truth_ind_1'] not selected by unique_inds must be duplicates so
            #get non-unique first matches by selecting indices of match_data['truth_ind_1'] not in unique_inds
            non_unique_first_match_inds = match_data['truth_ind_1'][ np.delete(np.arange(len(match_data['truth_ind_1'])), unique_inds) ]
            first_match_is_anothers_first_match = np.in1d(match_data['truth_ind_1'], non_unique_first_match_inds)
            first_match_is_anothers_later_match = np.zeros(len(match_data), dtype=bool)
            for i in range(2, n_matches+1):
                if "truth_ind_%d"%i in match_data.dtype.names:
                    first_match_is_anothers_later_match += np.in1d(match_data["truth_ind_1"], match_data["truth_ind_%d"%i])
            has_exclusive_first_match = ~first_match_is_anothers_first_match * ~first_match_is_anothers_later_match
            type_0 = has_close_match * ~has_second_close_match * has_exclusive_first_match
            n_obj = len(type_0)
            type_0_frac = float(type_0.sum())/n_obj
            self.logger.error("%d/%d (fraction %f) objects are type 0 i.e. have a single, exclusive close match"%(
                type_0.sum(), n_obj, type_0_frac))
            #Now we can assign the type_0 objects
            match_data["match_type"][type_0] = 0
            match_data["truth_ind_final"][type_0] = match_data["truth_ind_1"][type_0]
            #we can also do the type 1 objects
            type_1 = ~has_close_match
            match_data["match_type"][type_1] = 1
            self.logger.error("%d/%d (fraction %f) objects are type 1 i.e. have no close match"%(
                type_1.sum(), n_obj, float(type_1.sum())/n_obj))

            #Now the tricky ones - type 2 and 3.
            is_type_2_or_3 = ~type_0 * ~type_1
            if is_type_2_or_3.sum()>0:
                #These are indices in the sextractor catalog of type 2 or 3 objects
                type_2_or_3_inds = np.where(is_type_2_or_3)[0]

                #we need to get fluxes for these so we can loop through in order or brightness
                fluxes_2_or_3 = sex_data["FLUX_AUTO"][is_type_2_or_3]
                #set any crap to be fainter than the faintest non-crap
                is_crap = ~np.isfinite(fluxes_2_or_3)
                fluxes_2_or_3[is_crap] = np.min(fluxes_2_or_3[~is_crap])-1.
                #Now sort the type 2/3 indices by flux
                type_2_or_3_inds_sorted = type_2_or_3_inds[np.argsort(fluxes_2_or_3)]
                #get a list of assigned truth indices - so far this is just the type_0 ones
                assigned_truth_inds = list(match_data["truth_ind_final"][type_0])
                for ind in type_2_or_3_inds_sorted:
                    truth_ind = -99
                    truth_flux = np.min(truth_fluxes)-1.
                    for i in range(1, n_matches+1):
                        truth_ind_i = match_data["truth_ind_%d"%i][ind]
                        truth_flux_i = truth_fluxes[ind]
                        if truth_ind_i not in assigned_truth_inds:
                            if truth_flux_i>truth_flux:
                                truth_ind = truth_ind_i
                    if truth_ind != -99:
                        assigned_truth_inds.append(truth_ind_i)
                        match_data["truth_ind_final"][ind] = truth_ind
                        match_data["match_type"][ind] = 2
                    else:
                        match_data["match_type"][ind] = 3
            type_2 = match_data["match_type"]==2
            type_3 = match_data["match_type"]==3
            self.logger.error("%d/%d (fraction %f) objects are type 2"%(
                type_2.sum(), n_obj, float(type_2.sum())/n_obj))
            self.logger.error("%d/%d (fraction %f) objects are type 3"%(
                type_3.sum(), n_obj, float(type_3.sum())/n_obj))
                        
            #save this matching info catalog
            output_file = os.path.join(cat_dir, "%s_sex_truth_match.fits"%tilename)
            fitsio.write(output_file, match_data, clobber=True)
            tile_file_info["sex_truth_match_cat"] = output_file
        
            match_ind_truth = match_data["truth_ind_final"]
            match_ids = truth_data_tile["id"][match_ind_truth]

            #Now if we also want to make the comparison plots, loop through bands,
            #get truth info for bands, and use matching from above.
            if self.config["do_plots"]:
                for band in bands:
                    band_file_info = tile_file_info[band]
                    truth_data_band = fitsio.read(band_file_info["consolidated_truth_file"])
                    sex_data = fitsio.read( band_file_info["sex_cat"] )
                    assert np.all(sex_data["NUMBER"]==match_data["NUMBER"])
                    #just use nearest match for these comparisons
                    use_objs = np.ones(len(sex_data), dtype=bool)
                    #remove those with closest neighbour futher than self.config["match_dist_arcsec"]
                    use_objs[match_data["sep_arcsec_1"] > self.config["match_dist_arcsec"]] = False

                    #also remove flagged objects
                    flagged = sex_data["FLAGS"] > 3
                    use_objs[flagged] = False

                    #if set, remove objects with mag>config["max_mag"]
                    too_faint = sex_data["MAG_AUTO"]>self.config["max_mag"]
                    use_objs[too_faint] = False

                    #We may not have drawn all the objects drawn for a tile for a given band
                    #This means that not all match_ids will be in the truth catalog for the band
                    #so we need to remove those that aren't
                    truth_ids_band = (truth_data_band['id']).astype(int)
                    match_in_band = np.in1d(match_ids, truth_ids_band)
                    use_objs[~match_in_band] = False

                    sex_data_use, match_ids_use = sex_data[use_objs], match_ids[use_objs]
                    
                    truth_inds_band = find_index_of_y_in_x(truth_ids_band, match_ids_use)

                    #Generate a 3-panel figure with MAG_AUTO-mag_true in top panel,
                    #(MAG_AUTO-mag_true)/MAGERR_AUTO in the second panel, and
                    #the fraction of objects in each magnitude bin whose measured
                    #mag is within 1-sigma of the true mag in the third panel.
                    #All as a function of true mag.
                    fig_mag = pylab.figure(figsize=(4,12))
                    ax1 = fig_mag.add_subplot(311)
                    truth_mags = truth_data_band["mag"][truth_inds_band]
                    sex_mags = sex_data_use["MAG_AUTO"]
                    sex_mag_errs = sex_data_use["MAGERR_AUTO"]
                    #if requested, remove most extreme truth objects
                    if self.config["mag_sigma_clip"]>1.e-9:
                        _,lo,hi = sigmaclip(truth_mags, low=self.config["mag_sigma_clip"],
                                            high=self.config["mag_sigma_clip"])
                        use = (truth_mags>lo)*(truth_mags<hi)
                        truth_mags = truth_mags[use]
                        sex_mags = sex_mags[use]
                        sex_mag_errs = sex_mag_errs[use]
                    mag_diffs = sex_mags - truth_mags

                    #First just plot raw magnitude differences as a function of true mag
                    ax1.plot( truth_mags, mag_diffs, '+', alpha=0.1)
                    ax1.set_ylabel("MAG_AUTO - true mag")
                    ax1.set_xticks([])

                    #Now mag_diff/err
                    ax2 = fig_mag.add_subplot(312)
                    mag_diff_over_errs = mag_diffs / sex_mag_errs
                    #find outliers and plot at 4 sigma in red
                    _,lo,hi = sigmaclip(mag_diff_over_errs, low=4., high=4.)
                    outliers_high = mag_diff_over_errs>hi
                    outliers_low = mag_diff_over_errs<lo
                    #plot inliers as normal - these are not high or low outliers i.e. the following:
                    inliers = ~(outliers_high+outliers_low)
                    ax2.plot( truth_mags[inliers], mag_diff_over_errs[inliers], '+', alpha=0.1)
                    ax2.plot( truth_mags[outliers_high], hi*np.ones(outliers_high.sum()), 'r+', alpha=0.1)
                    ax2.plot( truth_mags[outliers_low], lo*np.ones(outliers_low.sum()), 'r+', alpha=0.1)
                    ax2.set_ylabel("MAG_AUTO bias/err")
                    ax2.set_xticks([])
                    #ax1.errorbar( truth_mags, sex_data["MAG_AUTO"]-truth_data_band["mag"], fmt='+' )

                    #Now fraction within 1-sigma
                    ax3 = fig_mag.add_subplot(313)
                    ax3.set_xlabel("True magnitude")
                    ax3.set_ylabel("mag diff < err fraction")
                    #also plot in bins the fraction of objects with mag difference less than magerr_auto
                    mag_start = 0.999*truth_mags.min()
                    mag_end = 1.001*truth_mags.max()
                    self.logger.error("""comparing %s-band measured and truth mags for tile %s 
                    in %d bins between %.3f and %.3f"""%(band, tilename,
                        self.config["num_mag_bins"], mag_start, mag_end))
                    mag_bins = np.linspace(mag_start, mag_end, self.config["num_mag_bins"]+1)
                    mag_bin_mids = 0.5*(mag_bins[:-1]+mag_bins[1:])
                    binned_mean_mag_diff = np.zeros(self.config["num_mag_bins"])
                    binned_std_mag_diff = np.zeros_like(binned_mean_mag_diff)
                    num_in_bin = np.zeros(self.config["num_mag_bins"], dtype=int)
                    frac_gt_err = np.zeros_like(binned_mean_mag_diff) #fraction at greater difference from zero than 1 sigma
                    mag_bin_inds = np.digitize( truth_mags, mag_bins ) - 1
                    for i in range(self.config["num_mag_bins"]):
                        use = mag_bin_inds==i
                        n_in_bin_i = use.sum()
                        if n_in_bin_i == 0:
                            continue
                        num_in_bin[i] = n_in_bin_i
                        binned_mean_mag_diff[i] = np.sum(mag_diffs[use]) / n_in_bin_i
                        binned_std_mag_diff[i] = np.sqrt( np.sum((mag_diffs - binned_mean_mag_diff[i])**2) / n_in_bin_i )
                        try:
                            frac_gt_err[i] = float((np.abs(mag_diffs[use]/sex_mag_errs[use])<1.).sum()) / n_in_bin_i
                        except ZeroDivisionError:
                            frac_gt_err[i] = 0.
                    print("binned mean mag_diff:", binned_mean_mag_diff)
                    print("binned std mag_diff:", binned_std_mag_diff)
                    ax3.bar( mag_bins[:-1], frac_gt_err, align='edge' )
                    ax3.set_xlim(ax2.get_xlim())
                    fig_mag.tight_layout()
                    figname = os.path.join(plot_dir, "%s_%s_sextractor_mag_truth.png"%(tilename, band))
                    fig_mag.savefig(figname)
                    #pylab.show()
                    pylab.close(fig_mag)
        return 0, stash

class CmTruthTest(Step):
    """Compare cm mag to expectation from
    truth catalogs
    """
    plot_dir = "plots_cm_truth_test"
    ra_col = "ra" #for matching
    dec_col = "dec"
    def __init__(self, config, base_dir, name="cm_truth_test", logger=None, verbosity=0, log_file=None):
        super(CmTruthTest, self).__init__(config, base_dir, name=name, logger=logger, verbosity=verbosity, log_file=log_file)
        if "id_column" not in self.config:
            self.config["id_column"] = "id"
        if "do_plots" not in self.config:
            self.config["do_plots"] = True
        if "match_dist_arcsec" not in self.config:
            self.config["match_dist_arcsec"] = 2. * 0.263
        if "cut_dict" not in self.config:
            self.config["cut_dict"] = { "cm_flags" : [0], "flags" : [0] }

    def get_mags_and_err(self, stash, meas_data, band):
        if "sof_bands" in stash:
            band_num = stash["sof_bands"].index(band)
        else:
            self.logger.error("sof_bands not found in stash - assuming same as stash['bands']")
            band_num = stash["bands"].index(band)
        mags = meas_data["cm_mag"][:, band_num]
        flux_var = meas_data["cm_flux_cov"][:, band_num, band_num]
        flux = meas_data["cm_flux"][:, band_num]
        mag_err = - 1.09 * np.abs( np.sqrt(flux_var)/flux )
        return mags, mag_err

    def execute(self, stash, new_params=None, comm=None):

        tilenames = stash["tilenames"]
        
        #make plot directory if it doesn't already exist
        if self.config["do_plots"]:
            plot_dir = os.path.join(stash["base_dir"], self.plot_dir)
            safe_mkdir(plot_dir)
        
        #If we haven't yet matched the SExtractor detections to the truth
        #catalog, do this now.
        if ("sex_truth_test",0) not in stash["completed_step_names"]:
            self.logger.error("running sex_truth_test")
            config = {"do_plots":False, "id_column":self.config["id_column"]}
            sex_match_step = SExtractorTruthTest(config, self.base_dir, logger=self.logger)
            sex_match_status, stash = sex_match_step.execute_step(stash, comm=comm, collect_stashes=True)
            stash["completed_step_names"].append(("sex_truth_test",sex_match_status))

        #Now loop through tiles doing stuff
        shear_data_this_proc = {}
        for tilename in tilenames:
            tile_file_info = stash["tile_info"][tilename]
            self.logger.error("computing metacal shear data for tile %s"%(tilename))


            #Read sof/mof data
            meas_data = fitsio.read(tile_file_info["sof_file"])

            #Get tile truth data 
            truth_data_tile = fitsio.read(tile_file_info["consolidated_truth_file"])

            #Get match data
            match_data = fitsio.read(tile_file_info["sex_truth_match_cat"])
            match_ind_truth = match_data["truth_ind_1"]
            match_ids = truth_data_tile["id"][match_ind_truth]

            #Now loop through bands doing plots
            if "sof_bands" in stash:
                bands = stash["sof_bands"]
            else:
                bands = stash["bands"]

            for band in bands:
                band_file_info = tile_file_info[band]
                truth_data_band = fitsio.read(band_file_info["consolidated_truth_file"])
                meas_data = fitsio.read(tile_file_info["sof_file"])

                assert np.all(meas_data["number"]==match_data["NUMBER"])

                #just use nearest match for these comparisons
                use_objs = np.ones(len(meas_data), dtype=bool)
                #remove those with closest neighbour further than self.config["match_dist_arcsec"]
                use_objs[match_data["sep_arcsec_1"] > self.config["match_dist_arcsec"]] = False
                
                #apply cuts - apply_cut_dict returns mask as second argument
                _,use = apply_cut_dict( meas_data, self.config["cut_dict"], verbose=True)
                use_objs[~use] = False

                #We may not have drawn all the objects drawn for a tile for a given band
                #This means that not all match_ids will be in the truth catalog for the band
                #so we need to remove those that aren't
                truth_ids_band = (truth_data_band['id']).astype(int)
                match_in_band = np.in1d(match_ids, truth_ids_band)
                use_objs[~match_in_band] = False

                meas_data_use, match_ids_use = meas_data[use_objs], match_ids[use_objs]
                truth_inds_band = find_index_of_y_in_x(truth_ids_band, match_ids_use)

                truth_mags = truth_data_band["mag"][truth_inds_band]
                meas_mags, meas_mag_errs = self.get_mags_and_err(stash, meas_data_use, band)

                mag_diffs = meas_mags-truth_mags
                print("mean truth mag:", truth_mags.mean())
                print("mean measured mag:", meas_mags.mean())
                print("mean mag diff:", mag_diffs.mean())
                print("std(mag_diff):", np.std(mag_diffs))
            
        return 0, stash

class CoaddNoiseTest(Step):
    """Compare noise properties of coadds to DES originals"""
    plot_dir = "plots_coadd_noise_test"
    def __init__(self, config, base_dir, name="coadd_noise_test", logger=None, verbosity=0, log_file=None):
        super(CoaddNoiseTest, self).__init__(config, base_dir, name=name, logger=logger, verbosity=verbosity, log_file=log_file)
    
    def execute(self, stash, new_params=None, comm=None):

        tilenames = self.config.get("tilenames", stash["tilenames"])
        bands = stash["bands"]
        #make plot directory if it doesn't already exist                                                                                                                          
        plot_dir = os.path.join(stash["base_dir"], self.plot_dir)
        safe_mkdir(plot_dir)
        for tilename in tilenames:
            tile_file_info = stash["tile_info"][tilename]
            #First inspect riz detection image and segmentation map
            #except I don't have those at nersc :-(
            self.logger.error("Would be good to check noise properties of riz detection image, but don't have those stored at nersc....")
            for band in bands:
                #get orig coadd stuff
                orig_coadd_file = get_orig_coadd_file(stash["desdata"], stash["desrun"], tilename, band)
                orig_coadd_fits = fitsio.FITS(orig_coadd_file)
                orig_coadd_img, orig_coadd_msk, orig_coadd_wgt  = orig_coadd_fits[1].read(), orig_coadd_fits[2].read(), orig_coadd_fits[3].read()
                orig_seg_dir = os.path.join( os.path.dirname(os.path.dirname(orig_coadd_file)), "seg" )
                orig_seg_file = (os.path.basename(orig_coadd_file)).replace(".fits.fz", "_segmap.fits")
                orig_seg_file = os.path.join( orig_seg_dir, orig_seg_file )
                orig_seg = fitsio.read(orig_seg_file)
                
                #get sim coadd stuff
                band_file_info = stash["tile_info"][tilename][band]
                sim_coadd_img = fitsio.read(band_file_info["coadd_file"], band_file_info["coadd_ext"])
                sim_coadd_msk = fitsio.read(band_file_info["coadd_mask_file"], band_file_info["coadd_mask_ext"])
                sim_coadd_wgt = fitsio.read(band_file_info["coadd_weight_file"], band_file_info["coadd_weight_ext"])
                sim_seg = fitsio.read(band_file_info["seg_file"])

                #Go ahead and plot noise distribution by removing pixels that are:
                #- non-zero in the mask plane
                #- non-zero in the segmentation map
                fig=pylab.figure()
                ax = fig.add_subplot(111)
                sim_coadd_pix_use = (sim_seg==0)*(sim_coadd_msk==0)
                _,lo,hi = sigmaclip((sim_coadd_img[sim_coadd_pix_use]).flatten())
                sim_bins = np.linspace(lo, hi, 100)
                _,sim_bins,_ = ax.hist(sim_coadd_img[sim_coadd_pix_use], bins=sim_bins, histtype='step', label='sim', density=True)
                orig_coadd_pix_use = (orig_seg==0)*(orig_coadd_msk==0)
                _,orig_bins,_ = ax.hist(orig_coadd_img[orig_coadd_pix_use], bins=sim_bins, histtype='step', label='DES', density=True)
                ax.set_xlabel("pixel value")
                ax.legend()
                ax.set_title("%s %s-band coadd noise distibrution"%(tilename, band))
                filename = os.path.join(self.plot_dir, "%s_%s_coadd_noise.png"%(tilename, band))
                self.logger.error("saving coadd noise plot to %s"%filename)
                fig.savefig(filename)
                pylab.close(fig)
        return 0, stash
