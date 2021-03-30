import os
import sys
import tempfile
import logging
import yaml
import galsim

from ..steps import GalSimRunner
from ..step import Step
from ..stash import Stash
from ..pipeline import Pipeline, DEFAULT_STEPS


TEST_DIR = os.getcwd()
CONFIG = """\
modules:
    - galsim.des
    - galsim_extra
    - montara
    - numpy

pipeline:
    steps: [galsim, single_band_swarp]

delete_images:
    delete_coadd: True
    delete_se: True
    save_tilenames:
    - DES0003-3832

desy3cols:
  shear_weight_grid: ${DESDATA}/y3_shape_w_grid_03_16_20_highsnr.txt

delete_meds:
    save_tilenames: [DES0003-3832]

all_tile_cats:
    tag:

meds:
    cutout_types: ['image','weight','seg','bmask']
    meds_dir: meds
    meds_run: y3v02
    sub_bkg: False
    add_psf_data: True
    use_joblib: True

sof:
    config_file: ${IMSIM_DIR}/ngmix_config/run-y3imsim-sof-psfinfile.yaml
    clobber: True
    use_joblib: True

single_band_swarp:
    config_file: ${IMSIM_DIR}/astromatic_config/Y3A1_v1_swarp.config
    swarp_cmd: swarp
    ref_mag_zp: 30.
    update:
        NTHREADS: 8
        PIXEL_SCALE : 0.263
        IMAGE_SIZE : 10000,10000

swarp:
    config_file: ${IMSIM_DIR}/astromatic_config/Y3A1_v1_swarp.config
    swarp_cmd: swarp
    center_from_header: True
    coadd_bands: ['r','i','z']
    mask_hdu : 1
    update:
        RESAMPLE : N
        COPY_KEYWORDS : BUNIT,TILENAME,TILEID
        PIXEL_SCALE : 0.263
        IMAGE_SIZE : 10000,10000
        COMBINE_TYPE : CHI-MEAN
        NTHREADS : 32
        BLANK_BADPIXELS : Y

#Options for SExtractor - can just provide a sextractor config file as below
#Fields can be updated using the update section e.g. here we update the detection
#threshold DETECT_THRESH.
sextractor:
    sex_cmd: sex
    #single_band_det: r
    config_file: ${IMSIM_DIR}/astromatic_config/Y3A1_v1_sex.config
    params_file : ${IMSIM_DIR}/astromatic_config/deblend.param
    filter_file : ${IMSIM_DIR}/astromatic_config/Y3A1_v1_gauss_3.0_7x7.conv
    star_nnw_file : ${IMSIM_DIR}/astromatic_config/Y3A1_v1_sex.nnw
    update:
        CHECKIMAGE_TYPE : SEGMENTATION,BACKGROUND,BACKGROUND_RMS
        DEBLEND_MINCONT : 0.001
        DETECT_THRESH : 1.1
        ANALYSIS_THRESH : 1.1

eval_variables:
    srun: e2e-test
    sstar_mag_col: &star_mag_col
        type: FormattedStr
        format: "mag_%s"
        items:
        - "$band"
    sgal_mag_col: &gal_mag_col
        type: FormattedStr
        format: "mag_%s_dered"
        items:
        - "$band"
    ftruth_g:
        type: List
        items:
        - 0.
        - '$float((@gal.items.0.ellip).g)'
        index: '@current_obj_type_index'

    ftruth_beta:
        type: List
        items:
        - 0.
        - '$float((@gal.items.0.ellip).beta.rad)'
        index: '@current_obj_type_index'
    sz_col: &z_col "photoz"

input:
    # Use analytic galaxies with size and flux parameters that match the distribution seen
    # in the COSMOS galaxies.
    catalog_sampler:
        file_name: /global/project/projectdirs/des/y3-image-sims/input_cosmos_v4.fits
        cuts:
            mag_i: [15., 25.]  #use only 15<mag_i<25. for now.
            isgal: [1,] #select galaxies only since we're simulating stars separately.
            mask_flags: [0,] #apply mask flags
            bdf_hlr: [0.,5.]
    desstar:
        file_name:
            type: FormattedStr
            format: /global/cscratch1/sd/maccrann/DES/image_sims/star_cats_v0/stars-%s.fits
            items:
            - "$tilename"
        mag_i_max: 25.
    des_piff:
        file_name: "$piff_path"

image:
    type: WideScattered
    border: 15
    random_seed: 1234
    nproc: 1

    # The number of objects across the full focal plane.
    nobjects:
        type: MixedNObjects
        ngalaxies:
            type: RandomPoisson
            mean: 170000
        use_all_stars: True

    #could read this from the image headers, but let's set them explicitly for now
    xsize: 2048
    ysize: 4096

    world_pos:
        type: RADec
        type: RADec
        ra:
            type: Degrees
            theta: { type: Random, min: "$ra_min_deg", max: "$ra_max_deg" }
        dec:
            type: Radians
            theta:
                type: RandomDistribution
                function: "math.cos(x)"  # Uniform on the sky means P(delta) ~ cos(delta)
                x_min: "$numpy.radians(dec_min_deg)"
                x_max: "$numpy.radians(dec_max_deg)"

#use Piff PSF for now
psf:
    type: DES_Piff
    use_substitute: "$is_blacklisted"
    no_smooth: False
    substitute_psf:
        type: Moffat
        beta: 3.
        fwhm: 1.

output:
    type: DESTile
    nproc: 32
    # The number of exposures to build
    bands: [g,r,i,z]
    desrun: y3v02
    desdata: /global/project/projectdirs/des/y3-image-sims
    noise_mode: from_weight
    add_bkg: False
    tilename: DES0003-3832
    blacklist_file: /global/homes/m/maccrann/DES/y3-wl_image_sims/input/piff_stuff/blacklist400.yaml

    #Save weight and badpix extensions too
    badpixfromfits:
        hdu: 1
        mask_hdu: 2
        mask_file: "$orig_image_path"
    weight:
        hdu: 2

    truth:
        #DESTile type fills in filename
        columns:
            num: obj_num
            half_light_radius:
                type: Eval
                str: "0.0 if @current_obj_type=='star' else hlr"
                fhlr: "@gal.items.0.half_light_radius"
            g1: "$(@stamp.shear).g1"
            g2: "$(@stamp.shear).g2"
            g: "$truth_g"
            beta: "$truth_beta"
            obj_type: "@current_obj_type"
            obj_type_index: "@current_obj_type_index"
            band: "band"
            mag_zp: "$mag_zp"
            laigle_number:
                type: Eval
                str: "-1 if @current_obj_type=='star' else int(laigle_number)"
                flaigle_number: { type: catalog_sampler_value, col: laigle_number }
            z:
                type: Eval
                str: "-1. if @current_obj_type=='star' else z_gal"
                fz_gal: { type: catalog_sampler_value, col: "$z_col" }
"""

def test_galsim_execute():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = tmpdir
        config_file_path = os.path.join(tmpdir, "cfg.yaml")
        with open(config_file_path, "w") as fp:
            fp.write(CONFIG)

        # creating galsim step. Similar to from_config_file() in pipeline.py. 
        config = galsim.config.ReadConfig(config_file_path)[0]
        galsim_logger = logging.getLogger("galsim_test")
        step_galsim = [GalSimRunner(config, base_dir, logger=galsim_logger)]

        # execute galsim step. 
        galsim_stash = Stash(base_dir, ["galsim"])
        status, stsh = step_galsim[0].execute_step(galsim_stash, new_params=None)

        assert stsh["tilename"] == "DES0003-3832"










































