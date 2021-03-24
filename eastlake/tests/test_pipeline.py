import os
import sys
import pytest
import tempfile
import logging
from collections import OrderedDict
from unittest import mock
import yaml
import galsim

from ..steps import GalSimRunner, SingleBandSwarpRunner
from ..step import Step
from ..stash import Stash
from ..pipeline import Pipeline, DEFAULT_STEPS, STEP_CLASSES


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

#No knots
gal:
    type: Sum
    items:
        - type: Exponential
          half_light_radius: { type: catalog_sampler_value, col: bdf_hlr }
          ellip:
              type: GBeta
              g: { type: Eval, str: "np.sqrt(g1**2 + g2**2)", fg1: { type: catalog_sampler_value, col: bdf_g1 }, fg2: { type: catalog_sampler_value, col: bdf_g2 } }
              beta: { type: Random }
          flux: { type: Eval, str: "1-fracdev", ffracdev: { type: catalog_sampler_value, col: bdf_fracdev } }

        - type: DeVaucouleurs
          half_light_radius: '@gal.items.0.half_light_radius'
          ellip: "@gal.items.0.ellip"
          flux: "$1-@gal.items.0.flux"

    flux:
        type: Eval
        #Input catalog has mag
        #convert to flux via flux = 10**(0.4*(mag_zp-mag))
        str: "10**(0.4*(mag_zp-mag))"
        fmag: { type: catalog_sampler_value, col: *gal_mag_col }

star:
    type: Gaussian  # Basically a delta function.
    sigma: 1.e-6
    flux:
        type: Eval
        str: "10**( 0.4 * (mag_zp - mag))"
        fmag: { type: DESStarValue, col: *star_mag_col }

stamp:
    type: MixedScene
    objects:
        # These give the probability of picking each kind of object.  The
        # choice of which one is picked for a given object is written to the
        # base dict as base['current_obj_type'] and is thus available as
        # @current_obj_type.  The actual constructed object is similarly
        # available as @current_obj.  And the type by number in this list
        # (starting with 0 for the first) is @current_obj_type_index.
        star: 0.2
        gal: 0.8
    obj_type: {type: Eval,
              str: "object_type_list[i]",
              ii: "$obj_num-start_obj_num"
              }
    draw_method: auto
    shear:
        type: G1G2
        g1: 0.02
        g2: 0.00
    gsparams:
        maximum_fft_size: 16384

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

CONFIG_NOPL = """\
modules:
    - galsim.des
    - galsim_extra
    - montara
    - numpy

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

def test_pipeline_state(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = tmpdir  # os.path.join(TEST_DIR,'foo')
        config = None
        steps = []
        for step_name in DEFAULT_STEPS:
            steps.append(Step(config, base_dir, name=step_name, logger=None, verbosity=None, log_file=None))
        pl = Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None,
                      name="pipeline", config=None, record_file="job_record.pkl")

        assert pl.name == 'pipeline'
        # calling get_logger(). check for print statements
        captured = capsys.readouterr()
        assert "log_file=" in captured.out
        assert "filemode=" in captured.out
        logging.basicConfig(format="%(message)s", level=logging.WARNING, stream=sys.stdout, filemode='w')
        log1 = logging.getLogger(pl.name)
        assert pl.logger == log1

        # logger is not None to start. Create pl2.
        pl2 = Pipeline(steps, base_dir, logger=log1, verbosity=1, log_file=None,
                       name="pipeline", config=None, record_file="job_record.pkl")
        assert pl2.logger == log1

        # base_dir not None.
        assert os.path.isdir(base_dir)
        assert pl.base_dir == tmpdir  # os.path.join(TEST_DIR, 'foo')
        # base_dir is None. When base_dir is None, base_dir should be created as cwd.
        assert Pipeline(steps, None, logger=None, verbosity=1, log_file=None, name="pipeline",
                        config=None, record_file="job_record.pkl").base_dir == TEST_DIR

        # test logging error. => ignore for now.

        # steps, step_name test. Use pl.
        assert pl.steps == steps
        assert pl.step_names == [s.name for s in steps]

        # if config is not None. Create pl3.
        pl3 = Pipeline(steps, base_dir, logger=log1, verbosity=1, log_file=None,
                       name="pipeline", config='bar', record_file="job_record.pkl")
        # check if config is dumped into f. => skip for now?

        # record_file check. Use pl.
        assert pl.record_file == os.path.join(base_dir, 'job_record.pkl')
        assert Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=None,
                        record_file=base_dir+'/job_record.pkl').record_file == os.path.join(base_dir, 'job_record.pkl')
        assert Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline",
                        config=None, record_file=None).record_file == os.path.join(base_dir, 'job_record.pkl')

        # init stash. Use pl.
        assert pl.stash == Stash(base_dir, [s.name for s in steps])


def test_pipeline_from_record_file():

    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = tmpdir
        config_file_path = os.path.join(tmpdir, "cfg.yaml")  
        job_record_file_path = os.path.join(tmpdir, 'job_record.pkl') 
        with open(config_file_path, "w") as fp:
            fp.write(CONFIG)
        
        # making an example job record file. 
        test_config = galsim.config.ReadConfig(config_file_path)[0]
        test_logger = logging.getLogger("pipeline")
        galsim_step = [GalSimRunner(test_config, base_dir, logger=test_logger)]
        pipe_prev = Pipeline(
            galsim_step, base_dir, logger=test_logger, verbosity=1, 
            log_file=None, name="pipeline", config=test_config,
            record_file=job_record_file_path,
        )
        pipe_prev._save_restart(False)

        # step_names = []
        # for step_name in DEFAULT_STEPS:
        #     step_names.append(Step(None, base_dir, name=step_name, logger=None, verbosity=None, log_file=None))

    	# when starting from the previous run. logger=None, record_file=None, base_dir=None.
    	# step_names is None. Test when step_names is not None later. 
        pipe_cont = Pipeline.from_record_file(
            config_file_path, job_record_file_path , base_dir=None, 
            logger=None, verbosity=1, log_file=None, name="pipeline_cont", 
            step_names=None, new_params=None, record_file=None, 
        )
        # When step_names is None, step_names is created in from_config_file(). 
        step_names = ["galsim", "single_band_swarp"]
        stsh = Stash.load(job_record_file_path, base_dir, step_names)
        assert pipe_cont.stash == stsh
        
        # Do I test line 122-123? stsh["env"] is empty. 
        # Do I test line 124? assert pipe_cont == Pipeline(...)


def test_pipeline_from_config_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_file_path = os.path.join(tmpdir, "cfg.yaml")
        with open(config_file_path, "w") as fp:
            fp.write(CONFIG)

        # job_record_file = os.path.join(tmpdir, 'job_record.pkl')

        base_dir = tmpdir
        pipe_conf = Pipeline.from_config_file(
            config_file_path, base_dir, logger=None, verbosity=1,
            log_file=None, name="pipeline", step_names=None, new_params=None,
            record_file=None,
        )
        with open(os.path.join(base_dir, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)
        assert pipe_conf.logger == logging.getLogger("pipeline")  # logger is None.
        # len(config) is 1.
        # assert config == galsim.config.ReadConfig(config_file_path)[0]
        
        # SKIP FOR NOW. multiple configs in one file.
        # config_file_path = os.path.join(tmpdir, "cfg.yaml")
        # with open(config_file_path, "a") as fp:
        #     fp.write("---\n")
        #     fp.write(CONFIG)
        
        # with pytest.raises(RuntimeError):
        #     Pipeline.from_config_file(
        #         config_file_path, base_dir, logger=None, verbosity=1,
        #         log_file=None, name="pipeline", step_names=None, new_params=None,
        #         record_file=None,
        #     )

        # SKIP FOR NOW. template is in config. Line 157-172. 
            # assert pipe_conf.config_dirname == tmpdir
            # cwd is not where template is and there is no template in cwd.
            # assert pipe_conf.template_file_to_use == config_file_path
            # assert pipe_conf.config['template'] == config_file_path

        # SKIP FOR NOW. new_params is not None. Line 177-178. 
        # pipe_conf = Pipeline.from_config_file(
        #     config_file_path, base_dir, logger=None, verbosity=1,
        #     log_file=None, name="pipeline", step_names=None, new_params='foo',
        #     record_file=None,
        # )
        # When new_params is not None, galsim.config.UpdateConfig() happens. No need to test anything?

        # if 'pipeline' is not in config. Line 180-181. 
        # step_names must exist if config['pipeline'] does not exist. 
        # In that case, step_names in from_config_file arguments go through the rest to make steps list. -> TEST THIS LATER
        config_nopl_path = os.path.join(tmpdir, "cfg_nopl.yaml")
        with open(config_nopl_path, "w") as fp:
            fp.write(CONFIG_NOPL)
        nopipe_config = Pipeline.from_config_file(
            config_nopl_path, base_dir, logger=None, verbosity=1,
            log_file=None, name="pipeline", step_names=["galsim"], new_params=None,
            record_file=None,
        )
        # loading up config,yaml that is saved on disk. 
        with open(os.path.join(base_dir, "config.yaml"), "r") as f:
            config_nopl = yaml.load(f, Loader=yaml.Loader)
        assert config_nopl['pipeline'] == {'ntiles': 1}

        # if step_names are None. config['pipeline'] must exist. 
        # testing Line 182-216. Use pipe_conf as a default Pipeline class. 
        assert pipe_conf.base_dir == os.path.abspath(base_dir)
        # test line 197 later. This is when elements in step_names are not in config. 
        # 'galsim' is going to be added to steps first, and then step_class(other steps) is going to be added.
        # assumes step_class is not in step_config. skip line 203-208 and go to line 209.
        assert isinstance(pipe_conf.steps[0], GalSimRunner)
        assert isinstance(pipe_conf.steps[1], SingleBandSwarpRunner)



@mock.patch("eastlake.pipeline.GalSimRunner")
@mock.patch("eastlake.pipeline.SingleBandSwarpRunner")
def test_pipeline_from_config_file_execute(galsim_step_mock, singlebandswarp_step_mock):
    galsim_step_mock.return_value.name = "galsim"
    galsim_step_mock.return_value.execute_step.return_value = (0, 1)
    singlebandswarp_step_mock.return_value.name = "single_band_swarp"
    singlebandswarp_step_mock.return_value.execute_step.return_value = (0, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_file_path = os.path.join(tmpdir, "cfg.yaml")
        with open(config_file_path, "w") as fp:
            fp.write(CONFIG)

        base_dir = tmpdir

        pl = Pipeline.from_config_file(
            config_file_path, base_dir, logger=None, verbosity=1,
            log_file=None, name="pipeline", step_names=["galsim", "single_band_swarp"], new_params=None,
            record_file=None,
        )

        pl.execute()
        galsim_step_mock.return_value.execute_step.assert_called()
        singlebandswarp_step_mock.return_value.execute_step.assert_called()

        # testing line 260. 
        # steps would be ["galsim", "single_band_swarp"]. 
        assert pl.stash["completed_step_names"] == [("galsim", 0), ("single_band_swarp", 0)]

















