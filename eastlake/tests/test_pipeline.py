import os
import sys
import pytest
from unittest import TestCase
import tempfile
import logging
from collections import OrderedDict

from ..steps import *
from ..step import Step
from ..stash import Stash
from ..pipeline import Pipeline


TEST_DIR = os.getcwd()
STEP_CLASSES = OrderedDict( [ ('galsim', GalSimRunner),
                              ('swarp', SWarpRunner),
                              ('sextractor', SExtractorRunner),
                              ('meds', MEDSRunner),
                              ('single_band_swarp', SingleBandSwarpRunner),
                              ('true_detection', TrueDetectionRunner),
                              ('delete_images', DeleteImages),
                              ('delete_meds', DeleteMeds),
                            ] )
DEFAULT_STEPS = ["galsim", "swarp", "sextractor", "meds"]



def test_pipeline_state(capsys): 
	base_dir = os.path.join(TEST_DIR,'foo')
	config = None
	steps = []
	for step_name in DEFAULT_STEPS:
		steps.append(Step(config, base_dir, name=step_name, logger=None, verbosity=None, log_file=None))
	pl = Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=None, record_file="job_record.pkl")

	assert pl.name == 'pipeline' 
	# calling get_logger(). check for print statements
	captured = capsys.readouterr()
	assert "log_file=" in captured.out 
	assert "filemode=" in captured.out 
	logging.basicConfig(format="%(message)s", level=logging.WARNING, stream=sys.stdout, filemode='w') 
	log1 = logging.getLogger(pl.name) 
	assert pl.logger == log1

	# logger is not None to start. Create pl2. 
	pl2 = Pipeline(steps, base_dir, logger=log1, verbosity=1, log_file=None, name="pipeline", config=None, record_file="job_record.pkl")
	assert pl2.logger == log1

	# base_dir not None.
	assert os.path.isdir(base_dir)
	assert pl.base_dir == os.path.join(TEST_DIR, 'foo')
	# base_dir is None. 
	assert Pipeline(steps, None, logger=None, verbosity=1, log_file=None, name="pipeline", config=None, record_file="job_record.pkl").base_dir == TEST_DIR

	# test logging error. => ignore for now. 

	# steps, step_name test. Use pl. 
	assert pl.steps == steps
	assert pl.step_names == [s.name for s in steps]

	# if config is not None. Create pl3. 
	pl3 = Pipeline(steps, base_dir, logger=log1, verbosity=1, log_file=None, name="pipeline", config='bar', record_file="job_record.pkl")
	# check if config is dumped into f. => skip for now?

	# record_file check. Use pl. 
	assert pl.record_file == os.path.join(base_dir, 'job_record.pkl')
	assert Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=None, record_file=base_dir+'/job_record.pkl').record_file == os.path.join(base_dir, 'job_record.pkl')
	assert Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=None, record_file=None).record_file == os.path.join(base_dir, 'job_record.pkl')

	# init stash. Use pl. 
	assert pl.stash == Stash(base_dir, [s.name for s in steps])

def test_pipeline_from_record_file():

	# parameters. 
	config_file = 'config.yaml'
	job_record_file = 'job_record.pkl'
	base_dir = os.path.join(TEST_DIR,'foo')
	steps = []
	for step_name in DEFAULT_STEPS:
		steps.append(Step(config, base_dir, name=step_name, logger=None, verbosity=None, log_file=None))

	# when starting from the previous run. logger=None, record_file=None, base_dir=None. 
	# What should I test here? 
	pl = Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=None, record_file=None)
	pipe_cont = pl.from_record_file(config_file, job_record_file, base_dir=None, logger=None, verbosity=1, log_file=None, name="pipeline_cont", step_names=steps, new_params=None, record_file=None)
	stsh = Stash.load()

def test_pipeline_from_config_file():

	config_file = './eastlake/e2e-008_test.yaml'
	job_record_file = 'job_record.pkl'
	base_dir = os.path.join(TEST_DIR,'foo')

	pl = Pipeline(steps, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", config=None, record_file=None)
	pipe_conf = pl.from_conf_file(config_file, base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", step_names=None, new_params=None, record_file=None)
	assert pipe_conf.logger == logging.getLogger('pipeline') # logger is None. 
	# len(config) is 1. 
	assert pipe_conf.config == galsim.config.ReadConfig(config_file)[0]
	# multiple config files. 
	with pytest.raises(AssertionError) as e:
        pipe_conf2 = pl.from_conf_file(['config.yaml', 'config2.yaml'], base_dir, logger=None, verbosity=1, log_file=None, name="pipeline", step_names=None, new_params=None, record_file=None)
        assert e.type is AssertionError

    # template is in config.
    assert pipe_conf.config_dirname == './eastlake'
    # cwd is not where template is and there is no template in cwd. 
    assert pipe_conf.template_file_to_use == './eastlake/e2e-008.yaml'
    assert pipe_conf.config['template'] == './eastlake/e2e-008.yaml'

    ## I dont know what to do for line 176-180. 

    # 





























