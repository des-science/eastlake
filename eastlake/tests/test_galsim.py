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
DEMO1_CONFIG = """\
# The gal field defines what kind of galaxy profile to use.
gal :
    # One of the simplest profiles is a Gaussian.
    type : Gaussian

    # Gaussian profiles have a number of possible size parameters, but
    # sigma is the most basic one.
    # The others are fwhm and half_light_radius.  At least one of these must be specified.
    sigma : 2  # arcsec

    # The default flux would be 1, but you would typically want to define the flux
    # to be something other than that.
    flux : 1.e5


# Technically, the psf field isn't required, but for astronomical images we always have a PSF
# so you'll usually want to define one.  (If it's omitted, the galaxy isn't convolved
# by anything, so effectively a delta function PSF.)
# We use a Gaussian again for simplicity, but one somewhat smaller than the galaxy.
psf :
    type : Gaussian
    sigma : 1  # arcsec
    # No need to specify a flux, since flux=1 is the right thing for a PSF.


# The image field specifies some other information about the image to be drawn.
image :
    # If pixel_scale isn't specified, then pixel_scale = 1 is assumed.
    pixel_scale : 0.2  # arcsec / pixel

    # If you want noise in the image (which is typical) you specify that here.
    # In this case we use gaussian noise.
    noise :
        type : Gaussian
        sigma : 30  # standard deviation of the counts in each pixel

    # You can also specify the size of the image if you want, but if you omit it
    # (as we do here), then GalSim will automatically size the image appropriately.


# Typically, you will want to specify the output format and file name.
# If this is omitted, the output will be to a fits file with the same root name as the
# config file (so demo1.fits in this case), but that's usually not a great choice.
# So at the very least, you would typically want to specify at least the file_name.
output :
    type : Fits
    dir : output_yaml
    file_name : demo1.fits
    tilename : DES0003-3832
"""

def test_galsim_execute():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = tmpdir
        config_file_path = os.path.join(tmpdir, "cfg.yaml")
        with open(config_file_path, "w") as fp:
            fp.write(DEMO1_CONFIG)

        # creating galsim step. Similar to from_config_file() in pipeline.py. 
        config = galsim.config.ReadConfig(config_file_path)[0]
        galsim_logger = logging.getLogger("galsim_test")
        step_galsim = [GalSimRunner(config, base_dir, logger=galsim_logger)]

        # execute galsim step. 
        galsim_stash = Stash(base_dir, ["galsim"])
        status, stsh = step_galsim[0].execute_step(galsim_stash, new_params=None)

        #assert stsh["tilename"] == "DES0003-3832"










































