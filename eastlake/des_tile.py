import galsim
import os
import numpy as np
import yaml

from galsim.config.output import OutputBuilder
from .fits import writeMulti
import astropy.io.fits as pyfits
from .utils import safe_mkdir
from .tile_setup import Tile
import shutil
from collections import OrderedDict

MODES = ["single-epoch", "coadd"]  # beast too?


def get_source_list_files(base_dir, desrun, tilename, bands):
    """Build paths to source file lists for a set of tiles.
    Parameters
    ----------
    base_dir : str
        The base path for the location of the files. Equivalent to the value
        of DESDATA but for outputs.
    desrun : str
        The DES run name.
    tilename : str
        name of the coadd tile.
    band : list of str
        A list of the desired bands (e.g., ['r', 'i', 'z']).
    Returns
    -------
    source_list_files : dict
        A dictionary keyed on `(tilename, band)` with a tuple of values
        `(im_list_file, wgt_list_file, msk_list_file, magzp_list_file)`
        giving the image list, weight image list, bit mask image list, and
        magnitude zero point list.
    """
    source_list_files = {}
    for band in bands:
        output_dir = os.path.join(base_dir, desrun, tilename, 'lists')
        magzp_list_file = os.path.join(
            output_dir,
            '%s_%s_fcut-flist-%s-magzp.dat' % (tilename, band, desrun))
        im_list_file = os.path.join(
            output_dir,
            '%s_%s_fcut-flist-%s.dat' % (tilename, band, desrun))
        wgt_list_file = os.path.join(
            output_dir,
            '%s_%s_fcut-flist-%s-wgt.dat' % (tilename, band, desrun))
        msk_list_file = os.path.join(
            output_dir,
            '%s_%s_fcut-flist-%s-msk.dat' % (tilename, band, desrun))
        source_list_files[band] = (
            im_list_file, wgt_list_file, msk_list_file, magzp_list_file)
    return source_list_files


def get_bkg_path(image_path, desdata=None):
    """Get the background image path from the image path.
    Parameters
    ----------
    image_path : str
        The path to the image.
    desdata : str, optional
        The path to the local DESDATA dir.
    Returns
    -------
    bkg_path : str
        The path to the background image.
    """
    bkg_dir = os.path.join(
        os.path.dirname(os.path.dirname(image_path)), "bkg")
    basename = os.path.basename(image_path)
    bkg_filename = "_".join(basename.split("_")[:4]) + "_bkg.fits.fz"
    pth = os.path.join(bkg_dir, bkg_filename)
    return _replace_desdata(pth, desdata)


def get_piff_path(image_path):
    """Get the Piff path from the image path.
    Parameters
    ----------
    image_path : str
        The path to the SE image.
    Returns
    -------
    piff_path : str
        The path to the Piff model.
    """
    img_bname = os.path.basename(image_path)
    piff_bname = img_bname.replace(
        '.fz', ''
    ).replace(
        'immasked.fits', 'piff.fits')
    expnum = int(piff_bname.split('_')[0][1:])

    if "PIFF_DATA_DIR" in os.environ and "PIFF_RUN" in os.environ:
        piff_path = os.path.join(
            os.environ['PIFF_DATA_DIR'],
            os.environ['PIFF_RUN'],
            str(expnum),
            piff_bname)
    else:
        raise ValueError(
            "You must define the env vars PIFF_DATA_DIR and PIFF_RUN to "
            "use Piff PSFs!")

    return piff_path


def get_psfex_path(image_path, desdata=None):
    """Get the PSFEx path from the image path.
    Parameters
    ----------
    image_path : str
        The path to the image.
    desdata : str, optional
        The path to the local DESDATA dir.
    Returns
    -------
    psfex_path : str
        The path to the psfex model.
    """
    psfex_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(image_path))), "psf")
    basename = os.path.basename(image_path)
    psfex_filename = "%s_psfexcat.psf" % ("_".join((basename.split("_"))[:-1]))
    pth = os.path.join(psfex_dir, psfex_filename)
    return _replace_desdata(pth, desdata)


def get_psfex_path_coadd(coadd_path, desdata=None):
    """Get the coadd PSFEx path from the image path.
    Parameters
    ----------
    image_path : str
        The path to the image.
    desdata : str, optional
        The path to the local DESDATA dir.
    Returns
    -------
    coadd_psfex_path : str
        The path to the coadd psfex model.
    """
    psfex_dir = os.path.join(
        os.path.dirname(os.path.dirname(coadd_path)), "psf")
    basename = os.path.basename(coadd_path)
    psfex_filename = "%s_psfcat.psf" % (basename.split(".")[0])
    pth = os.path.join(psfex_dir, psfex_filename)
    return _replace_desdata(pth, desdata)


def get_orig_coadd_file(desdata, desrun, tilename, band):
    """Get the path to the original coadd file.
    NOTE: This function will replace the NERSC DESDATA path with the input path
    if it they are not the same. This special case is useful for people who
    copy the simulation data off of the NERSC filesystem to another location.
    Parameters
    ----------
    desdata : str
        The path to the local DESDATA dir.
    desrun : str
        The DES run name.
    tilename : str
        The name of the coadd tile.
    band : str
        The desired band (e.g., 'r').
    Returns
    -------
    coadd_image_path : str
        The path to the original coadd image.
    """
    tile_data_file = os.path.join(
        desdata, desrun, tilename, "lists",
        "%s_%s_fileconf-%s.yaml" % (tilename, band, desrun))
    with open(tile_data_file, "rb") as f:
        # new pyyaml syntax
        tile_data = yaml.load(f, Loader=yaml.Loader)

    # special case here since sometimes we pull data from nersc and I cannot
    # seem to find code to remake the tile lists
    return _replace_desdata(tile_data["coadd_image_url"], desdata)


def get_output_coadd_path(desdata, desrun, tilename, band, base_dir, fz=False):
    """Get the coadd output image path.
    Parameters
    ----------
    desdata : str
        The path to the local DESDATA dir.
    desrun : str
        The DES run name.
    tilename : str
        The name of the coadd tile.
    band : str
        The desired band (e.g., 'r').
    base_dir : str
        The base path for the location of the files. Equivalent to the value
        of DESDATA but for outputs.
    fz : bool, optional
        If False, then the compression file type indicator '.fz' is removed
        from the name of the coadd file if it is present. Otherwise, the name
        of the coadd file is used as is. Default is False.
    Returns
    -------
    coadd_output_filename : str
        The name of the coadd output image file.
    """
    orig_coadd_path = get_orig_coadd_file(desdata, desrun, tilename, band)
    path_from_desdata = os.path.relpath(orig_coadd_path, desdata)
    with_fz = os.path.join(base_dir, path_from_desdata)
    if not fz:
        coadd_output_filename = (
            with_fz[:-3] if with_fz.endswith(".fits.fz") else with_fz)
    else:
        coadd_output_filename = with_fz
    return coadd_output_filename


def get_tile_center(desdata, desrun, tilename, band):
    """Get the center of the coadd tile from the coadd WCS header values.
    Parameters
    ----------
    desdata : str
        The path to the local DESDATA dir.
    desrun : str
        The DES run name.
    tilename : str
        The name of the coadd tile.
    band : str
        The desired band (e.g., 'r').
    Returns
    -------
    center : tuple of floats
        A tuple of floats with the values of ('CRVAL1', 'CRVAL2') from the
        coadd image header.
    """
    orig_coadd_file = get_orig_coadd_file(desdata, desrun, tilename, band)
    coadd_header = galsim.fits.FitsHeader(orig_coadd_file)
    return (str(coadd_header["CRVAL1"]), str(coadd_header["CRVAL2"]))


def get_truth_from_image_file(image_file, tilename):
    """Get the truth catalog path from the image path and tilename.
    Parameters
    ----------
    image_file : str
        The path to the image file.
    tilename : str
        The name of the coadd tile.
    Returns
    -------
    truth_path : str
        The path to the truth file.
    """
    return os.path.join(
        os.path.dirname(image_file),
        "truth_%s_%s.dat" % (tilename, os.path.basename(image_file)))


class Blacklist(object):
    """A class for storing a blacklist - a list of images
    to be treated differently e.g. there may not exist a
    useful accompanying Piff file
    """

    def __init__(self, blacklist_data):
        self.blacklist_data = blacklist_data

    @classmethod
    def from_file(cls, blacklist_file):
        # read the blacklist from the file
        with open(os.path.expandvars(blacklist_file), "r") as f:
            blacklist_data = yaml.load(f, Loader=yaml.Loader)
        return cls(blacklist_data)

    def img_file_is_blacklisted(self, img_file):
        """
        Determine whether an image is in the blacklist
        from its filename
        Parameters
        ----------
        img_file: str
            the image's filename
        Returns
        -------
        is_blacklisted: bool
            whether or not the image is in the blacklist
        """
        # Grab the exposure number and chip
        # number from the image filename
        img_file = os.path.basename(img_file)
        # image files have the format
        # "D<exp_num>_<band>_c<chip_num>_<other stuff>"
        exp_num = int(img_file.split("_")[0][1:])
        chip_num = int(img_file.split("_")[2][1:])
        is_blacklisted = self.is_blacklisted(exp_num, chip_num)
        return is_blacklisted

    def is_blacklisted(self, exp_num, chip_num):
        """Determine whether an image is in the blacklist
        from its exp_num and chip_num
        """
        is_blacklisted = (exp_num, chip_num) in self.blacklist_data
        return is_blacklisted
