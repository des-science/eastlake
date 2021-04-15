import os
import yaml


def _replace_desdata(pth, desdata):
    """Replace the NERSC DESDATA path if needed.

    Parameters
    ----------
    pth : str
        The path string on which to do replacement.
    desdata : str
        The desired DESDATA. If None, then the path is simply returned as is.

    Returns
    -------
    pth : str
        The path, possible with DESDATA in the path replaced with the desired
        one.
    """
    if desdata is None:
        return pth

    nersc_desdata = '/global/project/projectdirs/des/y3-image-sims'
    if (nersc_desdata in pth and
            os.path.normpath(desdata) != os.path.normpath(nersc_desdata)):
        return pth.replace(nersc_desdata, desdata)
    else:
        return pth


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
