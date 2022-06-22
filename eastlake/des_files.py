import os

import yaml
import galsim
import numpy as np
from coord import CelestialCoord

MAGZP_REF = 30.0
PIZZA_CUTTER_YAML_PATH_KEYS = [
    "bmask_path",
    "cat_path",
    "gaia_stars_file",
    "image_path",
    "psf_path",
    "seg_path",
    "bkg_path",
    "head_path",
    "piff_path",
    "psfex_path",
    "weight_path",
]


def replace_imsim_data(
    pth,
    imsim_data,
    old_imsim_data=None,
):
    """Replace the IMSIM_DATA path if needed.

    Parameters
    ----------
    pth : str
        The path string on which to do replacement.
    imsim_data : str
        The desired IMSIM_DATA.
    old_imsim_data : str, optional
        The IMSIM_DATA path to be replaced. If None, it will be inferred by splitting
        on "/sources-" and using the first part. If the old path cannot be inferred,
        will raise an error.

    Returns
    -------
    pth : str
        The path, possible with IMSIM_DATA in the path replaced with the desired
        one.
    """
    if old_imsim_data is None:
        # try to infer
        # paths look like this
        # /Users/beckermr/MEDS_DIR/test-y6-sims/DES0131-3206/sources-r/OPS/
        parts = pth.split("/sources-")
        if (
            len(parts) == 2
            and parts[1][1:6] in ["/OPS/", "/ACT/"]
            and os.path.basename(parts[0]).startswith("DES")
        ):
            old_imsim_data = os.path.dirname(os.path.dirname(parts[0]))
        else:
            raise RuntimeError("Could not infer imsim data from the path %s!" % pth)

    if (
        old_imsim_data in pth
        and os.path.normpath(imsim_data) != os.path.normpath(old_imsim_data)
    ):
        return os.path.join(imsim_data, os.path.relpath(pth, start=old_imsim_data))
    else:
        return pth


def get_pizza_cutter_yaml_path(imsim_data, desrun, tilename, band):
    """Return the path to a pizza cutter info file.

    Parameters
    ----------
    imsim_data : str
        The path to the local IMSIM_DATA dir.
    desrun : str
        The DES run name.
    tilename : str
        The name of the coadd tile.
    band : str
        The desired band (e.g., 'r').

    Returns
    -------
    pth : str
        The path to the file.
    """
    return os.path.join(
        imsim_data, desrun, "pizza_cutter_info",
        f"{tilename}_{band}_pizza_cutter_info.yaml",
    )


def replace_imsim_data_in_pizza_cutter_yaml(
    band_info, output_imsim_data, old_imsim_data=None
):
    """Replace the IMSIM_DATA path in a pizza cutter yaml file.

    **This function operates in-place!**

    Parameters
    ----------
    band_info : dict
        The pizza cutter info.
    output_imsim_data : str
        The desired IMSIM_DATA.
    old_imsim_data : str, optional
        The IMSIM_DATA path to be replaced. If None, it will be inferred by splitting
        on "/sources-" and using the first part. If the old path cannot be inferred,
        will raise an error.
    """

    for key in PIZZA_CUTTER_YAML_PATH_KEYS:
        if key in band_info:
            band_info[key] = replace_imsim_data(
                band_info[key], output_imsim_data,
                old_imsim_data=old_imsim_data,
            )
    for i in range(len(band_info["src_info"])):
        for key in PIZZA_CUTTER_YAML_PATH_KEYS:
            if key in band_info["src_info"][i]:
                band_info["src_info"][i][key] = replace_imsim_data(
                    band_info["src_info"][i][key], output_imsim_data,
                    old_imsim_data=old_imsim_data,
                )


def read_pizza_cutter_yaml(imsim_data, desrun, tilename, band):
    """Read the pizza-cutter YAML file for this tile and band.

    Parameters
    ----------
    imsim_data : str
        The path to the local IMSIM_DATA dir.
    desrun : str
        The DES run name.
    tilename : str
        The name of the coadd tile.
    band : str
        The desired band (e.g., 'r').

    Returns
    -------
    info : dict
        The information for this tile.
    """
    pth = get_pizza_cutter_yaml_path(imsim_data, desrun, tilename, band)
    with open(pth, "r") as fp:
        band_info = yaml.safe_load(fp)

    replace_imsim_data_in_pizza_cutter_yaml(band_info, imsim_data)

    return band_info


def get_tile_center(coadd_file):
    """Get the center of the coadd tile from the coadd WCS header values.

    Parameters
    ----------
    coadd_file : str
        The path the coadd file to read.

    Returns
    -------
    center : tuple of floats
        A tuple of floats with the values of ('CRVAL1', 'CRVAL2') from the
        coadd image header.
    """
    coadd_header = galsim.fits.FitsHeader(coadd_file)
    return (str(coadd_header["CRVAL1"]), str(coadd_header["CRVAL2"]))


class Tile(dict):
    """
    Class for storing tile info.

    To instantiate, use the from_tilename method.
    """
    def __init__(self, tile_info):
        self.update(tile_info)

    @classmethod
    def from_tilename(
        cls, tilename, bands=["g", "r", "i", "z"], desrun="y6-image-sims", imsim_data=None,
    ):
        """Create a Tile from a tilename and set of bands.

        Parameters
        ----------
        tilename : str
            The DES tilename.
        bands : list of str, optional
            The bands to use. Default is griz.
        desrun : str, optional
            The MEDS file version identifier. Default is "y6-image-sims".
        imsim_data : str, optional
            The local IMSIM_DATA directory with the data. Default of None
            reads this value from the environment.

        Returns
        -------
        tile_info : Tile
            The tile information for the tile and bands.
        """
        if imsim_data is None:
            imsim_data = os.environ["IMSIM_DATA"]

        tile_data = {}
        tile_data["tilename"] = tilename
        tile_data["desrun"] = desrun
        tile_data["imsim_data"] = imsim_data
        tile_data["bands"] = bands

        # Set up lists - useful for looping through in galsim
        band_list = []
        im_file_list = []
        bkg_file_list = []
        mag_zp_list = []
        psfex_file_list = []
        piff_file_list = []
        output_filenames = []
        pizza_cutter_yaml = {}

        coadd_file_list = []
        coadd_output_filenames = []
        coadd_band_list = []
        coadd_psfex_file_list = []
        coadd_mag_zp_list = []

        # Collect corners of all se images to get bounds for
        # tile
        se_image_corners_deg = {}
        all_corners = []

        # Get single-epoch image info for this tile
        for band in bands:
            # read band information for the tile
            band_info = read_pizza_cutter_yaml(imsim_data, desrun, tilename, band)
            pizza_cutter_yaml[band] = band_info

            # Get the image files and mag zeropoints
            image_files = []
            mag_zps = []
            for src in band_info["src_info"]:
                image_files.append(src["image_path"])
                mag_zps.append(src["magzp"])
            im_file_list += image_files
            mag_zp_list += mag_zps

            # coadd stuff
            coadd_file_list.append(band_info["image_path"])
            coadd_band_list.append(band)

            # read coadd magzero from hdr
            coadd_header = galsim.fits.FitsHeader(band_info["image_path"])
            coadd_mag_zp_list.append(coadd_header["MAGZERO"])
            assert np.allclose(coadd_header["MAGZERO"], MAGZP_REF)
            # and also get coadd psfex file
            coadd_psfex_file_list.append(band_info["psf_path"])

            # Get coadd center
            coadd_center_ra = float(coadd_header["CRVAL1"])*galsim.degrees
            coadd_center_dec = float(coadd_header["CRVAL2"])*galsim.degrees
            coadd_center = CelestialCoord(ra=coadd_center_ra,
                                          dec=coadd_center_dec)

            # get corners in ra/dec for each image
            # we'll use these to set the bounds in which to simulate
            # objects
            se_image_corners_deg[band] = []
            for f in image_files:
                # ext = 1 if f.endswith(".fz") else 0
                wcs, origin = galsim.wcs.readFromFitsHeader(f)
                xsize, ysize = 2048, 4096
                im_pos1 = origin
                im_pos2 = origin + galsim.PositionD(xsize, 0)
                im_pos3 = origin + galsim.PositionD(xsize, ysize)
                im_pos4 = origin + galsim.PositionD(0, ysize)
                corners = [wcs.toWorld(im_pos1),
                           wcs.toWorld(im_pos2),
                           wcs.toWorld(im_pos3),
                           wcs.toWorld(im_pos4)]
                # wrap w.r.t coadd center - this should ensure things don't confusingly
                # cross ra=0.
                corners = [CelestialCoord(ra=c.ra.wrap(
                    center=coadd_center.ra), dec=c.dec) for c in corners]
                corners_deg = [(c.ra/galsim.degrees, c.dec/galsim.degrees)
                               for c in corners]
                se_image_corners_deg[band].append(corners_deg)
                all_corners += corners

            # also get psfex file here
            psfex_files = [
                src["psf_path"]
                for src in band_info["src_info"]
            ]
            psfex_file_list += psfex_files

            # get piff stuff here
            piff_files = [
                src["piff_path"]
                for src in band_info["src_info"]
            ]
            piff_file_list += piff_files

            bkg_file_list += [
                src["bkg_path"]
                for src in band_info["src_info"]
            ]

            # fill out indexing info for bands, tile_nums and tilenames
            band_list += band * len(image_files)

        # The tile is 10000x10000 pixels, but the extent for images
        # contributing can be 4096 greater in the + and - x-direction
        # and 2048 pixels greater in the + and - y-direction.
        tile_npix_x = 10000 + 4096*2
        tile_npix_y = 10000 + 2048*2
        tile_dec_min = (coadd_center_dec/galsim.degrees - tile_npix_y /
                        2 * coadd_header["CD2_2"])*galsim.degrees
        tile_dec_max = (coadd_center_dec/galsim.degrees + tile_npix_y /
                        2 * coadd_header["CD2_2"])*galsim.degrees
        tile_dec_ranges_deg = (tile_dec_min/galsim.degrees, tile_dec_max/galsim.degrees)

        # Need to be careful with ra, since depending on convention it increases in
        # the opposite direction to pixel coordinates, and there may be issues with
        # wrapping around 0.
        tile_ra_min = (coadd_center.ra/galsim.degrees
                       - tile_npix_x/2 * coadd_header["CD1_1"]
                       / np.cos(coadd_center_dec/galsim.radians))*galsim.degrees
        tile_ra_max = (coadd_center.ra/galsim.degrees
                       + tile_npix_x/2 * coadd_header["CD1_1"]
                       / np.cos(coadd_center_dec/galsim.radians))*galsim.degrees

        # make sure these are between coadd_center.ra-pi and coadd_center.ra+pi with
        # the wrap function
        tile_ra_min = tile_ra_min.wrap(center=coadd_center.ra)
        tile_ra_max = tile_ra_max.wrap(center=coadd_center.ra)
        tile_ra_ranges_deg = (tile_ra_min/galsim.degrees,
                              tile_ra_max/galsim.degrees)
        # and set the ordering in tile_ra_ranges_deg according to which is larger
        if tile_ra_ranges_deg[0] > tile_ra_ranges_deg[1]:
            tile_ra_ranges_deg = (tile_ra_ranges_deg[1],
                                  tile_ra_ranges_deg[0])

        coadd_wcs, origin = galsim.wcs.readFromFitsHeader(coadd_header)
        xmin, xmax, ymin, ymax = 1., 10000., 1., 10000.
        coadd_corners = [
            coadd_wcs.toWorld(galsim.PositionD(a, b))
            for (a, b) in [
                (xmin, ymin), (xmax, ymin),
                (xmax, ymax), (xmin, ymax)]]
        coadd_corners_ra_deg = [c.ra.wrap(corners[0].ra) / galsim.degrees
                                for c in coadd_corners]
        coadd_corners_dec_deg = [c.dec / galsim.degrees
                                 for c in coadd_corners]
        coadd_corners_deg = [(ra, dec) for (ra, dec) in zip(coadd_corners_ra_deg,
                                                            coadd_corners_dec_deg)]
        coadd_ra_ranges_deg = (np.min(coadd_corners_ra_deg),
                               np.max(coadd_corners_ra_deg))
        coadd_dec_ranges_deg = (np.min(coadd_corners_dec_deg),
                                np.max(coadd_corners_dec_deg))

        # As well as the coadd bounds in ra/dec, it's also useful to store the bounds
        # of the single-epoch images which enter the tile, since we may wish to
        # simulate objects across this larger area. Get this from the corners
        # of the se images we collected above
        all_corners_ra_deg = [c.ra.wrap(all_corners[0].ra) / galsim.degrees
                              for c in all_corners]
        all_corners_dec_deg = [c.dec / galsim.degrees
                               for c in all_corners]
        se_ra_ranges_deg = (np.min(all_corners_ra_deg), np.max(all_corners_ra_deg))
        se_dec_ranges_deg = (np.min(all_corners_dec_deg), np.max(all_corners_dec_deg))

        # Also record area - useful if we want to generate objects with a given
        # number density
        ra0, ra1 = se_ra_ranges_deg[0], se_ra_ranges_deg[1]
        dec0, dec1 = se_dec_ranges_deg[0], se_dec_ranges_deg[1]
        se_area = (ra1 - ra0)*(np.sin(dec1) - np.sin(dec0))
        tile_data["se_area"] = se_area

        tile_data["tile_center"] = coadd_center
        tile_data["image_files"] = im_file_list
        tile_data["bkg_files"] = bkg_file_list
        tile_data["mag_zp_list"] = mag_zp_list
        tile_data["psfex_files"] = psfex_file_list
        tile_data['piff_files'] = piff_file_list
        tile_data["coadd_ra_ranges_deg"] = coadd_ra_ranges_deg
        tile_data["coadd_dec_ranges_deg"] = coadd_dec_ranges_deg
        tile_data["tile_ra_ranges_deg"] = tile_ra_ranges_deg
        tile_data["tile_dec_ranges_deg"] = tile_dec_ranges_deg
        tile_data["coadd_corners_deg"] = coadd_corners_deg
        tile_data["se_ra_ranges_deg"] = se_ra_ranges_deg
        tile_data["se_dec_ranges_deg"] = se_dec_ranges_deg
        tile_data["se_image_corners_deg"] = se_image_corners_deg
        tile_data["band_list"] = band_list
        tile_data["output_file_list"] = output_filenames
        tile_data["coadd_wcs"] = coadd_wcs

        # coadd stuff
        tile_data["coadd_file_list"] = coadd_file_list
        tile_data["coadd_output_file_list"] = coadd_output_filenames
        tile_data["coadd_psfex_files"] = coadd_psfex_file_list
        tile_data["coadd_mag_zp_list"] = coadd_mag_zp_list
        tile_data["coadd_band_list"] = coadd_band_list

        # pizza cutter yaml
        tile_data["pizza_cutter_yaml"] = pizza_cutter_yaml

        return cls(tile_data)
