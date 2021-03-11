#Tile setup
import galsim
import os
import yaml
import numpy as np
from coord import CelestialCoord

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


def get_orig_source_list_file(desdata, desrun, tilename, band):
    """Build path to the source file list.
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
    source_file : str
        The path to the source file list.
    """
    source_file = os.path.join(
        desdata, desrun,
        tilename, "lists",
        "%s_%s_fcut-flist-%s.dat" % (tilename, band, desrun))
    return source_file

def get_orig_se_image_files_and_magzps(desdata, desrun, tilename, band):
    """Get list of paths to single-epoch image files
    for a given tilename and band.
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
    image_files: list
        List of image files.
    """
    #Get source list file
    source_list_file = get_orig_source_list_file(desdata, desrun, tilename, band)
    # read in the DESDM dile list and patse the parts
    # we remove empty lines
    # the first item is the file name and the second is the
    # magnitude zero point
    with open(source_list_file, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if l.strip() != ""]
    image_files = [_replace_desdata(l.split()[0], desdata) for l in lines]
    mag_zps = [float(l.split()[1]) for l in lines]
    return image_files, mag_zps

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

class Tile(dict):
    """
    Class for storing tile info
    """
    def __init__(self, tile_info):
        self.update(tile_info)

    @classmethod
    def from_tilename(cls, tilename, desrun='y3v02', desdata=None,
                 bands=["g","r","i","z"]):

        if desdata is None:
            desdata = os.environ["DESDATA"]
        
        tile_data = {}
        tile_data["tilename"] = tilename
        tile_data["desrun"] = desrun
        tile_data["desdata"] = desdata
        tile_data["bands"] = bands

        #Set up lists - useful for looping through in galsim
        band_list = []
        im_file_list = []
        mag_zp_list = []
        psfex_file_list = []
        piff_file_list = []
        output_filenames = []

        coadd_file_list = []
        coadd_output_filenames = []
        coadd_band_list = []
        coadd_psfex_file_list = []
        coadd_mag_zp_list = []

        #Collect corners of all se images to get bounds for
        #tile
        se_image_corners_deg = {}
        all_corners = []

        # Get single-epoch image info for this tile
        for band in bands:
            #Get the image files and mag zeropoints
            image_files, mag_zps = get_orig_se_image_files_and_magzps(
                tile_data["desdata"], tile_data["desrun"], tilename, band)
            im_file_list += image_files
            mag_zp_list += mag_zps

            # coadd stuff
            orig_coadd_file = get_orig_coadd_file(
                tile_data["desdata"], tile_data["desrun"],
                tilename, band)
            coadd_file_list.append(orig_coadd_file)
            coadd_band_list.append(band)

            # read coadd magzero from hdr
            coadd_header = galsim.fits.FitsHeader(orig_coadd_file)
            coadd_mag_zp_list.append(coadd_header["MAGZERO"])
            # and also get coadd psfex file
            coadd_psfex_path = get_psfex_path_coadd(
                orig_coadd_file, desdata=desdata)
            coadd_psfex_file_list.append(coadd_psfex_path)

            # Get coadd center
            coadd_center_ra = coadd_header["RA_CENT"]*galsim.degrees
            coadd_center_dec = coadd_header["DEC_CENT"]*galsim.degrees
            coadd_center = CelestialCoord(ra=coadd_center_ra,
                                          dec=coadd_center_dec)
            
            #get corners in ra/dec for each image
            #we'll use these to set the bounds in which to simulate
            #objects
            se_image_corners_deg[band] = []
            for f in image_files:
                ext=1 if f.endswith(".fz") else 0
                wcs, origin = galsim.wcs.readFromFitsHeader(f)
                xsize, ysize = 2048, 4096
                im_pos1 = origin
                im_pos2 = origin + galsim.PositionD(xsize,0)
                im_pos3 = origin + galsim.PositionD(xsize,ysize)
                im_pos4 = origin + galsim.PositionD(0,ysize)
                corners = [wcs.toWorld(im_pos1),
                           wcs.toWorld(im_pos2),
                           wcs.toWorld(im_pos3),
                           wcs.toWorld(im_pos4)]
                #wrap w.r.t coadd center - this should ensure things don't confusingly
                #cross ra=0.
                corners = [CelestialCoord(ra=c.ra.wrap(center=coadd_center.ra), dec=c.dec) for c in corners]
                corners_deg = [(c.ra/galsim.degrees,c.dec/galsim.degrees) for c in corners]
                se_image_corners_deg[band].append(corners_deg)
                all_corners += corners

            # also get psfex file here
            psfex_files = [
                get_psfex_path(f, desdata=desdata)
                for f in image_files]
            psfex_file_list += psfex_files

            # get piff stuff here
            piff_files = [
                get_piff_path(f)
                for f in image_files]
            piff_file_list += piff_files

            # fill out indexing info for bands, tile_nums and tilenames
            band_list += band * len(image_files)


        #The tile is 10000x10000 pixels, but the extent for images
        #contributing can be 4096 greater in the + and - x-direction
        #and 2048 pixels greater in the + and - y-direction.
        tile_npix_x = 10000 + 4096*2
        tile_npix_y = 10000 + 2048*2
        tile_dec_min = (coadd_center_dec/galsim.degrees - tile_npix_y/2 * coadd_header["CD2_2"])*galsim.degrees
        tile_dec_max = (coadd_center_dec/galsim.degrees + tile_npix_y/2 * coadd_header["CD2_2"])*galsim.degrees
        tile_dec_ranges_deg = (tile_dec_min/galsim.degrees, tile_dec_max/galsim.degrees)

        #Need to be careful with ra, since depending on convention it increases in 
        #the opposite direction to pixel coordinates, and there may be issues with
        #wrapping around 0.
        tile_ra_min = (coadd_center.ra/galsim.degrees
                        - tile_npix_x/2 * coadd_header["CD1_1"]
                       / np.cos(coadd_center_dec/galsim.radians))*galsim.degrees
        tile_ra_max = (coadd_center.ra/galsim.degrees
                       + tile_npix_x/2 * coadd_header["CD1_1"]
                       / np.cos(coadd_center_dec/galsim.radians))*galsim.degrees
        
        #make sure these are between coadd_center.ra-pi and coadd_center.ra+pi with the wrap function
        tile_ra_min = tile_ra_min.wrap(center=coadd_center.ra)
        tile_ra_max = tile_ra_max.wrap(center=coadd_center.ra)
        tile_ra_ranges_deg = (tile_ra_min/galsim.degrees,
                              tile_ra_max/galsim.degrees)
        #and set the ordering in tile_ra_ranges_deg according to which is larger
        if tile_ra_ranges_deg[0]>tile_ra_ranges_deg[1]:
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
        coadd_corners_deg = [(ra,dec) for (ra,dec) in zip(coadd_corners_ra_deg,
                                                          coadd_corners_dec_deg)]
        coadd_ra_ranges_deg = (np.min(coadd_corners_ra_deg), np.max(coadd_corners_ra_deg))
        coadd_dec_ranges_deg = (np.min(coadd_corners_dec_deg), np.max(coadd_corners_dec_deg))

        #As well as the coadd bounds in ra/dec, it's also useful to store the bounds
        #of the single-epoch images which enter the tile, since we may wish to
        #simulate objects across this larger area. Get this from the corners
        #of the se images we collected above
        all_corners_ra_deg = [c.ra.wrap(all_corners[0].ra) / galsim.degrees
                             for c in all_corners]
        all_corners_dec_deg = [c.dec / galsim.degrees
                           for c in all_corners]
        se_ra_ranges_deg = (np.min(all_corners_ra_deg), np.max(all_corners_ra_deg))
        se_dec_ranges_deg = (np.min(all_corners_dec_deg), np.max(all_corners_dec_deg))

        #Also record area - useful if we want to generate objects with a given
        #number density
        ra0,ra1 = se_ra_ranges_deg[0], se_ra_ranges_deg[1]
        dec0,dec1 = se_dec_ranges_deg[0], se_dec_ranges_deg[1]
        se_area = (ra1 - ra0)*(np.sin(dec1) - np.sin(dec0))
        tile_data["se_area"] = se_area
        
        tile_data["tile_center"] = coadd_center
        tile_data["image_files"] = im_file_list
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

        return cls(tile_data)

    def write(self, filename, overwrite=False):
        #Write to a picke file
        if os.path.exists(filename):
            raise OSError("""output filename %d exists and
overwrite is False""")
        with open(filename, "w") as f:
            pickle.dump(f, self)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as f:
            tile_data = pickle.load(f)
        return cls(tile_data)
