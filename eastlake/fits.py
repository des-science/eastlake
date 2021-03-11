import astropy.io.fits as pyfits
from galsim import Image, GalSimIncompatibleValuesError, GalSimValueError
from galsim.fits import _parse_compression, _add_hdu, _write_file

#The below function is copied from galsim.fits, but I've added a header argument
#to allow saving with a header
def writeMulti(image_list, file_name=None, dir=None, hdu_list=None, clobber=True,
               compression='auto', header_list=None):
    """Write a Python list of images to a multi-extension FITS file.
    The details of how the images are written to file depends on the arguments.
    @param image_list   A Python list of Images.  (For convenience, some items in this list
                        may be HDUs already.  Any Images will be converted into pyfits HDUs.)
    @param file_name    The name of the file to write to.  [Either `file_name` or `hdu_list` is
                        required.]
    @param dir          Optionally a directory name can be provided if `file_name` does not
                        already include it. [default: None]
    @param hdu_list     A pyfits HDUList.  If this is provided instead of `file_name`, then the
                        image is appended to the end of the HDUList as a new HDU. In that case,
                        the user is responsible for calling either `hdu_list.writeto(...)` or
                        `galsim.fits.writeFile(...)` afterwards.  [Either `file_name` or `hdu_list`
                        is required.]
    @param clobber      See documentation for this parameter on the galsim.fits.write() method.
    @param compression  See documentation for this parameter on the galsim.fits.write() method.
    @param header_list  List of fits headers (one for each image)
    """

    if any(image.iscomplex for image in image_list if isinstance(image, Image)):
        raise GalSimValueError("Cannot write complex Images to a fits file. "
                               "Write image.real and image.imag separately.", image_list)

    file_compress, pyfits_compress = _parse_compression(compression,file_name)

    if file_name and hdu_list is not None:
        raise GalSimIncompatibleValuesError(
            "Cannot provide both file_name and hdu_list", file_name=file_name, hdu_list=hdu_list)
    if not (file_name or hdu_list is not None):
        raise GalSimIncompatibleValuesError(
            "Must provide either file_name or hdu_list", file_name=file_name, hdu_list=hdu_list)

    if hdu_list is None:
        hdu_list = pyfits.HDUList()

    for i,image in enumerate(image_list):
        if isinstance(image, Image):
            hdu = _add_hdu(hdu_list, image.array, pyfits_compress)
            if image.wcs:
                image.wcs.writeToFitsHeader(hdu.header, image.bounds)
            if header_list is not None:
                hdu.header.extend( header_list[i] )
        else:
            # Assume that image is really an HDU.  If not, this should give a reasonable error
            # message.  (The base type of HDUs vary among versions of pyfits, so it's hard to
            # check explicitly with an isinstance call.  For newer pyfits versions, it is
            # pyfits.hdu.base.ExtensionHDU, but not in older versions.)
            hdu_list.append(image)

    if file_name:
        _write_file(file_name, dir, hdu_list, clobber, file_compress, pyfits_compress)
