from __future__ import print_function, absolute_import
import galsim
import galsim.des

from ..des_piff import DES_Piff


class PSFForMeds(object):
    """Wrapper to interface galsim objects w/ the MEDS making code.

    This is a class to pass the PSF used by GalSim onto the MEDSMaker class,
    which can save PSF images to the MEDS file. The meds code expects
    PSFEx objects with a get_rec method - this class gives that capability
    to galsim PSFs. The get_sigma function raises errors, but this doesn't
    matter since it is not used.

    Parameters
    ----------
    psf : galsim.GSObject or a subclass
        The galsim object to draw.
    wcs : a galsim WCS object
        The WCS to use to convert the PSF from world coordinates to image
        coordinates.
    method : string
        The galsim method to use to draw the PSF. Use 'no_pixel' with PSFs
        derived from pixel-convolved images.
    npix : int
        The number of pixels on a side for PSF image. Make sure to make it
        an odd number.

    Methods
    -------
    get_rec(row, col)
        Get a reconstruction of the PSF.
    """
    def __init__(self, psf, wcs, method, npix=53):
        self.psf = psf
        self.wcs = wcs
        self.method = method
        self.npix = npix

    def get_rec_shape(self, row, col):
        """Get the shape of the PSF image at a position.

        Parameters
        ----------
        row : float
            The row at which to get the PSF image in the stamp in
            zero-offset image coordinates.
        col : float
            The col at which to get the PSF image in the stamp in
            zero-offset image coordinates.

        Returns
        -------
        psf_shape : tuple of ints
            The shape of the PSF image.
        """
        return (self.npix, self.npix)

    def get_rec(self, row, col):
        """Get the PSF at a position.

        Parameters
        ----------
        row : float
            The row at which to get the PSF image in the stamp in
            zero-offset image coordinates.
        col : float
            The col at which to get the PSF image in the stamp in
            zero-offset image coordinates.

        Returns
        -------
        psf : np.ndarray, shape (npix, npix)
            An image of the PSF.
        """
        # we add 1 to the positions here since the MEDS code uses
        # zero offset positions and galsim + DES stuff expects one-offset
        im_pos = galsim.PositionD(col+1, row+1)
        wcs = self.wcs.local(im_pos)
        if isinstance(self.psf, galsim.GSObject):
            psf_im = self.psf.drawImage(
                nx=self.npix, ny=self.npix, wcs=wcs, method=self.method).array
        elif isinstance(self.psf, galsim.des.DES_PSFEx):
            psf_at_pos = self.psf.getPSF(im_pos)
            psf_im = psf_at_pos.drawImage(
                wcs=wcs, nx=self.npix, ny=self.npix, method='no_pixel').array
        elif isinstance(self.psf, DES_Piff):
            psf_at_pos = self.psf.getPSF(im_pos, wcs)
            psf_im = psf_at_pos.drawImage(
                wcs=wcs, nx=self.npix, ny=self.npix, method='auto').array
        else:
            raise ValueError(
                'We did not recognize the PSF type! %s' % self.psf)

        # commented out to make sure this is never done
        # usually this does not help anything
        # leaving notes here for the scientists of the future
        # if self.snr is not None:
        #     npix = psf_im.shape[0] * psf_im.shape[1]
        #     sigma = psf_im.sum() / (self.snr * npix**0.5)
        #     print("adding psf noise, final psf s/n = %f" % (
        #         psf_im.sum()/sigma**2 / np.sqrt(npix/sigma**2)))
        #     noise = np.random.normal(scale=sigma, size=psf_im.shape)
        #     psf_im += noise

        return psf_im

    def get_center(self, row, col):
        """Get the center of the PSF in the stamp/cutout.

        Parameters
        ----------
        row : float
            The row at which to get the PSF center in the stamp in
            zero-offset image coordinates.
        col : float
            The col at which to get the PSF center in the stamp in
            zero-offset image coordinates.

        Returns
        -------
        cen : 2-tuple of floats
            The center of the PSF in zero-offset image coordinates.
        """
        return (self.npix-1.)/2., (self.npix-1.)/2.

    def get_sigma(self, row, col):
        # note this used to return -99
        raise NotImplementedError()
