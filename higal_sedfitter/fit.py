from __future__ import print_function
import os
import glob

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.utils.console import ProgressBar
import dust_emissivity

import higal_beams

__all__ = ['PixelFitter',
           'fit_modified_blackbody_tofiles',
           'fit_modified_blackbody_to_imagecube']

class PixelFitter(object):
    
    def __init__(self, tguess=20, bguess=1.75, nguess=1e22,
                 trange=[2.73,50], brange=[1,3], nrange=[1e20,1e25],
                 tfixed=False, bfixed=False, nfixed=False):
        """
        Initialize an SED fitter instance with a set of guesses.  
        The input parameters follow a template that is the same for each
        of temperature, beta, and column.

        Once initialized, `PixelFitter` can be called as a function of
        frequency (Hz), flux (MJy), and error (MJy)


        Parameters
        ----------
        guess : float
            The guessed value for T,N, or beta.  Temperature in Kelvin,
            column in :math:`cm^{-2}`
        range : tuple or list of length 2
            The minimum/maximum values of each parameter
        fixed : bool
            Is the parameter fixed at the guessed value?
        """
        
        import lmfit
        from collections import OrderedDict

        parlist = [(n,lmfit.Parameter(name=n,value=x))
                   for n,x in zip(('T','beta','N'),
                                  (tguess, bguess, nguess))]

        parameters = lmfit.Parameters(OrderedDict(parlist))

        parameters['beta'].vary = not bfixed
        parameters['beta'].min = brange[0]
        parameters['beta'].max = brange[1]

        parameters['T'].vary = not tfixed
        parameters['T'].min = trange[0]
        parameters['T'].max = trange[1]

        parameters['N'].vary = not nfixed
        parameters['N'].min = nrange[0]
        parameters['N'].max = nrange[1]

        self.parameters = parameters

    def __call__(self, frequency, flux, err):
        """
        Perform the fit and return a tuple of values & errors

        Parameters
        ----------
        frequency : `~astropy.units.quantity.Quantity`
            An array of frequencies to be converted to Hz and passed to the
            fitter
        flux : `~astropy.units.quantity.Quantity`
            An array of `u.MJy` equivalent flux values
        err : `~astropy.units.quantity.Quantity`
            An array of `u.MJy` equivalent flux values that specify the errors
            on flux
        """
        fitter = dust_emissivity.fit_sed.fit_sed_lmfit_hz
        lm = fitter(frequency.to(u.Hz, u.spectral()).value,
                    flux.to(u.erg/u.cm**2/u.s/u.Hz).value,
                    err=err.to(u.erg/u.cm**2/u.s/u.Hz).value,
                    guesses=self.parameters,
                    blackbody_function='modified')
        self.lm = lm
        self.vals = (lm.params['T'].value, lm.params['beta'].value,
                     lm.params['N'].value)
        self.errs = (lm.params['T'].stderr, lm.params['beta'].stderr,
                     lm.params['N'].stderr)

        return self.vals,self.errs

    def integral(self, fmin, fmax):
        """
        Compute the integral of the currently-fitted parameters

        Parameters
        ----------
        fmin, fmax : `~astropy.units.quantity.Quantity`
            Frequency-equivalent start and end points for the integral
        """
        integrator = dust_emissivity.blackbody.integrate_sed
        return integrator(fmin, fmax,
                          function=dust_emissivity.blackbody.modified_blackbody,
                          temperature=self.lm.params['T'].value*u.K,
                          column=self.lm.params['N'].value*u.cm**-2,
                          beta=self.lm.params['beta'].value).value



def fit_modified_blackbody_tofiles(filename_template,
                                   wavelengths=[70,160,250,350,500],
                                   bad_value=0,
                                   name_to_um=higal_beams.name_to_um,
                                   **kwargs
                                   ):
    """
    Fit a modified blackbody to each pixel in a set of files identified using a
    filename template

    Parameters
    ----------
    filename_template : str
        A filename template that accepts the HiGal wavelength strings, e.g.:
            ``destripe_l000_{0}_reg.fits``
        would format to
            ``destripe_l000_PMW_reg.fits``
        Wildcards are allowed, so you can also do
            ``HIGAL*_{0}_RM_smregrid45.fits``
    wavelengths : list
        The wavelengths, in microns, to include in the fit
    bad_value : float
        A value to mark as bad and ignore.  Some files have NaNs indicating bad
        points, others have zeros - this is to account for bad pixels that have
        values of zero
    name_to_um : dict
        A dictionary identifying the translation between the string that will
        be inserted into the file template and the wavelength.  There are two
        built in: `higal_beams.name_to_um` and `higal_beams.num_to_um`.
    kwargs : dict
        passed to `fit_modified_blackbody_toimagecube`
    """

    target_files = {name_to_um[x]: filename_template.format(x)
                    for x in name_to_um}
    wavelengths_sorted = sorted(wavelengths)

    for k in target_files:
        if not os.path.exists(target_files[k]):
            G = glob.glob(target_files[k])
            if len(G) == 1:
                target_files[k] = G[0]
            else:
                raise IOError("File {0} does not exist".format(target_files[k]))

    # Make an image cube appropriately sorted by wavelength
    image_cube = np.array([fits.getdata(target_files[wl])
                           for wl in wavelengths_sorted])
    image_cube[image_cube == bad_value] = np.nan

    # wl should be defined from above; the headers should (in principle) be
    # identical
    outheader = fits.getheader(target_files[wl])

    return fit_modified_blackbody_to_imagecube(image_cube, outheader,
                                               wavelengths, **kwargs)

def fit_modified_blackbody_to_imagecube(image_cube,
                                        outheader,
                                        wavelengths=[70,160,250,350,500],
                                        error_scaling=0.2,
                                        pixelfitter=None, ncores=4,
                                        clobber=True,
                                        integral=False, out_prefix="",):
    """
    Fit a modified blackbody to each pixel in an image cube.  Writes the
    results to files of the form ``{out_prefix}+T.fits``, ``{out_prefix}+beta.fits``, 
    ``{out_prefix}+N.fits``,  and optionally ``{out_prefix}+integral.fits``.

    Parameters
    ----------
    image_cube : `~numpy.ndarray`
        A cube constructed from the individual wavelengths of the Herschel image
    wavelengths : list
        The wavelengths, in microns, to include in the fit
    error_scaling : float or None
        The amount to scale the input fluxes by to determine the errors
    pixelfitter : :class:`PixelFitter` or None
        An instance of the :class:`PixelFitter` class to use for the fitting
        (this is how guesses are specified).  If None, will use defaults.
    ncores : int
        OPTIONAL / NOT PRESENTLY IMPLEMENTED
        Allows parallelization
    clobber : bool
        Overwrite existing output files?
    integral : bool
        Also include the integral of the modified blackbody, e.g. for
        luminosity determination?  This increases the execution time by
        a factor of 2
    out_prefix : str
        A prefix to prepend to the output file names
        
    Returns
    -------
    t_hdu,b_hdu,n_hdu : :class:`~astropy.io.fits.HDUList`
        HDUlists incorporating the best-fit values as the
        `~astropy.io.fits.PrimaryHDU` image and the errors
        as `~astropy.io.fits.ImageHDU`s in the first (ERROR)
        extension
    int_hdu : :class:`~astropy.io.fits.PrimaryHDU`
        (optional) An HDU containing an image of the integral
        in :math:`erg/s/cm^2`
    """

    if pixelfitter is None:
        pixelfitter = PixelFitter()

    wavelengths_sorted = sorted(wavelengths)

    # Only fit pixels with no NaNs
    ok_to_fit = (np.isnan(image_cube)).max(axis=0) == 0
    okcount = np.count_nonzero(ok_to_fit)
    if okcount == 0:
        raise ValueError("No valid pixels found.")
    okx,oky = np.where(ok_to_fit)

    frequencies = (wavelengths_sorted*u.um).to(u.Hz, u.spectral())

    timg, bimg, nimg = [np.empty(ok_to_fit.shape)+np.nan for ii in range(3)]
    terr, berr, nerr = [np.empty(ok_to_fit.shape)+np.nan for ii in range(3)]
    if integral:
        intimg = np.empty(ok_to_fit.shape)+np.nan

    def fitter(xy):
        x,y = xy
        vals,errs = pixelfitter(frequencies, image_cube[:, x, y]*u.MJy,
                                image_cube[:,x,y]*error_scaling*u.MJy)
        timg[x,y] = vals[0]
        bimg[x,y] = vals[1]
        nimg[x,y] = vals[2]

        terr[x,y] = errs[0]
        berr[x,y] = errs[1]
        nerr[x,y] = errs[2]

        if integral:
            intimg[x,y] = pixelfitter.integral(1*u.cm, 1*u.um)

        return vals,errs

    # Parallelized version: currently doesn't work =(
    #with FITS_tools.cube_regrid._map_context(ncores) as map:
    #   result = map(fitter, zip(okx,oky))

    pb = ProgressBar(okcount)
    for xy in zip(okx,oky):
        pb.update()
        fitter(xy)

    t_hdu = fits.HDUList([fits.PrimaryHDU(data=timg, header=outheader),
                          fits.ImageHDU(data=terr, header=outheader, name='ERROR')])
    t_hdu[0].header['BUNIT'] = 'K'
    t_hdu[1].header['BUNIT'] = 'K'
    b_hdu = fits.HDUList([fits.PrimaryHDU(data=bimg, header=outheader),
                          fits.ImageHDU(data=berr, header=outheader, name='ERROR')])
    b_hdu[0].header['BUNIT'] = ''
    b_hdu[1].header['BUNIT'] = ''
    n_hdu = fits.HDUList([fits.PrimaryHDU(data=nimg, header=outheader),
                          fits.ImageHDU(data=nerr, header=outheader, name='ERROR')])
    n_hdu[0].header['BUNIT'] = 'cm^(-2)'
    n_hdu[1].header['BUNIT'] = 'cm^(-2)'

    t_hdu.writeto(out_prefix+'T.fits', clobber=clobber)
    b_hdu.writeto(out_prefix+'beta.fits', clobber=clobber)
    n_hdu.writeto(out_prefix+'N.fits', clobber=clobber)

    if integral:
        int_hdu = fits.PrimaryHDU(data=intimg, header=outheader)
        int_hdu.header['BUNIT'] = 'erg*s^(-1)*cm^(-2)'
        int_hdu.writeto(out_prefix+'integral.fits', clobber=clobber)
        return t_hdu,b_hdu,n_hdu,int_hdu
    else:
        return t_hdu,b_hdu,n_hdu

