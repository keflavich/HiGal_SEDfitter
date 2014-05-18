=================
Hi-Gal SED fitter
=================

Docs to come.  All the good stuff happens in `fits.py
<higal_sedfitter/fits.py>` and `smooth.py <higal_sedfitter/smooth.py>`.


Requires

 * FITS_tools_
 * lmfit_
 * dust_emissivity_

Example
-------
.. code-block:: python

   >>> from astropy.io import fits
   >>> from astropy import units as u
   >>> from higal_sedfitter import smooth, PixelFitter, fit_modified_blackbody_tofiles
   >>> target_header = fits.getheader('destripe_l048_PLW_wgls_rcal.fits')
   >>> smooth.smooth_images(45*u.arcsec, skip_existing=False, regrid=True,
   ...                      target_header=target_header)

   >>> pixelfitter = PixelFitter(bfixed=True)
   >>> fit_modified_blackbody_tofiles('destripe_l048_{0}_wgls_rcal_smregrid45.fits',
   ...                            pixelfitter=pixelfitter,
   ...                            out_prefix='higalsedfit_70to500_l048_beta1.75',
   ...                            integral=True)
   
.. _FITS_tools: fits-tools.rtfd.org
.. _lmfit: lmfit.github.io/lmfit-py/
.. _dust_emissivity: https://github.com/keflavich/dust_emissivity
