from __future__ import print_function
import warnings
import glob
import os
import re

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import convolution
import FITS_tools.header_tools as FITS_header_tools
from FITS_tools.hcongrid import hcongrid

from higal_beams import name_to_um, beams

FWHM_TO_SIGMA = 1./np.sqrt(8*np.log(2))

def smooth_images(target_resolution, globs=["destripe*P[LMS]W*fits",
                                            "destripe*blue*fits",
                                            "destripe*red*fits"],
                  reject_regex='smooth|smregrid',
                  verbose=True,
                  skip_existing=True,
                  regrid=True,
                  target_header=None,
                  regrid_order=1,
                  clobber=False,
                  **kwargs):
    """
    Smooth a series of images to the same resolution.  The output files will be
    of the form ``{inputfilename}_smooth.fits`` and
    ``{inputfilename}_smregrid.fits``

    Parameters
    ----------
    target_resolution : `~astropy.units.Quantity`
        A degree-equivalent value that specifies the beam size in the output
        image
    globs : list
        A list of strings to pass into `~glob.glob`.  All files found will be
        smoothed and possibly regridded.
    reject_regex : str
        A regular expression to apply to each discovered file to choose whether
        to reject it.  For example, if you've run this function once, you'll
        have files named ``file.fits`` and ``file_smooth.fits`` that you don't
        want to re-smooth and re-regrid.
    verbose : bool
        Print messages at each step?
    skip_existing : bool
        If the output smooth file is found and this is True, skip and move on
        to the next
    regrid : bool
        Regrid the file?  If True, ``target_header`` is also required
    regrid_order : int
        The order of the regridding operation.  Regridding is performed with
        interpolation, so 0'th order means nearest-neighbor and 1st order means
        bilinear.
    clobber : bool
        Overwrite files if they exist?
    kwargs : dict
        Passed to `smooth_image`

    Raises
    ------
    ValueError
        If ``target_header`` is not specified but ``regrid`` is

    Returns
    -------
    Nothing.  All output is to disk
    """

    for fn in [x for g in globs for x in glob.glob(g)
               if not re.search(reject_regex, x)]:
        if verbose:
            print("Reading file {0}".format(fn),)

        outnum = int(target_resolution.to(u.arcsec).value)
        smoutfn = fn.replace(".fits", "_smooth{0:d}.fits".format(outnum))
        if os.path.exists(smoutfn):
            if skip_existing:
                if verbose:
                    print("Skipping {0}".format(fn))
                continue
        
        smhduL = smooth_image_toresolution(fn, smoutfn, target_resolution,
                                           verbose=verbose, clobber=clobber,
                                           **kwargs)
        smhdu = smhduL[0]

        if regrid:
            if target_header is None:
                raise ValueError("Must specify a target header if regridding.")
            newimage = hcongrid(smhdu.data, smhdu.header, target_header,
                                order=regrid_order)

            rgoutfn = fn.replace(".fits", "_smregrid{0:d}.fits".format(outnum))
            print("Regridding {0} to {1}".format(smoutfn, rgoutfn))
            newhdu = fits.PrimaryHDU(data=newimage, header=target_header)
            newhdu.writeto(rgoutfn, clobber=clobber)


def smooth_image_toresolution(fn, outfn, target_resolution, clobber=False,
                              verbose=True, write=True):
    """
    Smooth images with known beam size to a target beam size

    Parameters
    ----------
    fn : str
    outfn : str
    target_resolution : `~astropy.units.Quantity`
        A degree-equivalent value that specifies the beam size in the output
        image
    verbose : bool
        Print messages at each step?
    clobber : bool
        Overwrite files if they exist?
    write : bool
        Write the file to the output filename?

    Returns
    -------
    f : `~astropy.io.fits.PrimaryHDU`
        The smoothed FITS image with appropriately updated BMAJ/BMIN
    """

    f = fits.open(fn)
    header = f[0].header

    native_beamsize = header['BMAJ']*u.deg
    if header['BMIN'] != header['BMAJ']:
        warnings.warn("Asymmetric beam not well-supported: "
                      "the images should be smoothed to have symmetric beams "
                      "prior to using this task")

    kernelsize = ((target_resolution*FWHM_TO_SIGMA)**2 -
                  (native_beamsize*FWHM_TO_SIGMA)**2)**0.5
    if kernelsize.value < 0:
        raise ValueError("Cannot smooth to target resolution: "
                         "smaller than current resolution.")

    platescale = FITS_header_tools.header_to_platescale(f[0].header,
                                                        use_units=True)
    kernelsize_pixels = (kernelsize/platescale).decompose()
    if not kernelsize_pixels.unit.is_equivalent(u.dimensionless_unscaled):
        raise ValueError("Kernel size not in valid units")

    kernel = convolution.Gaussian2DKernel(kernelsize_pixels.value)

    if verbose:
        print("Convolving with {0}-pixel kernel".format(kernelsize_pixels))

    if kernelsize_pixels > 15:
        sm = convolution.convolve_fft(f[0].data, kernel, interpolate_nan=True)
    else:
        sm = convolution.convolve(f[0].data, kernel)

    f[0].data = sm
    comment = "Smoothed with {0:03f}\" kernel".format(kernelsize.to(u.arcsec).value)
    f[0].header['BMAJ'] = (target_resolution.to(u.deg).value, comment)
    f[0].header['BMIN'] = (target_resolution.to(u.deg).value, comment)

    if write:
        f.writeto(outfn, clobber=clobber)

    return f

def add_beam_information_to_higal_header(fn, clobber=True):
    """
    Given a Hi-Gal FITS file name, attempt to add beam information to its
    header.

    The beams are assumed to be symmetric using the larger of the two beam axes
    given in Traficante et al 2011.  This is not a valid assumption in general,
    but without knowing the scan position angle you can't really do better.

    Parameters
    ----------
    fn : str
        A filename corresponding to a Hi-Gal FITS file.  MUST have one of the
        standard HiGal strings in the name: blue, red, PSW, PMW, or PLW
    clobber : bool
        Overwrite existing file?  (has to be "True" to work!)
    """

    f = fits.open(fn)

    wl_names = [x for x in name_to_um if x in fn]
    if len(wl_names) != 1:
        raise ValueError("Found too few or too many matches!")
    wl_name = wl_names[0]

    f[0].header.append(fits.Card(keyword='BMAJ',
                                 value=beams[name_to_um[wl_name]].to(u.deg).value,
                                 comment='From Traficante 2011'))
    f[0].header.append(fits.Card(keyword='BMIN',
                                 value=beams[name_to_um[wl_name]].to(u.deg).value,
                                 comment='Assumed equal to BMAJ'))
    f[0].header.append(fits.Card(keyword='BPA', value=0))

    f[0].header.append(fits.Card(keyword='BEAMNOTE',
                                 value='2011MNRAS.416.2932T',
                                 comment='Source paper for beam'))

    f.writeto(fn, clobber=clobber)
