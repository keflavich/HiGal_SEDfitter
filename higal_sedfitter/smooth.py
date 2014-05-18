from __future__ import print_function
import warnings
import glob
import os
import re

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy import convolution
from astropy.utils.console import ProgressBar
import FITS_tools.header_tools as FITS_header_tools
from FITS_tools.hcongrid import hcongrid
import FITS_tools
import dust_emissivity

from higal_beams import name_to_um, beams

FWHM_TO_SIGMA = 1./np.sqrt(8*np.log(2))

def smooth_images(target_resolution, globs=["destripe*P[LMS]W*fits",
                                            "destripe*blue*fits",
                                            "destripe*red*fits"],
                  reject_regex='smooth',
                  verbose=True,
                  skip_existing=True,
                  regrid=True,
                  target_header=None,
                  regrid_order=1,
                  clobber=False,
                  **kwargs):
    """
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
        
        smhdu = smooth_image(fn, smoutfn, target_resolution, verbose=verbose,
                             clobber=clobber, **kwargs)[0]

        if regrid:
            if target_header is None:
                raise ValueError("Must specify a target header if regridding.")
            newimage = hcongrid(smhdu.data, smhdu.header, target_header,
                                order=regrid_order)

            rgoutfn = fn.replace(".fits", "_smregrid{0:d}.fits".format(outnum))
            print("Regridding {0} to {1}".format(smoutfn, rgoutfn))
            newhdu = fits.PrimaryHDU(data=newimage, header=target_header)
            newhdu.writeto(rgoutfn, clobber=clobber)


def smooth_image(fn, outfn, target_resolution, clobber=False, verbose=True,
                 write=True):
    """
    Smooth images with known
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
    f[0].header['BMAJ'] = (45/3600., comment)
    f[0].header['BMIN'] = (45/3600., comment)

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
    """

    f = fits.open(fn)

    wl_name = [x for x in name_to_um if x in fn][0]

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
