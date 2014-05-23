import glob
from astropy import units as u
from astropy.io import fits
from . import smooth,fit,higal_beams
from .fit import PixelFitter

def fit_wrapper(higal_field, resolution=45*u.arcsec):

    pixelfitter = PixelFitter(bfixed=True)

    fmt = {'field':higal_field,
           'res':int(resolution.to(u.arcsec).value)}

    for fn in glob.glob("HIGAL{field}*fits".format(**fmt)):
        smooth.add_beam_information_to_higal_header(fn, name_to_um=higal_beams.num_to_um)

    target_fn = glob.glob('HIGAL{field}*_500_RM.fits'.format(**fmt))[0]
    target_header = fits.getheader(target_fn)
    smooth.smooth_images_toresolution(resolution, skip_existing=False,
                                      globs=['HIGAL{field}*'.format(**fmt)],
                                      regrid=True,
                                      target_header=target_header, clobber=True)
    fit.fit_modified_blackbody_tofiles('HIGAL{field}*_{{0}}_RM_smregrid{res}.fits'.format(**fmt),
                                       pixelfitter=pixelfitter,
                                       name_to_um=higal_beams.num_to_um,
                                       out_prefix='HIGAL_L{field}_'.format(**fmt),
                                       integral=True)
