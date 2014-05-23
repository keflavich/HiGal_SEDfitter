HiGal SED fitter
================

The source code is hosted `on github <https://github.com/keflavich/HiGal_SEDfitter>`_

Installation
------------

Can be installed through pip:

.. code-block:: bash

   pip install https://github.com/keflavich/HiGal_SEDfitter/archive/master.zip


Example
-------

.. code-block:: python

    import glob
    from astropy import units as u
    from astropy.io import fits
    from higal_sedfitter import smooth,fit,higal_beams
    from higal_sedfitter.fit import PixelFitter

    pixelfitter = PixelFitter(bfixed=True)

    higal_field = '000'
    fmt = {'field':higal_field}

    for fn in glob.glob("HIGAL{field}*fits".format(**fmt)):
        smooth.add_beam_information_to_higal_header(fn, name_to_um=higal_beams.num_to_um)

    target_fn = glob.glob('HIGAL{field}*_500_RM.fits'.format(**fmt))[0]
    target_header = fits.getheader(target_fn)
    smooth.smooth_images_toresolution(45*u.arcsec, skip_existing=False,
                                      globs=['HIGAL{field}*'.format(**fmt)],
                                      regrid=True,
                                      target_header=target_header, clobber=True)
    fit.fit_modified_blackbody_tofiles('HIGAL{field}*_{{0}}_RM_smregrid45.fits'.format(**fmt),
                                       pixelfitter=pixelfitter,
                                       name_to_um=higal_beams.num_to_um,
                                       out_prefix='HIGAL_L{field}_'.format(**fmt),
                                       integral=True)



Reference/API
=============

.. automodapi:: higal_sedfitter.smooth
    :no-inheritance-diagram:

.. automodapi:: higal_sedfitter.fit
    :no-inheritance-diagram:
