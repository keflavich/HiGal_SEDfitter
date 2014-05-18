class PixelFitter(object):
    
    def __init__(self, tguess=20, bguess=1.75, nguess=1e22,
                 trange=[2.73,50], brange=[1,3], nrange=[1e20,1e25],
                 tfixed=False, bfixed=False, nfixed=False):
        
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
        fitter = dust_emissivity.fit_sed.fit_sed_lmfit_hz
        lm = fitter(frequency.to(u.Hz).value,
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
        integrator = dust_emissivity.blackbody.integrate_sed
        return integrator(fmin, fmax,
                          function=dust_emissivity.blackbody.modified_blackbody,
                          temperature=self.lm.params['T'].value*u.K,
                          column=self.lm.params['N'].value*u.cm**-2,
                          beta=self.lm.params['beta'].value).value



def fit_modified_blackbody_tofiles(filename_template,
                                   wavelengths=[70,160,250,350,500],
                                   error_scaling=0.2,
                                   pixelfitter=PixelFitter(),
                                   ncores=4,
                                   clobber=True,
                                   bad_value=0,
                                   integral=False,
                                   out_prefix="",
                                   ):

    target_files = {name_to_um[x]: filename_template.format(x)
                    for x in name_to_um}
    wavelengths_sorted = sorted(wavelengths)

    for v in target_files.values():
        if not os.path.exists(v):
            raise IOError("File {0} does not exist".format(v))

    # Make an image cube appropriately sorted by wavelength
    image_cube = np.array([fits.getdata(target_files[wl])
                           for wl in wavelengths_sorted])
    image_cube[image_cube == bad_value] = np.nan

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

    # wl should be defined from above; the headers should (in principle) be
    # identical
    outheader = fits.getheader(target_files[wl])
    t_hdu = fits.HDUList([fits.PrimaryHDU(data=timg, header=outheader),
                          fits.ImageHDU(data=terr, header=outheader, name='ERROR')])
    b_hdu = fits.HDUList([fits.PrimaryHDU(data=bimg, header=outheader),
                          fits.ImageHDU(data=berr, header=outheader, name='ERROR')])
    n_hdu = fits.HDUList([fits.PrimaryHDU(data=nimg, header=outheader),
                          fits.ImageHDU(data=nerr, header=outheader, name='ERROR')])

    t_hdu.writeto(out_prefix+'T.fits', clobber=clobber)
    b_hdu.writeto(out_prefix+'beta.fits', clobber=clobber)
    n_hdu.writeto(out_prefix+'N.fits', clobber=clobber)

    if integral:
        int_hdu = fits.PrimaryHDU(data=intimg, header=outheader)
        int_hdu.writeto(out_prefix+'integral.fits', clobber=clobber)
        return t_hdu,b_hdu,n_hdu,int_hdu

    else:
        return t_hdu,b_hdu,n_hdu

