import numpy as np
from astropy.io import fits
from astropy import units as u
from dust_emissivity.blackbody import modified_blackbody
from higal_sedfitter.fit import fit_modified_blackbody_to_imagecube

def test_bbfit():
    temperatures = [5,10,15,20,25]
    betas        = [1,1.25,1.5,1.75,2.0,2.25,2.5]
    columns      = [22,23,24]
    wavelengths = [#np.array([70,160,250,350,500,850,1100],dtype='float'),
        np.array([160,250,350,500,1100],dtype='float'),
        np.array([160,250,350,500],dtype='float')]
    errlevels = [0.1,0.2,0.05]

    temperature = 20.
    beta = 1.75
    column = 22
    wavelength = wavelengths[0]
    errlevel = errlevels[0]

    #for temperature,beta,column in itertools.product(temperatures,betas,columns):

    tguess=20
    bguess=1.5
    nguess=21
    tguess=temperature
    bguess=beta
    nguess=column

    flux = modified_blackbody_wavelength(wavelength, temperature,
            beta=beta, wavelength_units='microns', normalize=False, logN=column,
            logscale=16)

    mp = fit_blackbody(wavelength, flux, err=err,
            blackbody_function=modified_blackbody_wavelength,
            logscale=16, guesses=(tguess, bguess, nguess), wavelength_units='microns')

def test_fit_cube():
    """
    Performance test of cube fitting
    """

    for nt,nb in zip([5,50],[5,50]):

        wavelengths = ([70,160,250,350,500] * u.um).to(u.GHz, u.spectral())
        imagecube = np.array([modified_blackbody(wavelengths, tem, beta=beta)
                              for tem in np.linspace(15,100,nt)*u.K
                              for beta in np.linspace(1,2,nb)]).reshape([wavelengths.size,nt,nb])

        import time
        tt = time.time()
        fit_modified_blackbody_to_imagecube(imagecube, fits.Header(), ncores=4)
        ptime = (time.time()-tt)

        tt = time.time()
        fit_modified_blackbody_to_imagecube(imagecube, fits.Header(), ncores=1)
        stime = (time.time()-tt)
        print
        print 'nt,nb = %i,%i parallel map in %g secs' % (nt,nb,ptime)
        print 'nt,nb = %i,%i serial map in %g secs' % (nt,nb,stime)
