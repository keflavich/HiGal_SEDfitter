from astropy import units as u

# Values extracted from Traficante 2011 Table 2
# http://adsabs.harvard.edu/abs/2011MNRAS.416.2932T
beams = {70: 10.7*u.arcsec,
         160: 13.9*u.arcsec,
         250: 23.9*u.arcsec,
         350: 31.3*u.arcsec,
         500: 43.8*u.arcsec}

name_to_um = {'blue':70,
              'red':160,
              'PSW':250,
              'PMW':350,
              'PLW':500}

num_to_um = {'070':70,
             '160':160,
             '250':250,
             '350':350,
             '500':500}
