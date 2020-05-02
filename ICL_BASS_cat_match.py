import numpy as np
import pandas as pds
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits as fits

## SDSS cat.
dat_SDSS = fits.open('redmapper_dr8_public_v6.3_catalog.fits.gz')
ra_sdss = dat_SDSS[1].data['ra']
dec_sdss = dat_SDSS[1].data['dec']
sdss_z_lamda = dat_SDSS[1].data['Z_LAMBDA']
sdss_z_spec = dat_SDSS[1].data['Z_SPEC']

idx0 = (sdss_z_lamda >= 0.2) & (sdss_z_lamda <= 0.3)
z_lamda = sdss_z_lamda[idx0]
ra_lamda = ra_sdss[idx0]
dec_lamda = dec_sdss[idx0]
keys = ['ra', 'dec', 'z']
values = [ra_lamda, dec_lamda, z_lamda]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('sdss_z-lamda_cat.csv')

idx1 = (sdss_z_spec >= 0.2) & (sdss_z_spec <= 0.3)
z_spec = sdss_z_spec[idx1]
ra_spec = ra_sdss[idx1]
dec_spec = dec_sdss[idx1]
keys = ['ra', 'dec', 'z']
values = [ra_spec, dec_spec, z_spec]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('sdss_z-spec_cat.csv')

## BASS cat.
dat_BASS = fits.open('blocks-dr3-prior.fits')
ra_BASS = dat_BASS[1].data['ra']
dec_BASS = dat_BASS[1].data['dec']
catalog = SkyCoord(ra = ra_BASS*u.degree, dec = dec_BASS*u.degree)

## find sources in BASS cat. located in 0.2~z~0.3 (based on SDSS cat.)
c_lamda = SkyCoord(ra = ra_lamda*u.degree, dec = dec_lamda*u.degree)
idx, d2d, d3d = c_lamda.match_to_catalog_sky(catalog)
ra_bass_lamda = ra_BASS[idx]
dec_bass_lamda = dec_BASS[idx]
keys = ['ra', 'dec']
values = [ra_bass_lamda, dec_bass_lamda]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('BASS_z-lamda_match_cat.csv')

c_spec = SkyCoord(ra = ra_spec*u.degree, dec = dec_spec*u.degree)
idx, d2d, d3d = c_spec.match_to_catalog_sky(catalog)
ra_bass_spec = ra_BASS[idx]
dec_bass_spec = dec_BASS[idx]
keys = ['ra', 'dec']
values = [ra_bass_spec, dec_bass_spec]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('BASS_z-spec_match_cat.csv')

