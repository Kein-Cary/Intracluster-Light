import h5py
import numpy as np
import pandas as pds
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

## SDSS cat.
dat_SDSS = fits.open('/home/xkchen/mywork/ICL/BASS_cat/redmapper_dr8_public_v6.3_catalog.fits.gz')
ra_sdss = dat_SDSS[1].data['ra']
dec_sdss = dat_SDSS[1].data['dec']
sdss_z_lamda = dat_SDSS[1].data['Z_LAMBDA']
sdss_z_spec = dat_SDSS[1].data['Z_SPEC']

id_dec = (dec_sdss >= 32)

idx0 = (sdss_z_lamda >= 0.10) & (sdss_z_lamda <= 0.33)
idu = idx0 & id_dec
z_lamda = sdss_z_lamda[idu]
ra_lamda = ra_sdss[idu]
dec_lamda = dec_sdss[idu]
Da_lamd = Test_model.angular_diameter_distance(z_lamda).value
angle_lamd = ((rad2asec * (2 / h) ) / Da_lamd ) / 3600 ## in unit of degree

keys = ['ra', 'dec', 'z', 'angle_size']
values = [ra_lamda, dec_lamda, z_lamda, angle_lamd]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('/home/xkchen/mywork/ICL/BASS_cat/sdss_z-lambda_cat.csv')
## .h5
tmp_array = np.array([ra_lamda, dec_lamda, z_lamda])
with h5py.File('/home/xkchen/mywork/ICL/BASS_cat/sdss_z-lambda_cat.h5', 'w') as f:
	f['a'] = np.array(tmp_array)
with h5py.File('/home/xkchen/mywork/ICL/BASS_cat/sdss_z-lambda_cat.h5', ) as f:
	for ll in range(len(tmp_array)):
		f['a'][ll,:] = tmp_array[ll,:]

idx1 = (sdss_z_spec >= 0.10) & (sdss_z_spec <= 0.33)
idv = idx1 & id_dec
z_spec = sdss_z_spec[idv]
ra_spec = ra_sdss[idv]
dec_spec = dec_sdss[idv]
Da_spec = Test_model.angular_diameter_distance(z_spec).value
angle_spec = ((rad2asec * (2 / h) ) / Da_spec ) / 3600 ## in unit of degree

keys = ['ra', 'dec', 'z', 'angle_size']
values = [ra_spec, dec_spec, z_spec, angle_spec]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('/home/xkchen/mywork/ICL/BASS_cat/sdss_z-spec_cat.csv')
## .h5
tmp_array = np.array([ra_spec, dec_spec, z_spec])
with h5py.File('/home/xkchen/mywork/ICL/BASS_cat/sdss_z-spec_cat.h5', 'w') as f:
	f['a'] = np.array(tmp_array)
with h5py.File('/home/xkchen/mywork/ICL/BASS_cat/sdss_z-spec_cat.h5', ) as f:
	for ll in range(len(tmp_array)):
		f['a'][ll,:] = tmp_array[ll,:]


## BASS cat.
dat_BASS = fits.open('/home/xkchen/mywork/ICL/BASS_cat/blocks-dr3-prior.fits')
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
data.to_csv('/home/xkchen/mywork/ICL/BASS_cat/BASS_z-lambda_match_cat.csv')

c_spec = SkyCoord(ra = ra_spec*u.degree, dec = dec_spec*u.degree)
idx, d2d, d3d = c_spec.match_to_catalog_sky(catalog)
ra_bass_spec = ra_BASS[idx]
dec_bass_spec = dec_BASS[idx]
keys = ['ra', 'dec']
values = [ra_bass_spec, dec_bass_spec]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('/home/xkchen/mywork/ICL/BASS_cat/BASS_z-spec_match_cat.csv')
