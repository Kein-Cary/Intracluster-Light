# this file use to fast resample data
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

from itertools import product
from itertools import starmap
from multiprocessing import Pool

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7

Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

Rv = 3.1
sfd = SFDQuery()
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
A_lambd = np.array([5.155, 3.793, 2.751, 2.086, 1.479])
l_wave = np.array([3551, 4686, 6166, 7480, 8932])

sb_lim = np.array([24.35, 25, 24.5, 24, 22.9]) # SB limit at z_ref
zopt = np.array([22.46, 22.5, 22.5, 22.5, 22.52]) # zero point

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
lamb = catalogue[3]
def fill_name():
	"""
	# name is the file name, 
	name_u : u band file name
	the formation: 'frame-u-raXXX.XXX-decXXX.XXX-redshiftX.XXX.fits.bz2' 
	"""
	name_u = []
	name_g = []
	name_r = []
	name_i = []
	name_z = []
	for k in range(len(z)):
		name_u.append('frame-u-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_g.append('frame-g-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_r.append('frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_i.append('frame-i-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
		name_z.append('frame-z-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(ra[k], dec[k], z[k]))
	f_name = [name_u, name_g, name_r, name_i, name_z]
	return f_name, ra, dec, z

def multi_stack():


	return

def test():

	file_str, ra, dec, z = fill_name()
	raise
	result = map(multi_stack, file_str[0], ra, dec, z)
	result = list(result) # use map as a test run
	'''
	p = Pool(5)
	result = p.starmap(multi_stack)
	p.close()
	p.join()
	'''
	return 
def main():
	test()

if __name__ == '__main__' :

	main()