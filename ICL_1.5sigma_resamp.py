import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

from resamp import gen
from numba import vectorize
from extinction_redden import A_wave
from light_measure import light_measure, flux_recal
# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

# sample catalog
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

R0 = 1 # in unit Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['u', 'g', 'r', 'i', 'z']
mag_add = np.array([-0.04, 0, 0, 0, 0.02])

def resamp_15sigma():

	for ii in range(len(band)):
		print('Now band is %s' % band[ii])
		for k in range(len(z)):
			ra_g = ra[k]
			dec_g = dec[k]
			z_g = z[k]
			Da_g = Test_model.angular_diameter_distance(z_g).value

			data = fits.getdata(load + 'mask_data/A_plane/1.5sigma/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			head_mean = data[1]
			cx0 = data[1]['CRPIX1']
			cy0 = data[1]['CRPIX2']
			RA0 = data[1]['CRVAL1']
			DEC0 = data[1]['CRVAL2']

			wcs = awc.WCS(data[1])
			cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur/pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			b = L_ref/L_z0
			Rref = (R0*rad2asec/Da_ref)/pixel

			ox = np.linspace(0, img.shape[1]-1, img.shape[1])
			oy = np.linspace(0, img.shape[0]-1, img.shape[0])
			oo_grd = np.array(np.meshgrid(ox, oy))
			cdr = np.sqrt((oo_grd[0,:] - cx)**2 + (oo_grd[1,:] - cy)**2)
			idd = (cdr > Rp) & (cdr < 1.1 * Rp)
			cut_region = img[idd]
			id_nan = np.isnan(cut_region)
			idx = np.where(id_nan == False)
			bl_array = cut_region[idx]
			bl_array = np.sort(bl_array)
			back_lel = np.mean(bl_array)

			imgt = img - back_lel
			f_goal = flux_recal(imgt, z_g, z_ref)
			xn, yn, resam = gen(f_goal, 1, b, cx, cy)
			xn = np.int(xn)
			yn = np.int(yn)
			ix0 = np.int(cx0/b)
			iy0 = np.int(cy0/b)
			if b > 1:
				resam = resam[1:, 1:]
			elif b == 1:
				resam = resam[1:-1, 1:-1]
			else:
				resam = resam

			x0 = resam.shape[1]
			y0 = resam.shape[0]

			keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
					'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
			value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
			ff = dict(zip(keys,value))
			fil = fits.Header(ff)
			fits.writeto(load + 
				'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), resam, header = fil, overwrite=True)

		print('Now band %s have finished!!' % band[ii])
	return

def main():
	resamp_15sigma()

if __name__ == "__main__":
	main()
