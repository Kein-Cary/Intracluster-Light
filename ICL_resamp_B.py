import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

from resamp import gen
from resample_modelu import sum_samp, down_samp
from light_measure import flux_recal
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
import time
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
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/mpi_h5/sample_catalog.h5', 'r') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

R0 = 1 # in unit Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
#band = ['u', 'g', 'r', 'i', 'z']
#mag_add = np.array([-0.04, 0, 0, 0, 0.02])

band = ['r', 'g', 'i', 'z', 'u']
mag_add = np.array([ 0, 0, 0, 0.02, -0.04])

def resamp_Bpl(band_id, sub_z, sub_ra, sub_dec):
	ii = np.int(band_id)
	zn = len(sub_z)

	print('Now band is %s' % band[ii])
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		# use star catalog dr7 (only g, r, i band)
		data_B = fits.getdata(load + 
			'mask_data/B_plane/Zibetti/B_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
		img_B = data_B[0]

		head_mean = data_B[1]
		cx0 = data_B[1]['CRPIX1']
		cy0 = data_B[1]['CRPIX2']
		RA0 = data_B[1]['CRVAL1']
		DEC0 = data_B[1]['CRVAL2']
		wcs = awc.WCS(data_B[1])
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		Angur = (R0 * rad2asec/Da_g)
		Rp = Angur/pixel
		L_ref = Da_ref * pixel / rad2asec
		L_z0 = Da_g*pixel/rad2asec
		b = L_ref/L_z0
		Rref = (R0*rad2asec/Da_ref)/pixel

		f_goal = flux_recal(img_B, z_g, z_ref)
		ix0 = np.int(cx0/b)
		iy0 = np.int(cy0/b)
		'''
		xn, yn, resam = gen(f_goal, 1, b, cx, cy)
		if b > 1:
			resam = resam[1:, 1:]
		elif b == 1:
			resam = resam[1:-1, 1:-1]
		else:
			resam = resam
		'''
		if b > 1:
			resam, xn, yn = sum_samp(b, b, f_goal, cx, cy)
		else:
			resam, xn, yn = down_samp(b, b, f_goal, cx, cy)

		xn = np.int(xn)
		yn = np.int(yn)
		x0 = resam.shape[1]
		y0 = resam.shape[0]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)

		fits.writeto(load + 
			'resample/Zibetti/B_mask/frameB-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), resam, header = fil, overwrite=True)

	print('Now band %s have finished!!' % band[ii])
	return

def main():
	t0 = time.time()
	tot_N = len(z)

	#commd.Barrier()
	for tt in range(3):
		m, n = divmod(tot_N, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		resamp_Bpl(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()
	t1 = time.time() - t0
	print('total time = ', t1)

if __name__ == "__main__":
	main()
