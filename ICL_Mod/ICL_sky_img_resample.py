import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import astropy.wcs as awc
import astropy.io.fits as fits

import astropy.units as U
from astropy import cosmology as apcy
from light_measure import flux_recal
from resample_module import sum_samp, down_samp

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

rad2asec = U.rad.to(U.arcsec)

Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396  # the pixel size, in unit arcsec, 
## for SB claculation: m = ZP - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)

z_ref = 0.250  # reference redshift, for flux scaling and pixel resampling
Da_ref = Test_model.angular_diameter_distance(z_ref).value
R0 = 1  # in unit Mpc, cluster physical size
Angu_ref = (R0 / Da_ref) * rad2asec
Rp_ref = Angu_ref / pixel  # cluster size in unit of pixels, at z_ref

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/' ## save the catalogue data
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'  ## save the process data
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

def sky_resample(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		data = fits.open(load + 'sky/sky_arr/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[ii]) )
		img = data[0].data
		head = data[0].header
		cx0 = data[0].header['CRPIX1']
		cy0 = data[0].header['CRPIX2']
		RA0 = data[0].header['CRVAL1']
		DEC0 = data[0].header['CRVAL2']
		wcs = awc.WCS(head)
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		Angu_z = R0 * rad2asec / Da_g
		Rp_z = Angu_z / pixel
		L_ref = Da_ref * pixel / rad2asec
		L_z0 = Da_g * pixel / rad2asec
		b = L_ref / L_z0

		ix0 = np.int(cx0 / b)
		iy0 = np.int(cy0 / b)

		if b > 1:
			resam, xn, yn = sum_samp(b, b, img, cx, cy)
		else:
			resam, xn, yn = down_samp(b, b, img, cx, cy)

		xn = np.int(xn)
		yn = np.int(yn)
		x0 = resam.shape[1]
		y0 = resam.shape[0]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(tmp + 'test/resam-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), resam, header = fil, overwrite = True)

	return

def main():

	#for kk in range(len(band)):
	for kk in range( 3 ):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk], 'r') as f:
			cat = np.array(f['a'])
		ra, dec, z = cat[0,:], cat[1,:], cat[2,:]
		zN = len(z)

		Ns = 100
		np.random.seed(1)
		tt0 = np.random.choice(zN, size = Ns, replace = False)
		set_z, set_ra, set_dec = z[tt0], ra[tt0], dec[tt0]

		m, n = divmod(Ns, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sky_resample(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

if __name__ == "__main__":
	main()
