import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import random
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from scipy.ndimage import map_coordinates as mapcd
from resample_modelu import sum_samp, down_samp
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, Ellipse
from light_measure import light_measure, light_measure_Z0

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
import time

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
Rpp = (rad2asec / Da_ref) / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/redMap_random/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'

band = ['r', 'g', 'i', 'u', 'z']
sky_SB = [21.04, 22.01, 20.36, 22.30, 19.18] # ref_value from SDSS
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def rebuild_sky(band_id, z_set, ra_set, dec_set):

	kk = np.int(band_id)
	Nz = len(z_set)
	for jj in range(Nz):
		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]
		try:
			data = fits.open(dfile + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], ra_g, dec_g, z_g) )
			img = data[0].data
			sky0 = data[2].data['ALLSKY'][0]
			sky_x = data[2].data['XINTERP'][0]
			sky_y = data[2].data['YINTERP'][0]
			inds = np.array(np.meshgrid(sky_x, sky_y))
			t_sky = mapcd(sky0, [inds[1,:], inds[0,:]], order = 1, mode = 'nearest')
			sky_bl = t_sky * (data[0].header['NMGY'])

			## save the sky img
			hdu = fits.PrimaryHDU()
			hdu.data = sky_bl
			hdu.header = data[0].header
			hdu.writeto(load + 'random_cat/sky_img/rand_sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]), overwrite = True)
		except FileNotFoundError:
			continue
	return

def resamp_sky(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		try:
			data = fits.open(load + 'random_cat/sky_img/rand_sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[ii]) )
			img = data[0].data
			head = data[0].header
			cx0 = data[0].header['CRPIX1']
			cy0 = data[0].header['CRPIX2']
			RA0 = data[0].header['CRVAL1']
			DEC0 = data[0].header['CRVAL2']
			wcs = awc.WCS(head)
			cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = rad2asec / Da_g
			Rp = Angur / pixel
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
			fits.writeto(load + 'random_cat/sky_resamp_img/rand_resamp-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), 
				resam, header = fil, overwrite = True)
		except FileNotFoundError:
			continue
	return

def sky_edg_cut(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)

	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		try:
			data = fits.getdata(load + 'random_cat/sky_resamp_img/rand_resamp-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
				(band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			BCGx, BCGy = data[1]['CENTER_X'], data[1]['CENTER_Y']
			RA0, DEC0 = data[1]['CRVAL1'], data[1]['CRVAL2']

			xc, yc = np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)
			## keep the image size but set np.nan for egde pixels
			re_img = np.zeros( (img.shape[0], img.shape[1]), dtype = np.float) + np.nan
			( re_img[yc - np.int(Rpp): yc + np.int(Rpp), xc - np.int(1.3 * Rpp): xc + np.int(1.3 * Rpp)] ) = ( 
				img[yc - np.int(Rpp): yc + np.int(Rpp), xc - np.int(1.3 * Rpp): xc + np.int(1.3 * Rpp)] )

			New_bcgx = BCGx + 0
			New_bcgy = BCGy + 0

			Lx = re_img.shape[1]
			Ly = re_img.shape[0]
			Crx = xc + 0
			Cry = yc + 0

			keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
					'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
			value = ['T', 32, 2, Lx, Ly, Crx, Cry, New_bcgx, New_bcgy, RA0, DEC0, ra_g, dec_g, z_g, pixel]
			ff = dict(zip(keys,value))
			fil = fits.Header(ff)
			fits.writeto(load + 'random_cat/sky_edge-cut_img/rand_Edg_cut-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
			(band[ii], ra_g, dec_g, z_g), re_img, header = fil, overwrite=True)
		except FileNotFoundError:
			continue
	return

def main():

	with h5py.File(load + 'mpi_h5/redMapper_rand_cat.h5', 'r') as f:
		tmp_array = np.array(f['a'])
	ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])
	Ntot = len(z)

	for tt in range(3):
		m, n = divmod(Ntot, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		rebuild_sky(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()
	"""
	for tt in range(3):
		m, n = divmod(Ntot, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		resamp_sky(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

	for tt in range(3):
		m, n = divmod(Ntot, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		sky_edg_cut(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()
	"""
if __name__ == '__main__':
	main()
