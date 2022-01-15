import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

from light_measure import flux_recal
from resample_modelu import sum_samp, down_samp
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

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

R0 = 1 # in unit Mpc
pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
Rv = 3.1
sfd = SFDQuery()

def img_edg_cut(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]

		data = fits.getdata(load + 'resample/1_5sigma_larger_R/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
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
		fits.writeto(load + 'edge_cut/sample_img/Edg_cut-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
		(band[ii], ra_g, dec_g, z_g), re_img, header = fil, overwrite=True)

	return

def sky_edg_cut(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)

	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]

		data = fits.getdata(load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
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
		fits.writeto(load + 'edge_cut/sample_sky/Edg_cut-sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
		(band[ii], ra_g, dec_g, z_g), re_img, header = fil, overwrite=True)

	return

def main():

	with h5py.File(load + 'mpi_h5/sample_catalog.h5', 'r') as f:
		dat = np.array(f['a'])
	z, ra, dec = dat[0,:], dat[1,:], dat[2,:]
	zN = len(z)

	for tt in range( len(band) ):
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		img_edg_cut(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

	for tt in range( len(band) ):
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		sky_edg_cut(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

if __name__ == "__main__":
	main()
