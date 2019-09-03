import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from scipy.interpolate import interp2d as interp2
from matplotlib.patches import Circle, Ellipse

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
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/sky_add_img/'
band = ['r', 'i', 'g', 'u', 'z']

def sky_add(band_id, z_set, ra_set, dec_set):
	kk = np.int(band_id)
	Nz = len(z_set)
	#for kk in range(len(band)):
	for jj in range(Nz):
		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		data = fits.open(d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		head_inf = data[0].header
		wcs = awc.WCS(head_inf)
		cenx, ceny = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		R_ph = rad2asec / (Test_model.angular_diameter_distance(z_g).value)
		R_p = R_ph / pixel

		sky0 = data[2].data['ALLSKY'][0]
		sky_x = data[2].data['XINTERP'][0]
		sky_y = data[2].data['YINTERP'][0]
		x0 = np.linspace(0, sky0.shape[1] - 1, sky0.shape[1])
		y0 = np.linspace(0, sky0.shape[0] - 1, sky0.shape[0])
		f_sky = interp2(x0, y0, sky0)
		New_sky = f_sky(sky_x, sky_y)
		fact = img.size / sky0.size
		sky_bl = New_sky * (data[0].header['NMGY']) / fact
		cimg = img + sky_bl

		hdu = fits.PrimaryHDU()
		hdu.data = cimg
		hdu.header = data[0].header
		hdu.writeto(load + 'cimg-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]), overwrite = True)

		fig = plt.figure(figsize = (16, 8))
		fig.suptitle('image comparison ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[kk]) )
		cluster0 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
		cluster1 = Circle(xy = (cenx, ceny), radius = R_p, fill = False, ec = 'b', alpha = 0.5, label = 'cluster region[1Mpc]')
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)

		tf = ax0.imshow(img, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax0, orientation = 'horizontal', fraction = 0.05, pad = 0.01, label = '$flux[nmaggy]$')
		ax0.add_patch(cluster0)
		ax0.set_title('calibrated image')
		ax0.set_xlim(0, img.shape[1])
		ax0.set_ylim(0, img.shape[0])
		ax0.set_xticks([])

		tf = ax1.imshow(cimg, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax1, orientation = 'horizontal', fraction = 0.05, pad = 0.01, label = '$flux[nmaggy]$')
		ax1.add_patch(cluster1)
		ax1.set_title('sky-added image')
		ax1.set_xlim(0, cimg.shape[1])
		ax1.set_ylim(0, cimg.shape[0])
		ax1.set_xticks([])

		plt.tight_layout()
		plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_class/sky_add/cimage_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[kk]), dpi = 300)
		plt.close()
	return

def main():
	t0 = time.time()
	Ntot = len(z)
	commd.Barrier()
	for tt in range(len(band)):
		m, n = divmod(Ntot, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		sky_add(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		commd.Barrier()	
	t1 = time.time() - t0
	print('t = ', t1)
	raise
if __name__ == "__main__" :
	main()
