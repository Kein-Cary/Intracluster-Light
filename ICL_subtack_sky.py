import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import subprocess as subpro
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
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/sky_add_img/'
band = ['r', 'i', 'g', 'u', 'z']

def sky_03(band_id, ra_g, dec_g, z_g):

	param_sky = 'default_sky_mask.sex'
	out_cat = 'default_mask_A.param'
	out_load_sky = '/mnt/ddnfs/data_users/cxkttwl/PC/sky_mask_%d_cpus.cat' % rank

	data = fits.open(load + 'cimg-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[band_id]) )
	img = data[0].data
	head_inf = data[0].header
	wcs = awc.WCS(head_inf)
	cx_BCG, cy_BCG = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
	R_ph = rad2asec / (Test_model.angular_diameter_distance(z_g).value)
	R_p = R_ph / pixel

	file_source = load + 'cimg-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[band_id])
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_sky, out_load_sky, out_cat)
	print(cmd)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_sky)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])
	Kron = 6
	a = Kron * A
	b = Kron * B

	ddr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)
	ix = ddr >= 0.95 * R_p
	iy = ddr <= 1.15 * R_p
	iz = ix & iy
	s_cx = cx[iz]
	s_cy = cy[iz]
	s_a = a[iz]
	s_b = b[iz]
	s_phi = theta[iz]
	s_Num = len(s_b)

	mask_sky = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
	ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
	oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
	basic_coord = np.array(np.meshgrid(ox,oy))
	major = s_a / 2
	minor = s_b / 2
	senior = np.sqrt(major**2 - minor**2)
	for k in range(s_Num):
		xc = s_cx[k]
		yc = s_cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = s_phi[k]*np.pi/180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r +1), img.shape[1] - 1] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r +1), img.shape[0] - 1] )

		df1 = lr**2 - cr**2*np.cos(chi)**2
		df2 = lr**2 - cr**2*np.sin(chi)**2
		fr = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)**2*df1 +(basic_coord[1,:][lb0: lb1, la0: la1] - yc)**2*df2\
		- cr**2*np.sin(2*chi)*(basic_coord[0,:][lb0: lb1, la0: la1] - xc)*(basic_coord[1,:][lb0: lb1, la0: la1] - yc)
		idr = fr/(lr**2*sr**2)
		jx = idr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
		iv[iu] = np.nan
		mask_sky[lb0: lb1, la0: la1] = mask_sky[lb0: lb1, la0: la1] * iv

	mirro_sky = img * mask_sky
	dr = np.sqrt((basic_coord[0,:] - cx_BCG)**2 + (basic_coord[1,:] - cy_BCG)**2)
	idr = (dr >= R_p) & (dr <= 1.1 * R_p)
	pix_cut = mirro_sky[idr]
	BL = np.nanmean(pix_cut)

	plt.figure()
	plt.title('sample ra%.3f dec%.3f z%.3f %s band' % (ra_g, dec_g, z_g, band[band_id]) )
	plt.imshow(mirro_sky, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
	hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'r', linestyle = '-')
	hsc.circles(cx_BCG, cy_BCG, s = 1.1 * R_p, fc = '', ec = 'r', linestyle = '--')
	plt.xlim(0, img.shape[1])
	plt.ylim(0, img.shape[0])
	plt.savefig(
		'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_class/sky_03_mask/sky_03mask_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[band_id]), dpi = 300)
	plt.close()

	return BL

def sky_subtract(band_id, z_set, ra_set, dec_set):
	kk = np.int(band_id)
	Nz = len(z_set)
	#for kk in range(len(band)):
	for jj in range(Nz):
		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		data = fits.open(load + 'cimg-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]) )
		img = data[0].data
		head_inf = data[0].header
		wcs = awc.WCS(head_inf)
		cenx, ceny = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		R_ph = rad2asec / (Test_model.angular_diameter_distance(z_g).value)
		R_p = R_ph / pixel
		Back_lel = sky_03(kk, ra_g, dec_g, z_g)
		sub_img = img - Back_lel

		hdu = fits.PrimaryHDU()
		hdu.data = sub_img
		hdu.header = data[0].header
		hdu.writeto('/mnt/ddnfs/data_users/cxkttwl/ICL/data/' + 
			'/sky_sub_img/Revis-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]), overwrite = True)

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
		sky_subtract(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		commd.Barrier()	
	t1 = time.time() - t0
	print('t = ', t1)

if __name__ == "__main__":
	main()