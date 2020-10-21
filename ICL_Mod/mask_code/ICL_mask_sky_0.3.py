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
##
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
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'

A_lambd = np.array([5.155, 3.793, 2.751, 2.086, 1.479])
l_wave = np.array([3551, 4686, 6166, 7480, 8932])

sb_lim = np.array([25, 25, 24.5, 24, 22.9]) # SB limit at z_ref
zopt = np.array([22.46, 22.5, 22.5, 22.5, 22.52]) # zero point
Rv = 3.1
sfd = SFDQuery()

def sky_light():

	band_add = ['r', 'i', 'z']
	band_fil = ['u', 'g', 'r', 'i', 'z']
	load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'

	param_sky = 'default_sky_mask.sex'
	out_cat = 'default_mask_A.param'
	out_load_sky = './result/sky_mask_test.cat'

	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	for q in range(len(z)):

		for l in range(len(band_fil)):

			pro_f = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band_fil[l], ra[q], dec[q], z[q])
			ra_g = ra[q]
			dec_g = dec[q]
			z_g = z[q]

			data_f = fits.open(pro_f)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
			R_ph = rad2asec/(Test_model.angular_diameter_distance(z[q]).value)
			R_p = R_ph/pixel

			ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
			pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
			BEV = sfd(pos)
			Av = Rv * BEV * 0.86
			Al = A_wave(l_wave[l], Rv) * Av
			img = img*10**(Al / 2.5)

			# mask in sky only
			hdu = fits.PrimaryHDU()
			hdu.data = data_f[0].data
			hdu.header = head_inf
			hdu.writeto('source_data.fits', overwrite = True)
			file_source = './source_data.fits'
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
			a = Kron*A
			b = Kron*B

			ddr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)
			ix = ddr >= 0.95*R_p
			iy = ddr <= 1.15*R_p
			iz = ix & iy
			s_cx = cx[iz]
			s_cy = cy[iz]
			s_a = a[iz]
			s_b = b[iz]
			s_phi = theta[iz]
			s_Num = len(s_b)

			mask_sky = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
			ox = np.linspace(0, 2047, 2048)
			oy = np.linspace(0, 1488, 1489)
			basic_coord = np.array(np.meshgrid(ox,oy))
			major = s_a/2
			minor = s_b/2
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

			hdu = fits.PrimaryHDU()
			hdu.data = mirro_sky
			hdu.header = head_inf
			hdu.writeto(load + 'mask_data/sky_plane/0.3sigma/sky_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band_fil[l], ra_g, dec_g, z[q]),overwrite = True)

			plt.figure()
			ax = plt.imshow(mirro_sky, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
			hsc.ellipses(s_cx, s_cy, w = s_a, h = s_b, rot = s_phi, fc = '', ec = 'r', ls = '--', lw = 0.5)
			hsc.circles(comx, comy, s = comr, fc = '', ec = 'b', ls = '-', lw = 0.5)

			hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
			hsc.circles(cx_BCG, cy_BCG, s = 1.1*R_p, fc = '', ec = 'b', ls = '--')
			plt.scatter(cx_BCG, cy_BCG, s = 10, marker = 'X', facecolors = '', edgecolors = 'r', linewidth = 0.5, alpha = 0.5)
			plt.title('BL_03 mask img ra%.3f dec%.3f z%.3f in %s band' % (ra_g, dec_g, z_g, band_fil[l]))
			plt.xlim(0, mirro_sky.shape[1])
			plt.ylim(0, mirro_sky.shape[0])
			plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_class/sky_03_mask/BL_mask_%s_ra%.3f_dec%.3f_z%.3f.png'%(band_fil[l], ra[q], dec[q], z[q]), dpi = 300)
			plt.close()

		print(q)
	return

def main():
	sky_light()

if __name__ == "__main__":
	main()
