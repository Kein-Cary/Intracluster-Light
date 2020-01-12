import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pds
import numpy as np
import h5py
import time

import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from matplotlib.patches import Circle
from astropy import cosmology as apcy
from scipy.ndimage import map_coordinates as mapcd
from astropy.io import fits as fits
from scipy import interpolate as interp
from light_measure import light_measure_Z0, light_measure

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

band = ['r', 'i', 'g', 'u', 'z']
sky_SB = [21.04, 20.36, 22.01, 22.30, 19.18]

d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'

## image data flux histogram
N_sum = np.array([2013, 2002, 2008, 2008, 2009]) ## sky-selected samples
for tt in range(3):
	## sky-selected sample
	with h5py.File(load + 'sky_select_img/%s_band_%d_imgs_sky_select.h5' % (band[tt], N_sum[tt]), 'r') as f:
		set_array = np.array(f['a'])
	ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]
	zN = len(z)

	tt_N = 25
	np.random.seed(5) ## 4, 5, 3, 2
	tt0 = np.random.choice(zN, size = tt_N, replace = False)
	set_z = z[tt0]
	set_ra = ra[tt0]
	set_dec = dec[tt0]

	fig = plt.figure( figsize = (20, 20) )
	#fig.suptitle('%s band image flux PDF comparison' % band[tt] )
	gs = gridspec.GridSpec(5,5)

	for jj in range(tt_N):

		ra_g = set_ra[jj]
		dec_g = set_dec[jj]
		z_g = set_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		R_cen = (rad2asec * 0.1 / Da_g) / pixel
		#Z05 
		data_0 = fits.getdata(load + 
			'resample/Zibetti/A_mask/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[tt], ra_g, dec_g, z_g), header = True)
		img_Z05 = data_0[0]
		xc = data_0[1]['CENTER_X']
		yc = data_0[1]['CENTER_Y']

		id_nan = np.isnan(img_Z05)
		id_fals = id_nan == False
		## rule out BCG region
		xn = np.linspace(0, img_Z05.shape[1] - 1, img_Z05.shape[1])
		yn = np.linspace(0, img_Z05.shape[0] - 1, img_Z05.shape[0])
		grd = np.array(np.meshgrid(xn, yn))
		dr = np.sqrt((grd[0,:] - xc )**2 + (grd[1,:] - yc)**2)
		idx = dr >= R_cen
		idu = id_fals & idx
		flux_Z05 = img_Z05[idu]

		idy = dr <= Rpp
		idv = idx & idy & id_fals
		flux_Z05_inC = img_Z05[idv]

		#B11
		data_1 = fits.getdata(load + 
			'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[tt], ra_g, dec_g, z_g), header = True)
		img_B11 = data_1[0]
		xc = data_1[1]['CENTER_X']
		yc = data_1[1]['CENTER_Y']

		id_nan = np.isnan(img_B11)
		id_fals = id_nan == False
		## rule out BCG region
		xn = np.linspace(0, img_B11.shape[1] - 1, img_B11.shape[1])
		yn = np.linspace(0, img_B11.shape[0] - 1, img_B11.shape[0])
		grd = np.array(np.meshgrid(xn, yn))
		dr = np.sqrt((grd[0,:] - xc )**2 + (grd[1,:] - yc)**2)
		idx = dr >= R_cen
		idu = id_fals & idx
		flux_B11 = img_B11[idu]

		idy = dr <= Rpp
		idv = idx & idy & id_fals
		flux_B11_inC = img_B11[idv]

		bins = np.linspace(-0.05, 0.05, 25)

		ext_f05 = flux_Z05[(flux_Z05 >= -0.05) & (flux_Z05 <= 0.05)]
		ext_f11 = flux_B11[(flux_B11 >= -0.05) & (flux_B11 <= 0.05)]
		tot_n_Z05 = len(ext_f05) / len(flux_Z05)
		tot_n_B11 = len(ext_f11) / len(flux_B11)
		'''
		ext_f05 = flux_Z05_inC[(flux_Z05_inC >= -0.05) & (flux_Z05_inC <= 0.05)]
		ext_f11 = flux_B11_inC[(flux_B11_inC >= -0.05) & (flux_B11_inC <= 0.05)]
		tot_n_Z05 = len(ext_f05) / len(flux_Z05_inC)
		tot_n_B11 = len(ext_f11) / len(flux_B11_inC)
		'''
		ax = plt.subplot(gs[ jj // 5, jj % 5 ])
		ax.set_title('%s band ra%.3f dec%.3f z%.3f'% (band[tt], ra_g, dec_g, z_g) )

		ax.hist(flux_Z05, bins = bins, histtype = 'step', color = 'r', density = True, alpha = 0.5, label = 'Z05 sky subtracted [%.2f]' % tot_n_Z05)
		ax.hist(flux_B11, bins = bins, histtype = 'step', color = 'g', density = True, alpha = 0.5, label = 'B11 sky subtracted [%.2f]' % tot_n_B11)
		#ax.hist(flux_Z05_inC, bins = bins, histtype = 'step', color = 'r', density = True, alpha = 0.5, label = 'Z05 sky subtracted [%.2f]' % tot_n_Z05)
		#ax.hist(flux_B11_inC, bins = bins, histtype = 'step', color = 'g', density = True, alpha = 0.5, label = 'B11 sky subtracted [%.2f]' % tot_n_B11)

		ax.axvline(x = np.nanmean(flux_Z05), linestyle = '--', color = 'r', alpha = 0.5, label = 'Average')
		ax.axvline(x = np.nanmedian(flux_Z05), linestyle = ':', color = 'r', alpha = 0.5, label = 'Median')
		ax.axvline(x = np.nanmean(flux_B11), linestyle = '--', color = 'g', alpha = 0.5,)
		ax.axvline(x = np.nanmedian(flux_B11), linestyle = ':', color = 'g', alpha = 0.5,)

		ax.set_xlabel('flux [namggies]')
		ax.set_ylabel('PDF')
		ax.legend(loc = 1, fontsize = 7.5)

	plt.tight_layout()
	plt.savefig(home + 'fig_ZIT/%s_band_img_flux_hist.png' % band[tt], dpi = 300)
	#plt.savefig(home + 'fig_ZIT/%s_band_img_flux_hist_in_Cluster.png' % band[tt], dpi = 300)
	plt.close()

print('Done')
"""
## ski image flux hist.
N_sum = np.array([3308, 3309, 3295, 3308, 3305])
for tt in range(3):
	with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[tt], 'r') as f:
		sub_array = np.array(f['a'])
	ra, dec, z, rich, r_mag = sub_array[0,:], sub_array[1,:], sub_array[2,:], sub_array[3,:], sub_array[4,:]

	zN = len(z)
	tt_N = 25
	np.random.seed(5) ## 4, 5, 3, 2
	tt0 = np.random.choice(zN, size = tt_N, replace = False)
	set_z = z[tt0]
	set_ra = ra[tt0]
	set_dec = dec[tt0]

	fig = plt.figure( figsize = (20, 20) )
	gs = gridspec.GridSpec(5,5)

	for jj in range(tt_N):
		ra_g = set_ra[jj]
		dec_g = set_dec[jj]
		z_g = set_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		'''
		### reproduce sky image
		data = fits.getdata(load + 'sky/sky_resamp/resample_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[tt], ra_g, dec_g, z_g), header = True)
		#data = fits.getdata(load + 'sky/sky_arr/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[tt]), header = True)
		cx, cy = data[1]['CENTER_X'], data[1]['CENTER_Y']
		img = data[0]

		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan

		lx = np.linspace(0, img.shape[1] - 1, img.shape[1])
		ly = np.linspace(0, img.shape[0] - 1, img.shape[0])
		grd = np.array(np.meshgrid(lx, ly))
		dR = np.sqrt((grd[0,:] - cx)**2 + (grd[1,:] - cy)**2)
		id_in = dR <= Rpp		
		# flux within cluster
		id_nan = np.isnan(img)
		id_fals = id_nan == False
		id_pix = id_fals & id_in
		flux_sky = img[id_pix]
		# flux out the cluster
		id_out = id_in == False
		flux_out = img[ id_out & id_fals ]
		'''
		### original sky image
		data = fits.open(d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band[tt], ra_g, dec_g, z_g) )
		sky0 = data[2].data['ALLSKY'][0]
		sky_bl = sky0 * (data[0].header['NMGY'])

		ax = plt.subplot(gs[ jj // 5, jj % 5 ])
		ax.set_title('%s band ra%.3f dec%.3f z%.3f'% (band[tt], ra_g, dec_g, z_g) )
		'''
		ax.hist(flux_sky, bins = 25, density = True, color = 'b', alpha = 0.5, label = 'Cluster region')
		ax.axvline(x = np.nanmean(flux_sky), linestyle = '--', color = 'r', alpha = 0.5, label = 'Mean')
		ax.axvline(x = np.nanmedian(flux_sky), linestyle = '-.', color = 'r', alpha = 0.5, label = 'Median')
		ax.hist(flux_out, bins = 25, density = True, color = 'k', alpha = 0.5, label = 'Out region')
		ax.axvline(x = np.nanmean(flux_out), linestyle = '--', color = 'k', alpha = 0.5, )
		ax.axvline(x = np.nanmedian(flux_out), linestyle = '-.', color = 'k', alpha = 0.5, )
		'''
		ax.hist(sky_bl.flatten(), bins = 25, density = True, color = 'b', alpha = 0.5,)
		ax.axvline(x = np.nanmean(sky_bl), linestyle = '--', color = 'k', alpha = 0.5, label = 'Mean')
		ax.axvline(x = np.nanmedian(sky_bl), linestyle = '-.', color = 'k', alpha = 0.5, label = 'Median')

		ax.set_xlabel('flux [nmaggy]')
		ax.set_ylabel('PDF')
		ax.legend(loc = 1, fontsize = 7.5)

	plt.tight_layout()
	#plt.savefig(load + 'sky/%s_band_sky_flux_hist.png' % band[tt], dpi = 300)
	#plt.savefig(load + 'sky/%s_band_sky_flux_in_Cluster.png' % band[tt], dpi = 300)
	plt.savefig(load + 'sky/%s_band_original_sky.png' % band[tt], dpi = 300)
	plt.close()

print('Done!')
"""
