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
from scipy.stats import binned_statistic as binned

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

band = ['r', 'g', 'i', 'u', 'z']
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'

## image data flux histogram
#N_sum = np.array([2013, 2002, 2008, 2008, 2009]) ## sky-selected samples(1Mpc)
N_sum = np.array([1291, 1286, 1283, 1294, 1287]) ## 0.8Mpc
"""
for tt in range(3):
	## sky-selected sample
	#with h5py.File(load + 'sky_select_img/%s_band_%d_imgs_sky_select.h5' % (band[tt], N_sum[tt]), 'r') as f:
	with h5py.File(load + 'sky_select_img/%s_band_sky_0.8Mpc_select.h5' % band[tt], 'r') as f:
		set_array = np.array(f['a'])
	ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]
	zN = len(z)

	tt_N = 100
	np.random.seed(5) ## 4, 5, 3, 2
	tt0 = np.random.choice(zN, size = tt_N, replace = False)
	set_z = z[tt0]
	set_ra = ra[tt0]
	set_dec = dec[tt0]

	Nbin = 26
	a0, a1 = 1.0, 1.1
	ref_x0, ref_y0 = 2427, 1765

	pdf_set_Z05 = np.zeros((tt_N, Nbin - 1), dtype = np.float)
	pdf_set_B11 = np.zeros((tt_N, Nbin - 1), dtype = np.float)
	flux_set = np.zeros((tt_N, Nbin - 1), dtype = np.float)
	'''
	fig = plt.figure( figsize = (20, 20) )
	gs = gridspec.GridSpec(5,5)
	'''
	for jj in range(tt_N):

		ra_g = set_ra[jj]
		dec_g = set_dec[jj]
		z_g = set_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		R_cen = (rad2asec * 0.1 / Da_g) / pixel
		#################Z05
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
		idx = (dr >= a0 * Rpp) & (dr <= a1 * Rpp)  ## select pixels in 1~1.1Mpc
		idu = id_fals & idx
		flux_Z05 = img_Z05[idu]

		idy = dr <= Rpp
		idv = idy & id_fals
		flux_Z05_inC = img_Z05[idv]

		###############B11
		data_1 = fits.getdata(load + 
			'resample/1_5sigma_larger_R/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[tt], ra_g, dec_g, z_g), header = True)
		img_B11 = data_1[0]
		xc = data_1[1]['CENTER_X']
		yc = data_1[1]['CENTER_Y']

		# read the difference image of this sample
		with h5py.File(load + 'sky_select_img/test_set/0.8Mpc/sky_difference_img_%d_imgs_%s_band.h5' % (N_sum[tt], band[tt]), 'r') as f:
			differ_img = np.array(f['a'])
		la0, la1 = np.int(ref_y0 - yc), np.int(ref_y0 - yc + img_B11.shape[0])
		lb0, lb1 = np.int(ref_x0 - xc), np.int(ref_x0 - xc + img_B11.shape[1])
		differ_add = differ_img[la0 : la1, lb0 : lb1]

		#img_B11 = img_B11 + 0.
		img_B11 = img_B11 + differ_add
		id_nan = np.isnan(img_B11)
		id_fals = id_nan == False

		## rule out BCG region
		xn = np.linspace(0, img_B11.shape[1] - 1, img_B11.shape[1])
		yn = np.linspace(0, img_B11.shape[0] - 1, img_B11.shape[0])
		grd = np.array(np.meshgrid(xn, yn))
		dr = np.sqrt((grd[0,:] - xc )**2 + (grd[1,:] - yc)**2)
		idx = (dr >= a0 * Rpp) & (dr <= a1 * Rpp)  ## select pixels in 1~1.1Mpc
		idu = id_fals & idx
		flux_B11 = img_B11[idu]

		idy = dr <= Rpp
		idv = idy & id_fals
		flux_B11_inC = img_B11[idv]

		if tt == 2:
			f_lim = 0.1
			bins = np.linspace( -f_lim, f_lim, Nbin)
		else:
			f_lim = 0.06
			bins = np.linspace( -f_lim, f_lim, Nbin)
		flux_set[jj, :] = 0.5 * (bins[:-1] + bins[1:])

		ext_f05 = flux_Z05[(flux_Z05 >= -f_lim) & (flux_Z05 <= f_lim)]
		ext_f11 = flux_B11[(flux_B11 >= -f_lim) & (flux_B11 <= f_lim)]
		tot_n_Z05 = len(ext_f05) / len(flux_Z05)
		tot_n_B11 = len(ext_f11) / len(flux_B11)

		pdx0 = binned(flux_Z05, flux_Z05, statistic = 'count', bins = bins, )[0]
		pdx1 = binned(flux_B11, flux_B11, statistic = 'count', bins = bins, )[0]
		pdf_set_Z05[jj,:] = pdx0 / np.nanmax(pdx0)
		pdf_set_B11[jj,:] = pdx1 / np.nanmax(pdx1)

		#ext_f05 = flux_Z05_inC[(flux_Z05_inC >= -0.05) & (flux_Z05_inC <= 0.05)]
		#ext_f11 = flux_B11_inC[(flux_B11_inC >= -0.05) & (flux_B11_inC <= 0.05)]
		#tot_n_Z05 = len(ext_f05) / len(flux_Z05_inC)
		#tot_n_B11 = len(ext_f11) / len(flux_B11_inC)

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
		'''
	'''
	plt.tight_layout()
	plt.savefig(home + '%s_band_img_flux_arn_1Mpc_hist.png' % band[tt], dpi = 300)
	#plt.savefig(home + '%s_band_img_flux_hist.png' % band[tt], dpi = 300)
	#plt.savefig(home + '%s_band_img_flux_hist_in_Cluster.png' % band[tt], dpi = 300)
	plt.close()
	'''
	## stacking the histogram
	m_flux = np.nanmean(flux_set, axis = 0)
	m_pdf_Z05 = np.nanmean(pdf_set_Z05, axis = 0)
	m_pdf_B11 = np.nanmean(pdf_set_B11, axis = 0)

	avera_Z05 = np.sum(m_pdf_Z05 * m_flux) / np.sum(m_pdf_Z05)
	std_Z05 = np.sqrt( np.sum((m_flux - avera_Z05)**2 * m_pdf_Z05) / np.sum(m_pdf_Z05) - avera_Z05**2 )
	avera_B11 = np.sum(m_pdf_B11 * m_flux) / np.sum(m_pdf_B11)
	std_B11 = np.sqrt( np.sum((m_flux - avera_B11)**2 * m_pdf_B11) / np.sum(m_pdf_B11) - avera_B11**2 )

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('%s band flux histogram' % band[tt])
	ax.step(m_flux, m_pdf_Z05, where = 'mid', color = 'r', label = 'Z05', alpha = 0.5)
	ax.axvline(x = avera_Z05, linestyle = '--', color = 'r', label = 'Mean', alpha = 0.5)
	ax.step(m_flux, m_pdf_B11, where = 'mid', color = 'b', label = 'B11', alpha = 0.5)
	ax.axvline(x = avera_B11, linestyle = '--', color = 'b', alpha = 0.5)
	ax.text(0.04, 0.4, s = '$ \\sigma_{Z05} = %.3f$' % std_Z05 + '\n' + '$ \\sigma_{B11} = %.3f$' % std_B11)

	ax.set_xlabel('flux [nmaggies]')
	ax.set_ylabel('PDF')
	ax.legend(loc = 1)
	plt.savefig(home + '%s_band_mean_flux_hist_arn_1Mpc.png' % band[tt], dpi = 300)
	plt.close()
"""
for tt in range(3):
	## sky-selected sample

	with h5py.File(load + 'sky_select_img/%s_band_sky_0.8Mpc_select.h5' % band[tt], 'r') as f:
		set_array = np.array(f['a'])
	ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]
	zN = len(z)

	tt_N = 25
	np.random.seed(5) ## 4, 5, 3, 2
	tt0 = np.random.choice(zN, size = tt_N, replace = False)
	set_z = z[tt0]
	set_ra = ra[tt0]
	set_dec = dec[tt0]

	for jj in range(tt_N):

		ra_g = set_ra[jj]
		dec_g = set_dec[jj]
		z_g = set_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		Rpx = (rad2asec / Da_g) / pixel
		## Z05 method : subtracted sky imgs
		file = load + 'sky_sub_img/Revis-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[tt])
		data_Z05 = fits.open(file)
		img_Z05 = data_Z05[0].data
		head_inf_05 = data_Z05[0].header
		wcs_05 = awc.WCS(head_inf_05)
		cx_Z05, cy_Z05 = wcs_05.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		## B11 method : subtracted sky imgs
		file1 = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band[tt], ra_g, dec_g, z_g)
		data_B11 = fits.open(file1)
		img_B11 = data_B11[0].data
		head_inf_11 = data_B11[0].header
		wcs_11 = awc.WCS(head_inf_11)
		cx_B11, cy_B11 = wcs_11.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		plt.figure(figsize = (12, 6))
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)

		clust_0 = Circle(xy = (cx_Z05, cy_Z05), radius = Rpx, fill = False, ec = 'r', alpha = 0.5,)
		clust_1 = Circle(xy = (cx_B11, cy_B11), radius = Rpx, fill = False, ec = 'r', alpha = 0.5,)
		ax0.set_title('[Z05 method]%s band ra%.3f dec%.3f z%.3f' % (band[tt], ra_g, dec_g, z_g),)
		tf = ax0.imshow(img_Z05, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'flux[nmaggy]')
		ax0.add_patch(clust_0)
		ax0.set_xlim(0, img_Z05.shape[1])
		ax0.set_ylim(0, img_Z05.shape[0])

		ax1.set_title('[B11 method]%s band ra%.3f dec%.3f z%.3f' % (band[tt], ra_g, dec_g, z_g),)
		tf = ax1.imshow(img_B11, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = 'flux[nmaggy]')
		ax1.add_patch(clust_1)
		ax1.set_xlim(0, img_B11.shape[1])
		ax1.set_ylim(0, img_B11.shape[0])

		plt.tight_layout()
		plt.savefig(home + 'img_%s_band_ra%.3f_dec%.3f_z%.3f.png' % (band[tt], ra_g, dec_g, z_g), dpi = 300)
		plt.close()

print('Done')