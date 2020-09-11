import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

def combine_func(d_file, out_file, z_set, ra_set, dec_set, band, img_x, img_y, id_cen, rms_file=None, pix_con_file=None):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img (include file-name structure:'/xxx/xxx/xxx.xxx')
	img_x, img_y : BCG position (in image coordinate)
	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center
	rms_file : stacking img pixel variance 
	pix_con_file : the pixel counts in each stacking img pixel
	"""
	lis_N = len(z_set)

	if id_cen == 0:
		x0, y0 = 2427, 1765
		Nx = np.linspace(0, 4854, 4855)
		Ny = np.linspace(0, 3530, 3531)
		#?????
		lis_arr = np.zeros( (lis_N, len(Ny), len(Nx)), dtype = np.float32) + np.nan
		sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float32) * np.nan
		p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		pix_f2 = np.zeros((len(Ny), len(Nx)), dtype = np.float32)

	if id_cen == 1:
		x0, y0 = 1024, 744
		Nx = np.linspace(0, 2047, 2048)
		Ny = np.linspace(0, 1488, 1489)

		lis_arr = np.zeros( (lis_N, len(Ny), len(Nx)), dtype = np.float32) + np.nan
		sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float32) * np.nan
		p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		pix_f2 = np.zeros((len(Ny), len(Nx)), dtype = np.float32)

	for jj in range( lis_N ):

		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		dev_05_x = img_x[jj] - np.int(img_x[jj])
		dev_05_y = img_y[jj] - np.int(img_y[jj])
		if dev_05_x > 0.5:
			xn = np.int(img_x[jj]) + 1
		else:
			xn = np.int(img_x[jj])

		if dev_05_y > 0.5:
			yn = np.int(img_y[jj]) + 1
		else:
			yn = np.int(img_y[jj])

		file = d_file % (band, ra_g, dec_g, z_g)
		data_A = fits.open( file )
		img_A = data_A[0].data
		head = data_A[0].header

		if id_cen == 0:
			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + img_A.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + img_A.shape[1])

		if id_cen == 1:
			rnx, rny = np.int(img_A.shape[1] / 2), np.int(img_A.shape[0] / 2) ## image center
			la0 = np.int(y0 - rny)
			la1 = np.int(y0 - rny + img_A.shape[0])
			lb0 = np.int(x0 - rnx)
			lb1 = np.int(x0 - rnx + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		lis_arr[jj][la0: la1, lb0: lb1][idv] = img_A[idv]

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]

		## tmp array for rms
		pix_f2[la0: la1, lb0: lb1][idv] = pix_f2[la0: la1, lb0: lb1][idv] + img_A[idv]**2

		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	#combine_img = np.nanmedian(lis_arr, axis = 0)
	combine_img = np.nanmean(lis_arr, axis = 0)
	with h5py.File(out_file, 'w') as f:
		f['a'] = np.array( combine_img )

	## pixel rms
	pix_f2[id_zero] = np.nan
	rms_arr = pix_f2 / p_count_A - stack_img**2
	rms_arr = np.sqrt(rms_arr)

	if rms_file != None:
		with h5py.File(rms_file, 'w') as f:
			f['a'] = np.array(rms_arr)

	if pix_con_file != None:
		with h5py.File(pix_con_file, 'w') as f:
			f['a'] = np.array(p_count_A)

	return

def main():
	import matplotlib as mpl
	mpl.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.patches import Circle, Rectangle
	import matplotlib.gridspec as gridspec

	import astropy.units as U
	import astropy.constants as C

	import time
	import astropy.wcs as awc
	import subprocess as subpro
	import scipy.stats as sts

	from scipy import ndimage
	from astropy import cosmology as apcy
	from scipy.optimize import curve_fit
	from light_measure import light_measure_Z0_weit

	##### median combine test
	pixel = 0.396
	home = '/home/xkchen/data/SDSS/'
	band = ['r', 'g', 'i', 'u', 'z']
	mag_add = np.array([0, 0, 0, -0.04, 0.02])

	dat = pds.read_csv(home + 'selection/tmp/tot_clust_remain_cat.csv')
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	cat_brit = pds.read_csv(home + 'selection/tmp/cluster_to_bright_cat.csv')
	set_ra, set_dec, set_z = np.array(cat_brit.ra), np.array(cat_brit.dec), np.array(cat_brit.z)

	out_ra = ['%.3f' % ll for ll in set_ra]
	out_dec = ['%.3f' % ll for ll in set_dec]

	lis_ra, lis_dec, lis_z = [], [], []
	lis_x, lis_y = [], []
	for ll in range( len(z) ):
		identi = ('%.3f' % ra[ll] in out_ra) & ('%.3f' % dec[ll] in out_dec)
		if identi == True:
			continue
		else:
			lis_ra.append( ra[ll] )
			lis_dec.append( dec[ll] )
			lis_z.append( z[ll] )
			lis_x.append( clus_x[ll] )
			lis_y.append( clus_y[ll] )

	lis_ra = np.array(lis_ra)
	lis_dec = np.array(lis_dec)
	lis_z = np.array(lis_z)
	lis_x = np.array(lis_x)
	lis_y = np.array(lis_y)

	d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
	id_cen = 0 # BCG-stacking

	out_file = home + 'tmp_stack/jack/clust_tot_BCG-stack_median_img.h5'
	cont_file = home + 'tmp_stack/jack/clust_tot_BCG-stack_pix-cont_median.h5'
	combine_func(d_file, out_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, rms_file = None, pix_con_file = cont_file)

	### compare
	with h5py.File(home + 'tmp_stack/jack/clust_tot_BCG-stack_correct_SB.h5', 'r') as f:
		clus_I = np.array(f['sb'])
		clus_I_r = np.array(f['r'])
		clus_I_err = np.array(f['sb_err'])

	with h5py.File(home + 'tmp_stack/jack/clust_tot_BCG-stack_correct.h5', 'r') as f:
		D_stack_img = np.array(f['a'])


	with h5py.File(home + 'tmp_stack/jack/clust_tot_BCG-stack_pix-cont_median.h5', 'r') as f:
		tmp_cont = np.array(f['a'])
	with h5py.File(home + 'tmp_stack/jack/clust_tot_BCG-stack_median_img.h5', 'r') as f:
		tmp_img = np.array(f['a'])

	bins = 95
	xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)
	Intns, Angl_r, Intns_err = light_measure_Z0_weit(tmp_img, tmp_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins),)[:3]
	bcg_I_r, bcg_I, bcg_I_err = Angl_r.copy(), Intns / pixel**2, Intns_err / pixel**2

	plt.figure()
	ax = plt.subplot(111)

	ax.errorbar(clus_I_r, clus_I, yerr = clus_I_err, xerr = None, color = 'r', marker = 'None', ls = '-', 
		ecolor = 'r', alpha = 0.5, label = 'cluster [mean combine]')
	ax.errorbar(bcg_I_r, bcg_I, yerr = bcg_I_err, xerr = None, color = 'b', marker = 'None', ls = '-', 
		ecolor = 'b', alpha = 0.5, label = 'cluster [median combine]')

	ax.set_ylim(3e-4, 3e-2)
	ax.set_yscale('log')
	ax.set_xlim(1e1, 1e3)
	ax.set_xlabel('$ R[arcsec] $')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.legend(loc = 1, frameon = False, fontsize = 8)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(left = 0.15, right = 0.95,)
	plt.savefig('BCG_stack_SB_compare.png', dpi = 300)
	plt.close()

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.03, 0.09, 0.27, 0.85])
	ax1 = fig.add_axes([0.36, 0.09, 0.27, 0.85])
	ax2 = fig.add_axes([0.69, 0.09, 0.27, 0.85])

	ax0.set_title('mean combine')
	tf = ax0.imshow(D_stack_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -0.1, vmax = 0.1,)
	plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')

	ax0.set_title('median combine')
	tf = ax1.imshow(tmp_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -0.1, vmax = 0.1,)
	plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')

	ax2.set_title('mean combine - median combine')
	diff_img = D_stack_img - tmp_img
	tf = ax2.imshow(diff_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -1e-3, vmax = 1e-3,)
	plt.colorbar(tf, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')

	plt.tight_layout()
	plt.savefig('2D_img_compare.png', dpi = 300)
	plt.close()

	raise

if __name__ == "__main__":
	main()
