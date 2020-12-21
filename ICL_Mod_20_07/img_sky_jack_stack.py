import time

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from scipy import ndimage
from astropy import cosmology as apcy

from img_stack import stack_func
from img_edg_cut_stack import cut_stack_func
from light_measure import light_measure_Z0_weit, light_measure_weit
from light_measure import jack_SB_func
from light_measure import light_measure_rn_Z0_weit, light_measure_rn_weit

### constants transform
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
G = C.G.value

### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

### observation params (for SDSS case)
pixel = 0.396
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

#####################
def aveg_stack_img(N_sample, data_file, out_file,):

	tt = 0
	with h5py.File(data_file % (tt), 'r') as f:
		tmp_img = np.array(f['a'])

	Nx, Ny = tmp_img.shape[1], tmp_img.shape[0]
	mean_img = np.zeros((Ny, Nx), dtype = np.float32)
	mean_pix_cont = np.zeros((Ny, Nx), dtype = np.float32)

	for nn in range( N_sample ):

		with h5py.File(data_file % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		idnn = np.isnan(tmp_img)
		mean_img[idnn == False] = mean_img[idnn == False] + tmp_img[idnn == False]
		mean_pix_cont[idnn == False] = mean_pix_cont[idnn == False] + 1.

	idzero = mean_pix_cont == 0.
	mean_pix_cont[idzero] = np.nan
	mean_img[idzero] = np.nan
	mean_img = mean_img / mean_pix_cont

	with h5py.File(out_file, 'w') as f:
		f['a'] = np.array( mean_img )

	return

def jack_samp_stack(d_file, id_set, out_file):

	tt = 0
	with h5py.File(d_file % (tt), 'r') as f:
		tmp_img = np.array(f['a'])
	Nx, Ny = tmp_img.shape[1], tmp_img.shape[0]

	sum_array_A = np.zeros( (Ny,Nx), dtype = np.float32)
	count_array_A = np.ones( (Ny,Nx), dtype = np.float32) * np.nan
	p_count_A = np.zeros( (Ny,Nx), dtype = np.float32)

	for jj in id_set:

		with h5py.File(d_file % ( jj ), 'r') as f:
			sub_img = np.array(f['a'])

		id_nn = np.isnan(sub_img)
		idv = id_nn == False
		sum_array_A[idv] = sum_array_A[idv] + sub_img[idv]
		count_array_A[idv] = sub_img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = id_nan == False
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	with h5py.File(out_file, 'w') as f:
		f['a'] = np.array(stack_img)

	return

def SB_pros_func(flux_img, pix_cont_img, sb_file, N_img, n_rbins, id_Z0, z_ref):
	# get the R_max for SB measurement, and R_max will be applied to all subsample
	# (also, can be re-measured based on the stacking imgs)

	lim_r = 0

	for nn in range( N_img ):

		with h5py.File( flux_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

		id_nn = np.isnan(tmp_img)
		eff_y, eff_x = np.where(id_nn == False)
		dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
		dR_max = np.int( dR.max() ) + 1
		lim_r = np.max( [lim_r, dR_max] )

	r_bins = np.logspace(0, np.log10(lim_r), n_rbins)

	for nn in range( N_img ):

		with h5py.File( flux_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])

		with h5py.File( pix_cont_img % nn, 'r') as f:
			tmp_cont = np.array(f['a'])

		xn, yn = np.int( tmp_img.shape[1] / 2), np.int( tmp_img.shape[0] / 2)

		if id_Z0 == True:
			Intns, Angl_r, Intns_err, npix, nratio = light_measure_Z0_weit( tmp_img, tmp_cont, pixel, xn, yn, r_bins)
			sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
			r_arr = Angl_r
		else:
			Intns, phy_r, Intns_err, npix, nratio = light_measure_weit( tmp_img, tmp_cont, pixel, xn, yn, z_ref, r_bins)
			sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
			r_arr = phy_r

		with h5py.File( sb_file % nn, 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err_arr)
			f['nratio'] = np.array(nratio)
			f['npix'] = np.array(npix)

	return

def sky_jack_main_func(id_cen, N_bin, n_rbins, cat_ra, cat_dec, cat_z, img_x, img_y, img_file, band_str, sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_mean = 0, id_cut = False, N_edg = None, id_Z0 = True, z_ref = None, id_sub = True,
	sub_rms = None, J_sub_rms = None,):
	"""
	combining jackknife stacking process, and 
	save : sub-sample (sub-jack-sample) stacking image, pixel conunt array, surface brightness profiles
	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center,
		     2 - stacking by random center
	N_bin : number of jackknife sample
	n_rbins : the number of radius bins (int type)

	cat_ra, cat_dec, cat_z : catalog information about the stacking sample, ra, dec, z
	img_x, img_y : BCG position (in image coordinate)

	img_file : img-data name (include file-name structure:'/xxx/xxx/xxx.xxx')
	band_str : the band of imgs, 'str' type

	sub_img, sub_pix_cont, sub_sb (stacking img, pixel counts array, SB profile): 
	file name (including patch and file name: '/xxx/xxx/xxx.xxx') of individual sub-sample img stacking result 

	J_sub_img, J_sub_pix_cont, J_sub_sb (stacking img, pixel counts array, SB profile): 
	file name (including patch and file name: '/xxx/xxx/xxx.xxx') of jackknife sub-sample img stacking result

	jack_SB_file : file name of the final jackknife stacking SB profile ('/xxx/xxx/xxx.xxx')
	jack_img : mean of the jackknife stacking img ('/xxx/xxx/xxx.xxx')
	jack_cont_arr : mean of the pixel count array ('/xxx/xxx/xxx.xxx')

	id_cut : id_cut == True, cut img edge pixels before stacking, id_cut == False, just stacking original size imgs
	N_edg : the cut region width, in unit of pixel, only applied when id_cut == True, pixels in this region will be set as 
			'no flux' contribution pixels (ie. set as np.nan)
	
	id_Z0 : stacking imgs on observation coordinate (id_Z0 = True) 
			or not (id_Z0 = False, give radius in physical unit, kpc), default is True
	
	z_ref : reference redshift, (used when id_Z0 = False,) stacking imgs at z_ref.
	id_sub : if measure the individual sub-samples profile( id_sub = True), or not (id_sub = False)

	id_mean : 0, 1, 2.  0 - img_add = img; 
	1 - img_add = img - np.mean(img); 2 - img_add = img - np.median(img); Default is id_mean = 0
	
	sub_rms, J_sub_rms : pixel standard deviation of stacking images (for sub-sample and jackknife sub-sample)
	"""
	lis_ra, lis_dec, lis_z = cat_ra, cat_dec, cat_z
	lis_x, lis_y = img_x, img_y

	zN = len(lis_z)
	n_step = zN // N_bin
	id_arr = np.linspace(0, zN - 1, zN)
	id_arr = id_arr.astype(int)

	band_id = band.index( band_str )

	## img stacking
	for nn in range(N_bin):

		if nn == N_bin - 1:
			dot = id_arr[nn * n_step:]
		else:
			dot = id_arr[nn * n_step: (nn + 1) * n_step]

		set_z = lis_z[dot]
		set_ra = lis_ra[dot]
		set_dec = lis_dec[dot]

		set_x = lis_x[dot]
		set_y = lis_y[dot]

		sub_img_file = sub_img % nn
		sub_cont_file = sub_pix_cont % nn
		if sub_rms is not None:
			sub_rms_file = sub_rms % nn
		else:
			sub_rms_file = None

		if id_cut == False:
			stack_func(img_file, sub_img_file, set_z, set_ra, set_dec, band[ band_id ], set_x, set_y, id_cen,
				rms_file = sub_rms_file, pix_con_file = sub_cont_file, id_mean = id_mean,)
		if id_cut == True:
			cut_stack_func(img_file, sub_img_file, set_z, set_ra, set_dec, band[ band_id ], set_x, set_y, id_cen, N_edg,
				rms_file = sub_rms_file, pix_con_file = sub_cont_file, id_mean = id_mean,)

	for nn in range( N_bin ):

		id_arry = np.linspace(0, N_bin -1, N_bin)
		id_arry = id_arry.astype(int)
		jack_id = list(id_arry)
		jack_id.remove(jack_id[nn])
		jack_id = np.array(jack_id)

		d_file = sub_img
		jack_img_file = J_sub_img % nn
		jack_samp_stack(d_file, jack_id, jack_img_file)

		d_file = sub_pix_cont
		jack_cont_file = J_sub_pix_cont % nn
		jack_samp_stack(d_file, jack_id, jack_cont_file)

		if sub_rms is not None:
			d_file = sub_rms
			jack_rms_file = J_sub_rms % nn
			jack_samp_stack(d_file, jack_id, jack_rms_file)

	## SB measurement
	if id_sub == True:
		## sub-samples
		SB_pros_func(sub_img, sub_pix_cont, sub_sb, N_bin, n_rbins, id_Z0, z_ref)

	## jackknife sub-samples
	SB_pros_func(J_sub_img, J_sub_pix_cont, J_sub_sb, N_bin, n_rbins, id_Z0, z_ref)

	## final jackknife SB profile
	tmp_sb = []
	tmp_r = []
	for nn in range( N_bin ):
		with h5py.File(J_sub_sb % nn, 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

			idvx = npix < 1.
			sb_arr[idvx] = np.nan
			r_arr[idvx] = np.nan

			tmp_sb.append(sb_arr)
			tmp_r.append(r_arr)

	## only save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File(jack_SB_file, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

	## calculate the jackknife SB profile and mean of jackknife stacking imgs
	d_file = J_sub_img
	out_file = jack_img
	aveg_stack_img(N_bin, d_file, out_file)

	d_file = J_sub_pix_cont
	out_file = jack_cont_arr
	aveg_stack_img(N_bin, d_file, out_file)

	return

