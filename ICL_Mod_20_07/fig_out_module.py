import h5py
import numpy as np
import pandas as pds
import astropy.constants as C
import astropy.units as U

from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp

#constant
rad2arcsec = U.rad.to(U.arcsec)

# cosmology model
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

# band information of SDSS
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

### img grid
def cc_grid_img( img_data, N_stepx, N_stepy):

	binx = img_data.shape[1] // N_stepx ## bin number along 2 axis
	biny = img_data.shape[0] // N_stepy

	beyon_x = img_data.shape[1] - binx * N_stepx ## for edge pixels divid
	beyon_y = img_data.shape[0] - biny * N_stepy

	odd_x = np.ceil(beyon_x / binx)
	odd_y = np.ceil(beyon_y / biny)

	n_odd_x = beyon_x // odd_x
	n_odd_y = beyon_y // odd_y

	d_odd_x = beyon_x - odd_x * n_odd_x
	d_odd_y = beyon_y - odd_y * n_odd_y

	# get the bin width
	wid_x = np.zeros(binx, dtype = np.float32)
	wid_y = np.zeros(biny, dtype = np.float32)
	for kk in range(binx):
		if kk == n_odd_x :
			wid_x[kk] = N_stepx + d_odd_x
		elif kk < n_odd_x :
			wid_x[kk] = N_stepx + odd_x
		else:
			wid_x[kk] = N_stepx

	for kk in range(biny):
		if kk == n_odd_y :
			wid_y[kk] = N_stepy + d_odd_y
		elif kk < n_odd_y :
			wid_y[kk] = N_stepy + odd_y
		else:
			wid_y[kk] = N_stepy

	# get the bin edge
	lx = np.zeros(binx + 1, dtype = np.int32)
	ly = np.zeros(biny + 1, dtype = np.int32)
	for kk in range(binx):
		lx[kk + 1] = lx[kk] + wid_x[kk]
	for kk in range(biny):
		ly[kk + 1] = ly[kk] + wid_y[kk]

	patch_mean = np.zeros( (biny, binx), dtype = np.float )
	patch_pix = np.zeros( (biny, binx), dtype = np.float )
	patch_S0 = np.zeros( (biny, binx), dtype = np.float )
	patch_Var = np.zeros( (biny, binx), dtype = np.float )
	for nn in range( biny ):
		for tt in range( binx ):

			sub_flux = img_data[ly[nn]: ly[nn + 1], lx[tt]: lx[tt + 1] ]
			id_nn = np.isnan(sub_flux)

			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )
			patch_S0[nn,tt] = (ly[nn + 1] - ly[nn]) * (lx[tt + 1] - lx[tt])

	return patch_mean, patch_pix, patch_Var, patch_S0, lx, ly

def grid_img( img_data, N_stepx, N_stepy):

	ly = np.arange(0, img_data.shape[0], N_stepy)
	ly = np.r_[ly, img_data.shape[0] - N_stepy, img_data.shape[0] ]
	lx = np.arange(0, img_data.shape[1], N_stepx)
	lx = np.r_[lx, img_data.shape[1] - N_stepx, img_data.shape[1] ]

	lx = np.delete(lx, -1)
	lx = np.delete(lx, -2)
	ly = np.delete(ly, -1)
	ly = np.delete(ly, -2)

	patch_mean = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_pix = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_Var = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	for nn in range( len(ly) ):
		for tt in range( len(lx) ):

			sub_flux = img_data[ly[nn]: ly[nn] + N_stepy, lx[tt]: lx[tt] + N_stepx]
			id_nn = np.isnan(sub_flux)
			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )

	return patch_mean, patch_pix, patch_Var, lx, ly

def zref_BCG_pos_func( cat_file, z_ref, out_file, pix_size,):
	"""
	this part use for calculate BCG position after pixel resampling. 
	"""
	dat = pds.read_csv( cat_file )
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Da_z = Test_model.angular_diameter_distance(z).value
	Da_ref = Test_model.angular_diameter_distance(z_ref).value

	L_ref = Da_ref * pix_size / rad2arcsec
	L_z = Da_z * pix_size / rad2arcsec
	eta = L_ref / L_z

	ref_bcgx = clus_x / eta
	ref_bcgy = clus_y / eta

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ra, dec, z, ref_bcgx, ref_bcgy]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return

### stacking region select based on the grid variance
def SN_lim_region_select( stack_img, lim_img_id, lim_set, grd_len,):
	"""
	lim_img_id : use for marker selected region, 
				0 -- use blocks' mean value
				1 -- use blocks' Variance
				2 -- use blocks' effective pixel number 
				3 -- use blocks' effective pixel fraction
				(check the order of return array of cc_grid_img function)
	lim_set : threshold for region selection
	grd_len : block edges for imgs division
	"""
	id_nan = np.isnan( stack_img )
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = stack_img[y_low: y_up+1, x_low: x_up + 1]
	grid_patch = cc_grid_img(dpt_img, grd_len, grd_len,)
	lx, ly = grid_patch[-2], grid_patch[-1]

	lim_img = grid_patch[ lim_img_id ]
	id_region = lim_img >= lim_set

	Nx, Ny = lim_img.shape[1], lim_img.shape[0]
	tp_block = np.ones((Ny, Nx), dtype = np.float32)
	tp_block[id_region] = np.nan

	copy_img = dpt_img.copy()

	for ll in range( Ny ):
		for pp in range( Nx ):

			idx_cen = (pp >= 17) & (pp <= 25)
			idy_cen = (ll >= 12) & (ll <= 17)
			id_cen = idx_cen & idy_cen

			if id_cen == True:
				continue
			else:
				lim_x0, lim_x1 = lx[pp], lx[ pp + 1 ]
				lim_y0, lim_y1 = ly[ll], ly[ ll + 1 ]

				id_NAN = np.isnan( tp_block[ll,pp] )

				if id_NAN == True:
					copy_img[lim_y0: lim_y1, lim_x0: lim_x1] = np.nan
				else:
					continue

	SN_mask = np.ones((dpt_img.shape[0], dpt_img.shape[1]), dtype = np.float32)
	id_nn = np.isnan(copy_img)
	SN_mask[id_nn] = np.nan

	all_mask = np.ones((stack_img.shape[0], stack_img.shape[1]), dtype = np.float32)
	id_nn = np.isnan(stack_img)
	all_mask[id_nn] = np.nan
	all_mask[y_low: y_up+1, x_low: x_up + 1] = SN_mask

	return all_mask

def arr_jack_func(SB_array, R_array, N_sample,):
	"""
	SB_array : y-data for jackknife resampling
	R_array : x-data for jackknife resampling
	( SB_array, R_array : list type )

	N_sample : number of sub-samples
	"""
	dx_r = np.array(R_array)
	dy_sb = np.array(SB_array)

	n_r = dx_r.shape[1]
	Len = np.zeros( n_r, dtype = np.float32)
	for nn in range( n_r ):
		tmp_I = dy_sb[:,nn]
		idnn = np.isnan(tmp_I)
		Len[nn] = N_sample - np.sum(idnn)

	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_SB = np.nanmean(dy_sb, axis = 0)
	std_Stack_SB = np.nanstd(dy_sb, axis = 0)

	### only calculate r bins in which sub-sample number larger than one
	id_one = Len > 1
	Stack_R = Stack_R[ id_one ]
	Stack_SB = Stack_SB[ id_one ]
	std_Stack_SB = std_Stack_SB[ id_one ]
	N_img = Len[ id_one ]
	jk_Stack_err = np.sqrt(N_img - 1) * std_Stack_SB

	### limit the radius bin contribution at least 1/3 * N_sample
	id_min = N_img >= np.int(N_sample / 3)
	lim_r = Stack_R[id_min]
	lim_R = np.nanmax(lim_r)

	return Stack_R, Stack_SB, jk_Stack_err, lim_R

def arr_slope_func(fdens, r, wind_len, poly_order, id_log = True,):
	"""
	wind_len = 9
	poly_order = 3
	"""
	f_signal = signal.savgol_filter( fdens, window_length = wind_len, polyorder = poly_order, 
									deriv = 0, delta = 1.0, axis = -1, mode = 'interp', cval = 0.0,)
	sign_x = 0.5 * ( r[1:] + r[:-1])
	dx = r[1:] - r[:-1]

	df_dx = ( f_signal[1:] - f_signal[:-1] ) / dx
	df_dlogx = df_dx * np.log( 10 ) * sign_x

	mf = np.log(10) * 0.5 * ( f_signal[1:] + f_signal[:-1] )
	dlogf_dlogx = df_dlogx / mf

	if id_log == True:
		f_slope = dlogf_dlogx.copy()

	else:
		f_slope = df_dx.copy()

	return sign_x, f_slope

