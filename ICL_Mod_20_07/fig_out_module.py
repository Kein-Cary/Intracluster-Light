import h5py
import numpy as np
import pandas as pds
import astropy.constants as C
import astropy.units as U

from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp
from scipy import ndimage

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
DH = vc / H0

# band information of SDSS
band = ['r', 'g', 'i']
l_wave = np.array([6166, 4686, 7480])
mag_add = np.array([0, 0, 0 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]

#**************************#
def WCS_to_pixel_func(ra, dec, header_inf):
	"""
	according to SDSS Early Data Release paper (section 4.2.2 wcs)
	"""
	Ra0 = header_inf['CRVAL1']
	Dec0 = header_inf['CRVAL2']

	row_0 = header_inf['CRPIX2']
	col_0 = header_inf['CRPIX1']

	af = header_inf['CD1_1']
	bf = header_inf['CD1_2']

	cf = header_inf['CD2_1']
	df = header_inf['CD2_2']

	y1 = (ra - Ra0) * np.cos( Dec0 * np.pi / 180 )
	y2 = dec - Dec0

	delt_col = (bf * y2 - df * y1) / ( bf * cf - af * df )
	delt_row = (af * y2 - cf * y1) / ( af * df - bf * cf )

	id_col = col_0 + delt_col
	id_row = row_0 + delt_row

	return id_col, id_row

def pixel_to_WCS_func(x, y, header_inf):

	Ra0 = header_inf['CRVAL1']
	Dec0 = header_inf['CRVAL2']

	row_0 = header_inf['CRPIX2']
	col_0 = header_inf['CRPIX1']

	af = header_inf['CD1_1']
	bf = header_inf['CD1_2']

	cf = header_inf['CD2_1']
	df = header_inf['CD2_2']

	_delta = bf * cf - af * df

	delta_x = x - col_0
	delta_y = y - row_0

	delta_ra = _delta * ( delta_x * af + delta_y * bf ) / _delta
	delta_dec = _delta * ( delta_x * cf + delta_y * df ) / _delta

	dec = Dec0 + delta_dec
	ra = Ra0 + delta_ra / np.cos( Dec0 * np.pi / 180 )

	return ra, dec

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

def sersic_func(r, Ie, re, ndex):
	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

### === ### img grid
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

### === ### covariance and correlation matrix
def BG_sub_cov_func( jk_sub_sb, N_samples, BG_files, out_file, R_lim0, R_lim1):
	"""
	calculate the covariance matrix of BG-sub SB profiles
	"""
	from light_measure import cov_MX_func
	from img_BG_sub_SB_measure import cc_rand_sb_func

	tmp_r, tmp_sb = [], []

	for nn in range( N_samples ):
		with h5py.File( jk_sub_sb % nn, 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]
		idvx = npix < 1.
		sb_arr[idvx] = np.nan
		r_arr[idvx] = np.nan

		cat = pds.read_csv( BG_files )
		( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD) = ( np.array(cat['e_a'])[0], np.array(cat['e_b'])[0], np.array(cat['e_x0'])[0], 
														np.array(cat['e_A'])[0], np.array(cat['e_alpha'])[0],np.array(cat['e_B'])[0], 
														np.array(cat['offD'])[0] )
		I_e, R_e = np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]

		sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)
		full_r_fit = cc_rand_sb_func( r_arr, e_a, e_b, e_x0, e_A, e_alpha, e_B )
		full_BG = full_r_fit - offD + sb_2Mpc
		devi_sb = sb_arr - full_BG

		id_lim = (r_arr >= R_lim0) & (r_arr <= R_lim1)
		tmp_r.append( r_arr[id_lim] )
		tmp_sb.append( devi_sb[id_lim] )

	R_mean, cov_MX, cor_MX = cov_MX_func(tmp_r, tmp_sb, id_jack = True)

	with h5py.File( out_file, 'w') as f:
		f['R_kpc'] = np.array( R_mean )
		f['cov_MX'] = np.array( cov_MX )
		f['cor_MX'] = np.array( cor_MX )

	return

def BG_pro_cov( jk_sub_sb, N_samples, out_file, R_lim0):
	"""
	calculate the covariance matrix of SB profiles before BG-subtraction
	"""
	from light_measure import cov_MX_func

	tmp_r = []
	tmp_sb = []

	for mm in range( N_samples ):

		with h5py.File( jk_sub_sb % mm, 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

		idvx = npix < 1.
		sb_arr[idvx] = np.nan

		idux = r_arr >= R_lim0
		tt_r = r_arr[idux]
		tt_sb = sb_arr[idux]

		tmp_r.append( tt_r )
		tmp_sb.append( tt_sb )

	R_mean, cov_MX, cor_MX = cov_MX_func(tmp_r, tmp_sb, id_jack = True,)

	with h5py.File( out_file, 'w') as f:
		f['cov_Mx'] = np.array( cov_MX )
		f['cor_Mx'] = np.array( cor_MX )
		f['R_kpc'] = np.array( R_mean )

	return


### === ### 1D profile fitting
def color_func( flux_arr_0, flux_err_0, flux_arr_1, flux_err_1):

	mag_arr_0 = 22.5 - 2.5 * np.log10( flux_arr_0 )
	mag_arr_1 = 22.5 - 2.5 * np.log10( flux_arr_1 )

	color_pros = mag_arr_0 - mag_arr_1

	sigma_0 = 2.5 * flux_err_0 / (np.log(10) * flux_arr_0 )
	sigma_1 = 2.5 * flux_err_1 / (np.log(10) * flux_arr_1 )

	color_err = np.sqrt( sigma_0**2 + sigma_1**2 )

	return color_pros, color_err

def SB_to_Lumi_func(sb_arr, obs_z, band_str):
	"""
	sb_arr : need in terms of absolute magnitude, in AB system
	"""
	if band_str == 'r':
		Mag_dot = Mag_sun[0]

	if band_str == 'g':
		Mag_dot = Mag_sun[1]

	if band_str == 'i':
		Mag_dot = Mag_sun[2]

	# luminosity, in unit of  L_sun / pc^2
	Lumi = 10**( -0.4 * (sb_arr - Mag_dot + 21.572 - 10 * np.log10( obs_z + 1 ) ) )

	return Lumi

def arr_jack_func(SB_array, R_array, N_sample):
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

	### limit the radius bin contribution at least 1/10 * N_sample
	id_min = N_img >= np.int(N_sample / 10)
	lim_r = Stack_R[id_min]
	lim_R = np.nanmax(lim_r)

	return Stack_R, Stack_SB, jk_Stack_err, lim_R

### === ### 2D array, centeriod mean
def centric_2D_aveg(data, weit_data, pix_size, cx, cy, z0, R_bins):

	Da0 = Test_model.angular_diameter_distance(z0).value ## in unit 'Mpc'
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	theta = np.arctan2((pix_id[1,:]-cy), (pix_id[0,:]-cx))
	chi = theta * 180 / np.pi

	rbin = R_bins # have been divided bins, in unit of pixels
	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc

	aveg_arr = np.ones( (Ny, Nx), dtype = np.float32 )

	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	dr = np.sqrt( ( (2 * pix_id[0] + 1) / 2 - (2 * xn + 1) / 2 )**2 + ( (2 * pix_id[1] + 1) / 2 - (2 * yn + 1) / 2)**2 )

	for k in range( 1,len(rbin) - 1):

		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		N_phi = np.int(360 / d_phi) + 1
		phi = np.linspace(0, 360, N_phi)
		phi = phi - 180

		ir = (dr >= rbin[k]) & (dr < rbin[k + 1])
		bool_sum = np.sum(ir)

		if bool_sum == 0:
			continue
		else:
			weit_arr = weit_data[ir]

			samp_flux = data[ir]
			samp_chi = chi[ir]
			tot_flux = np.nansum(samp_flux * weit_arr) / np.nansum(weit_arr)

			aveg_arr[ir] = tot_flux + 0.

	## center pixel
	aveg_arr[yn, xn] = data[yn, xn]

	id_Nan = np.isnan( data )
	aveg_arr[id_Nan] = np.nan

	bins_edgs = np.r_[ np.sqrt( pix_size**2 / np.pi ), rbin[1:] * pix_size ]

	return bins_edgs, aveg_arr
