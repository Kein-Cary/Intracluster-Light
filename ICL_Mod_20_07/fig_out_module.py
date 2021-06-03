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

### === ### 2D signal combine
def ri_2D_signal( r_img_file, r_rms_file, i_img_file, i_rms_file, rand_r_img_file, rand_i_img_file, 
	r_BG_file, i_BG_file, fig_title, out_fig_file, z_ref, pixel):

	import matplotlib as mpl
	# mpl.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	from matplotlib.patches import Circle

	"""
	BG_file : parameters file of the Background estimation
	"""
	Da_ref = Test_model.angular_diameter_distance( z_ref ).value
	L_pix = Da_ref * 10**3 * pixel / rad2arcsec

	R1Mpc = 1000 / L_pix
	R2Mpc = 2000 / L_pix
	R3Mpc = 3000 / L_pix

	## r band img
	with h5py.File( r_img_file, 'r') as f:
		r_band_img = np.array( f['a'] )
	with h5py.File( r_rms_file, 'r') as f:
		r_band_rms = np.array( f['a'] )

	inves_r_rms2 = 1 / r_band_rms**2 

	## i band img
	with h5py.File( i_img_file, 'r') as f:
		i_band_img = np.array( f['a'] )
	with h5py.File( i_rms_file, 'r') as f:
		i_band_rms = np.array( f['a'] )

	## random imgs
	with h5py.File( rand_r_img_file, 'r') as f:
		r_rand_img = np.array( f['a'])

	with h5py.File( rand_i_img_file, 'r') as f:
		i_rand_img = np.array( f['a'])

	cat = pds.read_csv( r_BG_file )
	r_offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	r_sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	off_r_band_rand_img = r_rand_img / pixel**2 - r_offD + r_sb_2Mpc


	cat = pds.read_csv( i_BG_file )
	i_offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	i_sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	off_i_band_rand_img = i_rand_img / pixel**2 - i_offD + i_sb_2Mpc

	inves_i_rms2 = 1 / i_band_rms**2

	r_BG_sub_img = r_band_img / pixel**2 - off_r_band_rand_img
	i_BG_sub_img = i_band_img / pixel**2 - off_i_band_rand_img

	cen_x, cen_y = np.int( r_band_img.shape[1] / 2 ), np.int( r_band_img.shape[0] / 2 )
	weit_img = ( r_BG_sub_img * inves_r_rms2 + i_BG_sub_img * inves_i_rms2 ) / ( inves_r_rms2 + inves_i_rms2 )

	###
	cut_L = np.int( 1e3 / L_pix )

	cut_img = weit_img[ cen_y - cut_L: cen_y + cut_L, cen_x - cut_L: cen_x + cut_L ]

	filt_img_0 = ndimage.gaussian_filter( cut_img, sigma = 3,)
	mag_map_0 = 22.5 - 2.5 * np.log10( filt_img_0 )

	filt_img_1 = ndimage.gaussian_filter( cut_img, sigma = 7,)
	mag_map_1 = 22.5 - 2.5 * np.log10( filt_img_1 )

	filt_img_2 = ndimage.gaussian_filter( cut_img, sigma = 11,)
	mag_map_2 = 22.5 - 2.5 * np.log10( filt_img_2 )

	filt_img_3 = ndimage.gaussian_filter( cut_img, sigma = 17,)
	mag_map_3 = 22.5 - 2.5 * np.log10( filt_img_3 )

	filt_img_4 = ndimage.gaussian_filter( cut_img, sigma = 21,)
	mag_map_4 = 22.5 - 2.5 * np.log10( filt_img_4 )

	## color_lis
	color_str = []
	for jj in range( 7 ):
		color_str.append( mpl.cm.autumn_r(jj / 6) )

	me_map = mpl.colors.ListedColormap( color_str )
	c_bounds = [ 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 32.5]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	fig = plt.figure()
	ax = fig.add_axes([ 0.05, 0.10, 0.90, 0.80])
	ax1 = fig.add_axes([ 0.82, 0.10, 0.02, 0.80])

	ax.set_title( fig_title )

	ax.imshow( cut_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax.contour( mag_map_0, origin = 'lower', levels = [26, 100], alpha = 0.75,
		colors = [ color_str[0], color_str[-1] ] )

	cs = ax.contour( mag_map_1, origin = 'lower', levels = [27, 100], alpha = 0.75, 
		colors = [ color_str[1], color_str[-1] ] )

	cs = ax.contour( mag_map_2, origin = 'lower', levels = [28, 100], alpha = 0.75, 
		colors = [ color_str[2], color_str[-1] ] )

	cs = ax.contour( mag_map_3, origin = 'lower', levels = [29, 100], alpha = 0.75, 
		colors = [ color_str[3], color_str[-1] ] )

	cs = ax.contour( mag_map_4, origin = 'lower', levels = [30, 32, 100,], alpha = 0.75, 
		colors = [ color_str[4], color_str[5], color_str[-1] ] )

	cbs = mpl.colorbar.ColorbarBase( ax = ax1, cmap = me_map, norm = norm, extend = 'neither', ticks = [26, 27, 28, 29, 30, 32],
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['26', '27', '28', '29', '30', '32'] )

	clust = Circle(xy = (cut_L, cut_L), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax.add_patch(clust)
	clust = Circle(xy = (cut_L, cut_L), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5, label = '0.5Mpc')
	ax.add_patch(clust)

	ax.set_xlim(0, cut_L * 2)
	ax.set_ylim(0, cut_L * 2)

	## # of pixels pre 100kpc
	ax.set_xticklabels( labels = [] ) ## ignore the major axis_ticks
	ax.set_yticklabels( labels = [] )

	n200 = 200 / L_pix

	ticks_0 = np.arange( cut_L, 0, -1 * n200)
	ticks_1 = np.arange( cut_L, cut_L * 2, n200)
	ticks = np.r_[ ticks_0[::-1], ticks_1[1:] ]

	tick_R = np.r_[ np.arange(800, 0, -200), np.arange(0, 1000, 200) ]
	tick_lis = [ '%d' % ll for ll in tick_R ]

	ax.set_xticks( ticks, minor = True, )
	ax.set_xticklabels( labels = tick_lis, minor = True,)

	ax.set_yticks( ticks, minor = True )
	ax.set_yticklabels( labels = tick_lis, minor = True,)
	ax.tick_params( axis = 'both', which = 'major', direction = 'in',)

	ax.set_xlabel( 'kpc' )
	ax.set_ylabel( 'kpc' )

	ax.legend( loc = 1, fontsize = 8)

	plt.savefig( out_fig_file, dpi = 300)
	plt.close()

	return

def BG_sub_2D_signal( img_file, random_img_file, BG_file, z_ref, pixel, band_str, out_fig_name):

	import matplotlib as mpl
	# mpl.use('Agg')
	import matplotlib.pyplot as plt
	import matplotlib.gridspec as gridspec
	from matplotlib.patches import Circle

	Da_ref = Test_model.angular_diameter_distance( z_ref ).value
	L_pix = Da_ref * 10**3 * pixel / rad2arcsec
	R1Mpc = 1000 / L_pix

	## flux imgs
	with h5py.File( img_file, 'r') as f:
		tmp_img = np.array( f['a'])
	cen_x, cen_y = np.int( tmp_img.shape[1] / 2 ), np.int( tmp_img.shape[0] / 2 )

	idnn = np.isnan( tmp_img )
	idy_lim, idx_lim = np.where(idnn == False)
	x_lo_lim, x_up_lim = idx_lim.min(), idx_lim.max()
	y_lo_lim, y_up_lim = idy_lim.min(), idy_lim.max()

	## random imgs
	with h5py.File( random_img_file, 'r') as f:
		rand_img = np.array( f['a'])
	xn, yn = np.int( rand_img.shape[1] / 2 ), np.int( rand_img.shape[0] / 2 )

	idnn = np.isnan( rand_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
	y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()

	## BG-estimate params
	cat = pds.read_csv( BG_file )
	offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
	sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
	BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	idnn = np.isnan( BG_sub_img )
	idy_lim, idx_lim = np.where( idnn == False)
	x_lo_cut, x_up_cut = idx_lim.min(), idx_lim.max()
	y_lo_cut, y_up_cut = idy_lim.min(), idy_lim.max()


	cut_img = tmp_img[ y_lo_lim: y_up_lim + 1, x_lo_lim: x_up_lim + 1 ] / pixel**2
	id_nan = np.isnan( cut_img )
	cut_img[id_nan] = 0.

	cut_rand = rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ] / pixel**2
	id_nan = np.isnan( cut_rand )
	cut_rand[id_nan] = 0.

	cut_off_rand = shift_rand_img[ y_lo_eff: y_up_eff + 1, x_lo_eff: x_up_eff + 1 ]
	id_nan = np.isnan( cut_off_rand )
	cut_off_rand[id_nan] = 0.

	cut_BG_sub_img = BG_sub_img[ y_lo_cut: y_up_cut + 1, x_lo_cut: x_up_cut + 1 ]
	id_nan = np.isnan( cut_BG_sub_img )
	cut_BG_sub_img[id_nan] = 0.

	### figs of 2D signal
	color_str = []
	for jj in range( 9 ):
		color_str.append( mpl.cm.autumn_r( jj / 9 ) )

	color_lis = []
	for jj in np.arange(0, 90, 10):
		color_lis.append( mpl.cm.rainbow_r( jj / 80 ) )


	filt_rand = ndimage.gaussian_filter( cut_rand, sigma = 65,)
	filt_rand_mag = 22.5 - 2.5 * np.log10( filt_rand )

	filt_off_rand = ndimage.gaussian_filter( cut_off_rand, sigma = 65,)
	filt_off_rand_mag = 22.5 - 2.5 * np.log10( filt_off_rand )


	filt_img = ndimage.gaussian_filter( cut_img, sigma = 65,) # sigma = 105,)
	filt_mag = 22.5 - 2.5 * np.log10( filt_img )

	filt_BG_sub_img = ndimage.gaussian_filter( cut_BG_sub_img, sigma = 65,) # sigma = 105,)
	filt_BG_sub_mag = 22.5 - 2.5 * np.log10( filt_BG_sub_img )


	fig = plt.figure( figsize = (18, 12) )
	ax0 = fig.add_axes( [0.03, 0.55, 0.40, 0.40] )
	cb_ax0 = fig.add_axes( [0.41, 0.55, 0.02, 0.40] )

	ax1 = fig.add_axes( [0.52, 0.55, 0.40, 0.40] )
	cb_ax1 = fig.add_axes( [0.90, 0.55, 0.02, 0.40] )

	ax2 = fig.add_axes( [0.03, 0.05, 0.40, 0.40] )
	cb_ax2 = fig.add_axes( [0.41, 0.05, 0.02, 0.40] )

	ax3 = fig.add_axes( [0.52, 0.05, 0.40, 0.40] )
	cb_ax3 = fig.add_axes( [0.90, 0.05, 0.02, 0.40] )

	levels_0 = np.linspace(28, 29, 6)

	## cluster imgs before BG subtract
	ax0.set_title( 'stacking cluster image' )
	tf = ax0.imshow( cut_img, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax0.contour( filt_mag, origin = 'lower',  levels = levels_0, colors = color_str[:6], extent = (0, x_up_lim + 1 - x_lo_lim, 0, y_up_lim + 1 - y_lo_lim ), )

	#c_bounds = np.r_[ levels_0[0] - 0.01, levels_0 + 0.01 ]
	c_bounds = np.r_[ levels_0[0] - 0.1, levels_0 + 0.1 ]
	me_map = mpl.colors.ListedColormap( color_str[:6] )
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax0, cmap = me_map, norm = norm, extend = 'neither', ticks = levels_0,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in levels_0] )

	clust = Circle( xy = (cen_x - x_lo_lim, cen_y - y_lo_lim), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_lim, cen_y - y_lo_lim), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_lim, cen_y - y_lo_lim), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5,)
	ax0.add_patch(clust)

	ax0.set_xlim( 0, x_up_lim + 1 - x_lo_lim )
	ax0.set_ylim( 0, y_up_lim + 1 - y_lo_lim )

	ax0.set_xticklabels( labels = [] )
	ax0.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_lim, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_lim, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax0.set_xticks( x_ticks, minor = True, )
	ax0.set_xticklabels( labels = tick_lis, minor = True,)
	ax0.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_lim, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_lim, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax0.set_yticks( y_ticks, minor = True )
	ax0.set_yticklabels( labels = tick_lis, minor = True,)
	ax0.set_ylabel( 'Mpc' )
	ax0.tick_params( axis = 'both', which = 'major', direction = 'in',)

	## cluster imgs after BG-subtraction
	ax2.set_title( 'stacking cluster image - background image')	
	tf = ax2.imshow( cut_BG_sub_img, origin  ='lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax2.contour( filt_BG_sub_mag, origin = 'lower',  levels = levels_0, colors = color_str[:6], 
		extent = (0, x_up_cut + 1 - x_lo_cut, 0, y_up_cut + 1 - y_lo_cut ), )

	#c_bounds = np.r_[ levels_0[0] - 0.01, levels_0 + 0.01 ]
	c_bounds = np.r_[ levels_0[0] - 0.1, levels_0 + 0.1 ]
	me_map = mpl.colors.ListedColormap( color_str[:6] )
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax2, cmap = me_map, norm = norm, extend = 'neither', ticks = levels_0,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in levels_0] )


	clust = Circle( xy = (cen_x - x_lo_cut, cen_y - y_lo_cut), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax2.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_cut, cen_y - y_lo_cut), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5,label = '0.5Mpc')
	ax2.add_patch(clust)
	clust = Circle( xy = (cen_x - x_lo_cut, cen_y - y_lo_cut), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5, label = '2Mpc')
	ax2.add_patch(clust)

	ax2.legend( loc = 1 )
	ax2.set_xlim( 0, x_up_cut + 1 - x_lo_cut )
	ax2.set_ylim( 0, y_up_cut + 1 - y_lo_cut )

	## # of pixels pre 100kpc
	ax2.set_xticklabels( labels = [] ) ## ignore the major axis_ticks
	ax2.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_cut, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_cut, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax2.set_xticks( x_ticks, minor = True, )
	ax2.set_xticklabels( labels = tick_lis, minor = True,)
	ax2.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_cut, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_cut, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax2.set_yticks( y_ticks, minor = True )
	ax2.set_yticklabels( labels = tick_lis, minor = True,)
	ax2.set_ylabel( 'Mpc' )
	ax2.tick_params( axis = 'both', which = 'major', direction = 'in',)

	ax2.set_xlabel( 'Mpc' )
	ax2.set_ylabel( 'Mpc' )

	### random image
	if band_str == 'r':
		tt_lis = np.linspace( 28.54, 28.64, 6 )
	if band_str == 'g':
		tt_lis = np.linspace( 28.77, 28.86, 6 )
	if band_str == 'i':
		tt_lis = np.linspace( 28.15, 28.45, 6 )

	ax1.set_title( 'stacking random image' )
	ax1.imshow( cut_rand, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax1.contour( filt_rand_mag, origin = 'lower',  levels = tt_lis, colors = color_str[:6], 
		extent = (0, x_up_eff + 1 - x_lo_eff, 0, y_up_eff + 1 - y_lo_eff ), )

	me_map = mpl.colors.ListedColormap( color_str[:6] )
	c_bounds = np.r_[ tt_lis[0] - 0.01, tt_lis + 0.01]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax1, cmap = me_map, norm = norm, extend = 'neither', ticks = tt_lis,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in tt_lis] )

	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5, label = '1Mpc')
	ax1.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5, label = '0.5Mpc')
	ax1.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5, label = '2Mpc')
	ax1.add_patch(clust)

	ax1.set_xlim( 0, x_up_eff + 1 - x_lo_eff )
	ax1.set_ylim( 0, y_up_eff + 1 - y_lo_eff )

	# ticks set
	ax1.set_xticklabels( labels = [] )
	ax1.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_eff, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_eff, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax1.set_xticks( x_ticks, minor = True, )
	ax1.set_xticklabels( labels = tick_lis, minor = True,)
	ax1.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_eff, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_eff, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax1.set_yticks( y_ticks, minor = True )
	ax1.set_yticklabels( labels = tick_lis, minor = True,)
	ax1.set_ylabel( 'Mpc' )
	ax1.tick_params( axis = 'both', which = 'major', direction = 'in',)


	### shift random ( based on BG estimate fitting)
	ax3.set_title( 'background image [stacking random image - C]' )
	ax3.imshow( cut_off_rand, origin = 'lower', cmap = 'Greys', vmin = -2e-2, vmax = 3e-2,)

	cs = ax3.contour( filt_off_rand_mag, origin = 'lower',  levels = tt_lis, colors = color_str[:6], 
		extent = (0, x_up_eff + 1 - x_lo_eff, 0, y_up_eff + 1 - y_lo_eff),)

	me_map = mpl.colors.ListedColormap( color_str[:6] )
	c_bounds = np.r_[ tt_lis[0] - 0.01, tt_lis + 0.01]
	norm = mpl.colors.BoundaryNorm( c_bounds, me_map.N )

	cbs = mpl.colorbar.ColorbarBase( ax = cb_ax3, cmap = me_map, norm = norm, extend = 'neither', ticks = tt_lis,
		spacing = 'proportional', orientation = 'vertical', )
	cbs.set_label( 'SB [mag / $arcsec^2$]' )
	cbs.ax.set_yticklabels( ['%.2f' % ll for ll in tt_lis] )


	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = R1Mpc, fill = False, ec = 'k', ls = '-', linewidth = 1.25, alpha = 0.5,)
	ax3.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 0.5 * R1Mpc, fill = False, ec = 'k', ls = '--', linewidth = 1.25, alpha = 0.5,)
	ax3.add_patch(clust)
	clust = Circle( xy = ( xn - x_lo_eff, yn - y_lo_eff), radius = 2 * R1Mpc, fill = False, ec = 'k', ls = '-.', linewidth = 1.25, alpha = 0.5,)
	ax3.add_patch(clust)

	ax3.set_xlim( 0, x_up_eff + 1 - x_lo_eff )
	ax3.set_ylim( 0, y_up_eff + 1 - y_lo_eff )

	# ticks set
	ax3.set_xticklabels( labels = [] )
	ax3.set_yticklabels( labels = [] )

	n500 = 500 / L_pix

	x_ticks_0 = np.arange( xn - x_lo_eff, 0, -1 * n500)
	x_ticks_1 = np.arange( xn - x_lo_eff, cut_rand.shape[1], n500)
	x_ticks = np.r_[ x_ticks_0[::-1], x_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(x_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(x_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax3.set_xticks( x_ticks, minor = True, )
	ax3.set_xticklabels( labels = tick_lis, minor = True,)
	ax3.set_xlabel( 'Mpc' )

	y_ticks_0 = np.arange( yn - y_lo_eff, 0, -1 * n500)
	y_ticks_1 = np.arange( yn - y_lo_eff, cut_rand.shape[0], n500)
	y_ticks = np.r_[ y_ticks_0[::-1], y_ticks_1[1:] ]

	tick_R = np.r_[ np.arange( ( len(y_ticks_0) - 1 ) * 500, 0, -500), np.arange(0, 500 * ( len(y_ticks_1) ), 500) ]
	tick_lis = [ '%.1f' % (ll / 1e3) for ll in tick_R ]

	ax3.set_yticks( y_ticks, minor = True )
	ax3.set_yticklabels( labels = tick_lis, minor = True,)
	ax3.set_ylabel( 'Mpc' )
	ax3.tick_params( axis = 'both', which = 'major', direction = 'in',)

	plt.savefig( out_fig_name, dpi = 300)
	plt.close()

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

