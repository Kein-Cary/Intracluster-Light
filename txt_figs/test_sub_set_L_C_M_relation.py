import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from scipy import optimize
from scipy import stats as sts
from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

### ... mass profile measure test
from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

from tmp_color_to_mass import get_c2mass_func
from tmp_color_to_mass import jk_sub_SB_func, jk_sub_Mass_func
from tmp_color_to_mass import aveg_mass_pro_func
from tmp_color_to_mass import cumu_mass_func

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

band = ['r', 'g', 'i']
Mag_sun = [ 4.65, 5.11, 4.53 ]
L_wave = np.array([ 6166, 4686, 7480 ])
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

z_ref = 0.25
pixel = 0.396

Dl_ref = Test_model.luminosity_distance( z_ref ).value
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec

### === M/L - color
def lg_linea_func( x, a, b):
	lg_M2L = a + b * x
	return lg_M2L

def resi_func( po, x, y):

	a, b = po[:]
	lg_m2l = lg_linea_func( x, a, b)
	delta = lg_m2l - y
	return delta

def lg_bi_linear_func(x, A, B, C0, D0):
	"""
	x_arr : g-r, r-i, luminosity
	"""
	x_gr, x_ri, x_lgL = x[0], x[1], x[2]
	y = A * x_gr + B * x_ri + C0 * x_lgL + D0
	return y

def resi_bi_line_func( po, x, y):

	A, B, C0, D0 = po[:]
	pre_y = lg_bi_linear_func(x, A, B, C0, D0)
	delta = pre_y - y
	return delta

def cumu_mass_func(rp, surf_mass, N_grid = 100):

	try:
		NR = len(rp)
	except:
		rp = np.array([ rp ])
		NR = len(rp)

	intep_sigma_F = interp.interp1d( rp, surf_mass, kind = 'linear', fill_value = 'extrapolate',)

	cumu_mass = np.zeros( NR, )

	for ii in range( NR ):

		new_rp = np.logspace(0, np.log10( rp[ii] ), N_grid)
		new_mass = intep_sigma_F( new_rp )

		cumu_mass[ ii ] = integ.simps( 2 * np.pi * new_rp * new_mass, new_rp)

	return cumu_mass

def mass_ratio_func( N_samples, scale_f, ref_M_func, low_R_lim, up_R_lim, jk_sub_file, out_file):

	tmp_r, tmp_ratio = [], []

	for nn in range( N_samples ):

		o_dat = pds.read_csv( jk_sub_file % nn,)

		tt_r = np.array( o_dat['R'] )
		tt_M = np.array( o_dat['surf_mass'] )

		tt_M = tt_M * 10**scale_f

		idx_lim = ( tt_r >= np.nanmin( low_R_lim ) ) & ( tt_r <= np.nanmax( up_R_lim ) )

		lim_R = tt_r[ idx_lim ]
		lim_M = tt_M[ idx_lim ]

		com_M = ref_M_func( lim_R )

		sub_ratio = np.zeros( len(tt_r),)
		sub_ratio[ idx_lim ] = lim_M / com_M

		sub_ratio[ idx_lim == False ] = np.nan

		tmp_r.append( tt_r )
		tmp_ratio.append( sub_ratio )

	aveg_R, aveg_ratio, aveg_ratio_err = arr_jack_func( tmp_ratio, tmp_r, N_samples)[:3]

	keys = ['R', 'M/M_tot', 'M/M_tot-err']
	values = [ aveg_R, aveg_ratio, aveg_ratio_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_file )

	return


### === data load
cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
file_s = 'age_bin_fixed_BCG_M'

cat_path = '/home/xkchen/figs/'
BG_path = '/home/xkchen/jupyter/fixed_BCG_M/age_bin/BGs/'
out_path = '/home/xkchen/figs/M2L_fit_age_bin_fixed_bcgM/'

#... relation estimation
for mm in range( 1,2 ):

	#... lg_Mstar
	l_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm] )
	l_obs_z, l_rich  = np.array( l_dat['z']), np.array( l_dat['rich'])
	l_lgM, l_age = np.array( l_dat['lg_Mstar']), np.array( l_dat['BCG_age'] )

	#... mag
	pdat = pds.read_csv( '/home/xkchen/figs/%s_BCG-color.csv' % cat_lis[mm] )
	p_g_mag, p_r_mag, p_i_mag = np.array( pdat['g_mag'] ), np.array( pdat['r_mag'] ), np.array( pdat['i_mag'] )

	gr_ = p_g_mag - p_r_mag
	ri_ = p_r_mag - p_i_mag

	#... Lumi
	pdat = pds.read_csv( '/home/xkchen/figs/%s_BCG_cmag.csv' % cat_lis[mm] )
	i_cMag = np.array( pdat['i_cMag'] )
	i_Lumi = 10**( -0.4 * ( i_cMag - Mag_sun[2] ) )

	r_cMag = np.array( pdat['r_cMag'] )
	r_Lumi = 10**( -0.4 * ( r_cMag - Mag_sun[0] ) )

	g_cMag = np.array( pdat['g_cMag'] )
	g_Lumi = 10**( -0.4 * ( g_cMag - Mag_sun[1] ) )

	L_i = i_Lumi + 0.
	L_g = g_Lumi + 0. 
	L_r = r_Lumi + 0.

	i_Mag = i_cMag + 0.
	g_Mag = g_cMag + 0.
	r_Mag = r_cMag + 0.

	gr_arr = gr_ + 0.
	ri_arr = ri_ + 0.
	lg_Mstar = l_lgM + 0.

	### === fit M = f( g-r, g-i, L_i)
	def points_select():

		based_str = 'i'

		id_R_lim = [ 1 ]
		sum_dex = np.sum( id_R_lim )
		kk = 0

		cp_gr_arr = gr_arr + 0.
		cp_ri_arr = ri_arr + 0.
		cp_lgLi_arr = np.log10( L_i )
		cp_lg_Mstar = lg_Mstar - 2 * np.log10( h )

		sigma = 5
		kk = 0

		while sum_dex > 0:

			put_x = np.array([ cp_gr_arr, cp_ri_arr, cp_lgLi_arr ])

			p0 = [ -0.05, 0.5, 0.02, 0.2 ]
			res_lsq = optimize.least_squares( resi_bi_line_func, x0 = np.array( p0 ), loss = 'cauchy', 
				f_scale = 0.1, args = ( put_x, cp_lg_Mstar),)

			a_fit = res_lsq.x[0]
			b_fit = res_lsq.x[1]
			c_fit = res_lsq.x[2]
			d_fit = res_lsq.x[3]

			fit_M2L = lg_bi_linear_func( put_x, a_fit, b_fit, c_fit, d_fit)

			Var = np.sum( (fit_M2L - cp_lg_Mstar)**2 ) / len( cp_lg_Mstar )
			sigma = np.sqrt( Var )
			sp_R = sts.spearmanr( fit_M2L, cp_lg_Mstar)[0]

			#... fit relation between fitting and obs.
			po = [0.9, 10]
			popt, pcov = optimize.curve_fit( lg_linea_func, xdata = fit_M2L, ydata = cp_lg_Mstar, p0 = po,)
			_a0, _b0 = popt
			com_line = lg_linea_func( fit_M2L, _a0, _b0 )

			dR_com_l = np.abs( _b0 * fit_M2L + _a0 - cp_lg_Mstar) / np.sqrt(1 + _b0**2)
			id_R_lim = dR_com_l >= 2 * sigma

			sum_dex = np.sum( id_R_lim )

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('least square')
			ax.scatter( fit_M2L, cp_lg_Mstar, marker = 'o', s = 1.5, color = 'k', alpha = 0.15, zorder = 100)
			ax.plot( fit_M2L, fit_M2L, 'b--',)
			ax.scatter( fit_M2L[id_R_lim], cp_lg_Mstar[id_R_lim], marker = '*', s = 5.5, color = 'g',)

			ax.annotate( text = '$\\eta = %.3f$' % ( len(cp_gr_arr) / len(gr_arr) ), xy = (0.10, 0.75), 
				xycoords = 'axes fraction',)
			ax.annotate( text = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), 
				xycoords = 'axes fraction',)
			ax.set_xlabel('$ fit \; : \; \\lg \, (M_{\\ast} ) $')
			ax.set_ylabel('$ data \; : \; \\lg \, (M_{\\ast} ) $')
			plt.savefig('/home/xkchen/%s_lgM_diag_selected_%d.png' % (cat_lis[mm], kk), dpi = 300)
			plt.close()

			cp_gr_arr = cp_gr_arr[ id_R_lim == False ]
			cp_ri_arr = cp_ri_arr[ id_R_lim == False ]
			cp_lgLi_arr = cp_lgLi_arr[ id_R_lim == False ]
			cp_lg_Mstar = cp_lg_Mstar[ id_R_lim == False ]

			kk += 1

			out_arr = np.array([ cp_gr_arr, cp_ri_arr, cp_lg_Mstar, cp_lgLi_arr ]).T
			np.savetxt( out_path + '%s_M2L%s_selected_points.txt' % (cat_lis[mm], based_str), 
				out_arr, fmt = '%.8f, %.8f, %.8f, %.8f',)

	# points_select()

	based_str = 'i'

	points = np.loadtxt( out_path + '%s_M2L%s_selected_points.txt' % (cat_lis[mm], based_str), delimiter = ',')
	cp_gr_arr = points[:,0]
	cp_ri_arr = points[:,1]
	cp_lg_Mstar = points[:,2]
	cp_lg_Lumi = points[:,3]

	p0 = [ -0.05, 0.5, 0.02, 0.2 ]
	put_x = np.array([ cp_gr_arr, cp_ri_arr, cp_lg_Lumi ])
	res_lsq = optimize.least_squares( resi_bi_line_func, x0 = np.array( p0 ), loss = 'cauchy',
		f_scale = 0.1, args = ( put_x, cp_lg_Mstar),)

	a_fit = res_lsq.x[0]
	b_fit = res_lsq.x[1]
	c_fit = res_lsq.x[2]
	d_fit = res_lsq.x[3]

	keys = ['a', 'b', 'c', 'd']
	values = [ a_fit, b_fit, c_fit, d_fit ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( out_path + '%s_least-square_M-to-%s-band-Lumi&color.csv' % (cat_lis[mm], based_str),)

	lg_Lumi = np.log10( L_i )
	put_x = np.array( [ gr_arr, ri_arr, lg_Lumi ] )
	fit_M2L = lg_bi_linear_func( put_x, a_fit, b_fit, c_fit, d_fit)

	_lg_M = lg_Mstar - 2 * np.log10( h )
	Var = np.sum( (fit_M2L - _lg_M)**2 ) / len( _lg_M )
	sigma = np.sqrt( Var )
	sp_R = sts.spearmanr( fit_M2L, _lg_M)[0]

	dd_lgM.append( _lg_M )

	bin_x = np.linspace(11.1, 12.3, 7)
	bin_x = np.r_[ 10.8, bin_x ]

	bin_cen = 0.5 * ( bin_x[1:] + bin_x[:-1])
	bin_medi, bin_std = np.array([]), np.array([])
	bin_Ng = np.array([])

	for oo in range( len(bin_x) - 1):
		id_in = ( fit_M2L >= bin_x[oo] ) & ( fit_M2L <= bin_x[oo+1] )
		sub_lgM = _lg_M[id_in]

		bin_medi = np.r_[ bin_medi, np.median( sub_lgM ) ]
		bin_std = np.r_[ bin_std, np.std( sub_lgM ) ]
		bin_Ng = np.r_[ bin_Ng, np.sum(id_in) ]

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('least square')

	# ax.scatter( fit_M2L, cp_lg_Mstar, marker = 'o', s = 1.5, color = 'k', alpha = 0.15, zorder = 100)
	# ax.hist2d( fit_M2L, cp_lg_Mstar, bins = 50, density = True, cmap = 'rainbow', norm = mpl.colors.LogNorm(),)

	ax.scatter( fit_M2L, _lg_M, marker = 'o', s = 1.5, color = 'k', alpha = 0.15, zorder = 100)
	ax.plot( fit_M2L, fit_M2L, 'b--',)

	ax.errorbar( bin_cen[bin_Ng > 1], bin_medi[bin_Ng > 1], yerr = bin_std[bin_Ng > 1], color = 'g', marker = 'o', capsize = 2.5,)

	ax.annotate( text = fig_name[mm], xy = (0.65, 0.15), xycoords = 'axes fraction',)
	ax.annotate( text = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), xycoords = 'axes fraction',)
	ax.annotate( text = 'a = %.3f' % a_fit + '\n' + 'b = %.3f' % b_fit + '\n' + 'c = %.3f' % c_fit + 
		'\n' + 'd = %.3f' % d_fit, xy = (0.10, 0.75), xycoords = 'axes fraction',)
	ax.set_xlabel(
		'$ \\lg \, (M_{\\ast}) = a \\cdot (g-r) + b \\cdot (r-i) + c \\cdot \\lg(L_{%s}) + d $' % (based_str),)
	ax.set_ylabel('$ \\lg \, (M_{\\ast}) $' )
	ax.set_xlim(10.80, 12.25)
	ax.set_ylim(10.50, 12.50)
	plt.savefig('/home/xkchen/%s_M-to-L_estimate.png' % cat_lis[mm], dpi = 300)
	plt.close()

raise

N_samples = 30
band_str = 'gri'

"""
#... mass estimation
for mm in range( 2 ):

	fit_file = out_path + '%s_least-square_M-to-i-band-Lumi&color.csv' % cat_lis[mm]

	sub_SB_file = BG_path + '%s_' % cat_lis[mm] + '%s-band_jack-sub-%d_BG-sub_SB.csv'
	low_R_lim, up_R_lim = 1e0, 1.2e3
	out_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'
	jk_sub_Mass_func( N_samples, band_str, sub_SB_file, low_R_lim, up_R_lim, out_file, Dl_ref, z_ref, 
		fit_file = fit_file)

	jk_sub_m_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'
	jk_aveg_m_file = out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str)
	lgM_cov_file = out_path + '%s_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)
	M_cov_file = out_path + '%s_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % (cat_lis[mm], band_str)

	aveg_mass_pro_func(N_samples, band_str, jk_sub_m_file, jk_aveg_m_file, lgM_cov_file, M_cov_file = M_cov_file)
"""

"""
dpt_R, dpt_L, dpt_M = [], [], []
dpt_M_err, dpt_L_err = [], []
dpt_cumu_M, dpt_cumu_M_err = [], []

for mm in range( 2 ):

	dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)
	aveg_R = np.array(dat['R'])

	aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
	aveg_cumu_m, aveg_cumu_m_err = np.array(dat['cumu_mass']), np.array(dat['cumu_mass_err'])
	aveg_lumi, aveg_lumi_err = np.array(dat['lumi']), np.array(dat['lumi_err'])

	dpt_R.append( aveg_R )
	dpt_M.append( aveg_surf_m )
	dpt_M_err.append( aveg_surf_m_err )

	dpt_L.append( aveg_lumi )
	dpt_L_err.append( aveg_lumi_err )
	dpt_cumu_M.append( aveg_cumu_m )
	dpt_cumu_M_err.append( aveg_cumu_m_err )

plt.figure()
plt.title('%s-based surface mass density profile' % band_str )
plt.plot( dpt_R[0], dpt_M[0], ls = '--', color = 'b', alpha = 0.5, label = fig_name[0])
plt.fill_between( dpt_R[0], y1 = dpt_M[0] - dpt_M_err[0], y2 = dpt_M[0] + dpt_M_err[0], color = 'b', alpha = 0.12,)
plt.plot( dpt_R[1], dpt_M[1], ls = '-', color = 'r', alpha = 0.5, label = fig_name[1])
plt.fill_between( dpt_R[1], y1 = dpt_M[1] - dpt_M_err[1], y2 = dpt_M[1] + dpt_M_err[1], color = 'r', alpha = 0.12,)

plt.xlim( 3e0, 1e3)
plt.xscale('log')
plt.xlabel('R[kpc]', fontsize = 15)
plt.yscale('log')
plt.ylim( 5e3, 2e9)
plt.legend( loc = 1, frameon = False, fontsize = 15,)
plt.ylabel('$\\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)
plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
plt.savefig('/home/xkchen/%s_%s-band_based_surface_mass_profile.png' % (file_s, band_str), dpi = 300)
plt.close()
"""

##... mass profile comparison
band_str = 'gri'
"""
for mm in range( 2 ):

	#... mass profile
	m_dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)
	jk_R = np.array(m_dat['R'])
	surf_m, surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])
	surf_L = np.array( m_dat['lumi'] )

	N_grid = 250
	up_lim_R = 47.69

	cumu_M = cumu_mass_func( jk_R, surf_m, N_grid = N_grid )
	intep_Mf = interp.interp1d( jk_R, cumu_M, kind = 'cubic',)

	M_40 = intep_Mf( up_lim_R )

	#... catalog infor.
	p_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm] )
	p_obs_z, p_rich  = np.array( p_dat['z']), np.array( p_dat['rich'])
	p_lgM, p_age = np.array( p_dat['lg_Mstar']), np.array( p_dat['BCG_age'] )

	lg_Mean = np.log10( np.mean(10**p_lgM / h**2) )
	lg_Medi = np.log10( np.median(10**p_lgM / h**2) )

	devi_Mean = lg_Mean - np.log10( M_40 )
	devi_Medi = lg_Medi - np.log10( M_40 )

	print( 10**devi_Medi )
	print( 10**devi_Mean )

	_cc_surf_M = surf_m * 10**devi_Medi

	keys = ['R', 'correct_surf_M', 'surf_M_err']
	values = [ jk_R, _cc_surf_M, surf_m_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)	

	keys = ['lg_medi_devi', 'lg_mean_devi']
	values = [ devi_Medi, devi_Mean ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv(out_path + '%s_%s-band-based_M_calib-f.csv' % (cat_lis[mm], band_str),)
"""

dat = pds.read_csv( '/home/xkchen/figs/M2L_fit_test_M/' + 
	'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
tot_R = np.array(dat['R'])
tot_surf_m, tot_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
interp_M_f = interp.interp1d( tot_R, tot_surf_m, kind = 'linear',)

"""
## ratio to total sample
N_samples = 30
for mm in range( 2 ):

	off_dat = pds.read_csv( out_path + '%s_%s-band-based_M_calib-f.csv' % (cat_lis[mm], band_str),)
	scale_f = np.array( off_dat['lg_medi_devi'] )[0]

	low_R_lim, up_R_lim = tot_R.min(), tot_R.max()
	jk_sub_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'
	out_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'corrected_aveg-jack_mass-ratio.csv'

	mass_ratio_func( N_samples, scale_f, interp_M_f, low_R_lim, up_R_lim, jk_sub_file, out_file)

for mm in range( 2 ):

	scale_f = 0.
	low_R_lim, up_R_lim = tot_R.min(), tot_R.max()
	jk_sub_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'jack-sub-%d_mass-Lumi.csv'
	out_file = out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'aveg-jack_mass-ratio.csv'

	mass_ratio_func( N_samples, scale_f, interp_M_f, low_R_lim, up_R_lim, jk_sub_file, out_file)
"""

dt_R, dt_M, dt_Merr = [], [], []
for mm in range( 2 ):

	m_dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)
	jk_R = np.array(m_dat['R'])
	surf_m, surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])
	surf_L = np.array( m_dat['lumi'] )
	dt_R.append( jk_R )
	dt_M.append( surf_m )
	dt_Merr.append( surf_m_err )

dt_eta_R, dt_eta, dt_eta_err = [], [], []
for mm in range( 2 ):

	m_dat = pds.read_csv( out_path + '%s_%s-band-based_' % (cat_lis[mm], band_str) + 'aveg-jack_mass-ratio.csv' )
	tt_r, tt_eta, tt_eta_err = np.array(m_dat['R']), np.array(m_dat['M/M_tot']), np.array(m_dat['M/M_tot-err'])

	dt_eta_R.append( tt_r )
	dt_eta.append( tt_eta )
	dt_eta_err.append( tt_eta_err )


#...calibrated case
dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[0] )
lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )

dat = pds.read_csv( out_path + '%s_gri-band-based_corrected_aveg-jack_mass-Lumi.csv' % cat_lis[1] )
hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['correct_surf_M'] ), np.array( dat['surf_M_err'] )


lo_eat_dat = pds.read_csv( out_path + '%s_gri-band-based_' % cat_lis[0] + 'corrected_aveg-jack_mass-ratio.csv' )
lo_eta_R, lo_eta, lo_eta_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['M/M_tot']), np.array(lo_eat_dat['M/M_tot-err'])

hi_eat_dat = pds.read_csv( out_path + '%s_gri-band-based_' % cat_lis[1] + 'corrected_aveg-jack_mass-ratio.csv' )
hi_eta_R, hi_eta, hi_eta_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['M/M_tot']), np.array(hi_eat_dat['M/M_tot-err'])


line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

fig = plt.figure( figsize = (12.0, 4.8) )
ax0 = fig.add_axes([0.09, 0.41, 0.40, 0.56])
sub_ax0 = fig.add_axes([0.09, 0.13, 0.40, 0.28])

ax1 = fig.add_axes([0.58, 0.41, 0.40, 0.56])
sub_ax1 = fig.add_axes([0.58, 0.13, 0.40, 0.28])

for mm in range( 2 ):

	ax0.plot( dt_R[mm]/ 1e3, dt_M[mm], ls = line_s[mm], color = line_c[mm], alpha = 0.75, label = fig_name[mm])
	ax0.fill_between( dt_R[mm]/ 1e3, y1 = dt_M[mm] - dt_Merr[mm], y2 = dt_M[mm] + dt_Merr[mm], color = line_c[mm], alpha = 0.15,)

	sub_ax0.plot( dt_eta_R[mm]/ 1e3, dt_eta[mm], ls = line_s[mm], color = line_c[mm], alpha = 0.75)
	sub_ax0.fill_between( dt_eta_R[mm]/ 1e3, y1 = dt_eta[mm] - dt_eta_err[mm], y2 = dt_eta[mm] + dt_eta_err[mm], color = line_c[mm], alpha = 0.15)

ax0.plot( tot_R/ 1e3, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
ax0.fill_between( tot_R/ 1e3, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.15,)

ax0.set_xlim( 3e-3, 1e0 )
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_ylim( 5e3, 2e9 )
ax0.legend( loc = 3, frameon = False, fontsize = 16,)
ax0.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 16,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax0.annotate( text = 'Before calibration', xy = (0.55, 0.85), xycoords = 'axes fraction', fontsize = 16, color = 'k',)

sub_ax0.plot( tot_R / 1e3, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)
sub_ax0.set_xlim( ax0.get_xlim() )
sub_ax0.set_xscale( 'log' )
sub_ax0.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 16,)

sub_ax0.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax0.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

sub_ax0.set_ylim( 0.45, 1.20 )
sub_ax0.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{All \; clusters} $', fontsize = 16,)
sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax0.set_xticklabels( [] )

ax1.plot( lo_R / 1e3, lo_surf_M, ls = '--', color = line_c[0], alpha = 0.75, label = fig_name[0],)
ax1.fill_between( lo_R / 1e3, y1 = lo_surf_M - lo_surf_M_err, 
	y2 = lo_surf_M + lo_surf_M_err, color = line_c[0], alpha = 0.15,)

ax1.plot( hi_R/ 1e3, hi_surf_M, ls = '-', color = line_c[1], alpha = 0.75, label = fig_name[1],)
ax1.fill_between( hi_R/ 1e3, y1 = hi_surf_M - hi_surf_M_err, 
	y2 = hi_surf_M + hi_surf_M_err, color = line_c[1], alpha = 0.15,)

ax1.plot( tot_R/ 1e3, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
ax1.fill_between( tot_R/ 1e3, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.15,)

ax1.set_xlim( 3e-3, 1e0 )
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim( 5e3, 2e9 )
ax1.legend( loc = 3, frameon = False, fontsize = 16,)
ax1.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 16,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax1.annotate( text = 'After calibration', xy = (0.55, 0.85), xycoords = 'axes fraction', fontsize = 16, color = 'k',)

sub_ax1.plot( lo_eta_R / 1e3, lo_eta, ls = '--', color = line_c[0], alpha = 0.75,)
sub_ax1.fill_between( lo_eta_R / 1e3, y1 = lo_eta - lo_eta_err, y2 = lo_eta + lo_eta_err, color = line_c[0], alpha = 0.15,)

sub_ax1.plot( hi_eta_R / 1e3, hi_eta, ls = '-', color = line_c[1], alpha = 0.75,)
sub_ax1.fill_between( hi_eta_R / 1e3, y1 = hi_eta - hi_eta_err, y2 = hi_eta + hi_eta_err, color = line_c[1], alpha = 0.15,)

sub_ax1.plot( tot_R / 1e3, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale( 'log' )
sub_ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 16,)

sub_ax1.set_xticks([ 1e-2, 1e-1, 1e0])
sub_ax1.set_xticklabels( labels = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$'] )

sub_ax1.set_ylim( 0.45, 1.20 )
sub_ax1.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{All \; clusters} $', fontsize = 16,)
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 16,)
ax1.set_xticklabels( [] )

plt.savefig('/home/xkchen/separat-fit_%s_mass-reatio_compare.png' % file_s, dpi = 300)
plt.close()
