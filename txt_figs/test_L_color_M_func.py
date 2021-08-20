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

###... load data
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
file_s = 'BCG_Mstar_bin'

cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'

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

dpt_lgM = []
dpt_i_lumi, dpt_r_lumi, dpt_g_lumi = [], [], []
dpt_i_cMag, dpt_r_cMag, dpt_g_cMag = [], [], []
dpt_g2r, dpt_r2i = [], []

for mm in range( 2 ):

	#... lg_Mstar
	l_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[mm] )
	l_obs_z, l_rich  = np.array( l_dat['z']), np.array( l_dat['rich'])
	l_lgM, l_age = np.array( l_dat['lg_Mstar']), np.array( l_dat['BCG_age'] )
	dpt_lgM.append( l_lgM )

	#... mag
	pdat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG-color.csv' % cat_lis[mm] )
	p_g_mag, p_r_mag, p_i_mag = np.array( pdat['g_mag'] ), np.array( pdat['r_mag'] ), np.array( pdat['i_mag'] )

	gr_ = p_g_mag - p_r_mag
	ri_ = p_r_mag - p_i_mag

	dpt_g2r.append( gr_ )
	dpt_r2i.append( ri_ )

	#... Lumi
	pdat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/%s_BCG_cmag.csv' % cat_lis[mm] )
	i_cMag = np.array( pdat['i_cMag'] )
	i_Lumi = 10**( -0.4 * ( i_cMag - Mag_sun[2] ) )

	r_cMag = np.array( pdat['r_cMag'] )
	r_Lumi = 10**( -0.4 * ( r_cMag - Mag_sun[0] ) )

	g_cMag = np.array( pdat['g_cMag'] )
	g_Lumi = 10**( -0.4 * ( g_cMag - Mag_sun[1] ) )

	dpt_i_cMag.append( i_cMag )
	dpt_i_lumi.append( i_Lumi )

	dpt_r_cMag.append( r_cMag )
	dpt_r_lumi.append( r_Lumi )

	dpt_g_cMag.append( g_cMag )
	dpt_g_lumi.append( g_Lumi )

L_i = np.r_[ dpt_i_lumi[0], dpt_i_lumi[1] ]
L_g = np.r_[ dpt_g_lumi[0], dpt_g_lumi[1] ]
L_r = np.r_[ dpt_r_lumi[0], dpt_r_lumi[1] ]

i_Mag = np.r_[ dpt_i_cMag[0], dpt_i_cMag[1] ]
g_Mag = np.r_[ dpt_g_cMag[0], dpt_g_cMag[1] ]
r_Mag = np.r_[ dpt_r_cMag[0], dpt_r_cMag[1] ]

gr_arr = np.r_[ dpt_g2r[0], dpt_g2r[1] ]
ri_arr = np.r_[ dpt_r2i[0], dpt_r2i[1] ]
lg_Mstar = np.r_[ dpt_lgM[0], dpt_lgM[1] ]

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

		#... fitting lg_Mstar
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
		ax.annotate( text = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), 
			xycoords = 'axes fraction',)
		ax.set_xlabel('$ fit \; : \; \\lg \, (M_{\\ast} ) $')
		ax.set_ylabel('$ data \; : \; \\lg \, (M_{\\ast} ) $')
		plt.savefig('/home/xkchen/lgM_diag_selected_%d.png' % kk, dpi = 300)
		plt.close()

		cp_gr_arr = cp_gr_arr[ id_R_lim == False ]
		cp_ri_arr = cp_ri_arr[ id_R_lim == False ]
		cp_lgLi_arr = cp_lgLi_arr[ id_R_lim == False ]
		cp_lg_Mstar = cp_lg_Mstar[ id_R_lim == False ]

		kk += 1

		out_arr = np.array([ cp_gr_arr, cp_ri_arr, cp_lg_Mstar, cp_lgLi_arr ]).T
		np.savetxt( '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/M2L%s_selected_points.txt' % based_str, 
			out_arr, fmt = '%.8f, %.8f, %.8f, %.8f',)

# points_select()

"""
based_str = 'i'
points = np.loadtxt(
	'/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/M2L%s_selected_points.txt' % based_str, delimiter = ',')
cp_gr_arr = points[:,0]
cp_ri_arr = points[:,1]
cp_lg_Mstar = points[:,2]
cp_lg_Lumi = points[:,3]

put_x = np.array([ cp_gr_arr, cp_ri_arr, cp_lg_Lumi ])
p0 = [ -0.05, 0.5, 0.02, 0.2 ]
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
out_data.to_csv('/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-%s-band-Lumi&color.csv' % based_str )

_lg_Lumi = np.log10( L_i )
put_x = np.array( [ gr_arr, ri_arr, _lg_Lumi ] )

#... fitting lg_Mstar
fit_M2L = lg_bi_linear_func( put_x, a_fit, b_fit, c_fit, d_fit)

_lg_M = lg_Mstar - 2 * np.log10( h )
Var = np.sum( (fit_M2L - _lg_M)**2 ) / len( _lg_M )
sigma = np.sqrt( Var )
sp_R = sts.spearmanr( fit_M2L, _lg_M)[0]


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

ax.annotate( text = '$\\sigma = %.3f\, ; \; R = %.3f$' % (sigma, sp_R), xy = (0.65, 0.05), xycoords = 'axes fraction',)
ax.annotate( text = 'a = %.3f' % a_fit + '\n' + 'b = %.3f' % b_fit + '\n' + 'c = %.3f' % c_fit + 
	'\n' + 'd = %.3f' % d_fit, xy = (0.10, 0.75), xycoords = 'axes fraction',)
ax.set_xlabel(
	'$ \\lg \, (M_{\\ast}) = a \\cdot (g-r) + b \\cdot (r-i) + c \\cdot \\lg(L_{%s}) + d $' % (based_str),)
ax.set_ylabel('$ \\lg \, (M_{\\ast}) $' )
ax.set_xlim(10.80, 12.25)
ax.set_ylim(10.50, 12.50)
plt.savefig('/home/xkchen/M-to-L_estimate.png', dpi = 300)
plt.close()
"""


### ... mass profile measure test
from tmp_color_to_mass import get_c2mass_func
from fig_out_module import arr_jack_func
from light_measure import cov_MX_func

from tmp_color_to_mass import jk_sub_SB_func, jk_sub_Mass_func
from tmp_color_to_mass import aveg_mass_pro_func
from tmp_color_to_mass import cumu_mass_func

## fixed richness samples
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
file_s = 'BCG_Mstar_bin'
cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'
BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/BCG_M_bin/BGs/'

# cat_lis = ['younger', 'older']
# fig_name = ['Low $ t_{\\mathrm{age}} $ $ \\mid \\lambda $', 'High $ t_{\\mathrm{age}} $ $ \\mid \\lambda $']
# file_s = 'BCG_age_bin'
# cat_path = '/home/xkchen/mywork/ICL/data/cat_z_form/age_bin_cat/gri_common_cat/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_rich/age_bin_SBs/BGs/'


## fixed BCG Mstar samples
# cat_lis = [ 'low-rich', 'hi-rich' ]
# fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
# file_s = 'rich_bin_fixed_BCG_M'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/rich_bin_SBs/BGs/'

# cat_lis = [ 'low-age', 'hi-age' ]
# fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
# file_s = 'age_bin_fixed_BCG_M'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'


## total cluster sample
out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'

def surf_M_func():

	N_samples = 30

	band_str = 'gri'
	fit_file = '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv'

	for mm in range( 2 ):

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

	return

# surf_M_func()

def total_surf_M_func():
	BG_path = '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/BGs/'
	path = '/home/xkchen/tmp_run/data_files/jupyter/total_bcgM/SBs/'

	N_samples = 30

	band_str = 'gri'
	fit_file = '/home/xkchen/tmp_run/data_files/figs/M2L_Lumi_selected/least-square_M-to-i-band-Lumi&color.csv'

	sub_SB_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_jack-sub-%d_BG-sub_SB.csv'

	low_R_lim, up_R_lim = 1e0, 1.2e3
	out_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv'
	jk_sub_Mass_func( N_samples, band_str, sub_SB_file, low_R_lim, up_R_lim, out_file, Dl_ref, z_ref, 
		fit_file = fit_file)

	jk_sub_m_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_' % band_str + 'jack-sub-%d_mass-Lumi.csv'
	jk_aveg_m_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str

	lgM_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_log-surf-mass_cov_arr.h5' % band_str
	M_cov_file = out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_surf-mass_cov_arr.h5' % band_str	
	aveg_mass_pro_func( N_samples, band_str, jk_sub_m_file, jk_aveg_m_file, lgM_cov_file, M_cov_file = M_cov_file)

	dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
	aveg_R = np.array(dat['R'])

	aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
	aveg_cumu_m, aveg_cumu_m_err = np.array(dat['cumu_mass']), np.array(dat['cumu_mass_err'])
	aveg_lumi, aveg_lumi_err = np.array(dat['lumi']), np.array(dat['lumi_err'])

	plt.figure()
	plt.title('%s-based surface mass density profile' % band_str )
	plt.plot( aveg_R, aveg_surf_m, ls = '-', color = 'r', alpha = 0.5,)
	plt.fill_between( aveg_R, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err, 
		color = 'r', alpha = 0.12,)

	plt.xlim(1e0, 1e3)
	plt.xscale('log')
	plt.xlabel('R[kpc]', fontsize = 15)
	plt.yscale('log')
	plt.ylim(1e4, 3e9)
	plt.legend( loc = 1, frameon = False, fontsize = 15,)
	plt.ylabel('$\\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)
	plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
	plt.savefig('/home/xkchen/tot_%s-band_based_surface_mass_profile.png' % band_str, dpi = 300)
	plt.close()

# total_surf_M_func()


#... mass profile compare
line_c = [ 'b',  'r' ]
line_s = [ '--', '-' ]

band_str = 'gri'

base_lis = ['gi', 'ri', 'gri', 'gr', 'ir', 'gir', 'ig', 'rg', 'rig']

line_name = ['g-i + $L_{i}$', 'r-i + $L_{i}$', 'g-r + $L_{i}$', 
			 'g-r + $L_{r}$', 'r-i + $L_{r}$', 'g-i + $L_{r}$', 
			 'g-i + $L_{g}$', 'g-r + $L_{g}$', 'r-i + $L_{g}$' ]

tl_lis = ['b-', 'b--', 'b:', 'r-', 'r--', 'r:', 'g-', 'g--', 'g:']

for pp in range( 3 ):

	fig_tx = plt.figure()
	ax = fig_tx.add_axes( [0.12, 0.12, 0.80, 0.80] )

	for mm in range( 2 ):

		dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], base_lis[pp]),)
		aveg_R = np.array(dat['R'])
		aveg_surf_m, aveg_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])

		ax.plot( aveg_R, aveg_surf_m, ls = '--', color = line_c[mm], alpha = 0.65, 
			label = 'Bell; ' + line_name[pp],)
		ax.fill_between( aveg_R, y1 = aveg_surf_m - aveg_surf_m_err, y2 = aveg_surf_m + aveg_surf_m_err,
						color = line_c[mm], ls = '--', alpha = 0.15,)


		m_dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str),)

		me_jk_R = np.array(m_dat['R'])
		me_surf_m, me_surf_m_err = np.array( m_dat['surf_mass']), np.array(m_dat['surf_mass_err'])

		cumu_M = np.array( m_dat['cumu_mass'] )
		_test_cM_f = interp.interp1d( me_jk_R, cumu_M, kind = 'linear')
		cM_40 = _test_cM_f( 50 )

		print( np.mean(aveg_surf_m / me_surf_m) )

		eta = 1.
		ax.plot( me_jk_R, me_surf_m * eta, ls = '-', color = line_c[mm], alpha = 0.65, 
			label = 'Mine; ' + 'g-r + r-i + $L_{%s}$' % band_str[-1],)
		ax.fill_between( me_jk_R, y1 = me_surf_m * eta - me_surf_m_err, y2 = me_surf_m * eta + me_surf_m_err,
						color = line_c[mm], alpha = 0.15,)

	handles,labels = ax.get_legend_handles_labels()
	handles = [ handles[1], handles[3] ]
	labels = [ fig_name[0], fig_name[1] ]
	legend_1 = ax.legend( handles, labels, loc = 3, frameon = False, fontsize = 15)

	ax.legend( loc = 1, frameon = False, fontsize = 15, markerfirst = False,)
	ax.add_artist( legend_1 )

	ax.set_xlim( 3e0, 1.1e3)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]', fontsize = 15)
	ax.set_yscale('log')
	ax.set_ylim( 1e3, 3e9)
	ax.set_ylabel('$\\Sigma_{\\ast} [M_{\\odot} / kpc^2]$', fontsize = 15,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/%s-band-based_surf-mass_compare.png' % base_lis[pp], dpi = 300)
	plt.close()
