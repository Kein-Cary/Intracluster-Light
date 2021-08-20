import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp
import scipy.signal as signal

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from fig_out_module import arr_jack_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']
psf_FWHM = 1.32

### === ### data load and figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

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
# fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
# 			'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
# file_s = 'age_bin_fixed_BCG_M'
# cat_path = '/home/xkchen/tmp_run/data_files/figs/'
# BG_path = '/home/xkchen/tmp_run/data_files/jupyter/fixed_BCG_M/age_bin/BGs/'


##### ===== ##### color or M/Li profile
c_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
hi_c_r, hi_gr, hi_gr_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
hi_gr = signal.savgol_filter( hi_gr, 5, 1)
d_hi_gr = signal.savgol_filter( hi_gr, 5, 1, deriv = 1,)

c_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
lo_c_r, lo_gr, lo_gr_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
lo_gr = signal.savgol_filter( lo_gr, 5, 1)
d_lo_gr = signal.savgol_filter( lo_gr, 5, 1, deriv = 1,)

def smooth_slope_func(r_arr, color_arr, wind_L, order_dex, delta_x):

	id_nn = np.isnan( color_arr )

	if np.sum( id_nn ) == 0:
		dc_dr = signal.savgol_filter( color_arr, wind_L, order_dex, deriv = 1, delta = delta_x)
	else:
		_cp_r = r_arr[ id_nn == False ]
		_cp_color = color_arr[ id_nn == False ]

		_cp_color_F = interp.interp1d( _cp_r, _cp_color, kind = 'linear', fill_value = 'extrapolate',)

		color_arr[id_nn] = _cp_color_F( r_arr[id_nn] )
		dc_dr = signal.savgol_filter( color_arr, wind_L, order_dex, deriv = 1, delta = delta_x)
	return dc_dr

def color_slope_f():

	N_samples = 30

	### average jack-sub sample color
	for mm in range( 2 ):

		tmp_r, tmp_dgr, tmp_dgi, tmp_dri = [], [], [], []

		for ll in range( N_samples ):

			pdat = pds.read_csv( BG_path + '%s_jack-sub-%d_color_profile.csv' % (cat_lis[mm], ll) )
			tt_r, tt_gr, tt_gr_err = np.array( pdat['R'] ), np.array( pdat['g-r'] ), np.array( pdat['g-r_err'] )
			tt_gi, tt_gi_err = np.array( pdat['g-i'] ), np.array( pdat['g-i_err'] )
			tt_ri, tt_ri_err = np.array( pdat['r-i'] ), np.array( pdat['r-i_err'] )

			WL, p_order = 13, 1
			delt_x = 0.0635

			d_gr_dlgr = smooth_slope_func(tt_r, tt_gr, WL, p_order, delt_x)
			d_gi_dlgr = smooth_slope_func(tt_r, tt_gi, WL, p_order, delt_x)
			d_ri_dlgr = smooth_slope_func(tt_r, tt_ri, WL, p_order, delt_x)

			keys = [ 'R', 'd_gr_dlgr', 'd_gi_dlgr', 'd_ri_dlgr' ]
			values = [ tt_r, d_gr_dlgr, d_gi_dlgr, d_ri_dlgr, ]

			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( BG_path + '%s_jack-sub-%d_color_slope.csv' % (cat_lis[mm], ll),)

			tmp_r.append( tt_r )
			tmp_dgr.append( d_gr_dlgr )
			tmp_dgi.append( d_gi_dlgr )
			tmp_dri.append( d_ri_dlgr )

		aveg_R, aveg_dgr, aveg_dgr_err = arr_jack_func( tmp_dgr, tmp_r, N_samples )[:3]
		aveg_R, aveg_dgi, aveg_dgi_err = arr_jack_func( tmp_dgi, tmp_r, N_samples )[:3]
		aveg_R, aveg_dri, aveg_dri_err = arr_jack_func( tmp_dri, tmp_r, N_samples )[:3]

		keys = [ 'R_kpc', 'd_gr', 'd_gr_err', 'd_gi', 'd_gi_err', 'd_ri', 'd_ri_err' ]
		values = [ aveg_R, aveg_dgr, aveg_dgr_err, aveg_dgi, aveg_dgi_err, aveg_dri, aveg_dri_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( BG_path + '%s_color_slope.csv' % cat_lis[mm] )

	return

def M2L_slope_func():

	out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'
	N_samples = 30

	calib_cat = pds.read_csv( out_path + '%s_gri-band-based_M_calib-f.csv' % file_s )
	lo_shift, hi_shift = np.array(calib_cat['low_medi_devi'])[0], np.array(calib_cat['high_medi_devi'])[0]

	M_offset = [ lo_shift, hi_shift ]

	### average jack-sub sample color
	for mm in range( 2 ):

		jk_sub_m_file = out_path + '%s_gri-band-based_' % cat_lis[mm] + 'jack-sub-%d_mass-Lumi.csv'

		tmp_r, tmp_M2L = [], []
		tmp_d_M2L = []

		for nn in range( N_samples ):

			o_dat = pds.read_csv( jk_sub_m_file % nn,)
			tt_r = np.array( o_dat['R'] )
			tt_M = np.array( o_dat['surf_mass'] )
			tt_Li = np.array( o_dat['lumi'] )

			tt_M = tt_M * 10**M_offset[mm]
			tt_M2L = tt_M / tt_Li

			#. slope of M2Li
			WL, p_order = 13, 1
			delt_x = 0.0635
			tt_dM2L = smooth_slope_func( tt_r, tt_M2L, WL, p_order, delt_x)

			keys = [ 'R', 'M/Li', 'd_M/L_dlgr' ]
			values = [ tt_r, tt_M2L, tt_dM2L ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( out_path + '%s_jack-sub-%d_M2Li_slope.csv' % (cat_lis[mm], nn),)

			tmp_r.append( tt_r )
			tmp_M2L.append( tt_M2L )
			tmp_d_M2L.append( tt_dM2L )

		aveg_R, aveg_M2L, aveg_M2L_err = arr_jack_func( tmp_M2L, tmp_r, N_samples)[:3]
		aveg_R, aveg_dM2L, aveg_dM2L_err = arr_jack_func( tmp_d_M2L, tmp_r, N_samples)[:3]

		keys = ['R', 'M/Li', 'M/Li-err', 'd_M/Li', 'd_M/Li_err']
		values = [ aveg_R, aveg_M2L, aveg_M2L_err, aveg_dM2L, aveg_dM2L_err ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + '%s_gri-band_aveg_M2Li_profile.csv' % cat_lis[mm],)

	return

# color_slope_f()

# M2L_slope_func()

"""
tmp_dcR, tmp_dgr, tmp_dgr_err = [], [], []
for mm in range( 2 ):

	c_dat = pds.read_csv( BG_path + '%s_color_slope.csv' % cat_lis[mm],)
	tt_c_r = np.array( c_dat['R_kpc'] )

	tt_dgr, tt_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )
	# tt_dgi, tt_dgi_err = np.array( c_dat['d_gi'] ), np.array( c_dat['d_gi_err'] )
	# tt_dri, tt_dri_err = np.array( c_dat['d_ri'] ), np.array( c_dat['d_ri_err'] )

	tmp_dcR.append( tt_c_r )
	tmp_dgr.append( tt_dgr )
	tmp_dgr_err.append( tt_dgr_err )

fig = plt.figure( figsize = (10.6, 4.8) )
ax0 = fig.add_axes([0.08, 0.13, 0.38, 0.80])
ax1 = fig.add_axes([0.58, 0.13, 0.38, 0.80])

ax0.plot( lo_c_r, lo_gr, ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax0.fill_between( lo_c_r, y1 = lo_gr - lo_gr_err, y2 = lo_gr + lo_gr_err, color = 'b', alpha = 0.15,)

ax0.plot( hi_c_r, hi_gr, ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax0.fill_between( hi_c_r, y1 = hi_gr - hi_gr_err, y2 = hi_gr + hi_gr_err, color = 'r', alpha = 0.15,)

ax0.legend( loc = 3, fontsize = 18, frameon = False,)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 18)
ax0.set_xlim( 3e0, 1.1e3)

ax0.set_ylabel('$ g \; - \; r $', fontsize = 20,)
ax0.set_ylim( 0.8, 1.55 )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

ax1.plot( tmp_dcR[0], tmp_dgr[0], ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax1.fill_between( tmp_dcR[0], y1 = tmp_dgr[0] - tmp_dgr_err[0], y2 = tmp_dgr[0] + tmp_dgr_err[0], color = 'b', alpha = 0.15,)

ax1.plot( tmp_dcR[1], tmp_dgr[1], ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax1.fill_between( tmp_dcR[1], y1 = tmp_dgr[1] - tmp_dgr_err[1], y2 = tmp_dgr[1] + tmp_dgr_err[1], color = 'r', alpha = 0.15,)

ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 18)
ax1.set_xlim( 3e0, 1.1e3)

ax1.set_ylabel( '$ {\\rm d} (g-r) \, / \, {\\rm d} \\mathcal{lg}R $', fontsize = 18,)
ax1.set_ylim( -0.6, 0.)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

plt.savefig('/home/xkchen/%s_color_slope.png' % file_s, dpi = 300)
plt.close()
"""

## Mass to light ratio compare
out_path = '/home/xkchen/tmp_run/data_files/figs/M2L_fit_test_M/'

tmp_R, tmp_dm2l, tmp_dm2l_err = [], [], []
tmp_m2l, tmp_m2l_err = [], []
for mm in range( 2 ):

	c_dat = pds.read_csv( out_path + '%s_gri-band_aveg_M2Li_profile.csv' % cat_lis[mm],)
	tt_c_r = np.array( c_dat['R'] )
	tt_dm2l, tt_dm2l_err = np.array( c_dat['d_M/Li'] ), np.array( c_dat['d_M/Li_err'] )
	tt_m2l, tt_m2l_err = np.array( c_dat['M/Li'] ), np.array( c_dat['M/Li-err'] )

	tt_m2l = signal.savgol_filter( tt_m2l, 5, 1)

	tmp_R.append( tt_c_r )
	tmp_dm2l.append( tt_dm2l )
	tmp_dm2l_err.append( tt_dm2l_err )

	tmp_m2l.append( tt_m2l )
	tmp_m2l_err.append( tt_m2l_err )	

fig = plt.figure( figsize = (10.6, 4.8) )
ax0 = fig.add_axes([0.10, 0.13, 0.38, 0.80])
ax1 = fig.add_axes([0.60, 0.13, 0.38, 0.80])

ax0.plot( tmp_R[0], tmp_m2l[0], ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax0.fill_between( tmp_R[0], y1 = tmp_m2l[0] - tmp_m2l_err[0], y2 = tmp_m2l[0] + tmp_m2l_err[0], color = 'b', alpha = 0.15,)

ax0.plot( tmp_R[1], tmp_m2l[1], ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax0.fill_between( tmp_R[1], y1 = tmp_m2l[1] - tmp_m2l_err[0], y2 = tmp_m2l[1] + tmp_m2l_err[1], color = 'r', alpha = 0.15,)

ax0.legend( loc = 3, fontsize = 18, frameon = False,)
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 18)
ax0.set_xlim( 3e0, 1.1e3)

ax0.set_ylabel('$ M_{\\ast} / L_{i}$', fontsize = 20,)
ax0.set_ylim( 1.0, 2.5 )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

ax1.plot( tmp_R[0], tmp_dm2l[0], ls = '--', color = 'b', alpha = 0.75, label = fig_name[0] )
ax1.fill_between( tmp_R[0], y1 = tmp_dm2l[0] - tmp_dm2l_err[0], y2 = tmp_dm2l[0] + tmp_dm2l_err[0], color = 'b', alpha = 0.15,)

ax1.plot( tmp_R[1], tmp_dm2l[1], ls = '-', color = 'r', alpha = 0.75, label = fig_name[1] )
ax1.fill_between( tmp_R[1], y1 = tmp_dm2l[1] - tmp_dm2l_err[1], y2 = tmp_dm2l[1] + tmp_dm2l_err[1], color = 'r', alpha = 0.15,)

ax1.set_xscale('log')
ax1.set_xlabel('$R \; [\\mathrm{k}pc] $', fontsize = 18)
ax1.set_xlim( 3e0, 1.1e3)

ax1.set_ylabel( '$ {\\rm d} (M_{\\ast}/L_{i}) \, / \, {\\rm d} \\mathcal{lg}R $', fontsize = 18,)
ax1.set_ylim( -1.0, 0.5 )
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)

plt.savefig('/home/xkchen/%s_M2Li_slope.png' % file_s, dpi = 300)
plt.close()
