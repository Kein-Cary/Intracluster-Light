import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
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


### === ### cosmology model
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

Da_ref = Test_model.angular_diameter_distance( z_ref ).value
phyR_psf = np.array( psf_FWHM ) * Da_ref * 10**3 / rad2asec
phyR_psf = phyR_psf / 1e3


def hist2d_pdf_func(x, y, bins, levels, smooth = None, weights = None,):

	from scipy.ndimage import gaussian_filter

	H, X, Y = np.histogram2d( x.flatten(), y.flatten(), bins = bins, weights = weights)

	if smooth is not None:
		H = gaussian_filter(H, smooth)

	Hflat = H.flatten()
	inds = np.argsort(Hflat)[::-1]
	Hflat = Hflat[inds]
	sm = np.cumsum(Hflat)
	sm /= sm[-1]
	V = np.empty(len(levels))

	for i, v0 in enumerate(levels):
		try:
			V[i] = Hflat[sm <= v0][-1]
		except IndexError:
			V[i] = Hflat[0]
	V.sort()

	m = np.diff(V) == 0
	if np.any(m) and not quiet:
		logging.warning("Too few points to create valid contours")
	while np.any(m):
		V[np.where(m)[0][0]] *= 1.0 - 1e-4
		m = np.diff(V) == 0
	V.sort()

	# Compute the bin centers.
	X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

	# Extend the array for the sake of the contours at the plot edges.
	H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
	H2[2:-2, 2:-2] = H
	H2[2:-2, 1] = H[:, 0]
	H2[2:-2, -2] = H[:, -1]
	H2[1, 2:-2] = H[0]
	H2[-2, 2:-2] = H[-1]
	H2[1, 1] = H[0, 0]
	H2[1, -2] = H[0, -1]
	H2[-2, 1] = H[-1, 0]
	H2[-2, -2] = H[-1, -1]
	X2 = np.concatenate(
		[
			X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
			X1,
			X1[-1] + np.array([1, 2]) * np.diff(X1[-2:]),
		]
	)
	Y2 = np.concatenate(
		[
			Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
			Y1,
			Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:]),
		]
	)

	return H, H2, X2, Y2, V

### === ### data load and figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

#... overall sample properties
cat = pds.read_csv( '/home/xkchen/mywork/ICL/data/' + 'cat_z_form/clslowz_z0.17-0.30_bc03_cat.csv' )
ra, dec, z = np.array(cat['ra']), np.array(cat['dec']), np.array(cat['z'])
rich, z_form = np.array(cat['lambda']), np.array(cat['z_form'])
lg_Mstar = np.array( cat['lg_M*_photo_z'] ) # Mass unit : M_sun / h^2


# fig = plt.figure( figsize = ( 22.4, 20.4 ) )
# gx = gridspec.GridSpec( 3, 3, figure = fig, left = 0.06, bottom = 0.045, right = 0.99, top = 0.99, 
# 						wspace = 0.18, hspace = 0.14, width_ratios = [1,1,1], height_ratios = [1,1,1] )

fig = plt.figure( figsize = ( 22.0, 14.0 ) )
gx = gridspec.GridSpec( 2, 3, figure = fig, left = 0.06, bottom = 0.06, right = 0.98, top = 0.98, 
						wspace = 0.18, hspace = 0.14, width_ratios = [1,1,1], height_ratios = [1,1] )

id_dered = True
# id_dered = False

band_str = 'gri'
# band_str = 'ri'

for pp in range( 2 ):

	if pp == 0:
		## fixed richness samples
		cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
		fig_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']
		file_s = 'BCG_Mstar_bin'
		cat_path = '/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'

	if pp == 1:
		## fixed BCG Mstar samples
		cat_lis = [ 'low-rich', 'hi-rich' ]
		fig_name = [ 'Low $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 'High $ \\lambda $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $']
		file_s = 'rich_bin_fixed_BCG_M'
		cat_path = '/home/xkchen/tmp_run/data_files/figs/'

	if pp == 2:
		cat_lis = [ 'low-age', 'hi-age' ]
		fig_name = [ 'Low $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $', 
					 'High $ t_{\\mathrm{age}} $ $ \\mid M_{\\ast}^{\\mathrm{BCG}} $' ]
		file_s = 'age_bin_fixed_BCG_M'
		cat_path = '/home/xkchen/tmp_run/data_files/figs/'

	#=====# data
	BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
	out_path = '/home/xkchen/figs/re_measure_SBs/SM_profile/'


	#... SB profile
	nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []

	for kk in range( 3 ):
		with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
		tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

		nbg_low_r.append( tt_r )
		nbg_low_sb.append( tt_mag )
		nbg_low_err.append( tt_mag_err )

	nbg_low_r = np.array( nbg_low_r )
	nbg_low_r = nbg_low_r / 1e3

	nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []

	for kk in range( 3 ):
		with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
		tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

		nbg_hi_r.append( tt_r )
		nbg_hi_sb.append( tt_mag )
		nbg_hi_err.append( tt_mag_err )

	nbg_hi_r = np.array( nbg_hi_r )
	nbg_hi_r = nbg_hi_r / 1e3


	#...color profile
	mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[1] )
	hi_c_r, hi_gr, hi_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
	hi_gi, hi_gi_err = np.array( mu_dat['g-i'] ), np.array( mu_dat['g-i_err'] )

	hi_gr = signal.savgol_filter( hi_gr, 7, 3)
	hi_gi = signal.savgol_filter( hi_gi, 7, 3)
	hi_c_r = hi_c_r / 1e3

	mu_dat = pds.read_csv( BG_path + '%s_color_profile.csv' % cat_lis[0] )
	lo_c_r, lo_gr, lo_gr_err = np.array( mu_dat['R_kpc'] ), np.array( mu_dat['g-r'] ), np.array( mu_dat['g-r_err'] )
	lo_gi, lo_gi_err = np.array( mu_dat['g-i'] ), np.array( mu_dat['g-i_err'] )

	lo_gr = signal.savgol_filter( lo_gr, 7, 3)
	lo_gi = signal.savgol_filter( lo_gi, 7, 3)
	lo_c_r = lo_c_r / 1e3


	#...color slope
	c_dat = pds.read_csv( BG_path + '%s_color_slope.csv' % cat_lis[0],)
	lo_dgr, lo_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )
	lo_dc_r = np.array( c_dat['R_kpc'] )
	lo_dc_r = lo_dc_r / 1e3

	c_dat = pds.read_csv( BG_path + '%s_color_slope.csv' % cat_lis[1],)
	hi_dgr, hi_dgr_err = np.array( c_dat['d_gr'] ), np.array( c_dat['d_gr_err'] )
	hi_dc_r = np.array( c_dat['R_kpc'] )
	hi_dc_r = hi_dc_r / 1e3


	#... sample properties
	hi_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )
	hi_obs_z, hi_rich  = np.array( hi_dat['z']), np.array( hi_dat['rich'])
	hi_lgM, hi_age = np.array( hi_dat['lg_Mstar']), np.array( hi_dat['BCG_age'] )

	lo_dat = pds.read_csv( cat_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )
	lo_obs_z, lo_rich  = np.array( lo_dat['z']), np.array( lo_dat['rich'])
	lo_lgM, lo_age = np.array( lo_dat['lg_Mstar']), np.array( lo_dat['BCG_age'] )

	#. Mass unit : M_sun
	hi_lgM = hi_lgM - 2 * np.log10( h )
	lo_lgM = lo_lgM - 2 * np.log10( h )

	#... mass profiles
	if id_dered == True:
		#...Mass profile (with correction)
		dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[0], band_str),)
		lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )
		lo_R = lo_R / 1e3

		dat = pds.read_csv( out_path + '%s_%s-band-based_corrected_aveg-jack_mass-Lumi_with-dered.csv' % (cat_lis[1], band_str),)
		hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array( dat['medi_correct_surf_M'] ), np.array( dat['surf_M_err'] )
		hi_R = hi_R / 1e3

		#. mass ratio with R_limt correction
		lo_eat_dat = pds.read_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample_with-dered.csv' % (cat_lis[0], band_str),)
		lo_eta_R, lo_eta, lo_eta_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['M/M_tot']), np.array(lo_eat_dat['M/M_tot-err'])

		hi_eat_dat = pds.read_csv( out_path + '%s_%s-band_corrected-aveg-M-ratio_to_total-sample_with-dered.csv' % (cat_lis[1], band_str),)
		hi_eta_R, hi_eta, hi_eta_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['M/M_tot']), np.array(hi_eat_dat['M/M_tot-err'])


	if id_dered == False:
		#...Mass profile (without correction)
		dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[0], band_str),)
		lo_R, lo_surf_M, lo_surf_M_err = np.array( dat['R'] ), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		lo_R = lo_R / 1e3

		dat = pds.read_csv( out_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[1], band_str),)
		hi_R, hi_surf_M, hi_surf_M_err = np.array( dat['R'] ), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
		hi_R = hi_R / 1e3

		#. mass ratio without R_limt correction
		lo_eat_dat = pds.read_csv( out_path + '%s_%s-band_aveg_M-ratio_to_total-sample.csv' % (cat_lis[0], band_str),)
		lo_eta_R, lo_eta, lo_eta_err = np.array(lo_eat_dat['R']), np.array(lo_eat_dat['M/M_tot']), np.array(lo_eat_dat['M/M_tot-err'])

		hi_eat_dat = pds.read_csv( out_path + '%s_%s-band_aveg_M-ratio_to_total-sample.csv' % (cat_lis[1], band_str),)
		hi_eta_R, hi_eta, hi_eta_err = np.array(hi_eat_dat['R']), np.array(hi_eat_dat['M/M_tot']), np.array(hi_eat_dat['M/M_tot-err'])


	#... total sample for comparison
	dat = pds.read_csv( out_path + 'photo-z_tot-BCG-star-Mass_%s-band-based_aveg-jack_mass-Lumi.csv' % band_str,)
	tot_R = np.array(dat['R'])
	tot_surf_m, tot_surf_m_err = np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])
	tot_R  = tot_R / 1e3


	#=====# figs	
	ax0 = fig.add_subplot( gx[pp, 0] )
	ax1 = fig.add_subplot( gx[pp, 1] )
	
	t_gx = gx[pp, 2].subgridspec( 3, 1, hspace = 0)
	ax2 = fig.add_subplot( t_gx[:2] )
	bot_ax2 = fig.add_subplot( t_gx[2] )


	if file_s == 'BCG_Mstar_bin':

		pre_N = 7
		bins_edg = np.linspace( 1.3, 2.3, pre_N)

		medi_lgM = []
		std_lgM = []

		#. median points
		cen_rich = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )
		lg_rich = np.log10( rich )

		for ii in range( pre_N - 1):

			id_lim = ( lg_rich >= bins_edg[ii] ) & ( lg_rich < bins_edg[ii+1] )
			lim_lgM = lg_Mstar[ id_lim ] 

			medi_lgM.append( np.median( lim_lgM - 2 * np.log10( h ) ) )
			std_lgM.append( np.std( lim_lgM - 2 * np.log10( h ) ) )

		medi_lgM = np.array( medi_lgM )
		std_lgM = np.array( std_lgM )

		#. hist2D density estimate
		cp_lg_Mstar = lg_Mstar - 2 * np.log10( h ) # Mass unit : M_sun
		lel_x = [ np.log( 1 - 0.5), np.log( 1 - 0.95) ]
		levels = ( 1 - np.exp( lel_x[0] ), 1 - np.exp( lel_x[1] ),)

		H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( rich, cp_lg_Mstar, bins = [100, 100], 
		levels = levels, smooth = (2.5, 1.0), weights = None,)


		#... BCG-M bin, fixed richness
		_point_rich = np.array([20, 30, 40, 50, 100, 200])

		# - 2 * np.log10( h ) change mass unit from M_sun / h^2 to M_sun
		line_divi = 0.446 * np.log10( _point_rich ) + 10.518 - 2 * np.log10( h )


		#... rich bin, fixed BCG-M
		div_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/rich-bin_fixed-BCG-M_divid_line.csv' )
		lgM_x, tt_sep_line = np.array( div_dat['lgM'] ), np.array( div_dat['lg_rich'] ) # lgM_x is mass in unit M_sun / h^2

		cc_lgM_x = lgM_x - 2 * np.log10( h )
		tmp_line_func = interp.interp1d( cc_lgM_x, tt_sep_line, kind = 'linear', fill_value = 'extrapolate',)

		cp_lgMx = np.linspace( 10, 13, 100 )
		cp_line = tmp_line_func( cp_lgMx )


		# ax0.scatter( lo_rich, lo_lgM, s = 15, marker = '.', c = lo_age, cmap = 'rainbow', alpha = 0.75, vmin = 1, vmax = 11,)
		# ax0.scatter( hi_rich, hi_lgM, s = 15, marker = '.', c = hi_age, cmap = 'rainbow', alpha = 0.75, vmin = 1, vmax = 11,)

		ax0.scatter( rich, cp_lg_Mstar, s = 10, c = 'grey', marker = '.', alpha = 0.45,)

		# _cmap_lis = []
		# for ii in range( 9 ):
		# 	sub_color = mpl.cm.Greys_r( ii / 8 )
		# 	_cmap_lis.append( sub_color )
		# ax0.contourf( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), colors = [ _cmap_lis[5], _cmap_lis[7], _cmap_lis[8] ],)

		ax0.plot( _point_rich, line_divi, ls = '-', color = 'k', alpha = 0.75, linewidth = 2, label = 
				'$ \\langle {\\rm \\mathcal{lg} } \, M_{\\ast}^{\\mathrm{BCG} } \\rangle = 0.45 \, {\\rm \\mathcal{lg} } \, \\lambda $' + '$ \; + \; %.2f$' % (10.518 - 2 * np.log10(h) ),)

		# ax0.plot( 10**cp_line, cp_lgMx, ls = '--', color = 'k', alpha = 0.75, linewidth = 2,)

		ax0.errorbar( 10**cen_rich, medi_lgM, yerr = std_lgM, xerr = None, color = 'k', marker = 'o', ls = 'none', ecolor = 'k', 
			mec = 'k', mfc = 'k', capsize = 2, markersize = 7,)

		ax0.legend( loc = 3, frameon = False, fontsize = 22, markerfirst = True, handletextpad = 0.3,)

		ax0.set_xscale( 'log' )
		ax0.set_xlabel( '$ \\lambda $' , fontsize = 22,)
		ax0.set_ylabel( '$ {\\rm \\mathcal{lg} } \; [ M^{\\mathrm{BCG}}_{\\ast} \, / \, M_{\\odot} ] $', fontsize = 22,)

		ax0.text( 60, 11.9, s = 'High $ M_{\\ast}^{\\mathrm{BCG}} $', fontsize = 24, rotation = 10, color = 'k')
		ax0.text( 60, 11.3, s = 'Low $ M_{\\ast}^{\\mathrm{BCG}} $', fontsize = 24, rotation = 9, color = 'k')

		ax0.set_xlim( 20, 100 )
		ax0.set_ylim( 10.46, 12.25 )

		x_tick_arr = [ 20, 30, 40, 50, 60, 100 ]
		tick_lis = [ '%d' % ll for ll in x_tick_arr ]
		ax0.set_xticks( x_tick_arr )
		ax0.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
		ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 22,)

		# sub_ax0 = ax0.inset_axes( [ 0.65, 0.17, 0.30, 0.25] )
		# sub_ax1 = ax0.inset_axes( [ 0.65, 0.12, 0.30, 0.05] )

		# age_edgs = np.logspace( 0, np.log10(12), 50)
		# sub_ax0.hist( lo_age, bins = age_edgs, density = True, histtype = 'step', color = 'b', ls = '--', alpha = 0.75, )
		# sub_ax0.hist( hi_age, bins = age_edgs, density = True, histtype = 'step', color = 'r', alpha = 0.75, )
		# sub_ax0.set_xlim( 1, 11 )

		# cmap = mpl.cm.rainbow
		# norm = mpl.colors.Normalize( vmin = 1, vmax = 11 )

		# c_ticks = np.array([1, 3, 5, 7, 9, 11])
		# cbs = mpl.colorbar.ColorbarBase( ax = sub_ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, 
		# 								orientation = 'horizontal',)
		# cbs.set_label( '$ t_{\\mathrm{age}} \; [\\mathrm{G}yr] $', fontsize = 18,)

		# cmap.set_under('cyan')
		# cbs.ax.set_xticklabels( labels = ['%.0f' % ll for ll in c_ticks ] )

		# sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
		# sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
		# sub_ax0.set_xticks( [] )


	if file_s == 'rich_bin_fixed_BCG_M':

		pre_N = 6
		bins_edg = np.linspace( 10, 12, pre_N)

		medi_rich = []
		std_rich = []

		#. median points
		cen_lgM = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )
		for ii in range( pre_N - 1):

			id_lim = ( lg_Mstar >= bins_edg[ii] ) & ( lg_Mstar < bins_edg[ii+1] )
			lim_rich = rich[ id_lim ]

			medi_rich.append( np.median( lim_rich ) )
			std_rich.append( np.std( lim_rich ) )

		medi_rich = np.array( medi_rich )
		std_rich = np.array( std_rich )

		fit_F = np.polyfit( cen_lgM, np.log10( medi_rich ), deg = 2)
		Pf = np.poly1d( fit_F )
		fit_line = Pf( cen_lgM )
		cp_cen_lgM = cen_lgM - 2 * np.log10( h )


		#. hist2D density estimate
		cp_lg_Mstar = lg_Mstar - 2 * np.log10( h ) # Mass unit : M_sun
		lel_x = [ np.log( 1 - 0.5), np.log( 1 - 0.95) ]
		levels = ( 1 - np.exp( lel_x[0] ), 1 - np.exp( lel_x[1] ),)
		H_arr, H2_arr, X2_arr, Y2_arr, V_arr = hist2d_pdf_func( cp_lg_Mstar, rich, bins = [100, 100], levels = levels, smooth = (1.5, 1.0), weights = None,)

		#... divide line
		div_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/rich-bin_fixed-BCG-M_divid_line.csv' )
		lgM_x, line_divi = np.array( div_dat['lgM'] ), np.array( div_dat['lg_rich'] ) # lgM_x is mass in unit M_sun / h^2

		cc_lgM_x = lgM_x - 2 * np.log10( h )

		tmp_line_func = interp.interp1d( cc_lgM_x, line_divi, kind = 'linear', fill_value = 'extrapolate',)
		cp_lgMx = np.linspace( 10, 13, 100 )
		cp_line = tmp_line_func( cp_lgMx )

		#... divide line (BCG-M bin, fixed richness)
		# BCG-M bin, fixed richness
		_point_rich = np.array([20, 30, 40, 50, 100, 200])

		#... - 2 * np.log10( h ) change mass unit from M_sun / h^2 to M_sun
		tt_sep_line = 0.446 * np.log10( _point_rich ) + 10.518 - 2 * np.log10( h )


		# ax0.scatter( lo_lgM, lo_rich, s = 15, c = lo_age, marker = '.', cmap = 'rainbow', alpha = 0.75, vmin = 1, vmax = 11,)
		# ax0.scatter( hi_lgM, hi_rich, s = 15, c = hi_age, marker = '.', cmap = 'rainbow', alpha = 0.75, vmin = 1, vmax = 11,)

		ax0.scatter( cp_lg_Mstar, rich, s = 10, c = 'grey', marker = '.', alpha = 0.45,)

		# _cmap_lis = []
		# for ii in range( 9 ):
		# 	sub_color = mpl.cm.Greys_r( ii / 8 )
		# 	_cmap_lis.append( sub_color )
		# ax0.contourf( X2_arr, Y2_arr, H2_arr.T, np.concatenate([ V_arr, [ H_arr.max() * (1 + 1e-4) ] ] ), colors = [ _cmap_lis[5], _cmap_lis[7], _cmap_lis[8] ],)

		ax0.errorbar( cp_cen_lgM, medi_rich, yerr = std_rich, xerr = None, color = 'k', marker = 'o', ls = 'none', ecolor = 'k', 
			mec = 'k', mfc = 'k', capsize = 2, markersize = 7,)

		p_fh = Pf[2] * 4 * ( np.log10(h) )**2 + Pf[1] * (-2) * np.log10(h)

		# ax0.plot( cp_lgMx, 10**cp_line, ls = '-', color = 'k', alpha = 0.75, linewidth = 2, 
		# 	label = '$10^{ %.3f \, ( {\\rm \\mathcal{lg} } \, M^{\\mathrm{BCG}}_{\\ast} )^{2} \, %.3f \, {\\rm \\mathcal{lg} } \, M^{\\mathrm{BCG}}_{\\ast} \; + \; %.3f }$'
		# 			 % ( Pf[2], Pf[1], Pf[0] + p_fh),)

		ax0.plot( cp_lgMx, 10**cp_line, ls = '-', color = 'k', alpha = 0.75, linewidth = 2, 
			label = 
			'$ \\langle {\\rm \\mathcal{lg} } \, \\lambda \\rangle = %.2f \, {\\rm \\mathcal{lg} }^{2} \, M^{\\mathrm{BCG}}_{\\ast} \, %.2f \, {\\rm \\mathcal{lg} } \, M^{\\mathrm{BCG}}_{\\ast} \; + \; %.2f$' 
			% ( Pf[2], Pf[1], Pf[0] + p_fh),)

		# ax0.plot( tt_sep_line, _point_rich, color = 'k', ls = '--', alpha = 0.75, linewidth = 2,)

		ax0.legend( loc = 2, frameon = True, fontsize = 17.2, facecolor = 'w', edgecolor = 'w', framealpha = 0.8, 
					borderaxespad = 0.7, borderpad = 0.2, handletextpad = 0.3,)

		ax0.text( 10.6, 26, s = 'High $ \\lambda $', fontsize = 24, color = 'k')
		ax0.text( 10.6, 20, s = 'Low $ \\lambda $', fontsize = 24, color = 'k')

		ax0.set_ylim( 19.5, 100 )
		ax0.set_yscale( 'log' )
		ax0.set_xlim( 10.44, 12.25 )

		ax0.set_ylabel('$ \\lambda $', fontsize = 22,)
		ax0.set_xlabel('$ {\\rm \\mathcal{lg} } \; [ M^{\\mathrm{BCG}}_{\\ast} \, / \, M_{\\odot} ] $', fontsize = 22,)

		x_tick_arr = [ 20, 30, 40, 50, 60, 100 ]
		tick_lis = [ '%d' % ll for ll in x_tick_arr ]
		ax0.set_yticks( x_tick_arr )
		ax0.get_yaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
		ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

		ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 22,)

		# sub_ax0 = ax0.inset_axes( [ 0.10, 0.72, 0.30, 0.25] )
		# sub_ax1 = ax0.inset_axes( [ 0.10, 0.67, 0.30, 0.05] )

		# age_edgs = np.logspace( 0, np.log10(12), 50)

		# sub_ax0.hist( lo_age, bins = age_edgs, density = True, histtype = 'step', color = 'b',  ls = '--', )
		# sub_ax0.hist( hi_age, bins = age_edgs, density = True, histtype = 'step', color = 'r', )
		# sub_ax0.set_xlim( 1, 11 )

		# cmap = mpl.cm.rainbow
		# norm = mpl.colors.Normalize( vmin = 1, vmax = 11 )
		# c_ticks = np.array([1, 3, 5, 7, 9, 11])
		# cbs = mpl.colorbar.ColorbarBase( ax = sub_ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, 
		# 								orientation = 'horizontal',)
		# cbs.set_label( '$ t_{\\mathrm{age}} \; [\\mathrm{G}yr] $', fontsize = 18,)

		# cmap.set_under('cyan')
		# cbs.ax.set_xticklabels( labels = ['%.0f' % ll for ll in c_ticks ] )
		# sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
		# sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
		# sub_ax0.set_xticks( [] )


	if file_s == 'age_bin_fixed_BCG_M':

		# BCG-age bin, fixed richness
		div_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/age-bin_fixed-BCG-M_divid_line.csv' )
		lgM_x, line_divi = np.array( div_dat['lgM'] ), np.array( div_dat['lg_age'] )  # lgM_x is mass in unit M_sun / h^2

		cc_lgM_x = lgM_x - 2 * np.log10( h )

		tmp_line_func = interp.interp1d( cc_lgM_x, line_divi, kind = 'linear', fill_value = 'extrapolate',)
		cp_lgMx = np.linspace( 10, 13, 100 )
		cp_line = tmp_line_func( cp_lgMx )


		#... without setting color for points
		min_rich = 20.
		max_rich = 100.

		poiint_f = ax0.scatter( lo_lgM, lo_age, s = 15, c = np.log10( lo_rich ), marker = '.', vmin = np.log10( min_rich ), vmax = np.log10( max_rich ), 
								cmap = 'rainbow', alpha = 0.75, )
		ax0.scatter( hi_lgM, hi_age, s = 15, c = np.log10( hi_rich ), marker = '.', vmin = np.log10( min_rich ), vmax = np.log10( max_rich ), cmap = 'rainbow', alpha = 0.75, )

		ax0.plot( cp_lgMx, 10**cp_line, ls = '-', color = 'k',)

		ax0.set_ylabel('$ t_{\\mathrm{age}} \; [\\mathrm{G}yr] $', fontsize = 22,)
		ax0.set_xlabel('$ {\\rm \\mathcal{lg} } \; [ M^{\\mathrm{BCG}}_{\\ast} \, / \, M_{\\odot} ] $', fontsize = 22,)
		ax0.set_xlim( 10.75, 12.25 )
		ax0.set_ylim( 1.7, 10.8 )

		ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

		ax0.text( 11.84, 8.4, s = 'High $ t_{\\mathrm{age}} $', fontsize = 24, rotation = 27,)
		ax0.text( 11.88, 7.4, s = 'Low $ t_{\\mathrm{age}} $', fontsize = 24, rotation = 27,)
		ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 22,)

		# sub_ax0 = ax0.inset_axes( [ 0.66, 0.15, 0.30, 0.25] )
		# sub_ax1 = ax0.inset_axes( [ 0.66, 0.10, 0.30, 0.05] )

		# lgrich_edgs = np.logspace( 1.30, 2.28, 27)

		# sub_ax0.hist( lo_rich, bins = lgrich_edgs, density = True, histtype = 'step', color = 'b', ls = '--',)
		# sub_ax0.hist( hi_rich, bins = lgrich_edgs, density = True, histtype = 'step', color = 'r', )

		# sub_ax0.set_xlim( 2e1, 1e2 )
		# sub_ax0.set_xscale('log')
		# sub_ax0.set_yscale('log')

		# sub_ax0.set_xticklabels( labels = [], minor = True,)
		# sub_ax0.set_xticks( [ 100 ] )
		# sub_ax0.set_xticklabels( labels = [],)

		# sub_ax0.set_yticks( [1e-3, 1e-2] )
		# sub_ax0.set_yticklabels( labels = ['$10^{-3}$', '$10^{-2}$'],)

		# cmap = mpl.cm.rainbow
		# norm = mpl.colors.Normalize( vmin = np.log10( min_rich ), vmax = np.log10( max_rich ) )

		# c_ticks = np.array([ 20, 30, 50, 70, 100 ])
		# cbs = mpl.colorbar.ColorbarBase( ax = sub_ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, 
		# 								orientation = 'horizontal',)
		# cbs.set_label('$ \\lambda $', fontsize = 18,)
		# cmap.set_under('blue')
		# cmap.set_over('red')

		# cax = plt.colorbar( poiint_f, ax = sub_ax1, orientation = 'horizontal', ticks = np.array([ 20, 30, 50, 70, 100 ]),)

		# _ticks = np.array( [20, 30, 60, 100 ] )
		# label_str = ['%d' % ll for ll in _ticks ]
		# sub_ax1.xaxis.set_major_locator( ticker.FixedLocator( _ticks ) )
		# sub_ax1.xaxis.set_major_formatter( ticker.FixedFormatter( label_str ) )

		# sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
		# sub_ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 18,)
		# sub_ax0.set_xticks( [] )


	for kk in ( 2, 0, 1 ):

		ax1.plot(nbg_low_r[kk], nbg_low_sb[kk], ls = '--', color = color_s[kk], alpha = 0.75,)
		ax1.fill_between(nbg_low_r[kk], y1 = nbg_low_sb[kk] - nbg_low_err[kk], 
			y2 = nbg_low_sb[kk] + nbg_low_err[kk], color = color_s[kk], alpha = 0.15,)

		ax1.plot(nbg_hi_r[kk], nbg_hi_sb[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk])
		ax1.fill_between(nbg_hi_r[kk], y1 = nbg_hi_sb[kk] - nbg_hi_err[kk], 
			y2 = nbg_hi_sb[kk] + nbg_hi_err[kk], color = color_s[kk], alpha = 0.15,)

	legend_1 = ax1.legend( [ fig_name[0], fig_name[1] ], loc = 1, frameon = False, fontsize = 22,)
	legend_0 = ax1.legend( loc = 3, frameon = False, fontsize = 22, )
	ax1.add_artist( legend_1 )

	ax1.set_ylim( 21.5, 33.5 )
	ax1.invert_yaxis()

	ax1.set_xlim( 1e-2, 1e0 )
	ax1.set_xscale('log')
	ax1.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 22,)
	ax1.set_ylabel('$ \\mu \; [mag \, / \, arcsec^2] $', fontsize = 22,)

	x_tick_arr = [ 1e-2, 1e-1, 1e0]
	tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
	ax1.set_xticks( x_tick_arr )
	ax1.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
	ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 22,)


	ax2.plot( lo_R, lo_surf_M, ls = '--', color = line_c[0], alpha = 0.75, label = fig_name[0],)
	ax2.fill_between( lo_R, y1 = lo_surf_M - lo_surf_M_err, y2 = lo_surf_M + lo_surf_M_err, color = line_c[0], alpha = 0.12,)
	ax2.plot( hi_R, hi_surf_M, ls = '-', color = line_c[1], alpha = 0.75, label = fig_name[1],)
	ax2.fill_between( hi_R, y1 = hi_surf_M - hi_surf_M_err, y2 = hi_surf_M + hi_surf_M_err, color = line_c[1], alpha = 0.12,)

	ax2.plot( tot_R, tot_surf_m, ls = '-.', color = 'k', alpha = 0.75, label = '$\\mathrm{All} \; \\mathrm{clusters}$')
	ax2.fill_between( tot_R, y1 = tot_surf_m - tot_surf_m_err, y2 = tot_surf_m + tot_surf_m_err, color = 'k', alpha = 0.12,)

	ax2.axvline( x = 0.1, ls = ':', color = 'k', linewidth = 2.5, alpha = 0.4,)
	ax2.axvline( x = 0.2, ls = ':', color = 'k', linewidth = 2.5, alpha = 0.4,)
	ax2.text( 0.15, 1e4, s = '$R_{ \\mathrm{SOI} }$', fontsize = 24, rotation = 'vertical', color = 'k', alpha = 0.4, fontstyle = 'italic',)

	ax2.set_xlim( 1e-2, 1e0 )
	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_ylim( 5e3, 3e8 )
	ax2.legend( loc = 3, frameon = False, fontsize = 22, handletextpad = 0.2,)
	ax2.set_ylabel('$ \\Sigma_{\\ast} \; [M_{\\odot} \, / \, kpc^2]$', fontsize = 22,)

	ax2.yaxis.set_minor_locator( ticker.LogLocator( base = 10.0, subs = 'all', numticks = 100 ) )
	ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 22,)


	bot_ax2.plot( lo_eta_R / 1e3, lo_eta, ls = '--', color = line_c[0], alpha = 0.75,)
	bot_ax2.fill_between( lo_eta_R / 1e3, y1 = lo_eta - lo_eta_err, y2 = lo_eta + lo_eta_err, color = line_c[0], alpha = 0.12,)

	bot_ax2.plot( hi_eta_R / 1e3, hi_eta, ls = '-', color = line_c[1], alpha = 0.75,)
	bot_ax2.fill_between( hi_eta_R / 1e3, y1 = hi_eta - hi_eta_err, y2 = hi_eta + hi_eta_err, color = line_c[1], alpha = 0.12,)

	bot_ax2.plot( tot_R, tot_surf_m / tot_surf_m, ls = '-.', color = 'k', alpha = 0.75,)

	bot_ax2.axvline( x = 0.1, ls = ':', color = 'k', linewidth = 2.5, alpha = 0.4,)
	bot_ax2.axvline( x = 0.2, ls = ':', color = 'k', linewidth = 2.5, alpha = 0.4,)

	bot_ax2.set_xlim( ax2.get_xlim() )
	bot_ax2.set_xscale( 'log' )
	bot_ax2.set_xlabel('$R \; [\\mathrm{M}pc] $', fontsize = 22,)

	bot_ax2.set_ylim( 0.45, 1.55 )
	bot_ax2.set_ylabel( '$ \\Sigma_{\\ast} \, / \, \\Sigma_{\\ast}^{ \\mathrm{All} } $', fontsize = 22,)

	x_tick_arr = [ 1e-2, 1e-1, 1e0]
	tick_lis = ['$\\mathrm{0.01}$','$\\mathrm{0.1}$', '$\\mathrm{1}$']
	bot_ax2.set_xticks( x_tick_arr )
	bot_ax2.get_xaxis().set_major_formatter( ticker.FixedFormatter( tick_lis ) )
	bot_ax2.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	bot_ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 22,)
	ax2.set_xticklabels( [] )

# plt.savefig('/home/xkchen/%s_subsample_result.png' % band_str, dpi = 300)
plt.savefig('/home/xkchen/%s_subsample_result.pdf' % band_str, dpi = 300)
plt.close()

