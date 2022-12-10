"""
use to model SB profiles before background subtracted
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy.table import Table, QTable
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord
from pynverse import inversefunc
from scipy import optimize
import scipy.signal as signal
from scipy import special


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === ### func.s
def sersic_func( R, n, Ie, Re ):
	"""
	for n > 0.36
	"""
	bc0 = 4 / (405 * n)
	bc1 = 45 / (25515 * n**2)
	bc2 = 131 / (1148175 * n**3)
	bc3 = 2194697 / ( 30690717750 * n**4 )

	b_n = 2 * n - 1/3 + bc0 + bc1 + bc2 - bc3
	mf0 = -b_n * ( R / Re )**( 1 / n )
	Ir = Ie * np.exp( mf0 + b_n )
	return Ir

##. core-like funcs
def Modi_Ferrer_func( R, A0, R_bk, belta, alpha):

	mf0 = 1 - ( R / R_bk )**( 2 - belta)
	mf = mf0**alpha

	return mf * A0

def Empi_King_func( R, A0, R_t, R_c, alpha):

	mf0 = 1 + (R_t / R_c)**2
	mf1 = mf0**( 1 / alpha )

	mf2 = 1 / ( 1 - 1 / mf1 )**alpha

	mf4 = 1 + ( R / R_c)**2
	mf5 = mf4**( 1 / alpha )

	mf6 = 1 / mf5 - 1 / mf1

	mf = mf2 * mf6**alpha

	return mf * A0

def Nuker_func( R, A0, R_bk, alpha, belta, gamma):

	mf0 = 2**( (belta - gamma) / alpha )
	mf1 = mf0 / ( R / R_bk)**gamma

	mf2 = 1 + ( R / R_bk  )**alpha
	mf3 = mf2**( (gamma - belta) / alpha )

	mf = mf1 * mf3

	return mf * A0

##.
def err_fit_func(p, x, y, params, yerr):

	_SB0_arr = params[0]

	#.
	A0, R_bk, alpha, belta, gamma = p[:]

	mf0 = Nuker_func( x, A0, R_bk, alpha, belta, gamma)
	mf = mf0 * _SB0_arr

	delta = mf - y

	# cov_inv = np.linalg.pinv( cov_mx )
	# chi2 = delta.T.dot( cov_inv ).dot(delta)
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf

### === 
def power1_func( R, R0, alpha ):
	A0 = 1.
	mf = A0 * ( (R / R0)**2.5 + 1)**alpha * 2**(-alpha)
	return mf / mf[0] - 1

def cc_err_fit_func(p, x, y, params, yerr):

	_SB0_arr = params[0]

	#.
	A0, R_bk, alpha, belta, gamma, R1, delta = p[:]

	mf0 = Nuker_func( x, A0, R_bk, alpha, belta, gamma)
	mf1 = power1_func( x, R1, delta )

	mf = mf0 * _SB0_arr + mf1 * _SB0_arr

	delta = mf - y
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf


### === ### data load
##. sat_img without BCG
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'

bin_rich = [ 20, 30, 50, 210 ]

# R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m
R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m


#. for figs
color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##. background shuffle list order
list_order = 13

N_sample = 100

band_str = 'r'


##... BG-subtracted SBs
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	pt_r = np.array( dat['r'] )
	pt_sb = np.array( dat['sb'] )
	pt_err = np.array( dat['sb_err'] )

	#.
	nbg_R.append( pt_r )
	nbg_SB.append( pt_sb )
	nbg_err.append( pt_err )


##... modelling for the sat SB profile inner radii bins
def inner_fit():

	#.
	for tt in range( len(R_bins) - 2 ):

		kk_r = nbg_R[ tt ]
		kk_sb = nbg_SB[ tt ]
		kk_err = nbg_err[ tt ]

		cp_k_r = nbg_R[ -1 ]
		cp_k_sb = nbg_SB[ -1 ]
		cp_k_err = nbg_err[ -1 ]

		##.
		# dd_mx = cp_k_r <= 90  ## kpc
		# tmp_F = interp.interp1d( cp_k_r[ dd_mx ], cp_k_sb[ dd_mx ], kind = 'linear', fill_value = 'extrapolate')
		# tmp_F = interp.splrep( cp_k_r[ dd_mx ], cp_k_sb[ dd_mx ], s = 0)
		tmp_F = interp.splrep( cp_k_r, cp_k_sb, s = 0)

		##.
		tmp_eta = kk_sb / interp.splev( kk_r, tmp_F, der = 0)
		tmp_eta_err = kk_err / interp.splev( kk_r, tmp_F, der = 0)


		##.
		R_lim = 20    ##. kpc
		id_R_lim = kk_r <= R_lim

		fit_R = kk_r[ id_R_lim ]
		fit_sb = kk_sb[ id_R_lim ]
		fit_sb_err = kk_err[ id_R_lim ]

		fit_SB0 = interp.splev( fit_R, tmp_F, der = 0)

		##.
		R_bk = 10
		A0 = 1.5
		ap_0 = 0.5

		inti_params = [ fit_SB0 ]


		##.
		alpha = 2
		belta = 2
		gamma = 0

		po = [ A0, R_bk, alpha, belta, gamma ]

		bounds = [ [1e-4, 1e1], [1, 200], [-10, 10], [-2.5, 2.5], [-2.5, 2.5] ]

		E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
										method = 'L-BFGS-B', bounds = bounds,)

		# E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
		# 								method = 'Powell',)

		print( E_return )
		popt = E_return.x

		print( '*' * 10 )
		print( popt )

		A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit = popt[:]

		##.
		keys = ['A0', 'Rc', 'alpha', 'belta', 'gamma']
		values = [ A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill, index = ['k', 'v'])
		out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'sat_%.2f-%.2fR200m_%s-band_trunF_mod.csv' % (R_bins[tt], R_bins[tt + 1], band_str),)

		#.
		new_r = np.logspace( -1, 2, 200 )

		_SB0_arr = interp.splev( new_r, tmp_F, der = 0)

		mf0 = Nuker_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )

		mod_f = mf0 * _SB0_arr
		mod_eta = mod_f / _SB0_arr

		dd_SB0 = interp.splev( fit_R, tmp_F, der = 0)
		dd_mf = Nuker_func( fit_R, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )
		dd_sb = dd_mf * dd_SB0


		##.
		diff = dd_sb - fit_sb
		chi2 = np.sum( diff**2 / fit_sb_err**2 )
		chi2nv = chi2 / ( len( fit_sb ) - len( popt ) )


		#.
		fig = plt.figure( figsize = (18.2, 4.8) )
		gax = fig.add_axes([0.03, 0.11, 0.29, 0.85])
		hax = fig.add_axes([0.37, 0.11, 0.29, 0.85])
		ax = fig.add_axes([0.70, 0.11, 0.29, 0.85])

		gax.plot( new_r, mf0, 'r-', label = '$F_{trunc}$')

		gax.set_xlim( 1e0, 1e2 )
		gax.set_xscale('log')
		gax.set_xlabel('$R \; [kpc]$', fontsize = 13,)
		gax.legend( loc = 3, frameon = False, fontsize = 13,)
		gax.axvline( x = Rc0_fit, ls = '-.', color = 'k', alpha = 0.35)

		hax.errorbar( kk_r, kk_sb, yerr = kk_err, marker = '.', ls = '--', color = 'b',
			ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.65, label = fig_name[ tt ],)

		hax.errorbar( cp_k_r, cp_k_sb, yerr = cp_k_err, marker = '.', ls = '-', color = 'r',
			ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.65, label = fig_name[ -1 ] + '$\,(\\mu_{0})$',)

		hax.plot( new_r, mod_f, 'k:', alpha = 0.75, label = '$F_{trunc} \\times \\mu_{0}$', lw = 3,)

		hax.annotate( s = '$\\chi^{2} / \\nu = %.5f$' % chi2nv, color = 'k', xy = (0.03, 0.05), 
					xycoords = 'axes fraction', fontsize = 13,)
		hax.axvline( x = Rc0_fit, ls = '-.', color = 'k', alpha = 0.35)
		hax.axvline( x = R_lim, ls = ':', color = 'k', alpha = 0.35)

		hax.legend( loc = 1, fontsize = 13,)

		hax.set_xscale('log')
		hax.set_xlim( 1e0, 1e2 )
		hax.set_xlabel('$R \; [kpc]$', fontsize = 13)

		hax.set_ylim( 1e-3, 5e0 )
		hax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 13,)
		hax.set_yscale('log')

		ax.plot( kk_r, tmp_eta, ls = '-', color = 'b', alpha = 0.75,)
		ax.fill_between( kk_r, y1 = tmp_eta - tmp_eta_err, y2 = tmp_eta + tmp_eta_err, color = 'b', alpha = 0.12,)

		ax.plot( new_r, mod_eta, 'k:', alpha = 0.75, lw = 3,)
		ax.axvline( x = Rc0_fit, ls = '-.', color = 'k', alpha = 0.35)
		ax.axvline( x = R_lim, ls = ':', color = 'k', alpha = 0.35)

		ax.set_xlim( 1e0, 1e2 )

		ax.set_xscale('log')
		ax.set_xlabel('$R \; [kpc]$', fontsize = 13)

		ax.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
							fontsize = 13, labelpad = 8)
		ax.set_ylim( 0.36, 0.9 )

		#.
		ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
		ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
		gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
		hax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

		plt.savefig('/home/xkchen/shape_compare_%d.png' % tt, dpi = 300)
		plt.close()


	##.
	fig = plt.figure( figsize = (10.4, 4.8) )
	ax0 = fig.add_axes([0.06, 0.10, 0.42, 0.80])
	ax1 = fig.add_axes([0.55, 0.10, 0.42, 0.80])

	#.
	for tt in range( len(R_bins) - 2 ):

		#.
		dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
					'sat_%.2f-%.2fR200m_%s-band_trunF_mod.csv' % (R_bins[tt], R_bins[tt + 1], band_str),)

		A0_fit, Rc0_fit = np.array( dat['A0'] )[0], np.array( dat['Rc'] )[0]
		alpha_fit, belta_fit, gamma_fit = np.array( dat['alpha'] )[0], np.array( dat['belta'] )[0], np.array( dat['gamma'] )[0]

		##.
		kk_r = nbg_R[ tt ]
		kk_sb = nbg_SB[ tt ]
		kk_err = nbg_err[ tt ]

		cp_k_r = nbg_R[ -1 ]
		cp_k_sb = nbg_SB[ -1 ]
		cp_k_err = nbg_err[ -1 ]

		##.
		tmp_F = interp.splrep( cp_k_r, cp_k_sb, s = 0)
		tmp_eta = kk_sb / interp.splev( kk_r, tmp_F, der = 0)
		tmp_eta_err = kk_err / interp.splev( kk_r, tmp_F, der = 0)


		#.
		new_r = np.logspace( -1, 2, 200 )

		_SB0_arr = interp.splev( new_r, tmp_F, der = 0)

		mf0 = Nuker_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )

		mod_f = mf0 * _SB0_arr
		mod_eta = mod_f / _SB0_arr

		##.
		ax0.plot( new_r, mf0, ls = '-', color = color_s[tt], alpha = 0.65, 
			label = '$A_{0}, R_{c}, \\alpha, \\beta, \\gamma=%.3f, \, %.3f, \, %.3f, \, %.3f, \, %.3f$' % (A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit),)
		ax0.axvline( x = Rc0_fit, ls = ':', color = color_s[tt], alpha = 0.65,)

		ax1.plot( kk_r, tmp_eta, ls = '-', color = color_s[tt], alpha = 0.65, label = fig_name[tt])
		ax1.fill_between( kk_r, y1 = tmp_eta - tmp_eta_err, y2 = tmp_eta + tmp_eta_err, color = color_s[tt], alpha = 0.12,)
		ax1.plot( new_r, mod_eta, ls = '--', color = color_s[tt], alpha = 0.65,)
		ax1.axvline( x = Rc0_fit, ls = ':', color = color_s[tt], alpha = 0.65,)

	ax0.set_xlim( 1e0, 1e2 )
	ax0.set_xscale('log')
	ax0.set_xlabel('$R \; [kpc]$', fontsize = 13,)
	ax0.set_ylim( 0.22, 1.02 )
	ax0.set_ylabel('$F_{trunc}$', fontsize = 13,)
	ax0.legend( loc = 1, fontsize = 11,)

	ax1.set_xlim( 1e0, 1e2 )
	ax1.set_xscale('log')
	ax1.set_xlabel('$R \; [kpc]$', fontsize = 13)

	ax1.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
						fontsize = 13, labelpad = 8)
	ax1.set_ylim( 0.36, 0.94 )

	# ax1.legend( loc = 1, fontsize = 12,)
	legend_2 = ax1.legend( labels = ['Observation', 'Model'], 
				loc = 2, frameon = False, fontsize = 13,)

	ax1.legend( loc = 3, frameon = False, fontsize = 13,)
	ax1.add_artist( legend_2 )

	#.
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
	ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
	ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	plt.savefig('/home/xkchen/Sat_%s-band_SB_ratio_model.png' % band_str, dpi = 300)
	plt.close()

	return

# inner_fit()

# raise


###... modelling for the sat SB profile with large radii bins
fit_pairs = []

#.
for tt in range( len(R_bins) - 2 ):

	kk_r = nbg_R[ tt ]
	kk_sb = nbg_SB[ tt ]
	kk_err = nbg_err[ tt ]

	cp_k_r = nbg_R[ -1 ]
	cp_k_sb = nbg_SB[ -1 ]
	cp_k_err = nbg_err[ -1 ]

	##.
	# dd_mx = cp_k_r <= 90  ## kpc
	# tmp_F = interp.interp1d( cp_k_r[ dd_mx ], cp_k_sb[ dd_mx ], kind = 'linear', fill_value = 'extrapolate')
	# tmp_F = interp.splrep( cp_k_r[ dd_mx ], cp_k_sb[ dd_mx ], s = 0)
	tmp_F = interp.splrep( cp_k_r, cp_k_sb, s = 0)

	##.
	tmp_eta = kk_sb / interp.splev( kk_r, tmp_F, der = 0)
	tmp_eta_err = kk_err / interp.splev( kk_r, tmp_F, der = 0)


	##.
	R_lim = 80    ##. 80, 90, 100 kpc
	id_R_lim = kk_r <= R_lim

	fit_R = kk_r[ id_R_lim ]
	fit_sb = kk_sb[ id_R_lim ]
	fit_sb_err = kk_err[ id_R_lim ]

	fit_SB0 = interp.splev( fit_R, tmp_F, der = 0)


	##.
	R_bk = 10
	A0 = 1.5
	ap_0 = 0.5

	inti_params = [ fit_SB0 ]

	##.
	alpha = 2
	belta = 2
	gamma = 0

	po = [ A0, R_bk, alpha, belta, gamma, R_bk, ap_0 ]

	bounds = [ [1e-4, 1e1], [5, 250], [-10, 10], [-2.5, 2.5], [-2.5, 2.5], [5, 200], [-2.5, 2.5] ]

	E_return = optimize.minimize( cc_err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
									method = 'L-BFGS-B', bounds = bounds,)

	# E_return = optimize.minimize( cc_err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
	# 								method = 'Powell',)

	print( E_return )
	popt = E_return.x

	print( '*' * 10 )
	print( popt )

	A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit, Rc1_fit, delta_fit = popt[:]
	fit_pairs.append( popt )

	#.
	new_r = np.logspace( -1, 2, 200 )

	_SB0_arr = interp.splev( new_r, tmp_F, der = 0)


	mf0 = Nuker_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )
	mf1 = power1_func( new_r, Rc1_fit, delta_fit )

	mod_f = mf0 * _SB0_arr + mf1 * _SB0_arr
	mod_eta = mod_f / _SB0_arr

	dd_SB0 = interp.splev( fit_R, tmp_F, der = 0)
	
	dd_mf0 = Nuker_func( fit_R, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )
	dd_mf1 = power1_func( fit_R, Rc1_fit, delta_fit )

	dd_sb = dd_mf0 * dd_SB0 + dd_mf1 * dd_SB0


	##.
	diff = dd_sb - fit_sb
	chi2 = np.sum( diff**2 / fit_sb_err**2 )
	chi2nv = chi2 / ( len( fit_sb ) - len( popt ) )

	##.
	fig = plt.figure( figsize = (18.2, 4.8) )
	gax = fig.add_axes([0.025, 0.11, 0.29, 0.85])
	hax = fig.add_axes([0.365, 0.11, 0.29, 0.85])
	ax = fig.add_axes([0.70, 0.11, 0.29, 0.85])

	gax.plot( new_r, mf0, 'r-', label = '$F_{trunc}$')
	gax.plot( new_r, mf1, 'b-', label = '$F_{cumu}$')

	gax.set_xscale('log')
	gax.set_xlabel('$R \; [kpc]$', fontsize = 13)
	gax.set_ylim( 0, 1 )

	gax.legend( loc = 2, frameon = False, fontsize = 14,)

	gax.axvline( x = Rc0_fit, ls = ':', color = 'r', alpha = 0.45,)
	gax.axvline( x = Rc1_fit, ls = ':', color = 'b', alpha = 0.45,)
	gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
	gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


	hax.errorbar( kk_r, kk_sb, yerr = kk_err, marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.65, label = fig_name[ tt ],)

	hax.errorbar( cp_k_r, cp_k_sb, yerr = cp_k_err, marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.65, label = fig_name[ -1 ] + ' ($\\mu_{0}$)',)

	hax.plot( new_r, mod_f, 'k--', alpha = 0.5, label = '$F_{trunc} \\times \\mu_{0}$' + '+$F_{cumu} \\times \\mu_{0}$', lw = 3,)

	hax.plot( new_r, mf0 * _SB0_arr, 'r--', alpha = 0.5, label = '$F_{trunc} \\times \\mu_{0}$',)
	hax.plot( new_r, mf1 * _SB0_arr, 'b--', alpha = 0.5, label = '$F_{cumu} \\times \\mu_{0}$',)


	hax.annotate( s = '$\\chi^{2} / \\nu = %.5f$' % chi2nv, color = 'k', xy = (0.05, 0.45),
				xycoords = 'axes fraction', fontsize = 13,)

	hax.axvline( x = Rc0_fit, ls = ':', color = 'r', alpha = 0.45,)
	hax.axvline( x = Rc1_fit, ls = ':', color = 'b', alpha = 0.45,)
	hax.axvline( x = R_lim, ls = '-.', color = 'k', alpha = 0.25,)

	hax.legend( loc = 3, frameon = True, fontsize = 13,)

	hax.set_xscale('log')
	hax.set_xlim( 1e0, 1e2 )
	hax.set_xlabel('$R \; [kpc]$', fontsize = 13)

	hax.set_ylim( 5e-4, 5e0 )
	hax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 13,)
	hax.set_yscale('log')
	hax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)


	ax.plot( kk_r, tmp_eta, ls = '-', color = 'b', alpha = 0.5,)
	ax.fill_between( kk_r, y1 = tmp_eta - tmp_eta_err, y2 = tmp_eta + tmp_eta_err, color = 'b', alpha = 0.12,)

	ax.plot( new_r, mod_eta, 'k--', alpha = 0.75, lw = 3,)

	ax.axvline( x = Rc0_fit, ls = ':', color = 'r', alpha = 0.45,)
	ax.axvline( x = Rc1_fit, ls = ':', color = 'b', alpha = 0.45,)
	ax.axvline( x = R_lim, ls = '-.', color = 'k', alpha = 0.25,)

	ax.set_xlim( 1e0, 1e2 )

	ax.set_xscale('log')
	ax.set_xlabel('$R \; [kpc]$', fontsize = 13)

	ax.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
						fontsize = 13, labelpad = 8)
	ax.set_ylim( 0.38, 0.9 )

	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
	ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	plt.savefig('/home/xkchen/SB_ratio_fitting_%d.png' % tt, dpi = 300)
	plt.close()


##.???
fig = plt.figure( figsize = (18.2, 4.8) )
ax0 = fig.add_axes([0.04, 0.11, 0.29, 0.85])
ax1 = fig.add_axes([0.37, 0.11, 0.29, 0.85])
ax2 = fig.add_axes([0.70, 0.11, 0.29, 0.85])

#.
for tt in range( len(R_bins) - 2 ):

	##.
	kk_r = nbg_R[ tt ]
	kk_sb = nbg_SB[ tt ]
	kk_err = nbg_err[ tt ]

	cp_k_r = nbg_R[ -1 ]
	cp_k_sb = nbg_SB[ -1 ]
	cp_k_err = nbg_err[ -1 ]

	##.
	tmp_F = interp.splrep( cp_k_r, cp_k_sb, s = 0)
	tmp_eta = kk_sb / interp.splev( kk_r, tmp_F, der = 0)
	tmp_eta_err = kk_err / interp.splev( kk_r, tmp_F, der = 0)

	##.
	A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit, Rc1_fit, delta_fit = fit_pairs[ tt ][:]

	new_r = np.logspace( -1, 2, 200 )

	_SB0_arr = interp.splev( new_r, tmp_F, der = 0)

	mf0 = Nuker_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )
	mf1 = power1_func( new_r, Rc1_fit, delta_fit )

	mod_f = mf0 * _SB0_arr + mf1 * _SB0_arr
	mod_eta = mod_f / _SB0_arr


	##.
	ax0.plot( new_r, mf0, ls = '-', color = color_s[tt], alpha = 0.65, 
		label = '$A_{0}, R_{c}, \\alpha, \\beta, \\gamma=%.3f, \, %.3f, \, %.3f, \, %.3f, \, %.3f$' % 
		(A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit),)
	ax0.axvline( x = Rc0_fit, ls = ':', color = color_s[tt], alpha = 0.45,)

	ax1.plot( new_r, mf1, ls = '-', color = color_s[tt], alpha = 0.65, 
		label = '$R_{c}, \\beta =%.3f, \, %.3f$' % (Rc1_fit, delta_fit),)
	ax1.axvline( x = Rc1_fit, ls = '--', color = color_s[tt], alpha = 0.45,)


	ax2.plot( kk_r, tmp_eta, ls = '-', color = color_s[tt], alpha = 0.5, label = fig_name[tt],)
	ax2.fill_between( kk_r, y1 = tmp_eta - tmp_eta_err, y2 = tmp_eta + tmp_eta_err, color = color_s[tt], alpha = 0.12,)
	ax2.plot( new_r, mod_eta, ls = '--', color = color_s[tt], alpha = 0.75, lw = 3,)

	ax2.axvline( x = Rc0_fit, ls = ':', color = color_s[tt], alpha = 0.45,)
	ax2.axvline( x = Rc1_fit, ls = '--', color = color_s[tt], alpha = 0.45,)

##.
ax0.set_xlim( 1e0, 1e2 )
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [kpc]$', fontsize = 13)
ax0.set_ylim( 0.22, 1.02 )
ax0.set_ylabel( '$F_{trunc}$', fontsize = 13)

ax0.legend( loc = 2, frameon = True, fontsize = 12,)
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

#.
ax1.set_xlim( 1e0, 1e2 )
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [kpc]$', fontsize = 13,)
ax1.set_ylim( 0, 1 )
ax1.set_ylabel( '$F_{cumu}$', fontsize = 13,)

ax1.legend( loc = 2, frameon = False, fontsize = 13,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

#.
# ax2.legend( loc = 3, frameon = False, fontsize = 14,)

legend_2 = ax2.legend( labels = ['Observation', 'Model'], 
			loc = 2, frameon = False, fontsize = 13,)

ax2.legend( loc = 3, frameon = False, fontsize = 13,)
ax2.add_artist( legend_2 )

ax2.set_xlim( 1e0, 1e2 )
ax2.set_xscale('log')
ax2.set_xlabel('$R \; [kpc]$', fontsize = 13)

ax2.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
					fontsize = 13, labelpad = 8)
ax2.set_ylim( 0.36, 0.94 )
ax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax2.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

plt.savefig('/home/xkchen/Sat_%s-band_SB_ratio_model.png' % band_str, dpi = 300)
plt.close()

raise


### === ### ratio profile indicated Rt compare
##.
marks = ['s', '>', 'o']
mark_size = [10, 25, 35]
crit_eta = [0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95]

##. read sample catalog and get the average centric distance
R_aveg = []
R_sat_arr = []

for dd in range( len( R_bins ) - 1 ):

	dd_Rs = np.array([])

	for qq in range( 3 ):

		cat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/' + 
			'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
			% ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

		x_Rc = np.array( cat['R2Rv'] )   ## R / R200m
		cp_x_Rc = x_Rc + 0

		dd_Rs = np.r_[ dd_Rs, cp_x_Rc ]

	##.
	R_aveg.append( np.mean( dd_Rs ) )
	R_sat_arr.append( dd_Rs )


##.
fig = plt.figure( figsize = (10.8, 4.8) )
ax0 = fig.add_axes([0.06, 0.10, 0.40, 0.80])
ax1 = fig.add_axes([0.54, 0.10, 0.40, 0.80])

tcp_R = []

#.
for tt in range( len(R_bins) - 2 ):

	##.
	A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit, Rc1_fit, delta_fit = fit_pairs[ tt ][:]

	tcp_R.append( Rc0_fit )

	new_r = np.logspace( -1, 2, 200 )

	_SB0_arr = interp.splev( new_r, tmp_F, der = 0)

	mf0 = Nuker_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )
	mf1 = power1_func( new_r, Rc1_fit, delta_fit )

	#.
	ax0.plot( new_r, mf0, ls = '-', color = color_s[tt], alpha = 0.65, 
		label = '$A_{0}, R_{c}, \\alpha, \\beta, \\gamma=%.3f, \, %.3f, \, %.3f, \, %.3f, \, %.3f$' % 
		(A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit),)
	ax0.axvline( x = Rc0_fit, ls = ':', color = color_s[tt], alpha = 0.45,)

ax0.set_xlim( 1e0, 1e2 )
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [kpc]$', fontsize = 13,)
ax0.set_ylim( 0.22, 1.02 )
ax0.set_ylabel('$F_{trunc}$', fontsize = 13,)
ax0.legend( loc = 2, fontsize = 10,)

ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


##.. Rt estimate based on 'Fang+2016 Model'
Rc = []

pat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/' + 
		'Extend_BCGM_gri-common_Rs-bin_over-rich_sat_Rt.csv',)

for dd in range( len( R_bins ) - 1 ):
	tt_Rc = np.array( pat[ fig_name[dd] ] )[0]
	Rc.append( tt_Rc )

ax1.plot( R_aveg[:-1], tcp_R, marker = marks[1], color = 'k', ls = '-', alpha = 0.75, label = '$R_{c}$ of $F_{trunc}$',)
ax1.plot( R_aveg, Rc, marker = marks[0], ls = ':', color = 'k', alpha = 0.75, label = 'Fang+2016 Model',)

#.
for oo in range( len( crit_eta ) ):

	id_set = oo

	##. estimate with ratio decrease
	pat = fits.open( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/' + 
					'Extend_BCGM_gri-common_Rs-bin_over-rich_r-band_polyfit_Rt_test.fits',)

	p_table = pat[1].data

	cp_Rc = []

	for tt in range( len(R_bins) - 1 ):

		Rt_arr = np.array( p_table[ fig_name[tt] ] )
		cp_Rc.append( Rt_arr[ id_set ] )

	##.
	ax1.plot( R_aveg, cp_Rc, marker = marks[2], ls = '--', color = mpl.cm.rainbow( oo / len( crit_eta ) ), alpha = 0.75, 
			label = 'Drops by %d%%' % (crit_eta[ id_set ] * 100),)

ax1.legend( loc = 4, frameon = False, fontsize = 11, markerfirst = False,)

ax1.set_ylabel('$ R_{t} \; [kpc \, / \, h]$', fontsize = 13,)
# ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.set_yscale('log')

ax1.set_xlim( 0.1, 0.8 )
ax1.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$', fontsize = 13,)
ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

plt.savefig('/home/xkchen/trunc_F_Rt_compare.png', dpi = 300)
plt.close()


##.
fig = plt.figure( figsize = (10.8, 4.8) )
ax0 = fig.add_axes([0.08, 0.10, 0.40, 0.80])
ax1 = fig.add_axes([0.56, 0.10, 0.40, 0.80])

tcp_R = []

#.
for tt in range( len(R_bins) - 2 ):

	##.
	A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit, Rc1_fit, delta_fit = fit_pairs[ tt ][:]

	tcp_R.append( Rc1_fit )

	new_r = np.logspace( -1, 2, 200 )

	_SB0_arr = interp.splev( new_r, tmp_F, der = 0)

	mf0 = Nuker_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit, gamma_fit )
	mf1 = power1_func( new_r, Rc1_fit, delta_fit )

	#.
	ax0.plot( new_r, mf1, ls = '-', color = color_s[tt], alpha = 0.65, 
		label = '$R_{c}, \\beta =%.3f, \, %.3f$' % (Rc1_fit, delta_fit),)
	ax0.axvline( x = Rc1_fit, ls = ':', color = color_s[tt], alpha = 0.45,)

ax0.set_xlim( 1e0, 1e2 )
ax0.set_xscale('log')
ax0.set_xlabel('$R \; [kpc]$', fontsize = 13,)

ax0.set_ylim( 1e-5, 1.0 )
ax0.set_yscale('log')
ax0.set_ylabel('$F_{cumu}$', fontsize = 13,)
# ax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax0.legend( loc = 4, fontsize = 12,)

ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

##.. Rt estimate based on 'Fang+2016 Model'
Rc = []

pat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/' + 
		'Extend_BCGM_gri-common_Rs-bin_over-rich_sat_Rt.csv',)

for dd in range( len( R_bins ) - 1 ):
	tt_Rc = np.array( pat[ fig_name[dd] ] )[0]
	Rc.append( tt_Rc )

ax1.plot( R_aveg[:-1], tcp_R, marker = marks[1], color = 'k', ls = '-', alpha = 0.75, label = '$R_{c}$ of $F_{cumu}$',)
ax1.plot( R_aveg, Rc, marker = marks[0], ls = ':', color = 'k', alpha = 0.75, label = 'Fang+2016 Model',)

#.
for oo in range( len( crit_eta ) ):

	id_set = oo

	##. estimate with ratio decrease
	pat = fits.open( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/' + 
					'Extend_BCGM_gri-common_Rs-bin_over-rich_r-band_polyfit_Rt_test.fits',)

	p_table = pat[1].data

	cp_Rc = []

	for tt in range( len(R_bins) - 1 ):

		Rt_arr = np.array( p_table[ fig_name[tt] ] )
		cp_Rc.append( Rt_arr[ id_set ] )

	##.
	ax1.plot( R_aveg, cp_Rc, marker = marks[2], ls = '--', color = mpl.cm.rainbow( oo / len( crit_eta ) ), alpha = 0.75, 
			label = 'Drops by %d%%' % (crit_eta[ id_set ] * 100),)

ax1.legend( loc = 4, frameon = False, fontsize = 11, markerfirst = False,)

ax1.set_ylabel('$ R_{t} \; [kpc \, / \, h]$', fontsize = 13,)
# ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.set_yscale('log')

ax1.set_xlim( 0.1, 0.8 )
ax1.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$', fontsize = 13,)
ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

plt.savefig('/home/xkchen/cumu_F_Rt_compare.png', dpi = 300)
plt.close()

