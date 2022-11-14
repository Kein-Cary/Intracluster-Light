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
def Moffat_func(R, A0, R0, n):
	mf = A0 / ( 1 + (R / R0)**2 )**n
	return mf

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


### === 
def power1_func( R, A0, R0, alpha, belta ):
	mf = A0 * ( R / R0 + 1)**(-alpha) * ( R / R0)**(-belta)
	return mf

def err_fit_func(p, x, y, params, yerr):

	_SB0_arr = params[0]

	#.
	A0, Rc0, alpha, belta = p[:]

	mf0 = power1_func( x, A0, Rc0, alpha, belta )
	mf = mf0 * _SB0_arr

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


R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m

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
	bp_0 = 0.5

	inti_params = [ fit_SB0 ]


	### === 
	po = [ A0, R_bk, ap_0, bp_0 ]

	bounds = [ [1e-4, 1e2], [5, 50], [-5, 5], [-5, 5] ]

	# E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
	# 								method = 'L-BFGS-B', bounds = bounds,)

	E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
									method = 'Powell',)

	print( E_return )
	popt = E_return.x

	print( '*' * 10 )
	print( popt )

	A0_fit, Rc0_fit, alpha_fit, belta_fit = popt[:]

	#.
	new_r = np.logspace( -1, 2, 200 )

	_SB0_arr = interp.splev( new_r, tmp_F, der = 0)

	mf0 = power1_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit )
	mod_f = mf0 * _SB0_arr
	mod_eta = mod_f / _SB0_arr

	##.
	dd_SB0 = interp.splev( fit_R, tmp_F, der = 0)
	dd_mf = power1_func( fit_R, A0_fit, Rc0_fit, alpha_fit, belta_fit )
	dd_sb = dd_mf * dd_SB0

	diff = dd_sb - fit_sb
	chi2 = np.sum( diff**2 / fit_sb_err**2 )
	chi2nv = chi2 / ( len( fit_sb ) - len( popt ) )


	##.
	fig = plt.figure( figsize = (18.2, 4.8) )
	gax = fig.add_axes([0.03, 0.11, 0.29, 0.85])
	hax = fig.add_axes([0.37, 0.11, 0.29, 0.85])
	ax = fig.add_axes([0.70, 0.11, 0.29, 0.85])

	gax.plot( new_r, mf0, 'r-', label = '$F_{trunc}$')

	gax.set_xscale('log')
	gax.set_xlabel('$R \; [kpc]$', fontsize = 13)
	gax.legend( loc = 3, frameon = False, fontsize = 13,)
	gax.axvline( x = Rc0_fit, ls = ':', color = 'k', alpha = 0.25,)


	hax.errorbar( kk_r, kk_sb, yerr = kk_err, marker = '.', ls = '--', color = 'b',
		ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, alpha = 0.75, label = fig_name[ tt ],)

	hax.errorbar( cp_k_r, cp_k_sb, yerr = cp_k_err, marker = '.', ls = '-', color = 'r',
		ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = fig_name[ -1 ] + ' ($\\mu_{0}$)',)

	hax.plot( new_r, mod_f, 'k:', alpha = 0.5, label = '$F_{trunc} \\times \\mu_{0}$', lw = 2.5,)


	hax.annotate( s = '$\\chi^{2} / \\nu = %.5f$' % chi2nv, color = 'k', xy = (0.55, 0.85),
				xycoords = 'axes fraction', fontsize = 13,)
	hax.axvline( x = Rc0_fit, ls = ':', color = 'k', alpha = 0.25,)
	hax.axvline( x = R_lim, ls = '-.', color = 'k', alpha = 0.25,)

	hax.legend( loc = 3, frameon = False, fontsize = 13,)

	hax.set_xscale('log')
	hax.set_xlim( 1e0, 1e2 )
	hax.set_xlabel('$R \; [kpc]$', fontsize = 13)

	hax.set_ylim( 1e-3, 5e0 )
	hax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 13,)
	hax.set_yscale('log')
	hax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)


	ax.plot( kk_r, tmp_eta, ls = '-', color = 'b', alpha = 0.75,)
	ax.fill_between( kk_r, y1 = tmp_eta - tmp_eta_err, y2 = tmp_eta + tmp_eta_err, color = 'b', alpha = 0.12,)

	ax.plot( new_r, mod_eta, 'k:', alpha = 0.75,)

	ax.axvline( x = Rc0_fit, ls = ':', color = 'k', alpha = 0.25,)
	ax.axvline( x = R_lim, ls = '-.', color = 'k', alpha = 0.25,)

	ax.set_xlim( 1e0, 1e2 )

	ax.set_xscale('log')
	ax.set_xlabel('$R \; [kpc]$', fontsize = 13)

	ax.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
						fontsize = 13, labelpad = 8)
	ax.set_ylim( 0, 1.0 )

	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
	ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

	plt.savefig('/home/xkchen/SB_ratio_fitting_%d.png' % tt, dpi = 300)
	plt.close()

