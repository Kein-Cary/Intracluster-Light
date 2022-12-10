"""
model the inner SB profile with the same model as entire sample
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.interpolate as interp

from astropy import modeling as Model
from pynverse import inversefunc
from astropy import cosmology as apcy
from scipy import optimize
from scipy.stats import binned_statistic as binned
from scipy import integrate as integ
from scipy import signal
from scipy import stats as sts
from scipy import fftpack
from scipy import fft

#.
from light_measure import light_measure_weit
from light_measure import light_measure_Z0_weit


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

pixel = 0.396
band = ['r', 'g', 'i']

z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance( z_ref ).value
Dl_ref = Test_model.luminosity_distance( z_ref ).value
a_ref = 1 / (z_ref + 1)



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
def Moffat_func(R, A0, Rd, n):
	mf = A0 / ( 1 + (R / Rd)**2 )**n
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

def KPA_func( R, A0, R_c0, R_c1 ):

	mod_F = Model.functional_models.KingProjectedAnalytic1D( amplitude = A0, r_core = R_c0, r_tide = R_c1)
	mf = mod_F( R )

	return mf

def power1_func( R, R0, alpha ):
	A0 = 1.
	mf = A0 * ( (R / R0)**2.5 + 1)**alpha * 2**(-alpha)
	return mf / mf[0] - 1


### === 
# def err_fit_func(p, x, y, params, yerr):

# 	_SB0_arr = params[0]

# 	#.
# 	A0, Rc0, alpha = p[:]

# 	mf0 = Moffat_func( x, A0, Rc0, alpha )
# 	mf = mf0 * _SB0_arr

# 	delta = mf - y
# 	chi2 = np.sum( delta**2 / yerr**2 )

# 	if np.isfinite( chi2 ):
# 		return chi2
# 	return np.inf

def err_fit_func(p, x, y, params, yerr):

	_SB0_arr = params[0]

	#.
	A0, Rc0, Rc1 = p[:]

	mf0 = KPA_func( x, A0, Rc0, Rc1 )
	mf = mf0 * _SB0_arr

	delta = mf - y
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf


def cc_err_fit_func(p, x, y, params, yerr):

	_SB0_arr = params[0]

	#.
	A0, Rc0, alpha, R1, belta = p[:]
	mf0 = Moffat_func( x, A0, Rc0, alpha )
	mf1 = power1_func( x, R1, belta )

	mf = mf0 * _SB0_arr + mf1 * _SB0_arr

	delta = mf - y
	chi2 = np.sum( delta**2 / yerr**2 )

	if np.isfinite( chi2 ):
		return chi2
	return np.inf



### === data load
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/nobcg_BGsub_SBs/'

bin_rich = [ 20, 30, 50, 210 ]

R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

band_str = 'r'

list_order = 13


##... figs set
color_s = ['b', 'g', 'r', 'm', 'k']

fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

#.
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
samp_name = ['Low $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $', 'High $ M_{\\ast}^{\\mathrm{BCG}} \\mid \\lambda $']

band_str = 'r'

qq = 1  ##. 0, 1

##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for ll in range( len(R_bins) - 1 ):

	#.
	dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' 
					% (cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str),)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )


##... inner SB profiles central raius

for tt in range( len(R_bins) - 2 ):

	kk_r = nbg_R[ tt ]
	kk_sb = nbg_SB[ tt ]
	kk_err = nbg_err[ tt ]

	cp_k_r = nbg_R[ -1 ]
	cp_k_sb = nbg_SB[ -1 ]
	cp_k_err = nbg_err[ -1 ]

	##.
	tmp_F = interp.splrep( cp_k_r, cp_k_sb, s = 0)

	##.
	dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB_ratio.csv'
						% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

	tmp_eta_R = np.array( dat['R'] )
	tmp_eta = np.array( dat['ratio'] )
	tmp_eta_err = np.array( dat['ratio_err'] )


	# ### === 
	# # R_lim = 50    ##. kpc  ##. for high BCG_Mstar bin
	# R_lim = 60      ##. kpc  ##. for low BCG_Mstar bin

	# id_R_lim = kk_r <= R_lim

	# fit_R = kk_r[ id_R_lim ]
	# fit_sb = kk_sb[ id_R_lim ]
	# fit_sb_err = kk_err[ id_R_lim ]

	# fit_SB0 = interp.splev( fit_R, tmp_F, der = 0)


	# ##.
	# R_bk = 10
	# A0 = 1.5
	# ap_0 = 0.5
	# bp_0 = 0.5

	# inti_params = [ fit_SB0 ]

	# ### === 
	# po = [ A0, R_bk, ap_0, R_bk, bp_0 ]
	# bounds = [ [1e-4, 1e2], [5, 300], [-2.5, 2.5], [5, 200], [-2.5, 2.5] ]

	# E_return = optimize.minimize( cc_err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
	# 								method = 'L-BFGS-B', bounds = bounds,)

	# # E_return = optimize.minimize( cc_err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
	# # 								method = 'Powell',)

	# print( E_return )
	# popt = E_return.x

	# print( '*' * 10 )
	# print( popt )

	# A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit = popt[:]

	# ##.
	# keys = ['A_Moffat', 'Rd_Moffat', 'n_Moffat', 'Rc', 'beta']
	# values = [ A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit ]
	# fill = dict( zip( keys, values) )
	# out_data = pds.DataFrame( fill, index = ['k', 'v'])
	# out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
	# 			'%s_sat_%.2f-%.2fR200m_%s-band_SB_trunF-model_fit.csv' % (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

	# ##.
	# new_r = np.logspace( -1, 2, 200 )

	# _SB0_arr = interp.splev( new_r, tmp_F, der = 0)

	# mf0 = Moffat_func( new_r, A0_fit, Rc0_fit, alpha_fit )
	# mf1 = power1_func( new_r, Rc1_fit, beta_fit )

	# mod_f = mf0 * _SB0_arr + mf1 * _SB0_arr
	# mod_eta = mod_f / _SB0_arr

	# ##.
	# dd_SB0 = interp.splev( fit_R, tmp_F, der = 0)

	# dd_mf0 = Moffat_func( fit_R, A0_fit, Rc0_fit, alpha_fit )
	# dd_mf1 = power1_func( fit_R, Rc1_fit, beta_fit )

	# dd_sb = dd_mf0 * dd_SB0 + dd_mf1 * dd_SB0

	# diff = dd_sb - fit_sb
	# chi2 = np.sum( diff**2 / fit_sb_err**2 )
	# chi2nv = chi2 / ( len( fit_sb ) - len( popt ) )


	### === 
	R_lim = 10   ##. 10, 20, 30 kpc

	id_R_lim = kk_r <= R_lim

	fit_R = kk_r[ id_R_lim ]
	fit_sb = kk_sb[ id_R_lim ]
	fit_sb_err = kk_err[ id_R_lim ]

	fit_SB0 = interp.splev( fit_R, tmp_F, der = 0)

	##.
	R_bk = 10
	A0 = 1.5

	inti_params = [ fit_SB0 ]

	### === 
	po = [ A0, R_bk, 2 * R_bk ]

	bounds = [ [1e-4, 1e2], [0, 30], [5, 200] ]

	# E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
	# 								method = 'L-BFGS-B', bounds = bounds,)

	E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( fit_R, fit_sb, inti_params, fit_sb_err), 
									method = 'Powell',)

	print( E_return )
	popt = E_return.x

	print( '*' * 10 )
	print( popt )

	A0_fit, Rc0_fit, Rc1_fit = popt[:]

	##.
	keys = [ 'A_power', 'R_core', 'R_tidal' ]
	values = [ A0_fit, Rc0_fit, Rc1_fit ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill, index = ['k', 'v'])
	out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
				'%s_sat_%.2f-%.2fR200m_%s-band_SB_inner-KPA_fit.csv' 
				% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

	#.
	new_r = np.logspace( -1, 2, 200 )

	_SB0_arr = interp.splev( new_r, tmp_F, der = 0)

	mf0 = KPA_func( new_r, A0_fit, Rc0_fit, Rc1_fit )
	mod_f = mf0 * _SB0_arr
	mod_eta = mod_f / _SB0_arr

	##.
	dd_SB0 = interp.splev( fit_R, tmp_F, der = 0)
	dd_mf = KPA_func( fit_R, A0_fit, Rc0_fit, Rc1_fit )
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


	ax.plot( tmp_eta_R, tmp_eta, ls = '-', color = 'b', alpha = 0.75,)
	ax.fill_between( tmp_eta_R, y1 = tmp_eta - tmp_eta_err, y2 = tmp_eta + tmp_eta_err, color = 'b', alpha = 0.12,)

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

	plt.savefig('/home/xkchen/%s_sat_%.2f-%.2fR200m_%s-band_SB-ratio_fit.png' 
			% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)
	plt.close()


raise



### === ### infered R compare
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/cat/'

##. galaxy catalog information
R_sat = []
aveg_Rsat = []

for qq in range( 2 ):

	R_aveg = []
	R_sat_arr = []

	for tt in range( len(R_bins) - 1 ):

		cat = pds.read_csv( cat_path + '%s_clust_frame-lim_Pm-cut_%.2f-%.2fR200m_mem_cat.csv' 
						% (cat_lis[qq], R_bins[tt], R_bins[tt + 1]),)

		x_Rc = np.array( cat['R2Rv'] )   ## R / R200m

		#.
		R_aveg.append( np.median( x_Rc ) )
		R_sat_arr.append( x_Rc )

	#.
	R_sat.append( R_sat_arr )
	aveg_Rsat.append( R_aveg )


##.
fit_pairs_arr = []

for qq in range( 2 ):

	fit_pairs = []

	for tt in range( len(R_bins) - 2 ):

		# ##.
		# dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
		# 					'%s_sat_%.2f-%.2fR200m_%s-band_SB_trunF-model_fit.csv' 
		# 					% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

		# A0_fit = np.array( dat['A_Moffat'] )[0]
		# Rc0_fit = np.array( dat['Rd_Moffat'] )[0]
		# alpha_fit = np.array( dat['n_Moffat'] )[0]

		# Rc1_fit = np.array( dat['Rc'] )[0]
		# beta_fit = np.array( dat['beta'] )[0]

		# popt = [ A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit ]


		##.
		dat = pds.read_csv(	'/home/xkchen/figs/extend_bcgM_cat_Sat/BCG_Mstar_bin/SB_model/' + 
							'%s_sat_%.2f-%.2fR200m_%s-band_SB_inner-KPA_fit.csv' 
							% (cat_lis[qq], R_bins[tt], R_bins[tt + 1], band_str),)

		A0_fit = np.array( dat['A_power'] )[0]
		Rc0_fit = np.array( dat['R_core'] )[0]
		Rc1_fit = np.array( dat['R_tidal'] )[0]

		popt = [ A0_fit, Rc0_fit, Rc1_fit ]

		#.
		fit_pairs.append( popt )

	fit_pairs_arr.append( np.array( fit_pairs ) )


##. observed SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for qq in range( 2 ):

	tmp_R, tmp_SB, tmp_err = [], [], []

	for ll in range( len(R_bins) - 1 ):

		#.
		dat = pds.read_csv( out_path + '%s_clust_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' 
						% (cat_lis[qq], R_bins[ll], R_bins[ll+1], band_str),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		tmp_R.append( tt_r )
		tmp_SB.append( tt_sb )
		tmp_err.append( tt_sb_err )

	#.
	nbg_R.append( tmp_R )
	nbg_SB.append( tmp_SB )
	nbg_err.append( tmp_err )


##. Rt infered from Moffat func.
crit_f = np.array( [ 0.15, 0.30, 0.45, 0.60, 0.75, 0.90 ] )

mft_Rt_arr = []

for qq in range( 2 ):

	mft_Rt = []

	for tt in range( len(R_bins) - 2 ):

		# ##.
		# A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit = fit_pairs_arr[qq][ tt ][:]

		# tt_Am = (1 - crit_f) * Moffat_func( 0, A0_fit, Rc0_fit, alpha_fit )

		# tt_R = inversefunc( Moffat_func, args = (A0_fit, Rc0_fit, alpha_fit), y_values = tt_Am )
		# mft_Rt.append( tt_R )


		##.
		A0_fit, Rc0_fit, Rc1_fit = fit_pairs_arr[qq][ tt ][:]

		tt_Am = (1 - crit_f) * KPA_func( 0, A0_fit, Rc0_fit, Rc1_fit )

		tt_R = inversefunc( KPA_func, args = ( A0_fit, Rc0_fit, Rc1_fit ), y_values = tt_Am, domain = [0, 100],)
		mft_Rt.append( tt_R )

	mft_Rt = np.array( mft_Rt )

	mft_Rt_arr.append( mft_Rt )



##. figs ratio profile
fig = plt.figure( figsize = (10.4, 4.8) )
ax0 = fig.add_axes([0.10, 0.11, 0.42, 0.80])
ax1 = fig.add_axes([0.52, 0.11, 0.42, 0.80])

axgs = [ ax0, ax1 ]

for qq in range( 2 ):

	cp_k_r, cp_k_sb = nbg_R[qq][-1], nbg_SB[qq][-1]

	#.
	for tt in range( len(R_bins) - 2 ):

		##.
		kk_r = nbg_R[qq][ tt ]
		kk_sb = nbg_SB[qq][ tt ]
		kk_err = nbg_err[qq][ tt ]

		##.
		tmp_F = interp.splrep( cp_k_r, cp_k_sb, s = 0)
		tmp_eta = kk_sb / interp.splev( kk_r, tmp_F, der = 0)
		tmp_eta_err = kk_err / interp.splev( kk_r, tmp_F, der = 0)

		##.
		new_r = np.logspace( -1, 2, 200 )

		# A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit = fit_pairs_arr[qq][ tt ][:]
		# mf0 = Moffat_func( new_r, A0_fit, Rc0_fit, alpha_fit )

		A0_fit, Rc0_fit, Rc1_fit = fit_pairs_arr[qq][ tt ][:]
		mf0 = KPA_func( new_r, A0_fit, Rc0_fit, Rc1_fit )

		##.
		ax = axgs[ qq ]

		ax.plot( kk_r, tmp_eta, ls = '-', color = color_s[tt], alpha = 0.75, label = fig_name[tt],)
		ax.fill_between( kk_r, y1 = tmp_eta - tmp_eta_err, y2 = tmp_eta + tmp_eta_err, color = color_s[tt], alpha = 0.12,)

		ax.plot( new_r, mf0, ls = '--', color = color_s[tt], alpha = 0.5, lw = 3,)

		ax.annotate( s = samp_name[qq], xy = (0.45, 0.90), xycoords = 'axes fraction', fontsize = 12,)

		if qq == 0:
			legend_2 = ax.legend( labels = ['Observed', 'Model'], 
						loc = 2, frameon = False, fontsize = 12,)

			ax.legend( loc = 3, frameon = False, fontsize = 12,)
			ax.add_artist( legend_2 )

#.
ax = axgs[0]
ax.set_ylim( 0.35, 0.95 )

ax.set_xlim( 1e0, 1e2 )
ax.set_xscale('log')
ax.set_xlabel('$R \; [kpc]$', fontsize = 12)

ax.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
					fontsize = 13, labelpad = 8)

ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax = axgs[1]
ax.set_ylim( axgs[0].get_ylim() )
ax.set_yticklabels( labels = [] )

ax.set_xlim( 1e0, 1e2 )
ax.set_xscale('log')
ax.set_xlabel('$R \; [kpc]$', fontsize = 12)

ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

plt.savefig('/home/xkchen/Sat_%s-band_SB_ratio_model.png' % band_str, dpi = 300)
plt.close()

# raise


##...
marks = [ 's', '>', 'o' ]
mark_size = [ 10, 25, 35 ]

fig = plt.figure( )
ax0 = fig.add_axes([0.14, 0.12, 0.85, 0.80])

# for tt in range( len(R_bins) - 2 ):

# 	if tt == 0:

# 		l1, = ax0.plot( aveg_Rsat[0][:-1], mft_Rt_arr[0][:,3], marker = marks[0], ls = '--', color = 'b', 
# 			alpha = 0.75, label = '%d%% Reduction' % (crit_f[3] * 100),)

# 		ax0.plot( aveg_Rsat[0][:-1], mft_Rt_arr[0][:,4], marker = marks[0], ls = '--', color = 'g', 
# 			alpha = 0.75, label = '%d%% Reduction' % (crit_f[4] * 100),)

# 		ax0.plot( aveg_Rsat[0][:-1], mft_Rt_arr[0][:,5], marker = marks[0], ls = '--', color = 'r', 
# 			alpha = 0.75, label = '%d%% Reduction' % (crit_f[5] * 100),)

# 	else:
# 		l1, = ax0.plot( aveg_Rsat[0][:-1], mft_Rt_arr[0][:,3], marker = marks[0], ls = '--', color = 'b', alpha = 0.75,)
# 		ax0.plot( aveg_Rsat[0][:-1], mft_Rt_arr[0][:,4], marker = marks[0], ls = '--', color = 'g', alpha = 0.75,)
# 		ax0.plot( aveg_Rsat[0][:-1], mft_Rt_arr[0][:,5], marker = marks[0], ls = '--', color = 'r', alpha = 0.75,)

# 	l2, = ax0.plot( aveg_Rsat[1][:-1], mft_Rt_arr[1][:,3], marker = marks[2], ls = '-', color = 'b', alpha = 0.75)
# 	ax0.plot( aveg_Rsat[1][:-1], mft_Rt_arr[1][:,4], marker = marks[2], ls = '-', color = 'g', alpha = 0.75)
# 	ax0.plot( aveg_Rsat[1][:-1], mft_Rt_arr[1][:,5], marker = marks[2], ls = '-', color = 'r', alpha = 0.75)

l1, = ax0.plot( aveg_Rsat[0][:-1], fit_pairs_arr[0][:,2], ls = '--', color = 'k', marker = 's',)
l2, = ax0.plot( aveg_Rsat[1][:-1], fit_pairs_arr[1][:,2], ls = '-', color = 'k', marker = 's',)

#.
legend_0 = plt.legend( handles = [l1, l2], labels = [samp_name[0], samp_name[1] ],
			loc = 2, frameon = False, fontsize = 12, markerfirst = False,)

ax0.legend( loc = 4, frameon = False, fontsize = 12, markerfirst = False,)
ax0.add_artist( legend_0 )

# ax0.set_ylim( 1e1, 2e3 )
ax0.set_ylabel('$R_{t} \; of \; F_{KP} \; [kpc]$', fontsize = 12,)
# ax0.set_yscale('log')

ax0.set_xlim( 0., 0.55 )
ax0.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$', fontsize = 12,)
ax0.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/BCG_Mstar_bin_%s-band_Moffat_Rt_compare.png' % band_str, dpi = 300)
plt.close()

