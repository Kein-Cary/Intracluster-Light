import sys
sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')

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

from astropy import modeling as Model
from astropy.table import Table, QTable
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy import integrate as integ
from astropy.coordinates import SkyCoord
from pynverse import inversefunc
from scipy import optimize
import scipy.signal as signal
from scipy import special

#.
from getFrames_py3 import getHandyFrames, get_cax


##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25
a_ref = 1 / (1 + z_ref)

band = ['r', 'g', 'i']


### === funcs
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

##.
def power1_func( R, A0, R0, alpha, belta ):
	# mf = A0 * ( R / R0 + 1)**(-alpha) * ( R / R0)**(-belta)
	# mf = A0 * ( R / R0 + 1)**(-alpha) * ( R / R0)**(alpha + belta)
	mf = A0 / ( (R / R0)**alpha + (R / R0)**belta )
	return np.log10( mf )


### === ### data load
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'

bin_rich = [ 20, 30, 50, 210 ]

R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m

N_sample = 100

##. background shuffle list order
list_order = 13

#.
band_str = 'r'

##.
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1]) 
								+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )

##. inner SB profile models
fit_pairs = []

for tt in range( len(R_bins) - 2 ):

	##. fitting including SB profile at radii beyond 30 kpc
	# dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
	# 			'sat_%.2f-%.2fR200m_%s-band_SB_trunF-model_fit.csv' % (R_bins[tt], R_bins[tt + 1], band_str),)

	# A0_fit = np.array( dat['A_Moffat'] )[0]
	# Rc0_fit = np.array( dat['Rd_Moffat'] )[0]
	# alpha_fit = np.array( dat['n_Moffat'] )[0]

	# Rc1_fit = np.array( dat['Rc'] )[0]
	# beta_fit = np.array( dat['beta'] )[0]

	# popt = [ A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit ]


	##. inner Moffat fitting
	# dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
	# 				'sat_%.2f-%.2fR200m_%s-band_SB_inner-Moffat_fit.csv' % (R_bins[tt], R_bins[tt + 1], band_str),)

	# A0_fit = np.array( dat['A_Moffat'] )[0]
	# Rc0_fit = np.array( dat['Rd_Moffat'] )[0]
	# alpha_fit = np.array( dat['n_Moffat'] )[0]

	# popt = [ A0_fit, Rc0_fit, alpha_fit ]


	# ##. inner Double powerlaw fitting
	# dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
	# 			'sat_%.2f-%.2fR200m_%s-band_SB_inner-2Power_fit.csv' % (R_bins[tt], R_bins[tt + 1], band_str),)

	# A0_fit = np.array( dat['A_power'] )[0]
	# Rc0_fit = np.array( dat['Rc_power'] )[0]
	# alpha_fit = np.array( dat['alpha'] )[0]
	# belta_fit = np.array( dat['belta'] )[0]

	# popt = [ A0_fit, Rc0_fit, alpha_fit, belta_fit ]


	##. inner Double powerlaw fitting
	dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/trunF_mod/' + 
				'sat_%.2f-%.2fR200m_%s-band_SB_inner-KPA_fit.csv' % (R_bins[tt], R_bins[tt + 1], band_str),)

	A0_fit = np.array( dat['A_power'] )[0]
	Rc0_fit = np.array( dat['R_core'] )[0]
	Rc1_fit = np.array( dat['R_tidal'] )[0]

	popt = [ A0_fit, Rc0_fit, Rc1_fit ]

	#.
	fit_pairs.append( popt )


##. Rt infered from Moffat func.
mft_Rt = []

# crit_f = np.array( [ 0.15, 0.30, 0.45, 0.60, 0.75, 0.90 ] )
# crit_f = np.array( [ 0.20, 0.30, 0.40, 0.50, 0.60, 0.70 ] )
crit_f = np.array( [ 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95 ] )

#.
for tt in range( len(R_bins) - 2 ):

	##.
	# #A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit = fit_pairs[ tt ][:]
	# A0_fit, Rc0_fit, alpha_fit = fit_pairs[ tt ][:]

	# tt_Am = (1 - crit_f) * Moffat_func( 0, A0_fit, Rc0_fit, alpha_fit )
	# tt_R = inversefunc( Moffat_func, args = (A0_fit, Rc0_fit, alpha_fit), y_values = tt_Am )

	##.
	# A0_fit, Rc0_fit, alpha_fit, belta_fit = fit_pairs[ tt ][:]
	# tp_r = np.logspace( -1, 2.5, 1000 )
	# tp_F = power1_func( tp_r, A0_fit, Rc0_fit, alpha_fit, belta_fit )

	# id_vm = tp_r >= 2.
	# us_R = tp_r[ id_vm ]
	# us_F = tp_F[ id_vm ]

	# _tk_F = interp.interp1d( us_R, us_F, kind = 'cubic', fill_value = 'extrapolate',)

	# tt_Am = (1 - crit_f) * us_F[0]
	# tt_R = inversefunc( power1_func, args = (A0_fit, Rc0_fit, alpha_fit, belta_fit), y_values = tt_Am, 
	# 					domain = [2, 500],)

	##.
	A0_fit, Rc0_fit, Rc1_fit = fit_pairs[tt][:]
	tt_Am = (1 - crit_f) * KPA_func( 0, A0_fit, Rc0_fit, Rc1_fit )
	tt_R = inversefunc( KPA_func, args = (A0_fit, Rc0_fit, Rc1_fit), y_values = tt_Am, domain = [0, 100],)

	mft_Rt.append( tt_R )

mft_Rt = np.array( mft_Rt )


##. read sample catalog and get the average centric distance
##. Fang+2016 prediction
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



## === figs. set
# color_s = ['b', 'g', 'r', 'm', 'k']
color_s = ['c', 'b', 'g', 'r', 'k']

#.
fig_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \, / \, R_{200m}\\leq %.2f$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \, / \, R_{200m}\\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$R \, / \, R_{200m}{=}[%.2f, \; %.2f]$' % (R_bins[dd], R_bins[dd + 1]),)

#.
marks = ['s', '>', 'o']
lines_c = ['b', 'g', 'r']
mark_size = [10, 25, 35]


##.
fig = plt.figure( figsize = (5.8, 10.8) )
ax1 = fig.add_axes( [0.15, 0.53, 0.83, 0.42] )
sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.42] )

ax1.errorbar( nbg_R[-1], nbg_SB[-1], yerr = nbg_err[-1], marker = '', ls = '-', color = color_s[-1],
	ecolor = color_s[-1], mfc = 'none', mec = color_s[-1], capsize = 1.5, alpha = 0.75, label = fig_name[-1],)

_kk_tmp_F = interp.interp1d( nbg_R[-1], nbg_SB[-1], kind = 'cubic', fill_value = 'extrapolate',)

for mm in range( 1, len(R_bins) - 2 ):

	#.
	# A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit = fit_pairs[mm][:]
	# A0_fit, Rc0_fit, alpha_fit = fit_pairs[mm][:]
	# A0_fit, Rc0_fit, alpha_fit, belta_fit = fit_pairs[mm][:]
	A0_fit, Rc0_fit, Rc1_fit = fit_pairs[mm][:]

	#.
	new_r = np.logspace( -1, 2.7, 300 )

	_SB0_arr = _kk_tmp_F( new_r )

	# mf0 = Moffat_func( new_r, A0_fit, Rc0_fit, alpha_fit )
	# mf0 = power1_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit )
	mf0 = KPA_func( new_r, A0_fit, Rc0_fit, Rc1_fit )

	#.
	ax1.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '', ls = '-', color = color_s[mm], 
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

	sub_ax1.plot( nbg_R[mm], nbg_SB[mm] / _kk_tmp_F( nbg_R[mm] ), ls = '-', color = color_s[mm], alpha = 0.75,)
	sub_ax1.fill_between( nbg_R[mm], y1 = (nbg_SB[mm] - nbg_err[mm]) / _kk_tmp_F( nbg_R[mm] ), 
				y2 = (nbg_SB[mm] + nbg_err[mm]) / _kk_tmp_F( nbg_R[mm] ), color = color_s[mm], alpha = 0.12,)

	if mm == 1:
		# sub_ax1.plot( new_r, mf0, ls = '--', color = color_s[mm], alpha = 0.75, label = 'Moffat')
		# sub_ax1.plot( new_r, mf0, ls = '--', color = color_s[mm], alpha = 0.75, label = 'Double Powerlaw')
		sub_ax1.plot( new_r, mf0, ls = '--', color = color_s[mm], alpha = 0.75, label = 'King model')

	else:
		sub_ax1.plot( new_r, mf0, ls = '--', color = color_s[mm], alpha = 0.75,)

#.
ax1.annotate( s = '$%s$-band' % band_str, xy = (0.03, 0.35), xycoords = 'axes fraction', fontsize = 13,)
ax1.legend( loc = 3, frameon = False, fontsize = 13,)

ax1.set_xlim( 1.5e0, 5e1 )
ax1.set_xscale('log')

ax1.set_ylim( 2e-3, 5e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 13,)
ax1.set_yscale('log')

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 13,)

sub_ax1.legend( loc = 3, frameon = False, fontsize = 13,)
sub_ax1.set_ylabel('$F \, = \, \\mu \; / \; \\mu \,$ (%s)' % fig_name[-1], labelpad = 8, fontsize = 13,)
sub_ax1.set_ylim( 0.375, 0.85 )

sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)
ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/sat_%s-band_SB_SB_ratio.png' % band_str, dpi = 300)
plt.close()

# raise


##. Inner Moffat compare
fig = plt.figure( figsize = (5.8, 5.4) )
ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.84] )

for mm in range( 1, len(R_bins) - 2 ):

	new_r = np.logspace( -1, 2.7, 300 )

	##.
	# A0_fit, Rc0_fit, alpha_fit, Rc1_fit, beta_fit = fit_pairs[mm][:]
	# A0_fit, Rc0_fit, alpha_fit = fit_pairs[mm][:]

	# mf0 = Moffat_func( new_r, A0_fit, Rc0_fit, alpha_fit )
	# off_y = mf0[0] / 1.

	##.
	# A0_fit, Rc0_fit, alpha_fit, belta_fit = fit_pairs[mm][:]
	# mf0 = power1_func( new_r, A0_fit, Rc0_fit, alpha_fit, belta_fit )
	# tp_r = np.logspace( -1, 2.5, 1000 )
	# tp_F = power1_func( tp_r, A0_fit, Rc0_fit, alpha_fit, belta_fit )

	# id_vm = tp_r >= 2.
	# us_R = tp_r[ id_vm ]
	# us_F = tp_F[ id_vm ]

	# off_y = us_F[0] / 1.

	##.
	A0_fit, Rc0_fit, Rc1_fit = fit_pairs[mm][:]

	mf0 = KPA_func( new_r, A0_fit, Rc0_fit, Rc1_fit )
	off_y = mf0[0] / 1.

	#.
	off_mf0 = mf0 / off_y

	# crit_R = inversefunc( Moffat_func, args = (A0_fit, Rc0_fit, alpha_fit), y_values = 0.25 * off_y )
	# crit_R = inversefunc( power1_func, args = (A0_fit, Rc0_fit, alpha_fit, belta_fit), y_values = 0.40 * off_y, 
	# 				domain = [2, 500],)

	crit_R = inversefunc( KPA_func, args = (A0_fit, Rc0_fit, Rc1_fit), y_values = 0.25 * off_y, domain = [0, 100],)

	ax1.plot( new_r, off_mf0, ls = '-', color = color_s[mm], label = fig_name[mm],)
	ax1.axvline( x = crit_R, ls = ':', color = color_s[mm],)

#.
ax1.set_xlim( 1e0, 2e2 )
ax1.set_xscale('log')
ax1.set_xlabel('$R \; [kpc]$', fontsize = 13,)

ax1.legend( loc = 3, frameon = False, fontsize = 13,)
ax1.set_ylabel('$F \, / \, F(0)$', labelpad = 8, fontsize = 13,)
ax1.set_ylim( 0., 1.01 )

ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 13,)

plt.savefig('/home/xkchen/sat_%s-band_Moffat_compare.png' % band_str, dpi = 300)
plt.close()


##.
fig = plt.figure( figsize = (5.8, 5.4) )
ax1 = fig.add_axes( [0.155, 0.11, 0.83, 0.84] )

# #.
# for oo in range( len( crit_f ) ):

# 	id_set = oo

# 	for tt in range( len(R_bins) - 2 ):

# 		if tt == 0:
# 			l2, = ax1.plot( R_aveg[1:-1], mft_Rt[:,oo][1:], marker = 'o', ls = 'none', 
# 				color = mpl.cm.rainbow(oo/9), alpha = 0.75, 
# 				label = '$F(R_{t}) \, / \, F_{0}{=}%d \\%%$' % (crit_f[ id_set ] * 100))

# 		else:
# 			l2, = ax1.plot( R_aveg[1:-1], mft_Rt[:,oo][1:], marker = 'o', ls = '-', 
# 				color = mpl.cm.rainbow(oo/9), alpha = 0.75,)


# #.
# for tt in range( len(R_bins) - 2 ):

# 	oo = 4

# 	if tt == 0:
# 		ax1.errorbar( R_aveg[1:-1], mft_Rt[:,oo][1:], yerr = mft_Rt[:,oo][1:] * 0.1, marker = 'o', ls = 'none', 
# 					color = 'r', ecolor = 'r', markersize = 15, mfc = 'r', mec = 'r', capsize = 1.5, alpha = 0.75, 
# 				label = '$F(R_{t}) \, / \, F_{0}{=}%d \\%%$' % (crit_f[ oo ] * 100),)

# 	else:
# 		ax1.errorbar( R_aveg[1:-1], mft_Rt[:,oo][1:], yerr = mft_Rt[:,oo][1:] * 0.1, marker = 'o', ls = 'none', 
# 					color = 'r', ecolor = 'r', markersize = 15, mfc = 'r', mec = 'r', capsize = 1.5, alpha = 0.75,)

##.. Rt estimate based on 'Fang+2016 Model'
line_name = []

for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		line_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		line_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		line_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

#.
Rc = []

pat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/theory_Rt/' + 
		'Extend_BCGM_gri-common_Rs-bin_over-rich_sat_Rt.csv',)

for dd in range( len( R_bins ) - 1 ):

	#.
	tt_Rc = np.array( pat[ line_name[dd] ] )[0]   ##. kpc / h

	Rc.append( tt_Rc / h )

# ax1.plot( R_aveg[1:-1], np.array( fit_pairs )[1:,2], ls = '--', color = 'r', marker = 'o', markersize = 12,
# 				label = '$R_{t} \; of \; F_{KP}$')

##. 10kpc limit fitting
yerrs = [ [13.22, 8.08, 11.33, 11.99 ], [29.71, 22.22, 24.38, 15.04] ]
y_valu = [ 52.33, 48.55, 58.63, 76.58 ]

##. 15kpc limit fitting
# yerrs = [ [14.53, 11.85, 10.10, 11.90 ], [26.30, 23.60, 20.45, 12.43] ]
# y_valu = [ 59.30, 56.77, 61.63, 80.78 ]

ax1.errorbar( R_aveg[:-1], y_valu, yerr = ( yerrs[0], yerrs[1] ), ls = '-', color = 'k', marker = 'o', markersize = 12,
				label = 'This work')

ax1.plot( R_aveg[1:], Rc[1:], marker = '', ls = '--', color = 'gray', alpha = 0.75, markersize = 12, 
				label = 'Predicted by $M_{\\rm{sub{-}halo} }^{\\rm{WL}}$',)

ax1.legend( loc = 4, frameon = False, fontsize = 12, markerfirst = False,)

# ax1.set_ylim( 2e1, 2e2 )
ax1.set_ylabel('$ R_{t} \; [kpc]$', fontsize = 12,)
ax1.set_yscale('log')

ax1.set_xlim( 0.15, 0.66 )

ax1.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$', fontsize = 12,)
ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/Infer_Rt_compare.png', dpi = 300)
plt.close()
