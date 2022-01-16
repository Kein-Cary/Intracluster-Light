import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage
import scipy.signal as signal

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov
from light_measure import light_measure_weit

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

### === ###
def sersic_func(r, Ie, re, ndex):

	belta = 2 * ndex - 0.324
	fn = -1 * belta * ( r / re )**(1 / ndex) + belta
	Ir = Ie * np.exp( fn )
	return Ir

### === ### SB measure compare to Z05
rand_path = '/home/xkchen/figs/re_measure_SBs/random_ref_SB/'

color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]

BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'
path = '/home/xkchen/figs/re_measure_SBs/SBs/'

com_path = '/home/xkchen/figs/re_measure_SBs/SBs_Z05_err/'

"""
## .. measure SB profile error with Z05 method
for kk in range( 3 ):

	with h5py.File(path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_img_z-ref.h5' % band[kk], 'r') as f:
		tmp_img = np.array( f['a'] )

	cen_x, cen_y = np.int( tmp_img.shape[1] / 2 ), np.int( tmp_img.shape[0] / 2 )

	#. pixel counts array
	with h5py.File(path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_pix-cont_z-ref.h5' % band[kk], 'r') as f:
		tmp_count = np.array( f['a'] )

	#. calculate the BG-sub images
	with h5py.File( rand_path + 'random_%s-band_rand-stack_Mean_jack_img_z-ref-aveg.h5' % band[kk], 'r') as f:
		rand_img = np.array( f['a'])

	xn, yn = np.int( rand_img.shape[1] / 2 ), np.int( rand_img.shape[0] / 2 )	

	## BG-estimate params
	BG_file = BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-profile_params_diag-fit.csv' % band[kk]

	cat = pds.read_csv( BG_file )
	offD, I_e, R_e = np.array(cat['offD'])[0], np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]

	#. surface brightness at 2Mpc
	sb_2Mpc = sersic_func( 2e3, I_e, R_e, 2.1)

	#. the BG fitting relation is in unit : flux / arcsec^2
	shift_rand_img = rand_img / pixel**2 - offD + sb_2Mpc
	BG_sub_img = tmp_img / pixel**2 - shift_rand_img

	#. change the BG-sub image pixel unit to flux
	calib_img = BG_sub_img * pixel**2


	#. Measure error in Z05
	R_bins = np.logspace(0, 3.3, 55)

	Intns, phy_r, Intns_err, npix, nratio = light_measure_weit( calib_img, tmp_count, pixel, cen_x, cen_y, z_ref, R_bins )

	id_nul = npix < 1.

	R_obs, mu_obs, mu_err = phy_r[ id_nul == False ], Intns[ id_nul == False ], Intns_err[ id_nul == False ]

	keys = ['r', 'sb', 'sb_err']
	values = [ R_obs, mu_obs, mu_err ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( com_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub-SB_with-Z05-err.csv' % band[ kk ] )

"""

### === ### comparison
tt_jk_R, tt_jk_SB, tt_jk_SB_err = [], [], []

for kk in range( 3 ):

	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_jk_R.append( tt_r )
	tt_jk_SB.append( tt_sb )
	tt_jk_SB_err.append( tt_err )


tt_Z5_R, tt_Z5_SB, tt_Z5_SB_err = [], [], []

for kk in range( 3 ):

	dat = pds.read_csv(com_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub-SB_with-Z05-err.csv' % band[ kk ])
	tt_r = np.array( dat['r'])
	tt_sb = np.array( dat['sb'])
	tt_err = np.array( dat['sb_err'])

	tt_Z5_R.append( tt_r )
	tt_Z5_SB.append( tt_sb )
	tt_Z5_SB_err.append( tt_err )


fig = plt.figure( figsize = (5.8, 5.4) )
ax = fig.add_axes( [0.13, 0.53, 0.83, 0.42] )
sub_ax = fig.add_axes( [0.13, 0.11, 0.83, 0.42] )

for kk in range( 3 ):

	ax.plot( tt_jk_R[kk], tt_jk_SB[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk],)
	ax.fill_between( tt_jk_R[kk], y1 = tt_jk_SB[kk] - tt_jk_SB_err[kk], y2 = tt_jk_SB[kk] + tt_jk_SB_err[kk], 
					color = color_s[kk], alpha = 0.12,)

	ax.plot( tt_Z5_R[kk], tt_Z5_SB[kk], ls = '--', color = color_s[kk], alpha = 0.75,)
	ax.fill_between( tt_Z5_R[kk], y1 = tt_Z5_SB[kk] - tt_Z5_SB_err[kk], y2 = tt_Z5_SB[kk] + tt_Z5_SB_err[kk], 
					color = color_s[kk], alpha = 0.12,)

	sub_ax.plot( tt_jk_R[kk], tt_jk_SB_err[kk], ls = '-', color = color_s[kk], alpha = 0.75)
	sub_ax.plot( tt_Z5_R[kk], tt_Z5_SB_err[kk], ls = '--', color = color_s[kk], alpha = 0.75)


legend_1 = ax.legend( [ 'Jackknife err', 'Z05 err'], loc = 1, frameon = False, fontsize = 12, markerfirst = False,)
legend_0 = ax.legend( loc = 3, frameon = False, fontsize = 12,)
ax.add_artist( legend_1 )

ax.set_xlim( 1e0, 1e3 )
ax.set_xscale('log')

ax.set_ylim( 1e-4, 2e1 )
ax.set_yscale('log')
ax.set_ylabel('$\\mu \; [nanomaggies \; arcsec^{-2}]$', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xlabel( '$R \; [\\mathrm{k}pc] $', fontsize = 12,)
sub_ax.set_xscale( 'log' )
sub_ax.set_yscale( 'log' )
sub_ax.set_ylabel('$\\sigma_{\\mu}$', fontsize = 12)

sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/total-sample_SB-err_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
ax = fig.add_axes( [0.13, 0.32, 0.83, 0.63] )
sub_ax = fig.add_axes( [0.13, 0.11, 0.83, 0.21] )

for kk in range( 3 ):

	ax.plot( tt_jk_R[kk], tt_jk_SB_err[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk],)
	ax.plot( tt_Z5_R[kk], tt_Z5_SB_err[kk], ls = '--', color = color_s[kk], alpha = 0.75, )

	_kk_intep_F = interp.interp1d( tt_Z5_R[kk], tt_Z5_SB_err[kk], kind = 'linear', fill_value = 'extrapolate',)

	sub_ax.plot( tt_jk_R[kk], tt_jk_SB_err[kk] / _kk_intep_F( tt_jk_R[kk] ), ls = '-', color = color_s[kk], alpha = 0.75,)

legend_1 = ax.legend( [ 'Jackknife err', 'Z05 err'], loc = 1, frameon = False, fontsize = 12, markerfirst = False,)
legend_0 = ax.legend( loc = 3, frameon = False, fontsize = 12, )
ax.add_artist( legend_1 )

ax.set_xlim( 1e0, 1e3 )
ax.set_xscale('log')

ax.set_ylim( 1e-5, 1e0 )
ax.set_yscale('log')
ax.set_ylabel('$\\sigma_{\\mu} \; [nanomaggies \; arcsec^{-2}]$', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xlabel( '$R \; [\\mathrm{k}pc] $', fontsize = 12,)
sub_ax.set_xscale( 'log' )
sub_ax.set_ylabel('$\\sigma_{\\mu}^{jk} \, / \, \\sigma_{\\mu}^{Z05}$', fontsize = 12)
sub_ax.axhline( y = 2, ls = ':', color = 'k',)
sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/total-sample_g2r_compare_c1.png', dpi = 300)
plt.close()


