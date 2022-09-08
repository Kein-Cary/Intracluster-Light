import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

import scipy.interpolate as interp
from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
#.
from light_measure import light_measure_weit
from img_sat_fig_out_mode import arr_jack_func
from img_sat_BG_sub_SB import aveg_BG_sub_func, stack_BG_sub_func


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


### === data load
R_str = 'scale'
R_bins = np.array( [0, 0.24, 0.40, 0.56, 1] )   ### times R200m


BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/BGs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/noBG_SBs/'

cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
cp_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'


band_str = 'r'


##. over all field galaxy
dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%s-band_aveg-jack_BG-sub_SB.csv' % band_str )

all_nbg_R = np.array( dat['r'] )
all_nbg_SB = np.array( dat['sb'] )
all_nbg_err = np.array( dat['sb_err'] )


##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( out_path + 'ctrl-galx_Extend_BCGM_gri-common_cat-mapped_%.2f-%.2fR200m_%s-band' % 
						(R_bins[tt], R_bins[tt + 1], band_str) + '_aveg-jack_BG-sub_SB.csv',)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	nbg_R.append( tt_r )
	nbg_SB.append( tt_sb )
	nbg_err.append( tt_sb_err )

##.
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

	dat = pds.read_csv( cp_out_path + 'Extend_BCGM_gri-common_all_%.2f-%.2fR200m' % (R_bins[tt], R_bins[tt + 1])
									+ '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

	cp_nbg_R.append( tt_r )
	cp_nbg_SB.append( tt_sb )
	cp_nbg_err.append( tt_sb_err )


##.
color_s = ['b', 'g', 'c', 'r', 'm']
line_s = [ ':', '--', '-' ]

fig_name = []
for dd in range( len(R_bins) - 1 ):

	if dd == 0:
		fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

	elif dd == len(R_bins) - 2:
		fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

	else:
		fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


##. compare to individual mapped control sample
fig = plt.figure( )
ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

for mm in range( len(R_bins) - 1 ):

	l2 = ax1.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '.', ls = '--', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75,)

	l3 = ax1.errorbar( cp_nbg_R[mm], cp_nbg_SB[mm], yerr = cp_nbg_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

	##.
	_kk_tmp_F = interp.interp1d( nbg_R[mm], nbg_SB[mm], kind = 'cubic', fill_value = 'extrapolate',)

	sub_ax1.plot( cp_nbg_R[mm], cp_nbg_SB[mm] / _kk_tmp_F( cp_nbg_R[mm] ), ls = '--', color = color_s[mm], alpha = 0.75,)

	sub_ax1.fill_between( cp_nbg_R[mm], y1 = (cp_nbg_SB[mm] - cp_nbg_err[mm]) / _kk_tmp_F( cp_nbg_R[mm]), 
			y2 = (cp_nbg_SB[mm] + cp_nbg_err[mm]) / _kk_tmp_F( cp_nbg_R[mm]), color = color_s[mm], alpha = 0.12,)

legend_2 = ax1.legend( handles = [ l2, l3], labels = [ 'Control, subsample-mapped', 'RedMaPPer Member' ], 
			loc = 1, frameon = False, fontsize = 12,)

ax1.legend( loc = 3, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlim( 2e0, 5e1 )
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.annotate( text = '%s-band' % band_str, xy = (0.03, 0.45), xycoords = 'axes fraction', fontsize = 12,)

ax1.set_ylim( 2e-3, 5e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

sub_ax1.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control}$', labelpad = 8, fontsize = 12,)
sub_ax1.set_ylim( 0.6, 1.3 )

sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/contrl-galx_%s-band_BG-sub_SB_compare.png' % band_str, dpi = 300)
plt.close()


raise

##. compare to over all galaxies of control sample
fig = plt.figure( )
ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

l1 = ax1.errorbar( all_nbg_R, all_nbg_SB, yerr = all_nbg_err, marker = '.', ls = '--', color = 'k', 
		ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75,)

for mm in range( len(R_bins) - 1 ):

	l3 = ax1.errorbar( cp_nbg_R[mm], cp_nbg_SB[mm], yerr = cp_nbg_err[mm], marker = '.', ls = '-', color = color_s[mm],
		ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

	##.
	_kk_tmp_F = interp.interp1d( all_nbg_R, all_nbg_SB, kind = 'cubic', fill_value = 'extrapolate',)

	sub_ax1.plot( cp_nbg_R[mm], cp_nbg_SB[mm] / _kk_tmp_F( cp_nbg_R[mm] ), ls = '--', color = color_s[mm], alpha = 0.75,)

	sub_ax1.fill_between( cp_nbg_R[mm], y1 = (cp_nbg_SB[mm] - cp_nbg_err[mm]) / _kk_tmp_F( cp_nbg_R[mm]), 
			y2 = (cp_nbg_SB[mm] + cp_nbg_err[mm]) / _kk_tmp_F( cp_nbg_R[mm]), color = color_s[mm], alpha = 0.12,)

legend_2 = ax1.legend( handles = [l1, l3], labels = ['Control, All galaxies', 'RedMaPPer Member' ], 
			loc = 1, frameon = False, fontsize = 12,)

ax1.legend( loc = 3, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlim( 2e0, 5e1 )
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.annotate( text = '%s-band' % band_str, xy = (0.03, 0.45), xycoords = 'axes fraction', fontsize = 12,)

ax1.set_ylim( 2e-3, 5e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.set_yscale('log')

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xscale('log')
sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

# sub_ax1.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control}$', labelpad = 8, fontsize = 12,)
# sub_ax1.set_ylim( 0.6, 1.3 )

sub_ax1.set_ylabel('$\\mu_{Member} \; / \; \\mu_{Control,All}$', labelpad = 8, fontsize = 12,)
sub_ax1.set_ylim( 0.65, 2.1 )

sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/contrl-galx_%s-band_BG-sub_SB_compare.png' % band_str, dpi = 300)
plt.close()

