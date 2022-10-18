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

import scipy.signal as signal
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp

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
a_ref = 1 / (z_ref + 1)

band = ['r', 'g', 'i']


### === data load
##. img_tract with BCG
cc_BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_rebin_bcg_affect_test/BGs/'
cc_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_rebin_bcg_affect_test/noBG_SBs/'
cc_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_rebin_bcg_affect_test/SBs/'

##. img_tract without BCG
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_rebin_bcg_affect_test/nobcg_SBs/'
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_rebin_bcg_affect_test/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_rebin_bcg_affect_test/nobcg_BGsub_SBs/'

cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_rebin_bcg_affect_test/cat/'


### === results comparison
##. shuffle list
list_order = 13

ll = 0    ##. ll = 0, 1, 2

##. rich (30, 50)
if ll == 1:
    R_bins = np.array( [ 0, 300, 400, 550, 5000] )

##. rich < 30
if ll == 0:
    R_bins = np.array([0, 150, 300, 500, 2000])

##. rich > 50
if ll == 2:
    R_bins = np.array([0, 400, 600, 750, 2000])

# color_s = []
# for dd in range( 4 ):

#     color_s.append( mpl.cm.rainbow( dd / 3 ) )
color_s = ['b', 'g', 'm', 'r']


fig_name = []
for dd in range( len(R_bins) - 1 ):

    if dd == 0:
        fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

    elif dd == len(R_bins) - 2:
        fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

    else:
        fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)


bin_rich = [ 20, 30, 50, 210 ]

sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


tmp_R, tmp_sb, tmp_err = [], [], []

##... sat SBs
for tt in range( len(R_bins) - 1 ):

    sub_R, sub_sb, sub_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        with h5py.File( path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
                        '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_err )

    tmp_R.append( sub_R )
    tmp_sb.append( sub_sb )
    tmp_err.append( sub_err )

##... BG SBs
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    _sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
                '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        _sub_bg_R.append( tt_r )
        _sub_bg_sb.append( tt_sb )
        _sub_bg_err.append( tt_err )

    tmp_bg_R.append( _sub_bg_R )
    tmp_bg_SB.append( _sub_bg_sb )
    tmp_bg_err.append( _sub_bg_err )


##... BG-subtracted SB profiles
nbg_R, nbg_SB, nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    sub_R, sub_sb, sub_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1]) + 
                            '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

        tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_sb_err )

    nbg_R.append( sub_R )
    nbg_SB.append( sub_sb )
    nbg_err.append( sub_err )


##. extracted image with BCG
cp_R, cp_sb, cp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    sub_R, sub_sb, sub_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        with h5py.File( cc_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
                        '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_err )

    cp_R.append( sub_R )
    cp_sb.append( sub_sb )
    cp_err.append( sub_err )


cp_bg_R, cp_bg_SB, cp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    _sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        with h5py.File( cc_BG_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
                '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        _sub_bg_R.append( tt_r )
        _sub_bg_sb.append( tt_sb )
        _sub_bg_err.append( tt_err )

    cp_bg_R.append( _sub_bg_R )
    cp_bg_SB.append( _sub_bg_sb )
    cp_bg_err.append( _sub_bg_err )


cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    sub_R, sub_sb, sub_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        dat = pds.read_csv( cc_out_path 
                    + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1]) 
                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

        tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_sb_err )

    cp_nbg_R.append( sub_R )
    cp_nbg_SB.append( sub_sb )
    cp_nbg_err.append( sub_err )


##. figs
y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [1e-3, 1e0], [5e-3, 6e0] ]
y_lim_2 = [ [0.2, 1.4], [0.2, 1.6], [0.2, 1.8] ]

##. BG-sub SBs
for kk in range( 3 ):

    fig = plt.figure( )
    ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
    sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

    for mm in range( len(R_bins) - 1 ):

        l1 = ax1.errorbar( cp_nbg_R[mm][kk], cp_nbg_SB[mm][kk], yerr = cp_nbg_err[mm][kk], marker = '.', ls = '-', color = color_s[mm],
            ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5,)

        _kk_tmp_F = interp.interp1d( cp_nbg_R[mm][kk], cp_nbg_SB[mm][kk], kind = 'cubic', fill_value = 'extrapolate',)


        l2 = ax1.errorbar( nbg_R[mm][kk], nbg_SB[mm][kk], yerr = nbg_err[mm][kk], marker = '', ls = '--', color = color_s[mm], 
            ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

        sub_ax1.plot( nbg_R[mm][kk], ( nbg_SB[mm][kk] - _kk_tmp_F( nbg_R[mm][kk] ) ) / _kk_tmp_F( nbg_R[mm][kk] ), 
                    ls = '--', color = color_s[mm], alpha = 0.75,)
        # sub_ax1.fill_between( nbg_R[mm][kk], y1 = (nbg_SB[mm][kk] - nbg_err[mm][kk]) / _kk_tmp_F( nbg_R[mm][kk] ), 
        #             y2 = (nbg_SB[mm][kk] + nbg_err[mm][kk]) / _kk_tmp_F( nbg_R[mm][kk] ), color = color_s[mm], alpha = 0.12,)

    ax1.annotate( s = line_name[ll] + ', %s-band' % band[kk], xy = (0.55, 0.10), xycoords = 'axes fraction',)

    legend_2 = ax1.legend( handles = [l1, l2], 
                labels = ['image cut with BCG', 'image cut without BCG' ], loc = 1, frameon = False,)

    ax1.legend( loc = 3, frameon = False,)
    ax1.add_artist( legend_2 )

    ax1.set_xlim( 2e0, 4e1 )
    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    ax1.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    sub_ax1.set_xlim( ax1.get_xlim() )
    sub_ax1.set_xscale('log')
    sub_ax1.set_xlabel('$R \; [kpc]$')

    sub_ax1.set_ylabel('$ ( \\mu \; - \; \\mu_{ with \, BCG} ) \, / \, \\mu_{ with \, BCG} $', labelpad = 8)
    sub_ax1.axhline( y = 1, ls = '-', color = 'k', lw = 0.8, alpha = 0.25,)
    sub_ax1.set_ylim( -0.1, 0.1 )
    sub_ax1.axhline( y = 0, ls = '-.', color = 'k', alpha = 0.25,)

    sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.set_xticklabels( labels = [] )

    plt.savefig('/home/xkchen/%s_sat_%s-band_BG-sub-SB_affect_of_BCG.png' % (sub_name[ll], band[kk]), dpi = 300)
    plt.close()



##. BG SBs

for kk in range( 3 ):

    fig = plt.figure( )
    ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
    sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

    for mm in range( len(R_bins) - 1 ):

        l1 = ax1.errorbar( cp_bg_R[mm][kk], cp_bg_SB[mm][kk], yerr = cp_bg_err[mm][kk], marker = '.', ls = '-', color = color_s[mm],
            ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5,)

        _kk_tmp_F = interp.interp1d( cp_bg_R[mm][kk], cp_bg_SB[mm][kk], kind = 'cubic', fill_value = 'extrapolate',)


        l2 = ax1.errorbar( tmp_bg_R[mm][kk], tmp_bg_SB[mm][kk], yerr = tmp_bg_err[mm][kk], marker = '', ls = '--', color = color_s[mm], 
            ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5, label = fig_name[mm],)

        sub_ax1.plot( tmp_bg_R[mm][kk], (tmp_bg_SB[mm][kk] - _kk_tmp_F( tmp_bg_R[mm][kk] ) ) / _kk_tmp_F( tmp_bg_R[mm][kk] ), 
                    ls = '--', color = color_s[mm], alpha = 0.5,)

        # sub_ax1.fill_between( tmp_bg_R[mm][kk], y1 = (tmp_bg_SB[mm][kk] - tmp_bg_err[mm][kk]) / _kk_tmp_F( tmp_bg_R[mm][kk] ), 
        #             y2 = (tmp_bg_SB[mm][kk] + tmp_bg_err[mm][kk]) / _kk_tmp_F( tmp_bg_R[mm][kk] ), color = color_s[mm], alpha = 0.12,)

    ax1.annotate( s = line_name[ll] + ', %s-band' % band[kk], xy = (0.35, 0.10), xycoords = 'axes fraction',)

    legend_2 = ax1.legend( handles = [l1, l2], 
                labels = ['image cut with BCG', 'image cut without BCG' ], loc = 1, frameon = False,)

    ax1.legend( loc = 4, frameon = False,)
    ax1.add_artist( legend_2 )

    ax1.set_xlim( 2e0, 4e1 )
    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')
    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')

    sub_ax1.set_xlim( ax1.get_xlim() )
    sub_ax1.set_xscale('log')
    sub_ax1.set_xlabel('$R \; [kpc]$')

    sub_ax1.set_ylabel('$ ( \\mu \; - \; \\mu_{ with \, BCG} ) \, / \, \\mu_{ with \, BCG} $', labelpad = 8)
    sub_ax1.axhline( y = 1, ls = '-', color = 'k', lw = 0.8, alpha = 0.25,)
    sub_ax1.set_ylim( -1, 1 )
    sub_ax1.axhline( y = 0, ls = '-.', color = 'k', alpha = 0.25,)

    sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.set_xticklabels( labels = [] )

    plt.savefig('/home/xkchen/%s_sat_%s-band_BG-SB_affect_of_BCG.png' % (sub_name[ll], band[kk]), dpi = 300)
    plt.close()

raise

##. pre-BGsub SBs

for kk in range( 3 ):

    fig = plt.figure( )
    ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
    sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

    for mm in range( len(R_bins) - 1 ):

        l1 = ax1.errorbar( cp_R[mm][kk], cp_sb[mm][kk], yerr = cp_err[mm][kk], marker = '.', ls = '-', color = color_s[mm],
            ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5,)

        _kk_tmp_F = interp.interp1d( cp_R[mm][kk], cp_sb[mm][kk], kind = 'cubic', fill_value = 'extrapolate',)


        l2 = ax1.errorbar( tmp_R[mm][kk], tmp_sb[mm][kk], yerr = tmp_err[mm][kk], marker = '', ls = '--', color = color_s[mm], 
            ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.5, label = fig_name[mm],)

        sub_ax1.plot( tmp_R[mm][kk], tmp_sb[mm][kk] / _kk_tmp_F( tmp_R[mm][kk] ), ls = '--', color = color_s[mm], alpha = 0.5,)
        sub_ax1.fill_between( tmp_R[mm][kk], y1 = (tmp_sb[mm][kk] - tmp_err[mm][kk]) / _kk_tmp_F( tmp_R[mm][kk] ), 
                    y2 = (tmp_sb[mm][kk] + tmp_err[mm][kk]) / _kk_tmp_F( tmp_R[mm][kk] ), color = color_s[mm], alpha = 0.12,)

    ax1.annotate( s = line_name[ll] + ', %s-band' % band[kk], xy = (0.15, 0.10), xycoords = 'axes fraction',)

    legend_2 = ax1.legend( handles = [l1, l2], 
                labels = ['image cut with BCG', 'image cut without BCG' ], loc = 1, frameon = False,)

    ax1.legend( loc = 5, frameon = False,)
    ax1.add_artist( legend_2 )

    # ax1.set_xlim( 2e0, 4e1 )
    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    # ax1.set_ylim( 1e-3, 1e-1 )
    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    sub_ax1.set_xlim( ax1.get_xlim() )
    sub_ax1.set_xscale('log')
    sub_ax1.set_xlabel('$R \; [kpc]$')

    sub_ax1.set_ylabel('$\\mu \; / \; \\mu_{ with \, BCG}$', labelpad = 8)
    sub_ax1.axhline( y = 1, ls = '-', color = 'k',)
    sub_ax1.set_ylim( 0.9, 1.1 )

    sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.set_xticklabels( labels = [] )

    plt.savefig('/home/xkchen/%s_sat_%s-band_pre-BGsub-SB_affect_of_BCG.png' % (sub_name[ll], band[kk]), dpi = 300)
    plt.close()

