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

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
import scipy.interpolate as interp

from light_measure import light_measure_weit
from img_sat_fig_out_mode import arr_jack_func
from img_sat_BG_sub_SB import aveg_BG_sub_func, stack_BG_sub_func



### === ### cosmology model
Test_model = apcy.Planck15.clone( H0 = 67.74, Om0 = 0.311 )
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to( U.arcsec )
pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']


### === data load
##. sample information
bin_rich = [ 20, 30, 50, 210 ]

#. physical distance
R_phy = [ 0, 200, 400 ]

BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_SBs/BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/shufl_SBs/nBG_SBs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/SBs/'

sub_name = ['low-rich', 'medi-rich', 'high-rich']
cat_lis = ['inner', 'middle', 'outer']

N_sample = 100
N_shufl = 20


##. sub-samples
for ll in range( 1,2 ):

    for tt in range( 3 ):

        for kk in range( 3 ):

            band_str = band[ kk ]

            ##. 
            tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

            for dd in range( 12 ):
                
                with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
                                % (sub_name[ll], cat_lis[tt], band_str, dd), 'r' ) as f:

                    tt_r = np.array(f['r'])
                    tt_sb = np.array(f['sb'])
                    tt_err = np.array(f['sb_err'])

                tmp_bg_R.append( tt_r )
                tmp_bg_SB.append( tt_sb )
                tmp_bg_err.append( tt_err )

            ##. save the average
            tmp_bg_R = np.array( tmp_bg_R )
            tmp_bg_SB = np.array( tmp_bg_SB )
            tmp_bg_err = np.array( tmp_bg_err )

            tmp_std = np.std( tmp_bg_SB, axis = 0 )
            tmp_bg_R = np.mean( tmp_bg_R, axis = 0 )
            tmp_bg_SB = np.mean( tmp_bg_SB, axis = 0 )

            with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band_aveg_shufl_SB-pro.h5' % 
                            (sub_name[ll], cat_lis[tt], band_str), 'w') as f:
                f['r'] = np.array( tmp_bg_R )
                f['sb'] = np.array( tmp_bg_SB )
                f['sb_err'] = np.array( tmp_std )

            ##. BG_sub SB pros
            sat_sb_file = path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'

            bg_sb_file = BG_path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band_aveg_shufl_SB-pro.h5' % (sub_name[ll], cat_lis[tt], band_str)

            out_file = out_path + 'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str

            stack_BG_sub_func( sat_sb_file, bg_sb_file, band[ kk ], N_sample, out_file )


##. entire samples
for ll in range( 1,2 ):

    for kk in range( 3 ):

        band_str = band[ kk ]

        ##. 
        tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

        for dd in range( 12 ):

            with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' % 
                            (sub_name[ll], band_str, dd), 'r' ) as f:
                tt_r = np.array( f['r'] )
                tt_sb = np.array( f['sb'] )
                tt_err = np.array( f['sb_err'] )

            tmp_bg_R.append( tt_r )
            tmp_bg_SB.append( tt_sb )
            tmp_bg_err.append( tt_err )



        ##. save the average
        tmp_bg_R = np.array( tmp_bg_R )
        tmp_bg_SB = np.array( tmp_bg_SB )
        tmp_bg_err = np.array( tmp_bg_err )

        tmp_std = np.std( tmp_bg_SB, axis = 0 )
        tmp_bg_R = np.mean( tmp_bg_R, axis = 0 )
        tmp_bg_SB = np.mean( tmp_bg_SB, axis = 0 )

        with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_%s-band_aveg_shufl_SB-pro.h5' % (sub_name[ll], band_str), 'w') as f:
            f['r'] = np.array( tmp_bg_R )
            f['sb'] = np.array( tmp_bg_SB )
            f['sb_err'] = np.array( tmp_std )


        ##. BG_sub SBs
        sat_sb_file = path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band' % band_str + '_jack-sub-%d_SB-pro_z-ref.h5'
        bg_sb_file = BG_path + 'Extend_BCGM_gri-common_%s_%s-band_aveg_shufl_SB-pro.h5' % (sub_name[ll], band_str)
        out_file = out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str

        stack_BG_sub_func( sat_sb_file, bg_sb_file, band[ kk ], N_sample, out_file )        

# raise


### === figs
sub_name = ['low-rich', 'medi-rich', 'high-rich']
cat_lis = ['inner', 'middle', 'outer']


color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'g', 'r']
line_s = [ '--', '-', '-.']

fig_name = ['Inner', 'Middle', 'Outer']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


ll = 1   ##. 0, 1, 2

tmp_R, tmp_sb, tmp_err = [], [], []

##... sat SBs
for tt in range( 3 ):

    sub_R, sub_sb, sub_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        with h5py.File( path + 
            'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_err )

    tmp_R.append( sub_R )
    tmp_sb.append( sub_sb )
    tmp_err.append( sub_err )


tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( 3 ):

    _sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band_aveg_shufl_SB-pro.h5' % (sub_name[ll], cat_lis[tt], band_str), 'r') as f:

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

for tt in range( 3 ):

    sub_R, sub_sb, sub_err = [], [], []

    for kk in range( 3 ):

        band_str = band[ kk ]

        dat = pds.read_csv( out_path + 
            'Extend_BCGM_gri-common_%s_phyR-%s' % (sub_name[ ll ], cat_lis[ tt ]) + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

        tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_sb_err )

    nbg_R.append( sub_R )
    nbg_SB.append( sub_sb )
    nbg_err.append( sub_err )


### ... average of entire subsamples
tot_R, tot_SB, tot_SB_err = [], [], []

for kk in range( 3 ):

    with h5py.File( path + 
        'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band' % band[kk] + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

        tt_r = np.array(f['r'])
        tt_sb = np.array(f['sb'])
        tt_err = np.array(f['sb_err'])

    tot_R.append( tt_r )
    tot_SB.append( tt_sb )
    tot_SB_err.append( tt_err )


tot_bg_R, tot_bg_SB, tot_bg_err = [], [], []

for kk in range( 3 ):

    with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_%s-band_aveg_shufl_SB-pro.h5' % (sub_name[ll], band_str), 'r') as f:

        tt_r = np.array(f['r'])
        tt_sb = np.array(f['sb'])
        tt_err = np.array(f['sb_err'])

    tot_bg_R.append( tt_r )
    tot_bg_SB.append( tt_sb )
    tot_bg_err.append( tt_err )


tot_nbg_R, tot_nbg_SB, tot_nbg_err = [], [], []

for kk in range( 3 ):

    dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ ll ] + '_%s-band_aveg-jack_BG-sub_SB.csv' % band[kk],)
    tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

    tot_nbg_R.append( tt_r )
    tot_nbg_SB.append( tt_sb )
    tot_nbg_err.append( tt_sb_err )


## ... SDSS profMean
phto_R = []
tot_sdss_SB, tot_sdss_err = [], []

for tt in range( 3 ):

    sub_R, sub_sb, sub_err = [], [], []

    for kk in range( 3 ):
        #.
        pat = pds.read_csv( path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band_aveg-sdss-prof-SB.csv' % (sub_name[ll], cat_lis[tt], band[kk]),)

        tt_r = np.array( pat['R'] )
        tt_sb = np.array( pat['aveg_sb'] )
        tt_err = np.array( pat['aveg_sb_err'] )

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_err )

    #.
    phto_R.append( sub_R )
    tot_sdss_SB.append( sub_sb )
    tot_sdss_err.append( sub_err )



##. figs
for tt in range( 3 ):

    for kk in range( 3 ):

        fig = plt.figure()
        ax1 = fig.add_axes([0.15, 0.12, 0.8, 0.8 ])

        ax1.errorbar( tmp_R[tt][kk], tmp_sb[tt][kk], yerr = tmp_err[tt][kk], marker = '.', ls = '-', color = 'r',
            ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Satellite + Background',)

        ax1.errorbar( tmp_bg_R[tt][kk], tmp_bg_SB[tt][kk], yerr = tmp_bg_err[tt][kk], marker = '.', ls = '--', color = 'b',
            ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = 'Background',)

        for dd in range( 12 ):

            with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_phyR-%s_%s-band_shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' 
                            % (sub_name[ll], cat_lis[tt], band[kk], dd), 'r' ) as f:

                tt_r = np.array(f['r'])
                tt_sb = np.array(f['sb'])
                tt_err = np.array(f['sb_err'])

            ax1.plot( tt_r, tt_sb, ls = ':', color = mpl.cm.rainbow( dd / 11 ), alpha = 0.55, )

        ax1.legend( loc = 1, frameon = False,)

        ax1.set_xscale('log')
        ax1.set_xlabel('R [kpc]')

        ax1.annotate( s = '%s, %s-band' % (cat_lis[tt], band[kk]), xy = (0.75, 0.75), xycoords = 'axes fraction',)

        ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
        ax1.set_ylim( 2e-3, 1e-1 )
        ax1.set_yscale('log')

        plt.savefig(
            '/home/xkchen/%s_phyR-%s_sat_%s-band_BG_compare.png' % (sub_name[ll], cat_lis[ tt ], band[kk]), dpi = 300)
        plt.close()


for kk in range( 3 ):

    plt.figure()
    ax1 = plt.subplot(111)

    for mm in range( 3 ):

        l2 = ax1.errorbar( tmp_R[mm][kk], tmp_sb[mm][kk], yerr = tmp_err[mm][kk], marker = '.', ls = '-', color = color_s[mm],
            ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = cat_lis[mm],)

        l3, = ax1.plot( tmp_bg_R[mm][kk], tmp_bg_SB[mm][kk], ls = '--', color = color_s[mm], alpha = 0.75,)
        ax1.fill_between( tmp_bg_R[mm][kk], y1 = tmp_bg_SB[mm][kk] - tmp_bg_err[mm][kk], 
                            y2 = tmp_bg_SB[mm][kk] + tmp_bg_err[mm][kk], color = color_s[mm], alpha = 0.12)

    legend_2 = ax1.legend( handles = [l2, l3], 
                labels = ['Satellite + Background', 'Background' ], loc = 'upper center', frameon = False,)

    ax1.legend( loc = 1, frameon = False, fontsize = 12,)
    ax1.add_artist( legend_2 )

    ax1.legend( loc = 1, frameon = False,)

    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    ax1.annotate( s = line_name[ll] + ', %s-band' % band[kk], xy = (0.65, 0.45), xycoords = 'axes fraction',)

    ax1.set_ylim( 1e-3, 1e1)
    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    plt.savefig('/home/xkchen/%s_sat_%s-band_BG_compare.png' % (sub_name[ll], band[kk]), dpi = 300)
    plt.close()


for kk in range( 3 ):

    fig = plt.figure( )
    ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
    sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

    ax1.errorbar( phto_R[0][kk], tot_sdss_SB[0][kk], yerr = tot_sdss_err[0][kk], marker = '', ls = ':', color = 'k',
        ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75,)

    ax1.errorbar( phto_R[1][kk], tot_sdss_SB[1][kk], yerr = tot_sdss_err[1][kk], marker = '', ls = '--', color = 'k',
        ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75,)

    ax1.errorbar( phto_R[2][kk], tot_sdss_SB[2][kk], yerr = tot_sdss_err[2][kk], marker = '', ls = '-', color = 'k',
        ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75,)


    # kk_tmp_F_0 = interp.interp1d( nbg_R[0][kk], nbg_SB[0][kk], kind = 'cubic', fill_value = 'extrapolate')
    # kk_tmp_F_1 = interp.interp1d( nbg_R[1][kk], nbg_SB[1][kk], kind = 'cubic', fill_value = 'extrapolate')
    kk_tmp_F_2 = interp.interp1d( phto_R[2][kk], tot_sdss_SB[2][kk], kind = 'cubic', fill_value = 'extrapolate')

    sub_ax1.plot( phto_R[0][kk], tot_sdss_SB[0][kk] / kk_tmp_F_2( phto_R[0][kk] ), ls = ':', color = 'r', alpha = 0.75,)
    sub_ax1.plot( phto_R[1][kk], tot_sdss_SB[1][kk] / kk_tmp_F_2( phto_R[1][kk] ), ls = '--', color = 'r', alpha = 0.75,)
    sub_ax1.plot( phto_R[2][kk], tot_sdss_SB[2][kk] / kk_tmp_F_2( phto_R[2][kk] ), ls = '-', color = 'r', alpha = 0.75,)

    ax1.annotate( s = line_name[ll] + ', %s-band' % band[kk], xy = (0.15, 0.15), xycoords = 'axes fraction',)

    ax1.legend( loc = 1, frameon = False, fontsize = 12,)

    ax1.set_xlim( 2e0, 3e1 )

    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    ax1.set_ylim( 1e-2, 6e0 )

    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    sub_ax1.set_xlim( ax1.get_xlim() )
    sub_ax1.set_xscale('log')
    sub_ax1.set_xlabel('$R \; [kpc]$')

    sub_ax1.set_ylabel('$\\mu \; / \; \\mu_{outer}$', labelpad = 8)
    sub_ax1.set_ylim( 0.64, 1.05 )

    sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.set_xticklabels( labels = [] )

    plt.savefig('/home/xkchen/%s_sat_%s-band_BG-sub-SB_sdss-prof_compare.png' % (sub_name[ ll ], band[kk]), dpi = 300)
    plt.close()

raise

for kk in range( 3 ):

    fig = plt.figure( )
    ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
    sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

    ax1.errorbar( nbg_R[0][kk], nbg_SB[0][kk], yerr = nbg_err[0][kk], marker = '', ls = ':', color = 'r',
        ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = fig_name[0],)

    ax1.errorbar( nbg_R[1][kk], nbg_SB[1][kk], yerr = nbg_err[1][kk], marker = '', ls = '--', color = 'r',
        ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = fig_name[1],)

    ax1.errorbar( nbg_R[2][kk], nbg_SB[2][kk], yerr = nbg_err[2][kk], marker = '', ls = '-', color = 'r',
        ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, alpha = 0.75, label = fig_name[2],)

    _kk_tmp_F = interp.interp1d( nbg_R[2][kk], nbg_SB[2][kk], kind = 'cubic', fill_value = 'extrapolate')

    sub_ax1.plot( nbg_R[0][kk], nbg_SB[0][kk] / _kk_tmp_F( nbg_R[0][kk] ), ls = ':', color = 'r', alpha = 0.75,)
    sub_ax1.fill_between( nbg_R[0][kk], y1 = (nbg_SB[0][kk] - nbg_err[0][kk]) / _kk_tmp_F( nbg_R[0][kk] ), 
                y2 = (nbg_SB[0][kk] + nbg_err[0][kk]) / _kk_tmp_F( nbg_R[0][kk] ), color = 'r', alpha = 0.12,)

    sub_ax1.plot( nbg_R[1][kk], nbg_SB[1][kk] / _kk_tmp_F( nbg_R[1][kk] ), ls = '--', color = 'r', alpha = 0.75,)
    sub_ax1.fill_between( nbg_R[1][kk], y1 = (nbg_SB[1][kk] - nbg_err[1][kk]) / _kk_tmp_F( nbg_R[1][kk] ), 
                y2 = (nbg_SB[1][kk] + nbg_err[1][kk]) / _kk_tmp_F( nbg_R[1][kk] ), color = 'r', alpha = 0.12,)

    # sub_ax1.plot( nbg_R[2][kk], nbg_SB[2][kk] / _kk_tmp_F( nbg_R[2][kk] ), ls = '-', color = 'r', alpha = 0.75,)
    # sub_ax1.fill_between( nbg_R[2][kk], y1 = (nbg_SB[2][kk] - nbg_err[2][kk]) / _kk_tmp_F( nbg_R[2][kk] ), 
    #             y2 = (nbg_SB[2][kk] + nbg_err[2][kk]) / _kk_tmp_F( nbg_R[2][kk] ), color = 'r', alpha = 0.12,)

    daa = nbg_SB[1][kk] / _kk_tmp_F( nbg_R[1][kk] )
    sub_ax1.axhline( y = daa[0], ls = '-', color = 'k', alpha = 0.25,)


    ax1.annotate( s = line_name[ll] + ', %s-band' % band[kk], xy = (0.15, 0.15), xycoords = 'axes fraction',)

    ax1.legend( loc = 1, frameon = False, fontsize = 12,)

    ax1.set_xlim( 2e0, 3e1 )

    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    ax1.set_ylim( 1e-2, 4e0 )

    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    sub_ax1.set_xlim( ax1.get_xlim() )
    sub_ax1.set_xscale('log')
    sub_ax1.set_xlabel('$R \; [kpc]$')

    sub_ax1.set_ylabel('$\\mu \; / \; \\mu_{outer}$', labelpad = 8)
    sub_ax1.set_ylim( daa[0] - 0.2, daa[0] + 0.2 )

    sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.set_xticklabels( labels = [] )

    plt.savefig('/home/xkchen/%s_sat_%s-band_BG-sub_compare.png' % (sub_name[ ll ], band[kk]), dpi = 300)
    plt.close()
