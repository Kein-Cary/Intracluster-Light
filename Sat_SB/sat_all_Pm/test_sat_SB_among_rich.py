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
a_ref = 1 / ( 1 + z_ref)

band = ['r', 'g', 'i']


### === ### data load
##. sample information
bin_rich = [ 20, 30, 50, 210 ]

##. without Pm weight
# BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/BGs/'
# out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/noBG_SBs/'
# path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/SBs/'


##. with Pm weight
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/Pm_weit_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/Pm_weit_noBG_SB/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_all_Pm/Pm_weit_SBs/'


sub_name = ['low-rich', 'medi-rich', 'high-rich']

##. R_limmits
# R_str = 'phy'
# R_bins = np.array( [ 0, 300, 400, 550, 5000] )     ### kpc

R_str = 'scale'
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m


##. line name
if R_str == 'phy':

    fig_name = []
    for dd in range( len(R_bins) - 1 ):

        if dd == 0:
            fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

        elif dd == len(R_bins) - 2:
            fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

        else:
            fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

if R_str == 'scale':

    fig_name = []
    for dd in range( len(R_bins) - 1 ):

        if dd == 0:
            fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

        elif dd == len(R_bins) - 2:
            fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

        else:
            fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


###. shuffle list order
list_order = 13


sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


line_s = [':', '--', '-']

color_s = ['b', 'g', 'm', 'r', 'k']

### === Pm limits compare
for kk in range( 1 ):

    band_str = band[ kk ]

    ##. Pm >= 0.
    nbg_R, nbg_SB, nbg_err = [], [], []

    for tt in range( len(R_bins) - 1 ):

        sub_R, sub_sb, sub_err = [], [], []

        for ll in range( 3 ):

            if R_str == 'phy':
                dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

            if R_str == 'scale':
                dat = pds.read_csv( out_path + 'clus-sat_with-low-Pm_%s_%.2f-%.2fR200m' % (sub_name[ll], R_bins[tt], R_bins[tt + 1]) 
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

            tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

            sub_R.append( tt_r )
            sub_sb.append( tt_sb )
            sub_err.append( tt_sb_err )

        nbg_R.append( sub_R )
        nbg_SB.append( sub_sb )
        nbg_err.append( sub_err )

    ##. Pm >= 0.8
    cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

    for tt in range( len(R_bins) - 1 ):

        sub_R, sub_sb, sub_err = [], [], []

        for ll in range( 3 ):

            if R_str == 'phy':

                dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/' 
                                + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1])
                                + '_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' % (band_str, dd),)

            if R_str == 'scale':

                dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/' 
                                + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[ll], R_bins[tt], R_bins[tt + 1])
                                + '_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' % (band_str, dd),)

            tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

            sub_R.append( tt_r )
            sub_sb.append( tt_sb )
            sub_err.append( tt_sb_err )

        cp_nbg_R.append( sub_R )
        cp_nbg_SB.append( sub_sb )
        cp_nbg_err.append( sub_err )


    ##. figs
    for tt in range( len(R_bins) - 1 ):

        fig = plt.figure()
        ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
        sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

        l1 = ax1.errorbar( nbg_R[tt][0], nbg_SB[tt][0], yerr = nbg_err[tt][0], marker = '', ls = '-', color = color_s[0],
            ecolor = color_s[0], mfc = 'none', mec = color_s[0], capsize = 1.5, alpha = 0.75, label = line_name[0],)

        ax1.errorbar( nbg_R[tt][1], nbg_SB[tt][1], yerr = nbg_err[tt][1], marker = '', ls = '-', color = color_s[1],
            ecolor = color_s[1], mfc = 'none', mec = color_s[1], capsize = 1.5, alpha = 0.75, label = line_name[1],)

        ax1.errorbar( nbg_R[tt][2], nbg_SB[tt][2], yerr = nbg_err[tt][2], marker = '', ls = '-', color = color_s[2],
            ecolor = color_s[2], mfc = 'none', mec = color_s[2], capsize = 1.5, alpha = 0.75, label = line_name[2],)


        l2 = ax1.errorbar( cp_nbg_R[tt][0], cp_nbg_SB[tt][0], yerr = cp_nbg_err[tt][0], marker = '', ls = '--', color = color_s[0],
            ecolor = color_s[0], mfc = 'none', mec = color_s[0], capsize = 1.5, alpha = 0.75,)

        ax1.errorbar( cp_nbg_R[tt][1], cp_nbg_SB[tt][1], yerr = cp_nbg_err[tt][1], marker = '', ls = '--', color = color_s[1],
            ecolor = color_s[1], mfc = 'none', mec = color_s[1], capsize = 1.5, alpha = 0.75,)

        ax1.errorbar( cp_nbg_R[tt][2], cp_nbg_SB[tt][2], yerr = cp_nbg_err[tt][2], marker = '', ls = '--', color = color_s[2],
            ecolor = color_s[2], mfc = 'none', mec = color_s[2], capsize = 1.5, alpha = 0.75,)

        id_nn = np.isnan( cp_nbg_SB[tt][0]  )
        id_mx = id_nn == False
        _tmp_F_0 = interp.interp1d( cp_nbg_R[tt][0][ id_mx ], cp_nbg_SB[tt][0][ id_mx ], kind = 'cubic', fill_value = 'extrapolate')

        id_nn = np.isnan( cp_nbg_SB[tt][1]  )
        id_mx = id_nn == False
        _tmp_F_1 = interp.interp1d( cp_nbg_R[tt][1][ id_mx ], cp_nbg_SB[tt][1][ id_mx ], kind = 'cubic', fill_value = 'extrapolate')

        id_nn = np.isnan( cp_nbg_SB[tt][2]  )
        id_mx = id_nn == False
        _tmp_F_2 = interp.interp1d( cp_nbg_R[tt][2][ id_mx ], cp_nbg_SB[tt][2][ id_mx ], kind = 'cubic', fill_value = 'extrapolate')

        sub_ax1.plot( nbg_R[tt][0], nbg_SB[tt][0] / _tmp_F_0( nbg_R[tt][0] ), ls = '-', color = color_s[0], )
        sub_ax1.plot( nbg_R[tt][1], nbg_SB[tt][1] / _tmp_F_1( nbg_R[tt][1] ), ls = '-', color = color_s[1], )
        sub_ax1.plot( nbg_R[tt][2], nbg_SB[tt][2] / _tmp_F_2( nbg_R[tt][2] ), ls = '-', color = color_s[2], )


        legend_2 = ax1.legend( handles = [l1, l2], 
                    labels = ['$P_{m} >=0$', '$P_{m}>=0.8$'], loc = 1, frameon = False, fontsize = 12,)

        ax1.legend( loc = 3, frameon = False, fontsize = 12,)
        ax1.add_artist( legend_2 )

        ax1.annotate( s = fig_name[tt] + ', %s-band' % band_str, xy = (0.55, 0.05), xycoords = 'axes fraction', fontsize = 12,)

        ax1.set_xlim( 2e0, 4e1 )
        ax1.set_xscale('log')
        ax1.set_xlabel('R [kpc]', fontsize = 12,)

        ax1.set_ylim( 1e-3, 3e0 )
        ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
        ax1.set_yscale('log')
        
        sub_ax1.set_xlim( ax1.get_xlim() )
        sub_ax1.set_xscale('log')
        sub_ax1.set_xlabel('$R \; [kpc]$', fontsize = 12,)

        sub_ax1.set_ylabel('$\\mu \,(P_{m} > 0) / \; \\mu \,(P_{m} > 0.8)$', labelpad = 8, fontsize = 12,)
        sub_ax1.set_ylim( 0.45, 1.05 )

        sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
        sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
        ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
        ax1.set_xticklabels( labels = [] )

        plt.savefig('/home/xkchen/Sat_%s-band_%.2f-%.2fR200m_Pm_cut_compare.png' % (band_str, R_bins[tt], R_bins[tt + 1]), dpi = 300)
        plt.close()



for kk in range( 1 ):

    band_str = band[ kk ]

    nbg_R, nbg_SB, nbg_err = [], [], []

    for tt in range( len(R_bins) - 1 ):

        sub_R, sub_sb, sub_err = [], [], []

        for ll in range( 3 ):

            if R_str == 'phy':
                dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

            if R_str == 'scale':
                dat = pds.read_csv( out_path + 'clus-sat_with-low-Pm_%s_%.2f-%.2fR200m' % (sub_name[ll], R_bins[tt], R_bins[tt + 1]) 
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

            tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

            sub_R.append( tt_r )
            sub_sb.append( tt_sb )
            sub_err.append( tt_sb_err )

        nbg_R.append( sub_R )
        nbg_SB.append( sub_sb )
        nbg_err.append( sub_err )

    ##. inner bins
    for tt in range( len(R_bins) - 1 ):

        fig = plt.figure()
        ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
        sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

        ax1.errorbar( nbg_R[tt][0], nbg_SB[tt][0], yerr = nbg_err[tt][0], marker = '', ls = '-', color = color_s[0],
            ecolor = color_s[0], mfc = 'none', mec = color_s[0], capsize = 1.5, alpha = 0.75, label = line_name[0],)

        ax1.errorbar( nbg_R[tt][1], nbg_SB[tt][1], yerr = nbg_err[tt][1], marker = '', ls = '-', color = color_s[1],
                    ecolor = color_s[1], mfc = 'none', mec = color_s[1], capsize = 1.5, alpha = 0.75, label = line_name[1],)

        ax1.errorbar( nbg_R[tt][2], nbg_SB[tt][2], yerr = nbg_err[tt][2], marker = '', ls = '-', color = color_s[2],
                    ecolor = color_s[2], mfc = 'none', mec = color_s[2], capsize = 1.5, alpha = 0.75, label = line_name[2],)

        _kk_tmp_F = interp.interp1d( nbg_R[tt][2], nbg_SB[tt][2], kind = 'cubic', fill_value = 'extrapolate',)

        sub_ax1.plot( nbg_R[tt][0], nbg_SB[tt][0] / _kk_tmp_F( nbg_R[tt][0] ), ls = '-', color = color_s[0], alpha = 0.75,)
        sub_ax1.fill_between( nbg_R[tt][0], y1 = (nbg_SB[tt][0] - nbg_err[tt][0]) / _kk_tmp_F( nbg_R[tt][0] ), 
                    y2 = (nbg_SB[tt][0] + nbg_err[tt][0]) / _kk_tmp_F( nbg_R[tt][0] ), color = color_s[0], alpha = 0.12,)

        sub_ax1.plot( nbg_R[tt][1], nbg_SB[tt][1] / _kk_tmp_F( nbg_R[tt][1] ), ls = '-', color = color_s[1], alpha = 0.75,)
        sub_ax1.fill_between( nbg_R[tt][1], y1 = (nbg_SB[tt][1] - nbg_err[tt][1]) / _kk_tmp_F( nbg_R[tt][1] ), 
                    y2 = (nbg_SB[tt][1] + nbg_err[tt][1]) / _kk_tmp_F( nbg_R[tt][1] ), color = color_s[1], alpha = 0.12,)

        if (R_str == 'scale') & (tt != len(R_bins) - 2):

            cc_dat = pds.read_csv( out_path 
                    + 'clus-sat_with-low-Pm_%s_%.2f-%.2fR200m' % (sub_name[-2], R_bins[-2], R_bins[-1]) 
                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

            cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )
            cc_tt_err = np.array( cc_dat['sb_err'] )

            _cp_tmp_F = interp.interp1d( cc_tt_r, cc_tt_sb, kind = 'cubic', fill_value = 'extrapolate',)

            if ll == 2:
                ax1.errorbar( cc_tt_r, cc_tt_sb, yerr = cc_tt_err, marker = '', ls = '-.', color = 'k',
                        ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.25, label = line_name[-1] + ', ' + fig_name[-1],)
            else:
                ax1.errorbar( cc_tt_r, cc_tt_sb, yerr = cc_tt_err, marker = '', ls = '-.', color = 'k',
                        ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.25,)

        ax1.annotate( s = fig_name[tt] + ', %s-band' % band_str, xy = (0.55, 0.85), xycoords = 'axes fraction',)
        ax1.legend( loc = 3, frameon = False,)

        ax1.set_xlim( 2e0, 4e1 )
        ax1.set_xscale('log')
        ax1.set_xlabel('R [kpc]')

        ax1.set_ylim( 1e-3, 3e0 )
        ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
        ax1.set_yscale('log')

        sub_ax1.set_xlim( ax1.get_xlim() )
        sub_ax1.set_xscale('log')
        sub_ax1.set_xlabel('$R \; [kpc]$')

        sub_ax1.set_ylabel('$\\mu \; / \; \\mu \,(\\lambda \\geq 50)$', labelpad = 8)
        sub_ax1.set_ylim( 0.75, 1.1 )

        sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
        ax1.set_xticklabels( labels = [] )

        if R_str == 'phy':
            plt.savefig('/home/xkchen/sat_%d-%dkpc_%s-band_BG-sub_SB_compare.png' 
                    % (R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)

        if R_str == 'scale':
            plt.savefig('/home/xkchen/sat_%.2f-%.2fR200m_%s-band_BG-sub_SB_compare.png' 
                    % (R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)

        plt.close()
