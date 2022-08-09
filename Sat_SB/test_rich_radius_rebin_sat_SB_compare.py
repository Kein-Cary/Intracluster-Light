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
band = ['r', 'g', 'i']


### === ### data load
##. sample information
bin_rich = [ 20, 30, 50, 210 ]

##. sat_img with BCGs
# BG_path = '/home/xkchen/figs_cp/cc_rich_rebin/BGs/'
# out_path = '/home/xkchen/figs_cp/cc_rich_rebin/noBG_SBs/'
# path = '/home/xkchen/figs_cp/cc_rich_rebin/SBs/'

##. sat_img without BCGs
path = '/home/xkchen/figs_cp/cc_rich_rebin/nobcg_SBs/'
BG_path = '/home/xkchen/figs_cp/cc_rich_rebin/nobcg_BGs/'
out_path = '/home/xkchen/figs_cp/cc_rich_rebin/nobcg_BGsub_SBs/'


sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


###. compare the SB profile of entire satellite sample
list_order = 13

color_s = ['b', 'g', 'r']
line_s = [':', '--', '-']

y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [7e-4, 1e0], [5e-3, 6e0] ]

N_sample = 100

###. entire sample comparison
"""
for kk in range( 3 ):

    tot_R, tot_sb, tot_sb_err = [], [], []
    tot_bg_R, tot_bg_SB, tot_bg_err = [], [], []
    tot_nbg_R, tot_nbg_SB, tot_nbg_err = [], [], []

    for ll in range( 3 ):

        #.
        with h5py.File( path + 
            'Extend_BCGM_gri-common_%s' % sub_name[ll] + '_%s-band' % band[kk] + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        tot_R.append( tt_r )
        tot_sb.append( tt_sb )
        tot_sb_err.append( tt_err )

        #.
        with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_%s-band_' % (sub_name[ll], band[kk]) + 
                        'shufl-%d_BG_Mean_jack_SB-pro_z-ref.h5' % list_order, 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        tot_bg_R.append( tt_r )
        tot_bg_SB.append( tt_sb )
        tot_bg_err.append( tt_err )

        #.
        dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s' % sub_name[ll] + '_%s-band_aveg-jack_BG-sub_SB.csv' % band[kk],)
        tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

        tot_nbg_R.append( tt_r )
        tot_nbg_SB.append( tt_sb )
        tot_nbg_err.append( tt_sb_err )

    #. figs
    fig = plt.figure()
    ax1 = fig.add_axes([0.12, 0.10, 0.80, 0.85])

    for nn in range( 3 ):

        l2 = ax1.errorbar( tot_R[nn], tot_sb[nn], yerr = tot_bg_err[nn], marker = '.', ls = '-', color = color_s[nn],
            ecolor = color_s[nn], mfc = 'none', mec = color_s[nn], capsize = 1.5, label = line_name[ nn ],)

        l3, = ax1.plot( tot_bg_R[nn], tot_bg_SB[nn], ls = '--', color = color_s[nn], alpha = 0.75,)
        ax1.fill_between( tot_bg_R[nn], y1 = tot_bg_SB[nn] - tot_bg_err[nn], 
                            y2 = tot_bg_SB[nn] + tot_bg_err[nn], color = color_s[nn], alpha = 0.12)

    legend_2 = ax1.legend( handles = [l2, l3], 
                labels = ['Satellite + Background', 'Background' ], loc = 4, frameon = False,)

    ax1.legend( loc = 1, frameon = False,)
    ax1.add_artist( legend_2 )

    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    ax1.annotate( s = '%s-band' % band[kk], xy = (0.65, 0.65), xycoords = 'axes fraction',)

    ax1.set_ylim( y_lim_0[kk][0], y_lim_0[kk][1] )
    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    plt.savefig('/home/xkchen/all-sat_%s-band_BG_compare.png' % band[kk], dpi = 300)
    plt.close()


    fig = plt.figure()
    ax1 = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
    sub_ax1 = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

    for nn in range( 3 ):

        ax1.errorbar( tot_nbg_R[nn], tot_nbg_SB[nn], yerr = tot_nbg_err[nn], marker = '.', ls = line_s[ nn ], color = 'r',
            ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = line_name[ nn ],)

    _kk_tmp_F = interp.interp1d( tot_nbg_R[-1], tot_nbg_SB[-1], kind = 'linear',)

    for nn in range( 2 ):

        sub_ax1.plot( tot_nbg_R[nn], tot_nbg_SB[nn] / _kk_tmp_F( tot_nbg_R[nn] ), ls = line_s[ nn ], color = 'r',)

    ax1.legend( loc = 1, frameon = False,)

    ax1.set_xlim( 1e0, 5e1 )
    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    ax1.annotate( s = '%s-band' % band[kk], xy = (0.65, 0.75), xycoords = 'axes fraction',)

    ax1.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    sub_ax1.set_xlim( ax1.get_xlim() )
    sub_ax1.set_xscale('log')
    sub_ax1.set_xlabel('$R \; [kpc]$')

    sub_ax1.set_ylabel('$\\mu \; / \; \\mu_{ \\lambda \, \\geq \, 50}$', labelpad = 8)
    sub_ax1.set_ylim( 0.9, 1.12 )

    sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.set_xticklabels( labels = [] )

    plt.savefig('/home/xkchen/all-sat_%s-band_BG-sub_SB_compare.png' % band[kk], dpi = 300)
    plt.close()

raise
"""


###. ratio profile and slope record

for kk in range( 3 ):

    band_str = band[ kk ]

    for ll in range( 3 ):

        ##. rich < 30
        if ll == 0:
            R_bins = np.array([0, 150, 300, 500, 2000])

        ##. rich (30, 50)
        if ll == 1:
            R_bins = np.array( [ 0, 300, 400, 550, 5000] )

        ##. rich > 50
        if ll == 2:
            R_bins = np.array([0, 400, 600, 750, 2000])


        for tt in range( len(R_bins) - 2 ):

            ##. use for ratio compare
            cc_dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[-2], R_bins[-1]) + 
                                '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

            cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )

            id_rx = cc_tt_r < 80   ##.kpc
            _tt_tmp_F = interp.interp1d( cc_tt_r[ id_rx ], cc_tt_sb[ id_rx ], kind = 'cubic', fill_value = 'extrapolate',)


            tmp_r, tmp_ratio = [], []
            tmp_slope = []
            tmp_Rc = []

            for dd in range( N_sample ):

                dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1]) 
                                 + '_%s-band_jack-sub-%d_BG-sub-SB-pro_z-ref.h5' % (band_str, dd),)

                tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

                #. critical radius
                id_rx = tt_r <= 40
                sm_r = tt_r[ id_rx ]

                tt_eta = tt_sb[ id_rx ] / _tt_tmp_F( sm_r )

                sm_eta = signal.savgol_filter( tt_eta, 5, 2, deriv = 1)
                _tmp_interp_F = interp.interp1d( np.log10( sm_r ), sm_eta, kind = 'cubic', fill_value = 'extrapolate',)

                _new_R = np.logspace( np.log10( sm_r[0] ), np.log10( sm_r[-1] ), 35 )
                _new_eta = _tmp_interp_F( np.log10( _new_R ) )

                id_tag = _new_eta == np.min( _new_eta )
                cen_R_lim_0 = _new_R[ id_tag ][0]

                #.
                tmp_r.append( sm_r )
                tmp_ratio.append( tt_eta )
                tmp_slope.append( sm_eta )
                tmp_Rc.append( cen_R_lim_0 )

            ##.
            aveg_R_0, aveg_ratio, aveg_eta_err = arr_jack_func( tmp_ratio, tmp_r, N_sample )[:3]
            aveg_R_1, aveg_slope, aveg_slope_err = arr_jack_func( tmp_slope, tmp_r, N_sample )[:3]

            tmp_Rc = np.array( tmp_Rc )
            aveg_Rc = np.mean( tmp_Rc )
            std_aveg_Rc = np.std( tmp_Rc )


            out_Rc, out_Rc_err = np.ones( len( aveg_R_1), ) * aveg_Rc, np.ones( len( aveg_R_1), ) * std_aveg_Rc

            ##. save
            keys = [ 'R', 'ratio', 'ratio_err' ]
            values = [ aveg_R_0, aveg_ratio, aveg_eta_err ]
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )
            data.to_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1]) 
                        + '_%s-band_aveg-jack_BG-sub_SB_ratio.csv' % band_str,)

            keys = [ 'R', 'slope', 'slope_err', 'R_crit', 'std_R_crit' ]
            values = [ aveg_R_1, aveg_slope, aveg_slope_err, out_Rc, out_Rc_err ]
            fill = dict( zip( keys, values ) )
            data = pds.DataFrame( fill )
            data.to_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ll], R_bins[tt], R_bins[tt + 1]) 
                        + '_%s-band_aveg-jack_BG-sub_SB_ratio_slope.csv' % band_str,)

