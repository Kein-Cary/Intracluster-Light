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

##. sat_img without BCG
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGs/'
out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'


##. shuffle list
list_order = 13

sub_name = ['low-rich', 'medi-rich', 'high-rich']

bin_rich = [ 20, 30, 50, 210 ]
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']

line_s = [':', '--', '-']
color_s = ['b', 'g', 'm', 'r', 'k']


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

#.
if R_str == 'scale':

    fig_name = []
    for dd in range( len(R_bins) - 1 ):

        if dd == 0:
            fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

        elif dd == len(R_bins) - 2:
            fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

        else:
            fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)


### === figs
y_lim_0 = [ [1e-3, 4e0], [1e-3, 1e0], [1e-3, 7e0] ]
y_lim_1 = [ [2e-3, 4e0], [7e-4, 1e0], [5e-3, 6e0] ]


for kk in range( 1 ):

    band_str = band[ kk ]

    ##. 
    fig = plt.figure( figsize = (19.84, 4.8) )
    ax0 = fig.add_axes([0.05, 0.32, 0.28, 0.63])
    sub_ax0 = fig.add_axes([0.05, 0.11, 0.28, 0.21])

    ax1 = fig.add_axes([0.38, 0.32, 0.28, 0.63])
    sub_ax1 = fig.add_axes([0.38, 0.11, 0.28, 0.21])

    ax2 = fig.add_axes([0.71, 0.32, 0.28, 0.63])
    sub_ax2 = fig.add_axes([0.71, 0.11, 0.28, 0.21])

    axes = [ ax0, ax1, ax2 ]
    sub_axes = [ sub_ax0, sub_ax1, sub_ax2 ]

    #. data load
    for qq in range( 3 ):

        nbg_R, nbg_SB, nbg_err = [], [], []
        crit_R_0 = []
        crit_eta_0 = 0.15

        out_R_lim = 30

        for tt in range( len(R_bins) - 1 ):

            if R_str == 'phy':

                dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

                tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

                ##. use for ratio compare
                cc_dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[-2], R_bins[-1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

                cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )


            if R_str == 'scale':

                dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[qq], R_bins[tt], R_bins[tt + 1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

                tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

                ##. use for ratio compare
                cc_dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[qq], R_bins[-2], R_bins[-1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)

                cc_tt_r, cc_tt_sb = np.array( cc_dat['r'] ), np.array( cc_dat['sb'] )


            id_rx = cc_tt_r < 80   ##.kpc
            _tt_tmp_F = interp.interp1d( cc_tt_r[ id_rx ], cc_tt_sb[ id_rx ], kind = 'cubic', fill_value = 'extrapolate',)


            if tt == len(R_bins) - 2:

                #. critical radius
                id_rx = tt_r <= 50
                sm_r = tt_r[ id_rx ]

                tt_eta = tt_sb[ id_rx ] / tt_sb[ 0 ]

                sm_eta = signal.savgol_filter( tt_eta, 5, 2)
                relat_eta = (sm_eta[0] - sm_eta) / sm_eta[0]
                _tmp_interp_F = interp.interp1d( relat_eta, np.log10( sm_r ), kind = 'cubic', fill_value = 'extrapolate',)
                cen_R_lim_0 = 10**_tmp_interp_F( 1 - 1 / 100 )

            else:

                #. critical radius
                id_rx = tt_r <= out_R_lim
                sm_r = tt_r[ id_rx ]
                sm_eta = tt_sb[ id_rx ] / _tt_tmp_F( sm_r )

                _tmp_interp_F = interp.interp1d( sm_r, sm_eta, kind = 'cubic', fill_value = 'extrapolate',)
                new_R = np.logspace( np.log10( sm_r[0] ), np.log10( sm_r[-1] ), 35)
                new_eta = _tmp_interp_F( new_R )

                tag_eta = new_eta[0] * ( 1 - crit_eta_0 )
                diff = np.abs(new_eta - tag_eta)
                id_tag = diff == np.min( diff )
                cen_R_lim_0 = new_R[ id_tag ][0]


            nbg_R.append( tt_r )
            nbg_SB.append( tt_sb )
            nbg_err.append( tt_sb_err )

            crit_R_0.append( cen_R_lim_0 )

        print( crit_R_0 )

        #. save the crit_R
        keys = fig_name
        values = crit_R_0
        fill = dict( zip( keys, values ) )
        data = pds.DataFrame( fill, index = ['k', 'v'] )
        data.to_csv( out_path + 'Extend_BCGM_gri-common_%s_%s_%s-band_Rt_test.csv' % (sub_name[qq], R_str, band_str),)


        #.
        gax, sub_gax = axes[qq], sub_axes[qq]

        gax.errorbar( nbg_R[-1], nbg_SB[-1], yerr = nbg_err[-1], marker = '', ls = '--', color = 'k',
            ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75, label = fig_name[-1],)

        _kk_tmp_F = interp.interp1d( nbg_R[-1], nbg_SB[-1], kind = 'linear', fill_value = 'extrapolate',)

        for mm in range( len(R_bins) -2 ):

            gax.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '', ls = '-', color = color_s[mm], 
                ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

            sub_gax.plot( nbg_R[mm], nbg_SB[mm] / _kk_tmp_F( nbg_R[mm] ), ls = '-', color = color_s[mm], alpha = 0.75,)
            sub_gax.fill_between( nbg_R[mm], y1 = (nbg_SB[mm] - nbg_err[mm]) / _kk_tmp_F( nbg_R[mm] ), 
                        y2 = (nbg_SB[mm] + nbg_err[mm]) / _kk_tmp_F( nbg_R[mm] ), color = color_s[mm], alpha = 0.12,)


            gax.axvline( x = crit_R_0[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
            sub_gax.axvline( x = crit_R_0[mm], ls = '--', color = color_s[mm], alpha = 0.75,)


        gax.annotate( s = line_name[qq] + ', %s-band' % band_str, xy = (0.55, 0.85), xycoords = 'axes fraction', fontsize = 14,)
        gax.legend( loc = 3, frameon = False, fontsize = 13,)

        gax.set_xlim( 2e0, 5e1 )
        gax.set_xscale('log')

        gax.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
        gax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 14)
        gax.set_yscale('log')

        sub_gax.set_xlim( gax.get_xlim() )
        sub_gax.set_xscale('log')
        sub_gax.set_xlabel('$R \; [kpc]$', fontsize = 14)

        sub_gax.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
                            fontsize = 11, labelpad = 8)
        sub_gax.set_ylim( 0.45, 1.05 )

        gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
        sub_gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
        sub_gax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
        gax.set_xticklabels( labels = [] )

    plt.savefig('/home/xkchen/%s_%s-band_BG-sub_sat-SB.png' % (R_str, band_str), dpi = 300)
    plt.close()



for kk in range( 1 ):

    band_str = band[ kk ]

    ##. 
    fig = plt.figure( figsize = (19.84, 7.2) )
    ax0 = fig.add_axes([0.05, 0.53, 0.28, 0.45])
    sub_ax00 = fig.add_axes([0.05, 0.38, 0.28, 0.15])
    sub_ax01 = fig.add_axes([0.05, 0.23, 0.28, 0.15])
    sub_ax02 = fig.add_axes([0.05, 0.08, 0.28, 0.15])


    ax1 = fig.add_axes([0.38, 0.53, 0.28, 0.45])
    sub_ax10 = fig.add_axes([0.38, 0.38, 0.28, 0.15])
    sub_ax11 = fig.add_axes([0.38, 0.23, 0.28, 0.15])
    sub_ax12 = fig.add_axes([0.38, 0.08, 0.28, 0.15])


    ax2 = fig.add_axes([0.71, 0.53, 0.28, 0.45])
    sub_ax20 = fig.add_axes([0.71, 0.38, 0.28, 0.15])
    sub_ax21 = fig.add_axes([0.71, 0.23, 0.28, 0.15])
    sub_ax22 = fig.add_axes([0.71, 0.08, 0.28, 0.15])


    axes = [ ax0, ax1, ax2 ]
    sub_axes0 = [ sub_ax00, sub_ax10, sub_ax20 ]
    sub_axes1 = [ sub_ax01, sub_ax11, sub_ax21 ]
    sub_axes2 = [ sub_ax02, sub_ax12, sub_ax22 ]


    #. data load
    for qq in range( 3 ):

        nbg_R, nbg_SB, nbg_err = [], [], []

        for tt in range( len(R_bins) - 1 ):

            if R_str == 'phy':

                dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)
                tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

            if R_str == 'scale':

                dat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[qq], R_bins[tt], R_bins[tt + 1])
                                    + '_%s-band_aveg-jack_BG-sub_SB.csv' % band_str,)
                tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

            nbg_R.append( tt_r )
            nbg_SB.append( tt_sb )
            nbg_err.append( tt_sb_err )

        ##.
        crit_R = []

        R_dat = pds.read_csv( out_path + 
                    'Extend_BCGM_gri-common_%s_%s_%s-band_Rt_test.csv' % (sub_name[qq], R_str, band_str),)

        for tt in range( len(R_bins) - 1 ):

            tt_rc = np.array( R_dat[ fig_name[tt] ] )[0]

            crit_R.append( tt_rc )

        ##. ratio and slope profile
        sub_R, sub_ratio, sub_ratio_err = [], [], []
        sub_R1, sub_slope, sub_slope_err, sub_R_crit, sub_R_crit_std = [], [], [], [], []

        for tt in range( len(R_bins) - 2 ):

            if R_str == 'phy':
                cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
                                + '_%s-band_aveg-jack_BG-sub_SB_ratio.csv' % band_str,)

            if R_str == 'scale':
                cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
                                + '_%s-band_aveg-jack_BG-sub_SB_ratio.csv' % band_str,)

            tt_r, tt_eta, tt_eta_err = np.array( cat['R'] ), np.array( cat['ratio'] ), np.array( cat['ratio_err'] )

            sub_R.append( tt_r )
            sub_ratio.append( tt_eta )
            sub_ratio_err.append( tt_eta_err )


            if R_str == 'phy':
                cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
                                + '_%s-band_aveg-jack_BG-sub_SB_ratio_slope.csv' % band_str,)

            if R_str == 'scale':
                cat = pds.read_csv( out_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m' % (sub_name[qq], R_bins[tt], R_bins[tt + 1]) 
                                + '_%s-band_aveg-jack_BG-sub_SB_ratio_slope.csv' % band_str,)

            tt_r, tt_slope, tt_slope_err = np.array( cat['R'] ), np.array( cat['slope'] ), np.array( cat['slope_err'] )
            tt_Rc, tt_Rc_std = np.array( cat['R_crit'] )[0], np.array( cat['std_R_crit'] )[0]

            sub_R1.append( tt_r )
            sub_slope.append( tt_slope )
            sub_slope_err.append( tt_slope_err )

            sub_R_crit.append( tt_Rc )
            sub_R_crit_std.append( tt_Rc_std )

        #.
        sub_offset = []

        for tt in range( len(R_bins) - 2 ):

            _tt_off = sub_ratio[ tt ][0] / sub_ratio[-1][0]
            sub_offset.append( _tt_off )


        #.
        gax = axes[ qq ]
        sub_gax0 = sub_axes0[ qq ]
        sub_gax1 = sub_axes1[ qq ]
        sub_gax2 = sub_axes2[ qq ]

        gax.errorbar( nbg_R[-1], nbg_SB[-1], yerr = nbg_err[-1], marker = '', ls = '--', color = 'k',
            ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, alpha = 0.75, label = fig_name[-1],)

        _kk_tmp_F = interp.interp1d( nbg_R[-1], nbg_SB[-1], kind = 'linear', fill_value = 'extrapolate',)

        for mm in range( len(R_bins) -2 ):

            gax.errorbar( nbg_R[mm], nbg_SB[mm], yerr = nbg_err[mm], marker = '', ls = '-', color = color_s[mm], 
                ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, alpha = 0.75, label = fig_name[mm],)

            sub_gax0.plot( nbg_R[mm], nbg_SB[mm] / _kk_tmp_F( nbg_R[mm] ), ls = '-', color = color_s[mm], alpha = 0.75,)
            sub_gax0.fill_between( nbg_R[mm], y1 = (nbg_SB[mm] - nbg_err[mm]) / _kk_tmp_F( nbg_R[mm] ), 
                        y2 = (nbg_SB[mm] + nbg_err[mm]) / _kk_tmp_F( nbg_R[mm] ), color = color_s[mm], alpha = 0.12,)


            gax.axvline( x = crit_R[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
            sub_gax0.axvline( x = crit_R[mm], ls = '--', color = color_s[mm], alpha = 0.75,)

            ##.
            sub_gax1.plot( sub_R[mm], sub_ratio[mm] / sub_offset[mm], ls = '-', color = color_s[mm], alpha = 0.75,)
            sub_gax1.axvline( x = sub_R_crit[mm], ls = '--', color = color_s[mm], alpha = 0.75,)

            sub_gax2.plot( sub_R1[mm], sub_slope[mm], ls = '-', color = color_s[mm], alpha = 0.75,)
            sub_gax2.axvline( x = sub_R_crit[mm], ls = '--', color = color_s[mm], alpha = 0.75,)


        gax.annotate( s = line_name[qq] + ', %s-band' % band_str, xy = (0.55, 0.85), xycoords = 'axes fraction', fontsize = 14,)
        gax.legend( loc = 3, frameon = False, fontsize = 13,)

        gax.set_xlim( 2e0, 5e1 )
        gax.set_xscale('log')

        gax.set_ylim( y_lim_1[kk][0], y_lim_1[kk][1] )
        gax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 14)
        gax.set_yscale('log')

        sub_gax0.set_xlim( gax.get_xlim() )
        sub_gax0.set_xscale('log')
        sub_gax0.set_xlabel('$R \; [kpc]$', fontsize = 14)

        sub_gax0.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], 
                            fontsize = 13, labelpad = 8)
        sub_gax0.set_ylim( 0.45, 1.05 )


        sub_gax1.set_xlim( gax.get_xlim() )
        sub_gax1.set_xscale('log')

        # sub_gax1.set_ylabel('$\\mu \; / \; \\mu \, (R \\geq %.2f \, R_{200m})$' % R_bins[-2], labelpad = 8, fontsize = 11,)
        sub_gax1.set_ylim( 0.45, 1.05 )


        sub_gax2.set_xlim( gax.get_xlim() )
        sub_gax2.set_xscale('log')
        sub_gax2.set_xlabel('$R \; [kpc]$', fontsize = 14,)

        sub_gax2.set_ylabel('ratio slope', fontsize = 11,)
        sub_gax2.set_ylim( -0.04, 0.04 )


        gax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)

        sub_gax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
        sub_gax0.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

        sub_gax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
        sub_gax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

        sub_gax2.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 14,)
        sub_gax2.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

        gax.set_xticklabels( labels = [] )

    # plt.savefig('/home/xkchen/%s_%s-band_BG-sub_sat-SB.png' % (R_str, band_str), dpi = 300)
    plt.savefig('/home/xkchen/%s_%s-band_BG-sub_sat-SB.pdf' % (R_str, band_str), dpi = 300)
    plt.close()

raise


##. Rt compare
marks = ['s', '>', 'o']
mark_size = [10, 25, 35]

band_str = band[ 0 ]    ##. r-band


fig = plt.figure()
ax1 = fig.add_axes( [0.12, 0.11, 0.80, 0.85] )

for qq in range( 3 ):

    ##. R_crit with ratio decrease
    cp_Rc = []

    cc_dat = pds.read_csv( out_path + 
            'Extend_BCGM_gri-common_%s_%s_%s-band_Rt_test.csv' % (sub_name[qq], R_str, band_str),)

    for dd in range( len(R_bins) - 2 ):

        _dd_Rt = np.array( cc_dat[ fig_name[dd] ] )[0]

        cp_Rc.append( _dd_Rt )


    ##. read sample catalog and get the average centric distance
    R_aveg = []
    Rv_arr = []

    for dd in range( len( R_bins ) - 1 ):

        if R_str == 'phy':
            cat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv'
                            % ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

            x_Rc = np.array( cat['R_sat'] )   ## Mpc / h
            cp_x_Rc = x_Rc * 1e3 * a_ref / h  ## kpc

        if R_str == 'scale':
            cat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
                            % ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

            z_obs = np.array( cat['bcg_z'] )
            x_Rc = np.array( cat['R2Rv'] )   ## R / R200m

            x_Rsat = np.array( cat['R_sat'] )   ## Mpc / h
            x_Rv = x_Rsat / x_Rc    ## Mpc / h

            cp_x_Rc = x_Rc + 0
            cp_Rv = x_Rv * 1e3 / h     ## kpc

        R_aveg.append( np.mean( cp_x_Rc) )
        Rv_arr.append( np.mean( cp_Rv ) )

        print( Rv_arr )

    #.
    # l2, = ax1.plot( R_aveg[:-1], cp_Rc, ls = '-', color = color_s[qq], alpha = 0.75, label = line_name[qq],)

    l2, = ax1.plot( R_aveg[:-1], cp_Rc / np.array( Rv_arr )[:-1], ls = '-', color = color_s[qq], alpha = 0.75, label = line_name[qq],)

ax1.legend( loc = 2, frameon = False, fontsize = 12)
ax1.annotate( s = 'Ratio decrease', xy = (0.15, 0.55), xycoords = 'axes fraction',)

# ax1.set_ylim( 7.5, 18.5 )
# ax1.set_ylabel('$ R_{t} \; [kpc]$')
# ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax1.set_ylabel('$ R_{t} \, / \, \\bar{R}_{200m}$')
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )


if R_str == 'phy':
    ax1.set_xlabel('$\\bar{R}_{sat} \; [kpc]$')

if R_str == 'scale':
    ax1.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$')

ax1.set_xscale('log')
# ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax1.tick_params( axis = 'both', which = 'both', direction = 'in',)

# plt.savefig('/home/xkchen/%s_Rt_compare.png' % R_str, dpi = 300)
plt.savefig('/home/xkchen/%s_Rt2Rv_compare.png' % R_str, dpi = 300)

plt.close()

