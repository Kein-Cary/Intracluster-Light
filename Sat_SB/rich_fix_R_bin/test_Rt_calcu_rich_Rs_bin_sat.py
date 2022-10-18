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

##.
from img_sat_Rt_estimate import Ms_to_Mh_func
from img_sat_Rt_estimate import Mh_c_func
from img_sat_Rt_estimate import Mh_rich_R_func

##.
from Gauss_Legendre_factor import GaussLegendreQuadrature
from Gauss_Legendre_factor import GaussLegendreQuad_arr

from colossus.cosmology import cosmology as co_cosmos
from colossus.halo import profile_nfw


### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100

Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

#. setting of cosmology (for density profile calculation)
params = {'flat': True, 'H0': 67.74, 'Om0': 0.311, 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95}
co_cosmos.addCosmology('myCosmo', params = params )
my_cosmo = co_cosmos.setCosmology( 'myCosmo' )

#.
pixel = 0.396

band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])
Mag_sun = [ 4.65, 5.11, 4.53 ]


### === ### function
def R_func( R, a, b, c):
    return a + (R + b)**c

def err_fit_func( p, x, y, yerr ):

    pa, pb, pc = p[:]

    pf = R_func( x, pa, pb, pc )
    delta = pf - y

    chi2 = np.sum( delta**2 / yerr**2 )

    if np.isfinite( chi2 ):
        return chi2
    return np.inf



### === ### match for subsample
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'
out_path = '/home/xkchen/figs_cp/theory_Rt/'

##. halo mass of satellites~( Li et al. 2016)
ref_sub_Mh = [ 11.37, 11.92, 12.64 ]
ref_Mh_err_0 = [ 0.35, 0.19, 0.12 ]
ref_Mh_err_1 = [ 0.35, 0.18, 0.11 ]

ref_sat_Ms = [ 10.68, 10.72, 10.78 ]
ref_R_edg = [ 0.1, 0.3, 0.6, 0.9 ]


Li_dat = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/Li_Mh2Mstar_data_point.csv')
Li_xerr = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/Li_Mh2Mstar_data_Xerr.csv')
Li_yerr = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/Li_Mh2Mstar_data_Yerr.csv')

Li_R = np.array( Li_dat['R'] )


##.
def Li_data_fit():

    ##. Mh -- R relation
    fit_err = 0.5 * ( np.array( ref_Mh_err_0 ) + np.array( ref_Mh_err_1 ) )

    ##. 
    popt, pcov = optimize.curve_fit( R_func, Li_R, ref_sub_Mh, p0 = np.array([9, 0.4, 0.66]), sigma = fit_err, absolute_sigma = True)
    a_fit, b_fit, c_fit = popt[:]

    new_R = np.logspace( -2, 1, 50 )
    new_F = R_func( new_R, a_fit, b_fit, c_fit )

    ##.
    keys = ['a', 'b', 'c']
    values = [ a_fit, b_fit, c_fit ]
    fill = dict( zip( keys, values) )
    out_data = pds.DataFrame( fill, index = ['k', 'v'])
    out_data.to_csv( '/home/xkchen/figs_cp/theory_Rt/Li_data/R_Mh_fit_params.csv',)


    fig = plt.figure()
    ax = fig.add_axes( [0.13, 0.10, 0.80, 0.85] )

    ax.errorbar( Li_R, ref_sub_Mh, yerr = [ref_Mh_err_1, ref_Mh_err_0 ], marker = '.', ls = '-', color = 'r',
                ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Li+2016',)

    ax.plot( new_R, new_F, 'b-', label = 'Fitting')

    ax.set_xlabel('$R_{p} \; [Mpc / h]$', fontsize = 12,)
    ax.set_ylabel('$\\lg M_{h} \; [M_{\\odot} / h]$', fontsize = 12)
    ax.set_xscale('log')
    ax.set_xlim( 0.09, 1.1 )
    ax.set_ylim( 10.5, 13 )

    ax.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.1 * np.ones( 100, ), x2 = 0.3 * np.ones( 100, ), color = 'r', alpha = 0.12,)
    ax.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.3 * np.ones( 100, ), x2 = 0.6 * np.ones( 100, ), color = 'g', alpha = 0.12,)
    ax.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.6 * np.ones( 100, ), x2 = 0.9 * np.ones( 100, ), color = 'b', alpha = 0.12,)

    ax.legend( loc = 4, frameon = False, fontsize = 12)
    ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

    plt.savefig('/home/xkchen/sub_Mh_Li.png', dpi = 300)
    plt.close()


    ##.
    po = [ 9.78, 0.34, 0.18 ]

    E_return = optimize.minimize( err_fit_func, x0 = np.array( po ), args = ( Li_R, ref_sat_Ms, fit_err), method = 'Powell',)
    popt = E_return.x

    sa_fit, sb_fit, sc_fit = popt[:]
    new_F = R_func( new_R, sa_fit, sb_fit, sc_fit )

    #.
    keys = ['a', 'b', 'c']
    values = [ sa_fit, sb_fit, sc_fit ]
    fill = dict( zip( keys, values) )
    out_data = pds.DataFrame( fill, index = ['k', 'v'])
    out_data.to_csv( '/home/xkchen/figs_cp/theory_Rt/Li_data/R_Mstar_fit_params.csv',)


    fig = plt.figure()
    ax = fig.add_axes( [0.13, 0.10, 0.80, 0.85] )

    ax.plot( Li_R, ref_sat_Ms, 'r-o', label = 'Li+2016',)

    ax.plot( new_R, R_func( new_R, sa_fit, sb_fit, sc_fit), 'b-', )

    ax.set_xlabel('$R_{p} \; [Mpc / h]$', fontsize = 12,)
    ax.set_ylabel('$\\lg M_{\\ast} \; [M_{\\odot} / h]$', fontsize = 12)
    ax.set_xscale('log')
    ax.set_xlim( 0.09, 1.1 )
    ax.set_ylim( 10.5, 11 )

    ax.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.1 * np.ones( 100, ), x2 = 0.3 * np.ones( 100, ), color = 'r', alpha = 0.12,)
    ax.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.3 * np.ones( 100, ), x2 = 0.6 * np.ones( 100, ), color = 'g', alpha = 0.12,)
    ax.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.6 * np.ones( 100, ), x2 = 0.9 * np.ones( 100, ), color = 'b', alpha = 0.12,)

    ax.legend( loc = 4, frameon = False, fontsize = 12)
    ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

    plt.savefig('/home/xkchen/sub_Mstar_Li.png', dpi = 300)
    plt.close()

    return

# Li_data_fit()


##.
def fig_mass_infer():

    R_str = 'scale'
    R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m

    bin_rich = [ 20, 30, 50, 210 ]
    sub_name = ['low-rich', 'medi-rich', 'high-rich']

    ##.
    marker_s = ['o', 's', '^']
    color_s = ['b', 'g', 'r', 'c', 'm']

    fig_name = []

    for dd in range( len(R_bins) - 1 ):

        if dd == 0:
            fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

        elif dd == len(R_bins) - 2:
            fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

        else:
            fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

    line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


    ##. Li's data fit params
    cat = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/R_Mh_fit_params.csv')
    a_fit, b_fit, c_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]

    cat = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/R_Mstar_fit_params.csv')
    sa_fit, sb_fit, sc_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]

    new_R = np.logspace( -2, 1, 50 )


    fig = plt.figure( figsize = ( 10, 5 ) )
    ax0 = fig.add_axes( [0.09, 0.10, 0.40, 0.85] )
    ax1 = fig.add_axes( [0.58, 0.10, 0.40, 0.85] )

    ax0.errorbar( Li_R, ref_sub_Mh, yerr = [ref_Mh_err_1, ref_Mh_err_0 ], marker = 'o', ls = '', color = 'k',
                ecolor = 'k', mfc = 'k', mec = 'k', capsize = 1.5, label = 'Li+2016',)
    ax0.plot( new_R, R_func( new_R, a_fit, b_fit, c_fit), 'k-', label = 'Fitting')

    ax0.set_xlabel('$R_{p} \; [Mpc / h]$', fontsize = 12,)
    ax0.set_ylabel('$\\lg M_{h} \; [M_{\\odot} / h]$', fontsize = 12)
    ax0.set_xscale('log')
    ax0.set_xlim( 3e-2, 1.1 )
    ax0.set_ylim( 10.8, 12.8 )

    ax0.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.1 * np.ones( 100, ), x2 = 0.3 * np.ones( 100, ), color = 'b', alpha = 0.10,)
    ax0.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.3 * np.ones( 100, ), x2 = 0.6 * np.ones( 100, ), color = 'g', alpha = 0.10,)
    ax0.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.6 * np.ones( 100, ), x2 = 0.9 * np.ones( 100, ), color = 'r', alpha = 0.10,)


    ax1.plot( Li_R, ref_sat_Ms, 'ko', label = 'Li+2016',)
    ax1.plot( new_R, R_func( new_R, sa_fit, sb_fit, sc_fit), 'k-', label = 'Fitting' )

    ax1.set_xlabel('$R_{p} \; [Mpc / h]$', fontsize = 12,)
    ax1.set_ylabel('$\\lg M_{\\ast} \; [M_{\\odot} / h]$', fontsize = 12)
    ax1.set_xscale('log')
    ax1.set_xlim( 3e-2, 1.1 )
    ax1.set_ylim( 10.6, 10.9 )

    ax1.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.1 * np.ones( 100, ), x2 = 0.3 * np.ones( 100, ), color = 'b', alpha = 0.10,)
    ax1.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.3 * np.ones( 100, ), x2 = 0.6 * np.ones( 100, ), color = 'g', alpha = 0.10,)
    ax1.fill_betweenx( y = np.linspace(8, 20, 100), x1 = 0.6 * np.ones( 100, ), x2 = 0.9 * np.ones( 100, ), color = 'r', alpha = 0.10,)


    for tt in range( len(R_bins) - 1 ):

        for ll in range( 3 ):

            dat = pds.read_csv( cat_path + 
                'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
                % (bin_rich[ll], bin_rich[ll+1], R_bins[tt], R_bins[tt+1]),)

            tt_bcg_z = np.array( dat['bcg_z'] )
            tt_Rsat = np.array( dat['R_sat'] )

            aveg_R_sat = np.median( tt_Rsat )   ##. Mpc / h
            aveg_R_std = np.std( tt_Rsat )

            aveg_Mh = R_func( aveg_R_sat, a_fit, b_fit, c_fit )    ##. M_sun / h
            aveg_Ms = R_func( aveg_R_sat, sa_fit, sb_fit, sc_fit ) ##. M_sun / h

            if tt == 0:
                ax0.errorbar( aveg_R_sat, aveg_Mh, xerr = aveg_R_std, marker = marker_s[ ll ], color = color_s[ ll ], 
                        ecolor = color_s[ ll ], mfc = 'none', mec = color_s[ ll ], capsize = 1.5, label = line_name[tt],)

                ax1.errorbar( aveg_R_sat, aveg_Ms, xerr = aveg_R_std, marker = marker_s[ ll ], color = color_s[ ll ], 
                        ecolor = color_s[ ll ], mfc = 'none', mec = color_s[ ll ], capsize = 1.5, label = line_name[ ll ], )

            else:
                ax0.errorbar( aveg_R_sat, aveg_Mh, xerr = aveg_R_std, marker = marker_s[ ll ], color = color_s[ ll ], 
                        ecolor = color_s[ ll ], mfc = 'none', mec = color_s[ ll ], capsize = 1.5, )

                ax1.errorbar( aveg_R_sat, aveg_Ms, xerr = aveg_R_std, marker = marker_s[ ll ], color = color_s[ ll ], 
                        ecolor = color_s[ ll ], mfc = 'none', mec = color_s[ ll ], capsize = 1.5, )

    ax0.legend( loc = 2, frameon = False, fontsize = 12)
    ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

    ax1.legend( loc = 2, frameon = False, fontsize = 12)
    ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

    plt.savefig('/home/xkchen/sat_Mh_Ms_infer.png', dpi = 300)
    plt.close()

    return

# fig_mass_infer()


#### ==== #### R_t estimation for subsamples

##. richness binned subsample
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']

rich_arr, M_200m, R_200m = [], [], []

C_arr = []
Mvir_arr = []
Rvir_arr = []
z_arr = []


for kk in range( 3 ):

    dat = pds.read_csv( cat_path + 'clust_rich_%s-%s_cat.csv' % (bin_rich[kk], bin_rich[kk+1]),)

    sub_rich = np.array( dat['rich'] )
    sub_z = np.array( dat['z'] )

    ##. M_sun / h, Mpc / h
    sub_lgMh = np.array( dat['lg_Mh'] )
    sub_Rv = np.array( dat['R_vir'] )

    rich_arr.append( sub_rich )
    M_200m.append( sub_lgMh )
    R_200m.append( sub_Rv )

    kk_Mh = 10**sub_lgMh

    aveg_z = np.mean( sub_z )
    aveg_Mh = np.mean( kk_Mh )
    aveg_c = Mh_c_func( aveg_z, aveg_Mh )

    C_arr.append( aveg_c )
    Mvir_arr.append( aveg_Mh )
    Rvir_arr.append( np.mean( sub_Rv ) )
    z_arr.append( aveg_z )


##. Li's data fit params
cat = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/R_Mh_fit_params.csv')
a_fit, b_fit, c_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]

cat = pds.read_csv('/home/xkchen/figs_cp/theory_Rt/Li_data/R_Mstar_fit_params.csv')
sa_fit, sb_fit, sc_fit = np.array( cat['a'] )[0], np.array( cat['b'] )[0], np.array( cat['c'] )[0]


R_str = 'scale'
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m


##.
marker_s = ['o', 's', '^']
color_s = ['b', 'g', 'r', 'c', 'm']

fig_name = []

for dd in range( len(R_bins) - 1 ):

    if dd == 0:
        fig_name.append( '$R \\leq %.2f \, R_{200m}$' % R_bins[dd + 1] )

    elif dd == len(R_bins) - 2:
        fig_name.append( '$R \\geq %.2f \, R_{200m}$' % R_bins[dd] )

    else:
        fig_name.append( '$%.2f \\leq R \\leq %.2f \, R_{200m}$' % (R_bins[dd], R_bins[dd + 1]),)

line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


##. radius binned satellites

for ll in range( 3 ):

    pp_Rt = []
    pp_alpha_k = []


    fig = plt.figure( figsize = ( 10, 5 ) )
    ax0 = fig.add_axes( [0.09, 0.10, 0.40, 0.85] )
    ax1 = fig.add_axes( [0.58, 0.10, 0.40, 0.85] )


    for tt in range( len(R_bins) - 1 ):

        dat = pds.read_csv( cat_path + 
            'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
            % (bin_rich[ll], bin_rich[ll+1], R_bins[tt], R_bins[tt+1]),)

        tt_bcg_ra, tt_bcg_dec = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] )
        tt_bcg_z = np.array( dat['bcg_z'] )
        tt_ra, tt_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

        tt_Rsat = np.array( dat['R_sat'] )
        tt_R2Rv = np.array( dat['R2Rv'] )

        aveg_R_sat = np.median( tt_Rsat )   ##. Mpc / h
        aveg_R_std = np.std( tt_Rsat )

        aveg_Mh = R_func( aveg_R_sat, a_fit, b_fit, c_fit )    ##. M_sun / h
        aveg_Ms = R_func( aveg_R_sat, sa_fit, sb_fit, sc_fit ) ##. M_sun / h

        aveg_z = np.mean( tt_bcg_z )
        aveg_c = Mh_c_func( aveg_z, 10**aveg_Mh )


        ##. density profile of satellite halo
        halo_nfw = profile_nfw.NFWProfile( M = Mvir_arr[ll], c = C_arr[ll], z = z_arr[ll], mdef = '200m')
        sub_nfw = profile_nfw.NFWProfile( M = 10**aveg_Mh, c = aveg_c, z = aveg_z, mdef = '200m')

        ##. kpc / h
        nr = 200
        r_bins = np.logspace( -3, 3.2, nr )

        ##. assume the r_bins is projected distance, calculate the 3D distance and background
        dR0 = 1e3 * aveg_R_sat    ##. kpc / h
        dR1 = np.sqrt( (1e3 * Rvir_arr[ll] )**2 - dR0**2 )

        x_mm = np.logspace( -3, np.log10( dR1 ), 200 )
        r_mm = np.sqrt( dR0**2 + x_mm**2 )

        pm_F_halo = halo_nfw.density( r_mm )
        pm_F_weit = halo_nfw.density( r_mm ) * r_mm 


        ##.
        rho_m_z = my_cosmo.rho_m( aveg_z )
        rho_delta = 200 * rho_m_z

        pm_rho_h = pm_F_halo / rho_delta
        pm_rho_weit = pm_F_weit / rho_delta

        order = 7

        [ans, err] = GaussLegendreQuad_arr( x_mm, pm_rho_h, order, 0, x_mm[-1] )
        I0 = ans * rho_delta

        [ans, err] = GaussLegendreQuad_arr( x_mm, pm_rho_weit, order, 0, x_mm[-1] )
        I1 = ans * rho_delta

        ##. 3D centric distance of subhalo
        tag_R = I1 / I0


        ##.
        enclos_Mh_sat = sub_nfw.enclosedMass( r_bins )
        aveg_rho_sat = 3 * enclos_Mh_sat / ( 4 * np.pi * r_bins**3 )

        tag_enclos_mass = halo_nfw.enclosedMass( tag_R )
        tag_BG = 3 * tag_enclos_mass / (4 * np.pi * tag_R**3 )

        ##. slope of the host halo density profile
        r_enclos_M = halo_nfw.enclosedMass( r_bins )

        mean_rho = 3 * r_enclos_M / ( 4 * np.pi * r_bins**3 )

        ln_rho = np.log( mean_rho )
        ln_R = np.log( r_bins )

        diff_x = np.diff( ln_R )
        diff_y = np.diff( ln_rho )
        slop_Mh = diff_y / diff_x

        tmp_k_F = interp.interp1d( r_bins[1:], slop_Mh, kind = 'cubic', fill_value = 'extrapolate',)
        alpha_k = tmp_k_F( tag_R )

        ##.
        tmp_F = interp.interp1d( r_bins, aveg_rho_sat, kind = 'cubic', fill_value = 'extrapolate')
        c_rt = inversefunc( tmp_F, np.abs( alpha_k ) * tag_BG )

        #. alpha = 1 case
        # c_rt = inversefunc( tmp_F, 1.0 * tag_BG )


        pp_Rt.append( c_rt.min() )
        pp_alpha_k.append( alpha_k.min() )


        ##. figure results of subsamples
        ax0.plot( r_bins, aveg_rho_sat, ls = '-', color = color_s[tt], alpha = 0.75, label = fig_name[tt],)

        ax0.axhline( y = np.abs( alpha_k.min() ) * tag_BG, ls = ':', color = color_s[tt],)
        # ax0.axhline( y = 1.0 * tag_BG, ls = ':', color = color_s[tt],)   ##. alpha = 1 case

        ax0.axvline( x = c_rt, ls = '--', color = color_s[tt], ymin = 0.9, ymax = 1.)

        ax1.scatter( 0.5 * ( R_bins[tt] + R_bins[tt+1]), alpha_k, marker = 's', s = 75, color = color_s[tt], label = fig_name[tt],)


    ##. save the tidal radius
    keys = [ '%s' % ll for ll in fig_name ]
    values = pp_Rt
    fill = dict( zip( keys, values ), index = ('k', 'v') )
    data = pds.DataFrame( fill )
    data.to_csv( out_path + 
        'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_Rscal-bin-sat_Rt.csv' % (bin_rich[ll], bin_rich[ll+1]),)

    # data.to_csv( out_path + 
    #     'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_Rscal-bin-sat_Rt-alpha=1.csv' % (bin_rich[ll], bin_rich[ll+1]),)


    ax1.plot( 0.5 * ( R_bins[1:] + R_bins[:-1]), pp_alpha_k, 'k-', alpha = 0.65)
    ax1.legend( loc = 1, frameon = False, fontsize = 12,)
    ax1.set_xlabel('$R_{sat} / R_{200m}$', fontsize = 12,)
    ax1.set_ylim( -2.2, -1.55 )
    ax1.set_ylabel('$\\alpha \, = \, d \, \\ln \, \\bar{\\rho} \, / \, d \, \\ln \, r $', fontsize = 12,)
    ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
    ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )

    ax0.annotate( s = line_name[ll], xy = (0.65, 0.85), xycoords = 'axes fraction', fontsize = 12,)
    ax0.legend( loc = 3, frameon = False, fontsize = 12,)
    ax0.set_xlim( 1, 2e2 )
    ax0.set_xscale('log')
    ax0.set_xlabel('$r[kpc / h]$', fontsize = 12,)
    ax0.set_ylim( 1e3, 4e8 )
    ax0.set_yscale('log')
    ax0.set_ylabel('$\\bar{\\rho}(r) [M_\odot h^{2} kpc^{-3}]$', fontsize = 12,)
    ax0.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

    plt.savefig( '/home/xkchen/%s_halo_sat_ebclose_3D_aveg_rho.png' % sub_name[ll], dpi = 300)
    plt.close()

    print( ll )




### === Rt compare
marks = ['s', '>', 'o']
mark_size = [10, 25, 35]
color_s = ['b', 'g', 'r', 'm']

##.
crit_eta = [ 0.05, 0.15, 0.25, 0.50, 0.75, 0.90, 0.95 ]

for oo in range( len( crit_eta ) ):

    id_set = oo

    fig = plt.figure()
    ax1 = fig.add_axes( [0.12, 0.11, 0.80, 0.85] )

    for qq in range( 3 ):

        ##. read sample catalog and get the average centric distance
        R_aveg = []
        R_sat_arr = []

        for dd in range( len( R_bins ) - 1 ):

            if R_str == 'phy':
                cat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv'
                                % ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

                x_Rc = np.array( cat['R_sat'] )   ## Mpc / h
                cp_x_Rc = x_Rc * 1e3 * a_ref / h  ## kpc

            if R_str == 'scale':
                cat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
                                % ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

                x_Rc = np.array( cat['R2Rv'] )   ## R / R200m
                cp_x_Rc = x_Rc + 0

            R_aveg.append( np.mean( cp_x_Rc) )
            R_sat_arr.append( cp_x_Rc )

        print( R_aveg )
        print( [ len(ll) for ll in R_sat_arr ] )

        ##. R_t and R_t_err 
        Rc = []

        pat = pds.read_csv( out_path + 
            'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_Rscal-bin-sat_Rt.csv' % (bin_rich[qq], bin_rich[qq+1]),)

        for dd in range( len(R_bins) - 1 ):

            tt_Rc = np.array( pat[ fig_name[dd] ] )[0]

            Rc.append( tt_Rc )


        ##. estimate with ratio decrease
        # pat = fits.open( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/' + 
        #                 'Extend_BCGM_gri-common_%s_%s_r-band_smooth-exten_Rt_test.fits' % (sub_name[qq], R_str),)

        pat = fits.open( '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/' + 
                        'Extend_BCGM_gri-common_%s_%s_r-band_polyfit_Rt_test.fits' % (sub_name[qq], R_str),)

        p_table = pat[1].data

        cp_Rc = []

        for tt in range( len(R_bins) - 1 ):

            Rt_arr = np.array( p_table[ fig_name[tt] ] )

            cp_Rc.append( Rt_arr[ id_set ] )


        ax1.plot( R_aveg, Rc, marker = marks[qq], ls = '-', color = color_s[qq], label = line_name[qq], alpha = 0.75,)

        if qq == 0:
            ax1.plot( R_aveg[:4], cp_Rc[:4], marker = marks[qq], ls = '--', color = color_s[qq], alpha = 0.45, label = 'Ratio decrease')

        else:
            ax1.plot( R_aveg[:4], cp_Rc[:4], marker = marks[qq], ls = '--', color = color_s[qq], alpha = 0.45,)

    ax1.legend( loc = 2, frameon = False, fontsize = 12)
    ax1.annotate( s = 'Decrease by %.2f' % crit_eta[ id_set ], xy = (0.60, 0.05), xycoords = 'axes fraction', fontsize = 12,)

    ax1.set_ylabel('$ R_{t} \; [kpc \, / \, h]$', fontsize = 12,)
    # ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.set_yscale('log')

    ax1.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$', fontsize = 12,)
    ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
    ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

    plt.savefig('/home/xkchen/%s_Rt_%.2f-ratio-decay_compare.png' % (R_str, crit_eta[id_set]), dpi = 300)
    plt.close()

raise


##.
fig = plt.figure()
# fig = plt.figure( figsize = (5.8, 5.4) )
ax1 = fig.add_axes( [0.15, 0.32, 0.83, 0.63] )
sub_ax1 = fig.add_axes( [0.15, 0.11, 0.83, 0.21] )

for qq in range( 3 ):

    ##. read sample catalog and get the average centric distance
    R_aveg = []
    R_sat_arr = []

    for dd in range( len( R_bins ) - 1 ):

        if R_str == 'phy':
            cat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv'
                            % ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

            x_Rc = np.array( cat['R_sat'] )   ## Mpc / h
            cp_x_Rc = x_Rc * 1e3 * a_ref / h  ## kpc

        if R_str == 'scale':
            cat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' 
                            % ( bin_rich[qq], bin_rich[qq + 1], R_bins[dd], R_bins[dd + 1]),)

            x_Rc = np.array( cat['R2Rv'] )   ## R / R200m
            cp_x_Rc = x_Rc + 0

        R_aveg.append( np.mean( cp_x_Rc) )
        R_sat_arr.append( cp_x_Rc )

    print( R_aveg )
    print( [ len(ll) for ll in R_sat_arr ] )


    ##. R_t and R_t_err 
    Rc = []

    pat = pds.read_csv( out_path + 
        'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_Rscal-bin-sat_Rt.csv' % (bin_rich[qq], bin_rich[qq+1]),)

    for dd in range( len(R_bins) - 1 ):

        tt_Rc = np.array( pat[ fig_name[dd] ] )[0]

        Rc.append( tt_Rc )

    Rc = np.array( Rc )


    ##. R_t for alpha = 1
    mm_Rt = []

    pat = pds.read_csv( out_path + 
        'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_Rscal-bin-sat_Rt-alpha=1.csv' % (bin_rich[qq], bin_rich[qq+1]),)

    for dd in range( len(R_bins) - 1 ):

        tt_Rc = np.array( pat[ fig_name[dd] ] )[0]

        mm_Rt.append( tt_Rc )

    mm_Rt = np.array( mm_Rt )


    ax1.plot( R_aveg, Rc, marker = marks[qq], ls = '-', color = color_s[qq], label = line_name[qq], alpha = 0.75,)

    ax1.scatter( R_aveg, mm_Rt, marker = marks[qq], facecolors = 'none', edgecolors = color_s[qq],)
    ax1.plot( R_aveg, mm_Rt, ls = '--', color = color_s[qq], label = line_name[qq] + ', $\\alpha{=}1$', alpha = 0.75,)

    sub_ax1.plot( R_aveg, (mm_Rt - Rc) / mm_Rt, ls = '-', color = color_s[qq], alpha = 0.75,)

ax1.legend( loc = 2, frameon = False, fontsize = 12)

ax1.set_ylabel('$ R_{t} \; [kpc \, / \, h]$', fontsize = 12, labelpad = 12,)
ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

ax1.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$', fontsize = 12,)
ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

sub_ax1.set_ylabel('$ (R_{t, \; \\alpha{=}1 } - R_{t}) \, / \, R_{t, \; \\alpha{=}1 }$')
sub_ax1.yaxis.set_minor_locator( ticker.AutoMinorLocator() )

sub_ax1.set_xlim( ax1.get_xlim() )
sub_ax1.set_xlabel('$\\bar{R}_{sat} \, / \, R_{200m}$', fontsize = 12,)
sub_ax1.xaxis.set_minor_locator( ticker.AutoMinorLocator() )
sub_ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

ax1.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/%s_Rt_compare_to_alpha=1.png' % R_str, dpi = 300)
plt.close()

