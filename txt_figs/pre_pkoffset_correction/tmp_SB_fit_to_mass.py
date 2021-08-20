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

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle

from light_measure import cov_MX_func
from fig_out_module import BG_sub_cov_func, BG_pro_cov

from color_2_mass import get_c2mass_func
from color_2_mass import gi_band_m2l_func, gr_band_m2l_func, ri_band_m2l_func

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from fig_out_module import arr_jack_func
from fig_out_module import color_func

### === ### cosmology model
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

def sersic_func(r, Ie, re, ndex):
    belta = 2 * ndex - 0.324
    fn = -1 * belta * ( r / re )**(1 / ndex) + belta
    Ir = Ie * np.exp( fn )
    return Ir

def com_sersic(r, I1, r1, n1, I2, r2, n2):
    sb_1 = sersic_func(r, I1, r1, n1)
    sb_2 = sersic_func(r, I2, r2, n2)
    sb = sb_1 + sb_2
    return sb

### === 2sersic, free params
def err_fit_func(p, r, y, params, yerr):
    I1, r1, n1, I2, r2, n2 = p[:]

    cov_mx = params[0]
    cov_inv = np.linalg.pinv( cov_mx )
    model_sb = com_sersic(r, I1, r1, n1, I2, r2, n2)
    delta = model_sb - y

    chi2 = delta.T.dot( cov_inv ).dot(delta)
    # chi2 = np.sum( delta**2 / yerr**2 )
    return chi2

def cc_err_fit_func(p, r, y, params, yerr):
    pt_i, pt_r, pt_n = p[:]
    fx_i, fx_r, fx_n, cov_mx = params[:]

    cov_inv = np.linalg.pinv( cov_mx )
    model_sb = com_sersic(r, pt_i, pt_r, pt_n, fx_i, fx_r, fx_n)
    delta = model_sb - y

    chi2 = delta.T.dot( cov_inv ).dot(delta)
    # chi2 = np.sum( delta**2 / yerr**2 )
    return chi2

### === 2sersic, fix outter n = 2.1
def fix2_err_fit_func(p, r, y, params, yerr):
    I1, r1, n1, I2, r2 = p[:]

    n2, cov_mx = params[:]
    cov_inv = np.linalg.pinv( cov_mx )
    model_sb = com_sersic(r, I1, r1, n1, I2, r2, n2)
    delta = model_sb - y

    chi2 = delta.T.dot( cov_inv ).dot(delta)
    # chi2 = np.sum( delta**2 / yerr**2 )
    return chi2

def cc_fix2_err_fit_func(p, r, y, params, yerr):
    pt_i, pt_r, pt_n = p[:]
    fx_i, fx_r, fx_n, cov_mx = params[:]

    cov_inv = np.linalg.pinv( cov_mx )
    model_sb = com_sersic(r, pt_i, pt_r, pt_n, fx_i, fx_r, fx_n)
    delta = model_sb - y

    chi2 = delta.T.dot( cov_inv ).dot(delta)
    # chi2 = np.sum( delta**2 / yerr**2 )
    return chi2

### === ### cov_arr

mass_id = True

##mass-bin
if mass_id == True:
    path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin/'
    BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/mass_bin_BG/'

    cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
    fig_name = ['low $M_{\\ast}$', 'high $M_{\\ast}$']


##age-bin
if mass_id == False:
    path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin/'
    BG_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/age_bin_BG/'

    fig_name = [ 'younger', 'older' ]
    cat_lis = [ 'younger', 'older' ]


cov_path = '/home/xkchen/mywork/ICL/code/rig_common_cat/cov_arr/'

color_s = ['r', 'g', 'b']
line_c = ['b', 'r']
### 
nbg_low_r, nbg_low_sb, nbg_low_err = [], [], []
nbg_low_mag, nbg_low_mag_err = [], []

for kk in range( 3 ):
    with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[0], band[kk]), 'r') as f:
        tt_r = np.array(f['r'])
        tt_sb = np.array(f['sb'])
        tt_err = np.array(f['sb_err'])

    nbg_low_r.append( tt_r )
    nbg_low_sb.append( tt_sb )
    nbg_low_err.append( tt_err )

    mag_arr = 22.5 - 2.5 * np.log10( tt_sb )
    mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

    nbg_low_mag.append( mag_arr )
    nbg_low_mag_err.append( mag_err )

nbg_hi_r, nbg_hi_sb, nbg_hi_err = [], [], []
nbg_hi_mag, nbg_hi_mag_err = [], []

for kk in range( 3 ):
    with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[1], band[kk]), 'r') as f:
        tt_r = np.array(f['r'])
        tt_sb = np.array(f['sb'])
        tt_err = np.array(f['sb_err'])

    nbg_hi_r.append( tt_r )
    nbg_hi_sb.append( tt_sb )
    nbg_hi_err.append( tt_err )

    mag_arr = 22.5 - 2.5 * np.log10( tt_sb )
    mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

    nbg_hi_mag.append( mag_arr )
    nbg_hi_mag_err.append( mag_err )

### === ### fitting for low mass bin, g band
'''
mass_dex = 0 # test only for low mass sample or younger sample

for tt in range( 3 ):

    if mass_dex == 1:
        com_r, com_sb, com_err = nbg_hi_r[tt], nbg_hi_sb[tt], nbg_hi_err[tt]
        idvx = (com_r <= 1.0e3) & (com_r >= 10)
        fx, fy, ferr = com_r[idvx], com_sb[idvx], com_err[idvx]

    if mass_dex == 0:
        com_r, com_sb, com_err = nbg_low_r[tt], nbg_low_sb[tt], nbg_low_err[tt]
        idvx = (com_r <= 1.0e3) & (com_r >= 10)
        fx, fy, ferr = com_r[idvx], com_sb[idvx], com_err[idvx]

    ## cov_arr
    with h5py.File( cov_path + '%s_%s-band_BG-sub_cov-cor_arr.h5' % (cat_lis[mass_dex], band[tt]), 'r') as f:
        cov_MX = np.array(f['cov_MX'])


    Ne2 = 2.1
    params_arr = [ Ne2, cov_MX ]
    mu0, re_0, ne_0, mu1, re_1 = 5e-1, 20, 4, 2e-3, 500
    po = np.array( [ mu0, re_0, ne_0, mu1, re_1 ] )

    bonds = ( (0, 5e0), (5, 100), (1, 10), (0, 1e-2), (1e2, 3e3) )

    E_return = optimize.minimize( fix2_err_fit_func, x0 = np.array(po), args = (fx, fy, params_arr, ferr), method = 'L-BFGS-B', bounds = bonds,)

    popt = E_return.x

    print( E_return )
    print( popt )

    n_dim = len( po )
    Ie1, Re1, Ne1, Ie2, Re2 = popt

    fit_line = com_sersic( com_r, Ie1, Re1, Ne1, Ie2, Re2, Ne2)
    fit_1 = sersic_func( com_r, Ie1, Re1, Ne1 )
    fit_2 = sersic_func( com_r, Ie2, Re2, Ne2 )

    ## chi2 arr
    cov_inv = np.linalg.pinv( cov_MX )
    delta_v = fit_line[idvx] - fy

    chi2 = delta_v.T.dot( cov_inv ).dot( delta_v )
    chi2nv = chi2 / ( len(fy) - n_dim )


    ## component beside
    resi_I = com_sb - fit_1

    id_rx = ( com_r > 6 ) & ( com_r <= 1e3 )

    lim_I = resi_I[ id_rx ]
    lim_err = com_err[ id_rx ]

    lim_po = np.array( [ Ie2, Re2, Ne2 ] )
    popt, pcov = optimize.curve_fit( sersic_func, com_r[ id_rx ], lim_I, p0 = lim_po, sigma = lim_err, method = 'trf')

    print( popt )

    fit_2_c = sersic_func( com_r, popt[0], popt[1], popt[2] )
    fit_line_c = fit_2_c + fit_1

    delt_c = fit_line_c[ idvx ] - fy
    chi2_c = delt_c.dot( cov_inv ).dot( delt_c )
    chi2nv_c = chi2_c / ( len(fy) - n_dim )

    ## save the params
    # keys = [ 'Ie_0', 'Re_0', 'ne_0', 'Ie_1', 'Re_1', 'ne_1' ]
    # values = [ Ie1, Re1, Ne1, popt[0], popt[1], popt[2] ]
    # fill = dict(zip( keys, values) )
    # out_data = pds.DataFrame( fill, index = ['k', 'v'])
    # out_data.to_csv('/home/xkchen/figs/%s_%s-band_2-sersic_fit.csv' % (cat_lis[mass_dex], band[tt]) )


    plt.figure()
    ax = plt.subplot(111)
    ax.set_title( fig_name[mass_dex] + ', %s band' % band[tt])

    ax.plot(com_r, com_sb, ls = '-', color = 'k', alpha = 0.5, label = 'signal')
    ax.fill_between(com_r, y1 = com_sb - com_err, y2 = com_sb + com_err, color = 'k', alpha = 0.12,)

    ax.plot(com_r, fit_1, ls = '--', color = 'r', alpha = 0.5,)
    # ax.plot(com_r, fit_2, ls = ':', color = 'r', alpha = 0.5,)
    # ax.plot(com_r, fit_line, ls = '-', color = 'r', alpha = 0.5, label = 'fitting',)

    # ax.plot( com_r, resi_I, ls = '-', color = 'b', alpha = 0.5,)

    ax.plot( com_r, fit_2_c, ls = '--', color = 'g', alpha = 0.5,)
    ax.plot( com_r, fit_line_c, ls = '-', alpha = 0.5, color = 'g',)

    ax.text(1e2, 5e-1, s = '$I_{e} = %.5f$' % Ie1 + ',$R_{e} = %.3f$' % Re1 + '\n' + '$n_{0} = %.3f$' % Ne1, color = 'g', fontsize = 8,)
    # ax.text(1e2, 1e-1, s = '$I_{e} = %.5f$' % Ie2 + ',$R_{e} = %.3f$' % Re2 + '\n' + ',$n_{0} = %.3f$' % Ne2, color = 'r', fontsize = 8,)
    # ax.text(1e2, 5e-2, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nv, color = 'b',)

    ax.text(1e2, 1e-1, s = '$I_{e} = %.5f$' % popt[0] + ',$R_{e} = %.3f$' % popt[1] + '\n' + ',$n_{0} = %.3f$' % popt[2], color = 'r', fontsize = 8,)
    ax.text(1e2, 5e-2, s = '$\\chi^{2} / \\nu = %.5f$' % chi2nv_c, color = 'b',)

    ax.set_xlim(1e0, 2e3)
    ax.set_ylim(1e-5, 1e1)
    ax.set_yscale('log')
    ax.legend( loc = 1)
    ax.set_xscale('log')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
    ax.grid(which = 'both', axis = 'both', alpha = 0.25,)

    plt.subplots_adjust( left = 0.15,)
    plt.savefig('/home/xkchen/%s_%s-band_lim-out-n=2.1_BG-sub-SB_fit.jpg' % (cat_lis[mass_dex], band[tt]), dpi = 300)
    plt.close()
'''

### === ### mass estimation based on fitting params
def err_L_func( lumi_arr, fdens_arr, fdens_err):

    mag_err = 2.5 * fdens_err / ( np.log(10) * fdens_arr )

    err_L = np.log( 10 ) * lumi_arr * 0.4 * mag_err

    return err_L

def err_M_func( color_arr, lumi_arr, b_factor, color_err, lumi_err, band_str):

    if band_str == 'gi':
        lg_m2l = gi_band_m2l_func( color_arr, lumi_arr )

    if band_str == 'gr':
        lg_m2l = gr_band_m2l_func( color_arr, lumi_arr )

    if band_str == 'ri':
        lg_m2l = ri_band_m2l_func( color_arr, lumi_arr )

    _m2l = 10**( lg_m2l )
    err_M = (1 / h**2) * np.sqrt( _m2l**2 * lumi_err**2 + lumi_arr**2 * _m2l**2 * np.log(10)**2 * b_factor**2 * color_err**2 )

    return err_M

param_path = '/home/xkchen/jupyter/tmp_sersic/'
fit2M_path = '/home/xkchen/jupyter/tmp_fit_SB_to_M/'

Dl_ref = Test_model.luminosity_distance( z_ref ).value
'''
for mm in range( 2 ):

    tmp_r, tmp_sb, tmp_err = [], [], []
    tmp_cen_sb, tmp_out_sb = [], []

    for kk in range( 3 ):

        ## measured data
        with h5py.File( BG_path + 'photo-z_%s_%s-band_BG-sub_SB.h5' % (cat_lis[mm], band[kk]), 'r') as f:
            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        idvx = ( tt_r >= 10 ) & ( tt_r <= 1e3)

        ## load the fitting params
        # ... mass-bin
        if mass_id == True:
            if mm == 0: ## low mass bin
                m_dat = pds.read_csv( param_path + 'speci_fit_low_mass/%s_%s-band_2-sersic_fit.csv' % (cat_lis[mm], band[kk]),)
                ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                                                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])
            if mm == 1:
                m_dat = pds.read_csv( param_path + '%s_%s-band_lim-n=2.1_mcmc_fit.csv' % (cat_lis[mm], band[kk]),)
                ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                                                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])

        #...age bin
        if mass_id == False:
            if mm == 0:
                # m_dat = pds.read_csv( param_path + 'speci_fit_low_mass/%s_%s-band_2-sersic_fit.csv' % (cat_lis[mm], band[kk]),)
                m_dat = pds.read_csv( param_path + '%s_%s-band_lim-n=2.1_mcmc_fit_separate.csv' % (cat_lis[mm], band[kk]),)
                ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                                                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])

            if mm == 1:
                m_dat = pds.read_csv( param_path + '%s_%s-band_lim-n=2.1_mcmc_fit.csv' % (cat_lis[mm], band[kk]),)
                ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                                                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])
        ## fitting SB
        fit_sb_arr = com_sersic( tt_r[idvx], m_I0, m_R0, m_n0, m_I1, m_R1, m_n1)

        fit_cen_arr = sersic_func( tt_r[idvx], m_I0, m_R0, m_n0 )
        fit_out_arr = sersic_func( tt_r[idvx], m_I1, m_R1, m_n1 )

        tmp_r.append( tt_r[idvx] )
        tmp_sb.append( fit_sb_arr )
        tmp_err.append( tt_err[idvx] )

        tmp_cen_sb.append( fit_cen_arr )
        tmp_out_sb.append( fit_out_arr )

    ##estimate the err of Lumi profile and mass profile

    ## ... gi-band
    p_g2i, p_g2i_err = color_func( tmp_sb[1], tmp_err[1], tmp_sb[2], tmp_err[2] )

    p_i_mag = 22.5 - 2.5 * np.log10( tmp_sb[2] )
    p_i_Mag = p_i_mag - 5 * np.log10( Dl_ref * 10**6 / 10)

    out_lm_file = fit2M_path + '%s_gi-band-based_fit-based_mass-Lumi.csv' % cat_lis[mm]
    band_str = 'gi'
    get_c2mass_func( tmp_r[2], band_str, p_i_Mag, p_g2i, z_ref, out_file = out_lm_file )

    p_dat = pds.read_csv( out_lm_file )
    p_surf_mass = np.array( p_dat['surf_mass'] )
    p_surf_lumi = np.array( p_dat['lumi'] )

    b_g2i = 0.518

    err_Lumi = err_L_func( p_surf_lumi, tmp_sb[2], tmp_err[2] )
    err_mass = err_M_func( p_g2i, p_surf_lumi, b_g2i, p_g2i_err, err_Lumi, band_str)

    keys = [ 'R', 'surf_mass', 'surf_mass_err', 'lumi', 'lumi_err' ]
    values = [ tmp_r[2], p_surf_mass, err_mass, p_surf_lumi, err_Lumi ]
    fill = dict(zip( keys, values) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( fit2M_path + '%s_gi-band-based_fit-based_mass-Lumi_with-err.csv' % cat_lis[mm],)

    ## ... gr-band
    if (mm == 0) & (mass_id == False): ## age-bin case
        tmp_err[0] = tmp_err[0][:-1]
        tmp_r[0] = tmp_r[0][:-1]
        tmp_sb[0] = tmp_sb[0][:-1]

    p_g2r, p_g2r_err = color_func( tmp_sb[1], tmp_err[1], tmp_sb[0], tmp_err[0] )

    p_r_mag = 22.5 - 2.5 * np.log10( tmp_sb[0] )
    p_r_Mag = p_r_mag - 5 * np.log10( Dl_ref * 10**6 / 10)

    out_lm_file_0 = fit2M_path + '%s_gr-band-based_fit-based_mass-Lumi.csv' % cat_lis[mm]
    band_str = 'gr'
    get_c2mass_func( tmp_r[0], band_str, p_i_Mag, p_g2i, z_ref, out_file = out_lm_file_0 )

    p_dat = pds.read_csv( out_lm_file_0 )
    r_surf_mass = np.array( p_dat['surf_mass'] )
    r_surf_lumi = np.array( p_dat['lumi'] )

    b_g2r = 1.097

    err_Lumi = err_L_func( r_surf_lumi, tmp_sb[0], tmp_err[0] )
    err_mass = err_M_func( p_g2r, r_surf_lumi, b_g2r, p_g2r_err, err_Lumi, band_str)

    keys = [ 'R', 'surf_mass', 'surf_mass_err', 'lumi', 'lumi_err' ]
    values = [ tmp_r[0], r_surf_mass, err_mass, r_surf_lumi, err_Lumi ]
    fill = dict(zip( keys, values) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( fit2M_path + '%s_gr-band-based_fit-based_mass-Lumi_with-err.csv' % cat_lis[mm],)

    ## ...ri-band
    p_r2i, p_r2i_err = color_func( tmp_sb[0], tmp_err[0], tmp_sb[2], tmp_err[2] )
    out_lm_file_1 = fit2M_path + '%s_ri-band-based_fit-based_mass-Lumi.csv' % cat_lis[mm]
    band_str = 'ri'
    get_c2mass_func( tmp_r[2], band_str, p_i_Mag, p_r2i, z_ref, out_file = out_lm_file_1 )

    p_dat = pds.read_csv( out_lm_file_1 )
    p_surf_mass = np.array( p_dat['surf_mass'] )
    p_surf_lumi = np.array( p_dat['lumi'] )

    b_r2i = 1.114

    err_Lumi = err_L_func( p_surf_lumi, tmp_sb[2], tmp_err[2] )
    err_mass = err_M_func( p_r2i, p_surf_lumi, b_r2i, p_r2i_err, err_Lumi, band_str)

    keys = [ 'R', 'surf_mass', 'surf_mass_err', 'lumi', 'lumi_err' ]
    values = [ tmp_r[2], p_surf_mass, err_mass, p_surf_lumi, err_Lumi ]
    fill = dict(zip( keys, values) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( fit2M_path + '%s_ri-band-based_fit-based_mass-Lumi_with-err.csv' % cat_lis[mm],)    
'''

### === ### figs
cat_lis_0 = ['younger', 'older']
fig_name_0 = ['younger', 'older']

cat_lis_1 = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
fig_name_1 = ['low $M_{\\ast}^{BCG}$', 'high $M_{\\ast}^{BCG}$']

'''
## ... gi-band, gr-band, ri-band
for band_str in ('gi', 'gr', 'ri'):

    tmp_m_R, tmp_m_Lumi, tmp_m_Mass = [], [], []
    tmp_m_L_err, tmp_m_M_err = [], []

    tmp_a_R, tmp_a_Lumi, tmp_a_Mass = [], [], []
    tmp_a_L_err, tmp_a_M_err = [], []

    for mm in range( 2 ):

        c_dat = pds.read_csv( fit2M_path + '%s_%s-band-based_fit-based_mass-Lumi_with-err.csv' % (cat_lis_1[mm], band_str),)

        tmp_m_R.append( np.array(c_dat['R']) )

        tmp_m_Lumi.append( np.array(c_dat['lumi']) )
        tmp_m_L_err.append( np.array(c_dat['lumi_err']) )

        tmp_m_Mass.append( np.array(c_dat['surf_mass']) )
        tmp_m_M_err.append( np.array(c_dat['surf_mass_err']) )

        c_dat = pds.read_csv( fit2M_path + '%s_%s-band-based_fit-based_mass-Lumi_with-err.csv' % (cat_lis_0[mm], band_str),)

        tmp_a_R.append( np.array(c_dat['R']) )

        tmp_a_Lumi.append( np.array(c_dat['lumi']) )
        tmp_a_L_err.append( np.array(c_dat['lumi_err']) )

        tmp_a_Mass.append( np.array(c_dat['surf_mass']) )
        tmp_a_M_err.append( np.array(c_dat['surf_mass_err']) )


    plt.figure()
    plt.title( '%s-band based M(r) estimate' % band_str )
    # plt.plot( tmp_m_R[0], tmp_m_Mass[0], ls = '--', color = line_c[0], alpha = 0.5, label = fig_name_1[0],)
    # plt.fill_between( tmp_m_R[0], y1 = tmp_m_Mass[0] - tmp_m_M_err[0], y2 = tmp_m_Mass[0] + tmp_m_M_err[0], color = line_c[0], alpha = 0.12,)

    # plt.plot( tmp_m_R[1], tmp_m_Mass[1], ls = '--', color = line_c[1], alpha = 0.5, label = fig_name_1[1],)
    # plt.fill_between( tmp_m_R[1], y1 = tmp_m_Mass[1] - tmp_m_M_err[1], y2 = tmp_m_Mass[1] + tmp_m_M_err[1], color = line_c[1], alpha = 0.12,)

    plt.plot( tmp_a_R[0], tmp_a_Mass[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name_0[0],)
    plt.fill_between( tmp_a_R[0], y1 = tmp_a_Mass[0] - tmp_a_M_err[0], y2 = tmp_a_Mass[0] + tmp_a_M_err[0], color = line_c[0], alpha = 0.12,)

    plt.plot( tmp_a_R[1], tmp_a_Mass[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name_0[1],)
    plt.fill_between( tmp_a_R[1], y1 = tmp_a_Mass[1] - tmp_a_M_err[1], y2 = tmp_a_Mass[1] + tmp_a_M_err[1], color = line_c[1], alpha = 0.12,)

    plt.xlim(1e1, 1e3)
    plt.xscale('log')
    plt.xlabel('R[kpc]', fontsize = 15,)
    plt.yscale('log')

    plt.ylim(1e3, 2e8)

    if band_str == 'gr':
        plt.ylim(1e3, 2e9)
    plt.legend( loc = 1, frameon = False, fontsize = 15,)

    plt.ylabel('$\\Sigma [M_{\\odot} / kpc^2]$', fontsize = 15,)
    plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
    plt.savefig('/home/xkchen/figs/mass-age-bin_%s-band_fit-based_surface_mass_profile.png' % band_str, dpi = 300,)
    plt.close()


    plt.figure()
    # plt.plot( tmp_m_R[0], tmp_m_Lumi[0], ls = '--', color = line_c[0], alpha = 0.5, label = fig_name_1[0])
    # plt.fill_between( tmp_m_R[0], y1 = tmp_m_Lumi[0] - tmp_m_L_err[0], y2 = tmp_m_Lumi[0] + tmp_m_L_err[0], color = line_c[0], alpha = 0.12,)

    # plt.plot( tmp_m_R[1], tmp_m_Lumi[1], ls = '--', color = line_c[1], alpha = 0.5, label = fig_name_1[1])
    # plt.fill_between( tmp_m_R[1], y1 = tmp_m_Lumi[1] - tmp_m_L_err[1], y2 = tmp_m_Lumi[1] + tmp_m_L_err[1], color = line_c[1], alpha = 0.12,)

    plt.plot( tmp_a_R[0], tmp_a_Lumi[0], ls = '-', color = line_c[0], alpha = 0.5, label = fig_name_0[0])
    plt.fill_between( tmp_a_R[0], y1 = tmp_a_Lumi[0] - tmp_a_L_err[0], y2 = tmp_a_Lumi[0] + tmp_a_L_err[0], color = line_c[0], alpha = 0.12,)

    plt.plot( tmp_a_R[1], tmp_a_Lumi[1], ls = '-', color = line_c[1], alpha = 0.5, label = fig_name_0[1])
    plt.fill_between( tmp_a_R[1], y1 = tmp_a_Lumi[1] - tmp_a_L_err[1], y2 = tmp_a_Lumi[1] + tmp_a_L_err[1], color = line_c[1], alpha = 0.12,)

    plt.xlim(1e1, 1e3)
    plt.xscale('log')
    plt.xlabel('R[kpc]', fontsize = 15)
    plt.yscale('log')
    plt.ylim(5e2, 1e7)
    plt.legend( loc = 1, frameon = False, fontsize = 15,)
    plt.ylabel('SB $[L_{\\odot} / kpc^2]$', fontsize = 15,)
    plt.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
    plt.savefig('/home/xkchen/figs/mass-age-bin_%s-band_fit-based_%s-band_SB_profile.png' % (band_str, band_str[1]), dpi = 300)
    plt.close()
'''

### === ### compare with SB profile fitting (convert the fitting of SB profile into mass profile)
def sb2mass_func( R_arr, sb_arr_0, sb_arr_1, out_file, band_info, z_obs):

    Dl_obs = Test_model.luminosity_distance( z_obs ).value

    mag_arr_0 = 22.5 - 2.5 * np.log10( sb_arr_0 )
    mag_arr_1 = 22.5 - 2.5 * np.log10( sb_arr_1 )

    color_pros = mag_arr_0 - mag_arr_1

    p_mag = 22.5 - 2.5 * np.log10( sb_arr_1 )
    p_Mag = p_mag - 5 * np.log10( Dl_obs * 10**6 / 10)

    get_c2mass_func( R_arr, band_info, p_Mag, color_pros, z_obs, out_file = out_file )

    p_dat = pds.read_csv( out_file )
    p_surf_mass = np.array( p_dat['surf_mass'] )

    return p_surf_mass

for band_str in ('gi', 'gr', 'ri'):

    for mm in range( 2 ):

        dat = pds.read_csv( BG_path + '%s_%s-band-based_aveg-jack_mass-Lumi.csv' % (cat_lis[mm], band_str) )
        obs_R, surf_M, surf_M_err = np.array( dat['R'] ), np.array( dat['surf_mass'] ), np.array( dat['surf_mass_err'] )

        tmp_cen_sb, tmp_out_sb = [], []
        """
        ## load the fitting params (SB profile decomposition)
        for kk in range( 3 ):
            # ... mass-bin
            if mass_id == True:
                if mm == 0: ## low mass bin
                    m_dat = pds.read_csv( param_path + 'speci_fit_low_mass/%s_%s-band_2-sersic_fit.csv' % (cat_lis[mm], band[kk]),)
                    ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])
                if mm == 1:
                    m_dat = pds.read_csv( param_path + '%s_%s-band_lim-n=2.1_mcmc_fit.csv' % (cat_lis[mm], band[kk]),)
                    ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])

            #...age bin
            if mass_id == False:
                if mm == 0:
                    # m_dat = pds.read_csv( param_path + 'speci_fit_low_mass/%s_%s-band_2-sersic_fit.csv' % (cat_lis[mm], band[kk]),)
                    m_dat = pds.read_csv( param_path + '%s_%s-band_lim-n=2.1_mcmc_fit_separate.csv' % (cat_lis[mm], band[kk]),)
                    ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])

                if mm == 1:
                    m_dat = pds.read_csv( param_path + '%s_%s-band_lim-n=2.1_mcmc_fit.csv' % (cat_lis[mm], band[kk]),)
                    ( m_I0, m_R0, m_n0, m_I1, m_R1, m_n1 ) = ( np.array(m_dat['Ie_0'])[0], np.array(m_dat['Re_0'])[0], np.array(m_dat['ne_0'])[0], 
                        np.array(m_dat['Ie_1'])[0], np.array(m_dat['Re_1'])[0], np.array(m_dat['ne_1'])[0])

            fit_cen_arr = sersic_func( obs_R, m_I0, m_R0, m_n0 )
            fit_out_arr = sersic_func( obs_R, m_I1, m_R1, m_n1 )

            tmp_cen_sb.append( fit_cen_arr )
            tmp_out_sb.append( fit_out_arr )

        ## ... gi-band
        if band_str == 'gi':

            out_lm_file = '/home/xkchen/figs/tmp_fit-based_mass-Lumi.csv'
            surf_cen_mass = sb2mass_func( obs_R, tmp_cen_sb[1], tmp_cen_sb[2], out_lm_file, band_str, z_ref)

            out_lm_file = '/home/xkchen/figs/tmp_fit-based_mass-Lumi.csv'
            surf_out_mass = sb2mass_func( obs_R, tmp_out_sb[1], tmp_out_sb[2], out_lm_file, band_str, z_ref)

        ## ... gr-band
        if band_str == 'gr':

            out_lm_file = '/home/xkchen/figs/tmp_fit-based_mass-Lumi.csv'
            surf_cen_mass = sb2mass_func( obs_R, tmp_cen_sb[1], tmp_cen_sb[0], out_lm_file, band_str, z_ref)

            out_lm_file = '/home/xkchen/figs/tmp_fit-based_mass-Lumi.csv'
            surf_out_mass = sb2mass_func( obs_R, tmp_out_sb[1], tmp_out_sb[0], out_lm_file, band_str, z_ref)

        ## ... ri-band
        if band_str == 'ri':

            out_lm_file = '/home/xkchen/figs/tmp_fit-based_mass-Lumi.csv'
            surf_cen_mass = sb2mass_func( obs_R, tmp_cen_sb[0], tmp_cen_sb[2], out_lm_file, band_str, z_ref)

            out_lm_file = '/home/xkchen/figs/tmp_fit-based_mass-Lumi.csv'
            surf_out_mass = sb2mass_func( obs_R, tmp_out_sb[0], tmp_out_sb[2], out_lm_file, band_str, z_ref)

        sum_m_pros = surf_cen_mass + surf_out_mass
        """

        c_dat = pds.read_csv( fit2M_path + '%s_%s-band-based_fit-based_mass-Lumi_with-err.csv' % (cat_lis[mm], band_str),)
        sum_m_pros = np.array(c_dat['surf_mass'])
        sum_m_R = np.array(c_dat['R'])

        plt.figure()
        ax = plt.subplot(111)
        ax.errorbar( obs_R, surf_M, yerr = surf_M_err, xerr = None, color = 'r', marker = 'o', ls = 'none', ecolor = 'r', 
            alpha = 0.5, mec = 'r', mfc = 'r', label = 'signal')

        # ax.plot( obs_R, sum_m_pros, ls = '-', color = 'b', alpha = 0.5, label = 'fitting',)
        # ax.plot( obs_R, surf_cen_mass, ls = '--', color = 'b', alpha = 0.5, )
        # ax.plot( obs_R, surf_out_mass, ls = ':', color = 'b', alpha = 0.5, )

        ax.plot( sum_m_R, sum_m_pros, ls = '-', color = 'b', alpha = 0.5, label = 'fitting',)

        ax.legend( loc = 1, )

        ax.set_ylim( 1e3, 3e8)
        ax.set_yscale('log')
        ax.set_ylabel( '$ lg \\Sigma [M_{\\odot} / kpc^2]$' )

        ax.set_xlim( 1e1, 1e3)
        ax.set_xlabel( 'R [kpc]')
        ax.set_xscale( 'log' )
        plt.savefig('/home/xkchen/%s_%s-band_based_surf-mass_obs-fit_compare.png' % (cat_lis[mm], band_str), dpi = 300)
        plt.close()
