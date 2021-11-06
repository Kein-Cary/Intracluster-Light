import h5py
import numpy as np
import pandas as pds
from scipy import optimize

from scipy.interpolate import splev, splrep
import scipy.signal as signal
import scipy.interpolate as interp

from fig_out_module import arr_jack_func
from light_measure import jack_SB_func
from fig_out_module import color_func, BG_sub_cov_func, BG_pro_cov

## SB model for random image
def cc_inves_x2(x, x0, A, alpha, B):
    return A * (np.abs(x - x0))**(-1*alpha) + B

def cc_rand_sb_func(x, a, b, x0, A, alpha, B):
    pf0 = a * np.log10(x) + b
    pf1 = cc_inves_x2(np.log10(x), x0, A, alpha, B)
    pf = pf0 + pf1
    return pf

def err_func(p, x, y, yerr):
    a, b, x0, A, alpha, B = p[:]
    pf = cc_rand_sb_func(x, a, b, x0, A, alpha, B)
    return np.sum( (pf - y)**2 / yerr**2 )

## SB model for cluster component (in sersic formula, fixed sersic index)
def sersic_func(r, Ie, re):
    ndex = 2.1 # Zhang et a., 2019, for large scale, n~2.1
    belta = 2 * ndex - 0.324
    fn = -1 * belta * ( r / re )**(1 / ndex) + belta
    Ir = Ie * np.exp( fn )
    return Ir

def BG_sub_sb_func(N_sample, jk_sub_sb, sb_out_put, band_str, BG_file, trunk_R = 2e3,):
    """
    by default, trunk_R = 2Mpc, which meanse the model signal beyond 2Mpc will be treated as 
    background and subtracted from the observation
    """
    tmp_r, tmp_sb = [], []

    cat = pds.read_csv(BG_file)
    ( e_a, e_b, e_x0, e_A, e_alpha, e_B, offD) = ( np.array(cat['e_a'])[0], np.array(cat['e_b'])[0], 
            np.array(cat['e_x0'])[0], np.array(cat['e_A'])[0], np.array(cat['e_alpha'])[0], 
            np.array(cat['e_B'])[0], np.array(cat['offD'])[0] )

    I_e, R_e = np.array(cat['I_e'])[0], np.array(cat['R_e'])[0]
    sb_2Mpc = sersic_func( trunk_R, I_e, R_e )

    for kk in range( N_sample ):

        with h5py.File( jk_sub_sb % kk, 'r') as f:
            c_r_arr = np.array(f['r'])
            c_sb_arr = np.array(f['sb'])
            c_sb_err = np.array(f['sb_err'])
            npix = np.array(f['npix'])

        id_Nul = npix < 1
        c_r_arr[ id_Nul ] = np.nan
        c_sb_arr[ id_Nul ] = np.nan
        c_sb_err[ id_Nul ] = np.nan

        full_r_fit = cc_rand_sb_func(c_r_arr, e_a, e_b, e_x0, e_A, e_alpha, e_B)
        full_BG = full_r_fit - offD + sb_2Mpc
        devi_sb = c_sb_arr - full_BG

        tmp_r.append( c_r_arr )
        tmp_sb.append( devi_sb )

    tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_sample,)[4:]

    ## save BG-subtracted pros
    with h5py.File( sb_out_put, 'w') as f:
        f['r'] = np.array(tt_jk_R)
        f['sb'] = np.array(tt_jk_SB)
        f['sb_err'] = np.array(tt_jk_err)

    return

### === ### surface mass, color, and gradient profiles
def sub_color_func( N_samples, tt_r_file, tt_g_file, tt_i_file, sub_color_file, aveg_C_file, id_dered = False, Al_arr = None):
    """
    Al_arr : extinction array, in order r-band, g-band, i-band
    """
    tmp_r, tmp_gr, tmp_gi, tmp_ri = [], [], [], []

    for ll in range( N_samples ):

        p_r_dat = pds.read_csv( tt_r_file % ll )
        tt_r_R, tt_r_sb, tt_r_err = np.array( p_r_dat['R'] ), np.array( p_r_dat['BG_sub_SB'] ), np.array( p_r_dat['sb_err'] )

        p_g_dat = pds.read_csv( tt_g_file % ll )
        tt_g_R, tt_g_sb, tt_g_err = np.array( p_g_dat['R'] ), np.array( p_g_dat['BG_sub_SB'] ), np.array( p_g_dat['sb_err'] )

        p_i_dat = pds.read_csv( tt_i_file % ll )
        tt_i_R, tt_i_sb, tt_i_err = np.array( p_i_dat['R'] ), np.array( p_i_dat['BG_sub_SB'] ), np.array( p_i_dat['sb_err'] )


        idR_lim = tt_r_R <= 1.2e3
        tt_r_R, tt_r_sb, tt_r_err = tt_r_R[ idR_lim], tt_r_sb[ idR_lim], tt_r_err[ idR_lim]

        idR_lim = tt_g_R <= 1.2e3
        tt_g_R, tt_g_sb, tt_g_err = tt_g_R[ idR_lim], tt_g_sb[ idR_lim], tt_g_err[ idR_lim] 

        idR_lim = tt_i_R <= 1.2e3
        tt_i_R, tt_i_sb, tt_i_err = tt_i_R[ idR_lim], tt_i_sb[ idR_lim], tt_i_err[ idR_lim] 

        gr_arr, gr_err = color_func( tt_g_sb, tt_g_err, tt_r_sb, tt_r_err )
        gi_arr, gi_err = color_func( tt_g_sb, tt_g_err, tt_i_sb, tt_i_err )
        ri_arr, ri_err = color_func( tt_r_sb, tt_r_err, tt_i_sb, tt_i_err )

        if id_dered == True:
            Al_r, Al_g, Al_i = Al_arr[:]

            gr_arr = gr_arr + np.nanmedian( Al_r ) - np.nanmedian( Al_g )
            gi_arr = gi_arr + np.nanmedian( Al_i ) - np.nanmedian( Al_g )
            ri_arr = ri_arr + np.nanmedian( Al_i ) - np.nanmedian( Al_r )

        keys = [ 'R', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err']
        values = [ tt_g_R, gr_arr, gr_err, gi_arr, gi_err, ri_arr, ri_err ]
        fill = dict(zip( keys, values) )
        out_data = pds.DataFrame( fill )
        out_data.to_csv( sub_color_file % ll,)

        tmp_r.append( tt_g_R )
        tmp_gr.append( gr_arr )
        tmp_gi.append( gi_arr )
        tmp_ri.append( ri_arr )

    aveg_R_0, aveg_gr, aveg_gr_err = arr_jack_func( tmp_gr, tmp_r, N_samples )[:3]
    aveg_R_1, aveg_gi, aveg_gi_err = arr_jack_func( tmp_gi, tmp_r, N_samples )[:3]
    aveg_R_2, aveg_ri, aveg_ri_err = arr_jack_func( tmp_ri, tmp_r, N_samples )[:3]

    Len_x = np.max( [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ] )
    id_L = [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ].index( Len_x )

    cc_aveg_R = [ aveg_R_0, aveg_R_1, aveg_R_2 ][ id_L ]

    cc_aveg_gr = np.zeros( Len_x,)
    cc_aveg_gr[ :len(aveg_gr) ] = aveg_gr
    cc_aveg_gr_err = np.zeros( Len_x,)
    cc_aveg_gr_err[ :len(aveg_gr) ] = aveg_gr_err

    cc_aveg_gi = np.zeros( Len_x,)
    cc_aveg_gi[ :len(aveg_gi) ] = aveg_gi
    cc_aveg_gi_err = np.zeros( Len_x,)
    cc_aveg_gi_err[ :len(aveg_gi) ] = aveg_gi_err

    cc_aveg_ri = np.zeros( Len_x,)
    cc_aveg_ri[ :len(aveg_ri) ] = aveg_ri
    cc_aveg_ri_err = np.zeros( Len_x,)
    cc_aveg_ri_err[ :len(aveg_ri) ] = aveg_ri_err

    keys = [ 'R_kpc', 'g-r', 'g-r_err', 'g-i', 'g-i_err', 'r-i', 'r-i_err' ]
    values = [ cc_aveg_R, cc_aveg_gr, cc_aveg_gr_err, cc_aveg_gi, cc_aveg_gi_err, cc_aveg_ri, cc_aveg_ri_err ]
    fill = dict( zip( keys, values) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( aveg_C_file,)

    return

def B03_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_g, z_g,
    aveg_jk_sm_file, lgM_cov_file, c_inv = False, M_cov_file = None ):
    """
    band_str = 'gi', 'gr', 'ri', according to Bell 2003
    Dl_g, z_g : the redshift and corresponding luminosity distance
    """
    from color_2_mass import jk_sub_Mass_func
    from color_2_mass import aveg_mass_pro_func   

    #. surface mass of subsamples
    jk_sub_Mass_func( N_samples, band_str, sub_sb_file, low_R_lim, up_R_lim, sub_sm_file, Dl_g, z_g, c_inv = c_inv )

    #. average mass profile of subsamples
    aveg_mass_pro_func( N_samples, band_str, sub_sm_file, aveg_jk_sm_file, lgM_cov_file, M_cov_file = M_cov_file )

    return

def fit_surfM_func( N_samples, band_str, low_R_lim, up_R_lim, sub_sb_file, sub_sm_file, Dl_g, z_g,
    aveg_jk_sm_file, lgM_cov_file, fit_file, M_cov_file = None ):
    """
    mass setimate based M/L - color relation: lg_M = a*(g-r) + b*(r-i) + c*lg_Li + d
    parameters for the relation is in fit_file
    """
    from tmp_color_to_mass import get_c2mass_func
    from tmp_color_to_mass import jk_sub_Mass_func
    from tmp_color_to_mass import aveg_mass_pro_func

    #. surface mass of subsamples
    jk_sub_Mass_func( N_samples, band_str, sub_sb_file, low_R_lim, up_R_lim, sub_sm_file, Dl_g, z_g, fit_file = fit_file )

    #. average mass profile of subsamples
    aveg_mass_pro_func( N_samples, band_str, sub_sm_file, aveg_jk_sm_file, lgM_cov_file, M_cov_file = M_cov_file )

    return

#. slope profile
def smooth_slope_func(r_arr, color_arr, wind_L, order_dex, delta_x):

    id_nn = np.isnan( color_arr )

    if np.sum( id_nn ) == 0:
        dc_dr = signal.savgol_filter( color_arr, wind_L, order_dex, deriv = 1, delta = delta_x)

    else:
        _cp_r = r_arr[ id_nn == False ]
        _cp_color = color_arr[ id_nn == False ]

        _cp_color_F = interp.interp1d( _cp_r, _cp_color, kind = 'linear', fill_value = 'extrapolate',)

        color_arr[id_nn] = _cp_color_F( r_arr[id_nn] )
        dc_dr = signal.savgol_filter( color_arr, wind_L, order_dex, deriv = 1, delta = delta_x)
    return dc_dr

def sub_color_slope_func( N_samples, sub_arr_file, sub_slope_file, aveg_slope_file, wind_L, poly_order, d_lgR = False):
    """
    sub_arr_file, sub_slope_file, aveg_slope_file : .csv files
    """
    tmp_r, tmp_dgr, tmp_dgi, tmp_dri = [], [], [], []

    for ll in range( N_samples ):

        pdat = pds.read_csv( sub_arr_file % ll )
        tt_r, tt_gr, tt_gr_err = np.array( pdat['R'] ), np.array( pdat['g-r'] ), np.array( pdat['g-r_err'] )
        tt_gi, tt_gi_err = np.array( pdat['g-i'] ), np.array( pdat['g-i_err'] )
        tt_ri, tt_ri_err = np.array( pdat['r-i'] ), np.array( pdat['r-i_err'] )

        WL, p_order = wind_L, poly_order

        if d_lgR == True:
            delt_x = np.median( np.diff( np.log10( tt_r ) ) )
        else:
            delt_x = 1.0

        d_gr_dlgr = smooth_slope_func(tt_r, tt_gr, WL, p_order, delt_x)
        d_gi_dlgr = smooth_slope_func(tt_r, tt_gi, WL, p_order, delt_x)
        d_ri_dlgr = smooth_slope_func(tt_r, tt_ri, WL, p_order, delt_x)

        keys = [ 'R', 'd_gr_dlgr', 'd_gi_dlgr', 'd_ri_dlgr' ]
        values = [ tt_r, d_gr_dlgr, d_gi_dlgr, d_ri_dlgr, ]

        fill = dict(zip( keys, values) )
        out_data = pds.DataFrame( fill )
        out_data.to_csv( sub_slope_file % ll,)

        tmp_r.append( tt_r )
        tmp_dgr.append( d_gr_dlgr )
        tmp_dgi.append( d_gi_dlgr )
        tmp_dri.append( d_ri_dlgr )

    aveg_R_0, aveg_dgr, aveg_dgr_err = arr_jack_func( tmp_dgr, tmp_r, N_samples )[:3]
    aveg_R_1, aveg_dgi, aveg_dgi_err = arr_jack_func( tmp_dgi, tmp_r, N_samples )[:3]
    aveg_R_2, aveg_dri, aveg_dri_err = arr_jack_func( tmp_dri, tmp_r, N_samples )[:3]

    Len_x = np.max( [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ] )
    id_L = [ len(aveg_R_0), len(aveg_R_1), len(aveg_R_2) ].index( Len_x )

    cc_aveg_R = [ aveg_R_0, aveg_R_1, aveg_R_2 ][ id_L ]

    cc_aveg_dgr = np.zeros( Len_x,)
    cc_aveg_dgr[ :len( aveg_dgr ) ] = aveg_dgr
    cc_aveg_dgr_err = np.zeros( Len_x,)
    cc_aveg_dgr_err[ :len( aveg_dgr ) ] = aveg_dgr_err

    cc_aveg_dgi = np.zeros( Len_x,)
    cc_aveg_dgi[ :len( aveg_dgi ) ] = aveg_dgi
    cc_aveg_dgi_err = np.zeros( Len_x,)
    cc_aveg_dgi_err[ :len( aveg_dgi ) ] = aveg_dgi_err

    cc_aveg_dri = np.zeros( Len_x,)
    cc_aveg_dri[ :len( aveg_dri ) ] = aveg_dri
    cc_aveg_dri_err = np.zeros( Len_x,)
    cc_aveg_dri_err[ :len( aveg_dri ) ] = aveg_dri_err

    keys = [ 'R_kpc', 'd_gr', 'd_gr_err', 'd_gi', 'd_gi_err', 'd_ri', 'd_ri_err' ]
    values = [ cc_aveg_R, cc_aveg_dgr, cc_aveg_dgr_err, cc_aveg_dgi, cc_aveg_dgi_err, cc_aveg_dri, cc_aveg_dri_err ]
    fill = dict( zip( keys, values ) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( aveg_slope_file )

    return

#. Mass-to-Light ratio test
def M2L_slope_func( N_samples, sub_sm_file, sub_m2l_file, aveg_m2l_file, wind_L, poly_order, d_lgR = False):

    tmp_r, tmp_M2L, tmp_d_M2L = [], [], []

    for nn in range( N_samples ):

        o_dat = pds.read_csv( sub_sm_file % nn,)
        tt_r = np.array( o_dat['R'] )
        tt_M = np.array( o_dat['surf_mass'] )
        tt_Li = np.array( o_dat['lumi'] ) 

        tt_M2L = tt_M / tt_Li

        WL, p_order = wind_L, poly_order

        if d_lgR == True:
            delt_x = np.median( np.diff( np.log10( tt_r ) ) )
        else:
            delt_x = 1.0

        tt_dM2L = smooth_slope_func( tt_r, tt_M2L, WL, p_order, delt_x)

        keys = [ 'R', 'M/Li', 'd_M/L_dlgr' ]
        values = [ tt_r, tt_M2L, tt_dM2L ]
        fill = dict( zip( keys, values ) )
        out_data = pds.DataFrame( fill )
        out_data.to_csv( sub_m2l_file % nn,)

        tmp_r.append( tt_r )
        tmp_M2L.append( tt_M2L )
        tmp_d_M2L.append( tt_dM2L )

    aveg_R, aveg_M2L, aveg_M2L_err = arr_jack_func( tmp_M2L, tmp_r, N_samples)[:3]
    aveg_R, aveg_dM2L, aveg_dM2L_err = arr_jack_func( tmp_d_M2L, tmp_r, N_samples)[:3]

    keys = ['R', 'M/Li', 'M/Li-err', 'd_M/Li', 'd_M/Li_err']
    values = [ aveg_R, aveg_M2L, aveg_M2L_err, aveg_dM2L, aveg_dM2L_err ]
    fill = dict( zip( keys, values ) )
    out_data = pds.DataFrame( fill )
    out_data.to_csv( aveg_m2l_file,)

    return


