import h5py
import numpy as np
import pandas as pds
from scipy import optimize

from light_measure import jack_SB_func

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

