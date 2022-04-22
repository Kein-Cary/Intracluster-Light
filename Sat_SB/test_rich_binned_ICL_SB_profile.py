
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

from scipy import optimize
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
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/BGs/'

##. sample information
bin_rich = [ 20, 30, 50, 210 ]
sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


tmp_R, tmp_sb, tmp_err = [], [], []

for kk in range( 3 ):

    band_str = band[ kk ]
    sub_R, sub_sb, sub_err = [], [], []

    for tt in range( 3 ):

        with h5py.File( BG_path + 'photo-z_match_clust_%s_%s-band' % ( sub_name[tt], band_str) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

            tt_r = np.array(f['r'])
            tt_sb = np.array(f['sb'])
            tt_err = np.array(f['sb_err'])

        sub_R.append( tt_r )
        sub_sb.append( tt_sb )
        sub_err.append( tt_err )
        
    tmp_R.append( sub_R )
    tmp_sb.append( sub_sb )
    tmp_err.append( sub_err )


### == figs
line_c = [ 'b', 'g', 'r']
line_s = [ '--', '-', '-.']
ylim_arr = [ 1e1, 3e0, 2e1 ]

for tt in range( 3 ):

    fig = plt.figure()
    ax1 = fig.add_axes([0.12, 0.10, 0.8, 0.85])

    ax1.plot( tmp_R[tt][ 0 ], tmp_sb[tt][ 0 ], ls = '-', color = 'k', label = line_name[ 0 ],)
    ax1.fill_between( tmp_R[tt][0], y1 = tmp_sb[tt][0] - tmp_err[tt][0], 
                    y2 = tmp_sb[tt][0] + tmp_err[tt][0], color = 'k', alpha = 0.12,)

    ax1.plot( tmp_R[tt][ 1 ], tmp_sb[tt][ 1 ], ls = '--', color = 'k', label = line_name[ 1 ],)
    ax1.fill_between( tmp_R[tt][1], y1 = tmp_sb[tt][1] - tmp_err[tt][1], 
                    y2 = tmp_sb[tt][1] + tmp_err[tt][1], color = 'k', alpha = 0.12,)

    ax1.plot( tmp_R[tt][ 2 ], tmp_sb[tt][ 2 ], ls = ':', color = 'k', label = line_name[ 2 ],)
    ax1.fill_between( tmp_R[tt][2], y1 = tmp_sb[tt][2] - tmp_err[tt][2], 
                    y2 = tmp_sb[tt][2] + tmp_err[tt][2], color = 'k', alpha = 0.12,)

    ax1.legend( loc = 1, frameon = False,)

    ax1.set_xlim( 1e0, 3e3 )
    ax1.set_xscale('log')
    ax1.set_xlabel('R [kpc]')

    ax1.set_ylim( 2e-3, ylim_arr[tt] )
    ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
    ax1.set_yscale('log')

    ax1.annotate( s = '%s-band' % band[tt], xy = (0.75, 0.65), xycoords = 'axes fraction',)

    plt.savefig('/home/xkchen/rich_binned_%s-band_BCG_ICL.png' % band[tt], dpi = 300)
    plt.close()

