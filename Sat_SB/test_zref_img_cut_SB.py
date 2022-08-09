import matplotlib as mpl
import matplotlib.pyplot as plt

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


### === SB profiles compare
BG_path = '/home/xkchen/figs_cp/cc_rich_rebin/BGs/'
cp_path = '/home/xkchen/figs_cp/SB_pros_check/BGs/'
path = '/home/xkchen/figs_cp/cc_rich_rebin/SBs/'


#.
bin_rich = [ 20, 30, 50, 210 ]
R_bins = np.array( [ 0, 300, 400, 550, 5000] )

list_order = 13

sub_name = ['low-rich', 'medi-rich', 'high-rich']

#. medi-rich sample
ll = 1

band_str = 'r'

##. background SB profile
tmp_bg_R, tmp_bg_SB, tmp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    with h5py.File( BG_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
            '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

        tt_r = np.array(f['r'])
        tt_sb = np.array(f['sb'])
        tt_err = np.array(f['sb_err'])

    tmp_bg_R.append( tt_r )
    tmp_bg_SB.append( tt_sb )
    tmp_bg_err.append( tt_err )


cp_bg_R, cp_bg_SB, cp_bg_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    with h5py.File( cp_path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
            '_%s-band_shufl-%d_BG' % (band_str, list_order) + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

        tt_r = np.array(f['r'])
        tt_sb = np.array(f['sb'])
        tt_err = np.array(f['sb_err'])

    cp_bg_R.append( tt_r )
    cp_bg_SB.append( tt_sb )
    cp_bg_err.append( tt_err )


##. satellite SB profile
tmp_R, tmp_sb, tmp_err = [], [], []

for tt in range( len(R_bins) - 1 ):

    with h5py.File( path + 'Extend_BCGM_gri-common_%s_phyR_%d-%dkpc' % (sub_name[ ll ], R_bins[tt], R_bins[tt + 1]) + 
                    '_%s-band' % band_str + '_Mean_jack_SB-pro_z-ref.h5', 'r') as f:

        tt_r = np.array(f['r'])
        tt_sb = np.array(f['sb'])
        tt_err = np.array(f['sb_err'])

    tmp_R.append( tt_r )
    tmp_sb.append( tt_sb )
    tmp_err.append( tt_err )



##. figs
color_s = ['b', 'c', 'g', 'r', 'm']

fig_name = []
for dd in range( len(R_bins) - 1 ):

    if dd == 0:
        fig_name.append( '$R \\leq %d \, kpc$' % R_bins[dd + 1] )

    elif dd == len(R_bins) - 2:
        fig_name.append( '$R \\geq %d \, kpc$' % R_bins[dd] )

    else:
        fig_name.append( '$%d \\leq R \\leq %d \, kpc$' % (R_bins[dd], R_bins[dd + 1]),)

line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


plt.figure()
ax1 = plt.subplot(111)

for mm in range( len(R_bins) - 1 ):

    l2, = ax1.plot( tmp_bg_R[mm], tmp_bg_SB[mm], ls = '--', color = color_s[mm], alpha = 0.75, label = fig_name[mm],)
    ax1.fill_between( tmp_bg_R[mm], y1 = tmp_bg_SB[mm] - tmp_bg_err[mm], 
                        y2 = tmp_bg_SB[mm] + tmp_bg_err[mm], color = color_s[mm], alpha = 0.12)

    l3, = ax1.plot( cp_bg_R[mm], cp_bg_SB[mm], ls = '-', color = color_s[mm], alpha = 0.75,)
    ax1.fill_between( cp_bg_R[mm], y1 = cp_bg_SB[mm] - cp_bg_err[mm], 
                        y2 = cp_bg_SB[mm] + cp_bg_err[mm], color = color_s[mm], alpha = 0.12)

legend_2 = ax1.legend( handles = [l2, l3], 
            labels = ['previous', 'Now' ], loc = 2, frameon = False, fontsize = 12,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.annotate( s = line_name[ll] + ', %s-band' % band_str, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12,)

ax1.set_ylim( 9e-4, 3e-2 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.set_yscale('log')

plt.savefig('/home/xkchen/%s_sat_%s-band_BG_check.png' % (sub_name[ll], band_str), dpi = 300)
plt.close()


plt.figure()
ax1 = plt.subplot(111)

for mm in range( len(R_bins) - 1 ):

    l2 = ax1.errorbar( tmp_R[mm], tmp_sb[mm], yerr = tmp_err[mm], marker = '.', ls = '-', color = color_s[mm],
        ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm],)

    l3, = ax1.plot( cp_bg_R[mm], cp_bg_SB[mm], ls = '--', color = color_s[mm], alpha = 0.75,)
    ax1.fill_between( cp_bg_R[mm], y1 = cp_bg_SB[mm] - cp_bg_err[mm], 
                        y2 = cp_bg_SB[mm] + cp_bg_err[mm], color = color_s[mm], alpha = 0.12)

legend_2 = ax1.legend( handles = [l2, l3], 
            labels = ['Satellite + Background', 'Background' ], loc = 5, frameon = False, fontsize = 12,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]', fontsize = 12,)

ax1.annotate( s = line_name[ll] + ', %s-band' % band_str, xy = (0.65, 0.05), xycoords = 'axes fraction', fontsize = 12,)

ax1.set_ylim( 1e-3, 4e0 )
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax1.set_yscale('log')

plt.savefig('/home/xkchen/%s_sat_%s-band_BG_compare.png' % (sub_name[ll], band_str), dpi = 300)
plt.close()

