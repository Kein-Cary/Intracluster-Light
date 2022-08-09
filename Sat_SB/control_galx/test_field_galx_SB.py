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


### === data load
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin_contrl_galx/SBs/'


bin_rich = [ 20, 30, 50, 210 ]

sub_name = ['low-rich', 'medi-rich', 'high-rich']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']


##. R_limmits
# R_str = 'phy'
# R_bins = np.array( [ 0, 300, 400, 550, 5000] )     ### kpc

R_str = 'scale'
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m

cp_out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_BGsub_SBs/'
cp_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/nobcg_SBs/'


### === Background subtraction



### === SBs compare
band_str = 'r'

tmp_R, tmp_sb, tmp_err = [], [], []

for ll in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		with h5py.File( path + 
			'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_Mean_jack_SB-pro_z-ref.h5' % 
			(sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	tmp_R.append( sub_R )
	tmp_sb.append( sub_sb )
	tmp_err.append( sub_err )


##... sat SBs
cp_R, cp_sb, cp_err = [], [], []

for ll in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []

	for tt in range( len(R_bins) - 1 ):

		with h5py.File( cp_path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_Mean_jack_SB-pro_z-ref.h5' 
						% (sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str), 'r') as f:

			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	cp_R.append( sub_R )
	cp_sb.append( sub_sb )
	cp_err.append( sub_err )

#.
cp_nbg_R, cp_nbg_SB, cp_nbg_err = [], [], []

for ll in range( 3 ):

	sub_R, sub_sb, sub_err = [], [], []
	
	for tt in range( len(R_bins) - 1 ):

		dat = pds.read_csv( cp_out_path + 
						'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_aveg-jack_BG-sub_SB.csv' % 
						(sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str),)

		tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_sb_err )

	cp_nbg_R.append( sub_R )
	cp_nbg_SB.append( sub_sb )
	cp_nbg_err.append( sub_err )



##.. figs
color_s = ['b', 'g', 'c', 'r', 'm']

line_s = [ ':', '--', '-' ]

#.
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

##. 2D image
"""
for ll in range( 3 ):

	for tt in range( len(R_bins) - 1 ):

		with h5py.File( path + 'Extend_BCGM_gri-common_%s_%.2f-%.2fR200m_%s-band_Mean_jack_img_z-ref.h5' % 
						(sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str), 'r') as f:

			tmp_img = np.array( f['a'] )

		id_nn = np.isnan(tmp_img)
		eff_y, eff_x = np.where(id_nn == False)

		da0, da1 = eff_x.min(), eff_x.max()
		db0, db1 = eff_y.min(), eff_y.max()

		cut_img = tmp_img[db0: db1+1, da0: da1+1]

		fig = plt.figure( )
		ax = fig.add_axes([0.11, 0.1, 0.80, 0.84])

		ax.set_title( line_name[ ll ] + ', ' + fig_name[ tt ] )
		tf = ax.imshow( cut_img / pixel**2, origin = 'lower', cmap = 'bwr', vmin = -1e-1, vmax = 1e-1,)

		plt.colorbar( tf, ax = ax, pad = 0.01, label = 'SB $[nanomaggies \, / \, arcsec^{2}] $')

		plt.savefig('/home/xkchen/%s_contrl-galx_%.2f-%.2fR200m_%s-band_aveg-img.png' % 
					(sub_name[ ll ], R_bins[tt], R_bins[tt + 1], band_str), dpi = 300)
		plt.close()

raise
"""

#.
for ll in range( 3 ):

	plt.figure()
	ax1 = plt.subplot(111)

	for mm in range( len(R_bins) - 1 ):

		ax1.errorbar( tmp_R[ll][mm], tmp_sb[ll][mm], yerr = tmp_err[ll][mm], marker = '.', ls = '-', color = color_s[mm],
			ecolor = color_s[mm], mfc = 'none', mec = color_s[mm], capsize = 1.5, label = fig_name[mm],)

	ax1.legend( loc = 1, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = line_name[ll] + ', %s-band' % band_str, xy = (0.08, 0.05), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( 3e-3, 4e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/%s_contrl-galx_%s-band_SB_compare.png' % (sub_name[ll], band_str), dpi = 300)
	plt.close()

#.
for mm in range( len(R_bins) - 1 ):

	plt.figure()
	ax1 = plt.subplot(111)

	for ll in range( 3 ):

		ax1.errorbar( tmp_R[ll][mm], tmp_sb[ll][mm], yerr = tmp_err[ll][mm], marker = '.', ls = '-', color = color_s[ll],
			ecolor = color_s[ll], mfc = 'none', mec = color_s[ll], capsize = 1.5, label = line_name[ll],)

	ax1.legend( loc = 1, frameon = False, fontsize = 12,)

	ax1.set_xscale('log')
	ax1.set_xlabel('R [kpc]', fontsize = 12,)

	ax1.annotate( s = fig_name[mm] + ', %s-band' % band_str, xy = (0.08, 0.05), xycoords = 'axes fraction', fontsize = 12,)

	ax1.set_ylim( 3e-3, 4e0 )
	ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
	ax1.set_yscale('log')

	ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/contrl-galx_%.2f-%.2fR200m_%s-band_SB_compare.png' % (R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
	plt.close()

#.
for mm in range( len(R_bins) - 1 ):

	for ll in range( 3 ):

		plt.figure()
		ax1 = plt.subplot(111)

		ax1.errorbar( tmp_R[ll][mm], tmp_sb[ll][mm], yerr = tmp_err[ll][mm], marker = '.', ls = '-', color = 'b',
			ecolor = 'b', mfc = 'none', mec = 'b', capsize = 1.5, label = 'Control',)

		ax1.errorbar( cp_R[ll][mm], cp_sb[ll][mm], yerr = cp_err[ll][mm], marker = '.', ls = '-', color = 'r',
			ecolor = 'r', mfc = 'none', mec = 'r', capsize = 1.5, label = 'Member',)

		ax1.errorbar( cp_nbg_R[ll][mm], cp_nbg_SB[ll][mm], yerr = cp_nbg_err[ll][mm], marker = '.', ls = '--', color = 'k',
			ecolor = 'k', mfc = 'none', mec = 'k', capsize = 1.5, label = 'Member, BG-subtraction',)

		ax1.legend( loc = 1, frameon = False, fontsize = 12,)

		ax1.set_xscale('log')
		ax1.set_xlabel('R [kpc]', fontsize = 12,)

		ax1.annotate( s = fig_name[mm] + '\n' + line_name[ll] + ', %s-band' % band_str, 
					xy = (0.45, 0.45), xycoords = 'axes fraction', fontsize = 12,)

		ax1.set_ylim( 1e-3, 4e0 )
		ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$', fontsize = 12,)
		ax1.set_yscale('log')

		ax1.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

		plt.savefig('/home/xkchen/%s_contrl-galx_%.2f-%.2fR200m_%s-band_SB_compare.png' % (sub_name[ ll ], R_bins[mm], R_bins[mm + 1], band_str), dpi = 300)
		plt.close()

