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


### === 
BG_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/BGs/'


cat_lis = ['inner-mem', 'outer-mem']

#. ( divided by 0.191 * R200m )
path = '/home/xkchen/figs/extend_bcgM_cat_Sat/SBs/'
fig_name = ['$R_{Sat} \, \\leq \, 0.191 R_{200m}$', '$R_{Sat} \, > \, 0.191 R_{200m}$']


color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ ':', '-' ]


##... R2Rv divide sample
in_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_cat.csv')
in_Rsat = np.array( in_dat['R_sat'])  ## Mpc / h
in_R2Rv = np.array( in_dat['R2Rv'])
in_Rsat = in_Rsat * 1e3 / h

out_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_cat.csv')
out_Rsat = np.array( out_dat['R_sat'])  ## Mpc / h
out_R2Rv = np.array( out_dat['R2Rv'])
out_Rsat = out_Rsat * 1e3 / h


sR_mean = [ np.mean( in_R2Rv ), np.mean( out_R2Rv ) ]
sR_medi = [ np.median( in_R2Rv ), np.median( out_R2Rv ) ]


##... properties of cluster sample
cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')

lg_Mh = np.array( cat['lg_Mh'] )
R_virs = np.array( cat['R_vir'] )

R_virs = R_virs * 1e3 / h  ## kpc
Mh = 10**lg_Mh / h         ## M_sun

Rv_mean = np.mean( R_virs )
Rv_medi = np.median( R_virs )


tmp_R, tmp_sb, tmp_err = [], [], []


##... sat SBs
for mm in range( 2 ):

	sub_R, sub_sb, sub_err = [], [], []

	for kk in range( 3 ):

		#. 1D profiles
		with h5py.File( path + 'Extend_BCGM_gri-common_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		#. 
		R_obs, mu_obs, mu_err = tt_r, tt_sb, tt_err

		sub_R.append( tt_r )
		sub_sb.append( tt_sb )
		sub_err.append( tt_err )

	tmp_R.append( sub_R )
	tmp_sb.append( sub_sb )
	tmp_err.append( sub_err )


##... SB profile of BCG+ICL+BG (as the background)
tmp_bR, tmp_BG, tmp_BG_err = [], [], []

for mm in range( 2 ):

	_sub_bg_R, _sub_bg_sb, _sub_bg_err = [], [], []

	for kk in range( 3 ):

		with h5py.File( BG_path + 
				'photo-z_match_tot-BCG-star-Mass_%s_%s-band_Mean_jack_SB-pro_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:
			tt_r = np.array(f['r'])
			tt_sb = np.array(f['sb'])
			tt_err = np.array(f['sb_err'])

		_sub_bg_R.append( tt_r )
		_sub_bg_sb.append( tt_sb )
		_sub_bg_err.append( tt_err )

	tmp_bR.append( _sub_bg_R )
	tmp_BG.append( _sub_bg_sb )
	tmp_BG_err.append( _sub_bg_err )


cc_bg_R, cc_bg_SB, cc_bg_err = [], [], []

for kk in range( 3 ):

	with h5py.File( BG_path + 'photo-z_match_tot-BCG-star-Mass_%s-band_Mean_jack_SB-pro_z-ref.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	cc_bg_R.append( tt_r )
	cc_bg_SB.append( tt_sb )
	cc_bg_err.append( tt_err )


# fig = plt.figure( )
# ax = fig.add_axes( [0.13, 0.32, 0.85, 0.63] )
# sub_ax = fig.add_axes( [0.13, 0.11, 0.85, 0.21] )

# for kk in range( 3 ):

# 	ax.plot( tmp_bR[0][kk], tmp_BG[0][kk], ls = '--', color = color_s[kk],)
# 	ax.plot( tmp_bR[1][kk], tmp_BG[1][kk], ls = ':', color = color_s[kk],)

# 	ax.plot( cc_bg_R[kk], cc_bg_SB[kk], ls = '-', color = color_s[kk], label = '%s-band' % band[kk] )

# 	sub_ax.plot( cc_bg_R[kk], tmp_BG[0][kk] / cc_bg_SB[kk], ls = '--', color = color_s[kk],)
# 	sub_ax.plot( cc_bg_R[kk], tmp_BG[1][kk] / cc_bg_SB[kk], ls = ':', color = color_s[kk],)

# legend_2 = ax.legend( [ fig_name[0], fig_name[1], 'All' ], loc = 3, frameon = False,)
# ax.legend( loc = 1, frameon = False,)
# ax.add_artist( legend_2 )

# ax.set_xscale('log')
# ax.set_xlim( 1e0, 3e3 )

# ax.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
# ax.set_yscale('log')

# sub_ax.set_xlim( ax.get_xlim() )
# sub_ax.set_xscale('log')
# sub_ax.set_xlabel('$R \; [kpc]$')
# sub_ax.set_ylabel('$\\mu_{weighed} \, / \, \\mu_{w/o \, weight}$')
# sub_ax.set_ylim( 0.97, 1.03 )
# sub_ax.axhline( y = 1, ls = '-', color = 'k', alpha = 0.75,)
# sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
# ax.set_xticklabels( labels = [] )

# plt.savefig('/home/xkchen/BCG+ICL+BG.png', dpi = 300)
# plt.close()


##... BG estimation for satellite SB
for kk in range( 3 ):

	##... scaling case
	# interp_mu = interp.interp1d( cc_bg_R[kk] / Rv_mean, cc_bg_SB[kk], kind = 'linear',)

	# _kk_in_sb = tmp_sb[0][kk] - interp_mu( sR_mean[0] )
	# _kk_in_BG = np.ones( len(tmp_R[0][kk] ), ) * interp_mu( sR_mean[0] )

	# _kk_out_sb = tmp_sb[1][kk] - interp_mu( sR_mean[1] )
	# _kk_out_BG = np.ones( len(tmp_R[1][kk] ), ) * interp_mu( sR_mean[1] )


	in_interp_mu = interp.interp1d( tmp_bR[0][kk] / Rv_mean, tmp_BG[0][kk], kind = 'linear',)

	_kk_in_sb = tmp_sb[0][kk] - in_interp_mu( sR_mean[0] )
	_kk_in_BG = np.ones( len(tmp_R[0][kk] ), ) * in_interp_mu( sR_mean[0] )

	out_interp_mu = interp.interp1d( tmp_bR[1][kk] / Rv_mean, tmp_BG[1][kk], kind = 'linear',)

	_kk_out_sb = tmp_sb[1][kk] - out_interp_mu( sR_mean[1] )
	_kk_out_BG = np.ones( len(tmp_R[1][kk] ), ) * out_interp_mu( sR_mean[1] )


	keys = ['r', 'sb', 'sb_err', 'bg_sb' ]
	values = [ tmp_R[0][kk], _kk_in_sb, tmp_err[0][kk], _kk_in_BG ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_%s_%s-band_scaled-R_BG-sub_SB.csv' % (cat_lis[0], band[kk]),)

	keys = ['r', 'sb', 'sb_err', 'bg_sb' ]
	values = [ tmp_R[1][kk], _kk_out_sb, tmp_err[1][kk], _kk_out_BG ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_%s_%s-band_scaled-R_BG-sub_SB.csv' % (cat_lis[1], band[kk]),)


	##... average of individual satllite position
	# interp_mu_F = interp.interp1d( cc_bg_R[kk], cc_bg_SB[kk], kind = 'linear',)

	# _kk_in_BG = np.sum( interp_mu_F( in_Rsat ) ) / len( in_Rsat )
	# _kk_out_BG = np.sum( interp_mu_F( out_Rsat ) ) / len( out_Rsat )


	in_interp_mu_F = interp.interp1d( tmp_bR[0][kk], tmp_BG[0][kk], kind = 'linear',)
	out_interp_mu_F = interp.interp1d( tmp_bR[1][kk], tmp_BG[1][kk], kind = 'linear',)

	_kk_in_BG = np.sum( in_interp_mu_F( in_Rsat ) ) / len( in_Rsat )
	_kk_out_BG = np.sum( out_interp_mu_F( out_Rsat ) ) / len( out_Rsat )

	_kk_in_sb = tmp_sb[0][kk] - _kk_in_BG
	_kk_in_BG = np.ones( len(tmp_R[0][kk] ), ) * _kk_in_BG

	_kk_out_sb = tmp_sb[1][kk] - _kk_out_BG
	_kk_out_BG = np.ones( len( tmp_R[1][kk] ), ) * _kk_out_BG

	keys = ['r', 'sb', 'sb_err', 'bg_sb' ]
	values = [ tmp_R[0][kk], _kk_in_sb, tmp_err[0][kk], _kk_in_BG ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_%s_%s-band_aveg-Rsat_BG-sub_SB.csv' % (cat_lis[0], band[kk]),)

	keys = ['r', 'sb', 'sb_err', 'bg_sb' ]
	values = [ tmp_R[1][kk], _kk_out_sb, tmp_err[1][kk], _kk_out_BG ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_%s_%s-band_aveg-Rsat_BG-sub_SB.csv' % (cat_lis[1], band[kk]),)


##.. figs and comparison
in_nbg_R, in_nbg_SB, in_nbg_err = [], [], []
in_BGs = []

for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_inner-mem_%s-band_aveg-Rsat_BG-sub_SB.csv' % band[kk])

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )
	tt_bg = np.array( dat['bg_sb'] )[0]

	in_nbg_R.append( tt_r ) 
	in_nbg_SB.append( tt_sb )
	in_nbg_err.append( tt_sb_err )
	in_BGs.append( tt_bg )


out_nbg_R, out_nbg_SB, out_nbg_err = [], [], []
out_BGs = []

for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_outer-mem_%s-band_aveg-Rsat_BG-sub_SB.csv' % band[kk])

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )
	tt_bg = np.array( dat['bg_sb'] )[0]

	out_nbg_R.append( tt_r ) 
	out_nbg_SB.append( tt_sb )
	out_nbg_err.append( tt_sb_err )
	out_BGs.append( tt_bg )


cp_in_nbg_R, cp_in_nbg_SB, cp_in_nbg_err = [], [], []
cp_in_BGs = []

for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_inner-mem_%s-band_scaled-R_BG-sub_SB.csv' % band[kk])

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )
	tt_bg = np.array( dat['bg_sb'] )[0]

	cp_in_nbg_R.append( tt_r ) 
	cp_in_nbg_SB.append( tt_sb )
	cp_in_nbg_err.append( tt_sb_err )
	cp_in_BGs.append( tt_bg )


cp_out_nbg_R, cp_out_nbg_SB, cp_out_nbg_err = [], [], []
cp_out_BGs = []

for kk in range( 3 ):

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/Extend_BCGM_gri-common_outer-mem_%s-band_scaled-R_BG-sub_SB.csv' % band[kk])

	tt_r, tt_sb, tt_sb_err = np.array( dat['r'] ), np.array( dat['sb'] ), np.array( dat['sb_err'] )
	tt_bg = np.array( dat['bg_sb'] )[0]

	cp_out_nbg_R.append( tt_r ) 
	cp_out_nbg_SB.append( tt_sb )
	cp_out_nbg_err.append( tt_sb_err )
	cp_out_BGs.append( tt_bg )



##... figs
plt.figure()
ax1 = plt.subplot(111)

for kk in range( 3 ):

	ax1.errorbar( tmp_R[0][kk], tmp_sb[0][kk], yerr = tmp_err[0][kk], marker = '.', ls = line_s[0], color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.errorbar( tmp_R[1][kk], tmp_sb[1][kk], yerr = tmp_err[1][kk], marker = '.', ls = line_s[1], color = color_s[kk],
		ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

legend_2 = plt.legend( [ fig_name[0], fig_name[1] ], loc = 3, frameon = False,)

ax1.legend( loc = 1, frameon = False, fontsize = 12,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]')

ax1.set_ylim( 1e-3, 5e0)
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
ax1.set_yscale('log')

plt.savefig('/home/xkchen/sat_SB_compare.png', dpi = 300)
plt.close()


plt.figure()
ax1 = plt.subplot(111)

for kk in range( 3 ):

	ax1.errorbar( in_nbg_R[kk], in_nbg_SB[kk], yerr = in_nbg_err[kk], marker = '.', ls = line_s[0], 
		color = color_s[kk], ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.errorbar( out_nbg_R[kk], out_nbg_SB[kk], yerr = out_nbg_err[kk], marker = '.', ls = line_s[1], 
		color = color_s[kk], ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

	ax1.axhline( y = in_BGs[kk], ls = line_s[0], color = color_s[kk], xmin = 0.85, xmax = 1.0)
	ax1.axhline( y = out_BGs[kk], ls = line_s[1], color = color_s[kk], xmin = 0.85, xmax = 1.0)

legend_2 = plt.legend( [ fig_name[0], fig_name[1] ], loc = 3, frameon = False,)

ax1.legend( loc = 1, frameon = False,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]')

ax1.set_ylim( 1e-4, 5e0)
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
ax1.set_yscale('log')

plt.savefig('/home/xkchen/sat_aveg-Rsat_BG-sub_SB_compare.png', dpi = 300)
plt.close()


plt.figure()
ax1 = plt.subplot(111)

for kk in range( 3 ):

	ax1.errorbar( cp_in_nbg_R[kk], cp_in_nbg_SB[kk], yerr = cp_in_nbg_err[kk], marker = '.', ls = line_s[0], 
		color = color_s[kk], ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5,)

	ax1.errorbar( cp_out_nbg_R[kk], cp_out_nbg_SB[kk], yerr = cp_out_nbg_err[kk], marker = '.', ls = line_s[1], 
		color = color_s[kk], ecolor = color_s[kk], mfc = 'none', mec = color_s[kk], capsize = 1.5, label = '%s-band' % band[kk],)

	ax1.axhline( y = cp_in_BGs[kk], ls = line_s[0], color = color_s[kk], xmin = 0.85, xmax = 1.0)
	ax1.axhline( y = cp_out_BGs[kk], ls = line_s[1], color = color_s[kk], xmin = 0.85, xmax = 1.0)

legend_2 = plt.legend( [ fig_name[0], fig_name[1] ], loc = 3, frameon = False,)

ax1.legend( loc = 1, frameon = False,)
ax1.add_artist( legend_2 )

ax1.set_xscale('log')
ax1.set_xlabel('R [kpc]')

ax1.set_ylim( 1e-4, 5e0)
ax1.set_ylabel('$\\mu \; [nanomaggy \, / \, arcsec^{2}]$')
ax1.set_yscale('log')

plt.savefig('/home/xkchen/sat_scaled-R_BG-sub_SB_compare.png', dpi = 300)
plt.close()


raise


##... 2D image
def img_2D_compare():

	tmp_img = []

	##... sat SBs
	for mm in range( 2 ):
		
		sub_img = []

		for kk in range( 3 ):

			#. 2D image
			with h5py.File( path + 
				'Extend_BCGM_gri-common_%s_%s-band_Mean_jack_img_z-ref.h5' % (cat_lis[mm], band[kk]), 'r') as f:
				tt_img = np.array( f['a'] )

			idnn = np.isnan( tt_img )
			idy_lim, idx_lim = np.where( idnn == False)
			x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
			y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()

			xn, yn = 805 - x_lo_eff, 805 - y_lo_eff

			cut_img = tt_img[ y_lo_eff : y_up_eff, x_lo_eff : x_up_eff]

			sub_img.append( tt_img )

		tmp_img.append( sub_img )


	for kk in range( 3 ):

		tt_img_0 = tmp_img[0][kk]

		tt_img_1 = tmp_img[1][kk]

		diff_img = tt_img_0 - tt_img_1


		idnn = np.isnan( tt_img_0 )

		idy_lim, idx_lim = np.where( idnn == False)
		x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
		y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()

		cut_img_0 = tt_img_0[ y_lo_eff : y_up_eff, x_lo_eff : x_up_eff]


		idnn = np.isnan( tt_img_1 )

		idy_lim, idx_lim = np.where( idnn == False)
		x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
		y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()

		cut_img_1 = tt_img_1[ y_lo_eff : y_up_eff, x_lo_eff : x_up_eff]


		idnn = np.isnan( diff_img )

		idy_lim, idx_lim = np.where( idnn == False)
		x_lo_eff, x_up_eff = idx_lim.min(), idx_lim.max()
		y_lo_eff, y_up_eff = idy_lim.min(), idy_lim.max()

		cut_diffi = diff_img[ y_lo_eff : y_up_eff, x_lo_eff : x_up_eff]


		fig = plt.figure( figsize = (19.84, 4.8) )
		ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
		ax1 = fig.add_axes([0.36, 0.09, 0.30, 0.85])
		ax2 = fig.add_axes([0.70, 0.09, 0.25, 0.85])

		ax0.set_title( fig_name[0] + ',(%s)' % band[kk] )
		tf = ax0.imshow( cut_img_0 / pixel**2, origin = 'lower', cmap = 'Greys', vmin = -5e-2, vmax = 5e-2,)
		plt.colorbar( tf, ax = ax0, fraction = 0.040, pad = 0.01,)

		ax1.set_title( fig_name[1] )
		tf = ax1.imshow( cut_img_1 / pixel**2, origin = 'lower', cmap = 'Greys', vmin = -5e-2, vmax = 5e-2,)
		plt.colorbar( tf, ax = ax1, fraction = 0.040, pad = 0.01,)

		ax2.set_title( 'Difference' )
		tf = ax2.imshow( cut_diffi / pixel**2, origin = 'lower', cmap = 'bwr', vmin = -1e-2, vmax = 1e-2,)
		plt.colorbar( tf, ax = ax2, fraction = 0.040, pad = 0.01,)

		plt.savefig('/home/xkchen/%s-band_sat_2D_diffi.png' % band[kk], dpi = 300)
		plt.close()

	return

img_2D_compare()

