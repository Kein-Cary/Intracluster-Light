import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.patches import Circle, Ellipse, Rectangle

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
import scipy.signal as signal
from scipy.interpolate import splev, splrep

from surface_mass_density import sigmam, sigmac, input_cosm_model, cosmos_param, rhom_set

### === ### cosmology
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
Omega_b = Test_model.Ob0

pixel = 0.396
band = ['r', 'g', 'i']
L_wave = np.array([ 6166, 4686, 7480 ])


### ==== ### comparison on the entire sample
#... sample properties
def hist_view():

	cat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/Extend_BCGM_bin_cat.csv')
	ref_ra, ref_dec, ref_z = np.array( cat['ra'] ), np.array( cat['dec'] ), np.array( cat['z'] )
	rich, lg_Mstar = np.array( cat['rich'] ), np.array( cat['lg_Mbcg'] )

	ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg )


	lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'low_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
	lo_rich, lo_lgMstar = np.array( lo_dat['rich']), np.array( lo_dat['lg_Mstar'] )

	hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
						'high_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
	hi_rich, hi_lgMstar = np.array( hi_dat['rich']), np.array( hi_dat['lg_Mstar'] )

	tot_rich = np.r_[ lo_rich, hi_rich ]
	tot_lgMstar = np.r_[ lo_lgMstar, hi_lgMstar ]
	tot_z = np.r_[ lo_z, hi_z ]
	tot_ra = np.r_[ lo_ra, hi_ra ]
	tot_dec = np.r_[ lo_dec, hi_dec ]

	tot_coord = SkyCoord( ra = tot_ra * U.deg, dec = tot_dec * U.deg )


	#.. previous catalog
	pre_lo_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
							'low_BCG_star-Mass_r-band_photo-z-match_rgi-common_cat_params.csv')
	pre_lo_ra, pre_lo_dec, pre_lo_z = np.array( pre_lo_dat['ra'] ), np.array( pre_lo_dat['dec'] ), np.array( pre_lo_dat['z'] )
	pre_lo_rich, pre_lo_lgMstar = np.array( pre_lo_dat['rich']), np.array( pre_lo_dat['lg_Mstar'] )

	pre_hi_dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/' + 
							'high_BCG_star-Mass_r-band_photo-z-match_rgi-common_cat_params.csv')
	pre_hi_ra, pre_hi_dec, pre_hi_z = np.array( pre_hi_dat['ra'] ), np.array( pre_hi_dat['dec'] ), np.array( pre_hi_dat['z'] )
	pre_hi_rich, pre_hi_lgMstar = np.array( pre_hi_dat['rich']), np.array( pre_hi_dat['lg_Mstar'] )


	pre_tot_rich = np.r_[ pre_lo_rich, pre_hi_rich ]
	pre_tot_lgMstar = np.r_[ pre_lo_lgMstar, pre_hi_lgMstar ]
	pre_tot_z = np.r_[ pre_lo_z, pre_hi_z ]
	pre_tot_ra = np.r_[ pre_lo_ra, pre_hi_ra ]
	pre_tot_dec = np.r_[ pre_lo_dec, pre_hi_dec ]

	pre_coord = SkyCoord( ra = pre_tot_ra * U.deg, dec = pre_tot_dec * U.deg )

	idx, sep, d3d = tot_coord.match_to_catalog_sky( pre_coord )
	id_lim = sep.value > 2.7e-4

	dmp_ra, dmp_dec, dmp_z = tot_ra[ id_lim ], tot_dec[ id_lim ], tot_z[ id_lim ]

	keys = ['ra', 'dec', 'z']
	values = [ dmp_ra, dmp_dec, dmp_z ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/gri_diff_cat.csv',)


	raise

	plt.figure()
	plt.hist( tot_rich, bins = 45, density = True, histtype = 'step', color = 'r', alpha = 0.75, label = 'Now')
	plt.axvline( x = np.median(tot_rich), ls = '-', color = 'r', alpha = 0.75, label = 'Median')
	plt.axvline( x = np.mean(tot_rich), ls = '--', color = 'r', alpha = 0.75, label = 'Mean')

	plt.hist( pre_tot_rich, bins = 45, density = True, histtype = 'step', color = 'b', alpha = 0.75, label = 'Previous')
	plt.axvline( x = np.median(pre_tot_rich), ls = '-', color = 'b', alpha = 0.75,)
	plt.axvline( x = np.mean(pre_tot_rich), ls = '--', color = 'b', alpha = 0.75,)

	plt.legend( loc = 1 )
	plt.xlabel('$\\lambda$')
	plt.xscale('log')
	plt.yscale('log')
	plt.savefig('/home/xkchen/rich_compare.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( tot_lgMstar, bins = 45, density = True, histtype = 'step', color = 'r', alpha = 0.75, label = 'Now')
	plt.axvline( x = np.log10( np.median(10**tot_lgMstar) ), ls = '-', color = 'r', alpha = 0.75, label = 'Median')
	plt.axvline( x = np.log10( np.mean(10**tot_lgMstar) ), ls = '--', color = 'r', alpha = 0.75, label = 'Mean')

	plt.hist( pre_tot_lgMstar, bins = 45, density = True, histtype = 'step', color = 'b', alpha = 0.75, label = 'Previous')
	plt.axvline( x = np.log10( np.median(10**pre_tot_lgMstar) ), ls = '-', color = 'b', alpha = 0.75,)
	plt.axvline( x = np.log10( np.mean(10**pre_tot_lgMstar) ), ls = '--', color = 'b', alpha = 0.75,)

	plt.legend( loc = 1 )
	plt.xlabel('$\\lg \, M_{\\ast} \; [M_{\\odot} / h^{2}]$')
	plt.savefig('/home/xkchen/lgMstar_compare.png', dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( tot_z, bins = 45, density = True, histtype = 'step', color = 'r', alpha = 0.75, label = 'Now')
	plt.axvline( x = np.median(tot_z), ls = '-', color = 'r', alpha = 0.75, label = 'Median')
	plt.axvline( x = np.mean(tot_z), ls = '--', color = 'r', alpha = 0.75, label = 'Mean')

	plt.hist( pre_tot_z, bins = 45, density = True, histtype = 'step', color = 'b', alpha = 0.75, label = 'Previous')
	plt.axvline( x = np.median(pre_tot_z), ls = '-', color = 'b', alpha = 0.75,)
	plt.axvline( x = np.mean(pre_tot_z), ls = '--', color = 'b', alpha = 0.75,)

	plt.legend( loc = 1 )
	plt.xlabel('$\\lambda$')
	plt.savefig('/home/xkchen/z_compare.png', dpi = 300)
	plt.close()


hist_view()
raise


#... SB and SM profiles
BG_path = '/home/xkchen/figs/extend_bcgM_cat/BGs/'

dat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat/SM_pros/' + 
					'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi_with-dered.csv' )
obs_R, obs_SM, obs_SM_err = np.array(dat['R']), np.array(dat['surf_mass']), np.array(dat['surf_mass_err'])


nbg_tot_r = []
nbg_tot_mag, nbg_tot_mag_err = [], []

for kk in range( 3 ):
	with h5py.File( BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	nbg_tot_r.append( tt_r )
	nbg_tot_mag.append( tt_mag )
	nbg_tot_mag_err.append( tt_mag_err )

c_dat = pds.read_csv( BG_path + 'photo-z_tot-BCG-star-Mass_dered_color_profile.csv' )
dered_R, dered_g2r, dered_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
dered_r2i, dered_r2i_err = np.array( c_dat['r-i'] ), np.array( c_dat['r-i_err'] )
sm_dered_g2r = signal.savgol_filter( dered_g2r, 7, 3)
sm_dered_r2i = signal.savgol_filter( dered_r2i, 7, 3)

#... 
pre_BG_path = '/home/xkchen/figs/re_measure_SBs/BGs/'

p_dat = pds.read_csv( '/home/xkchen/figs/re_measure_SBs/SM_profile/' + 
					  'photo-z_tot-BCG-star-Mass_gri-band-based_aveg-jack_mass-Lumi.csv',)
p_obs_R, p_obs_SM, p_obs_SM_err = np.array(p_dat['R']), np.array(p_dat['surf_mass']), np.array(p_dat['surf_mass_err'])


pre_tot_r = []
pre_tot_mag, pre_tot_mag_err = [], []

for kk in range( 3 ):
	with h5py.File( pre_BG_path + 'photo-z_tot-BCG-star-Mass_%s-band_BG-sub_SB.h5' % band[kk], 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
		tt_err = np.array(f['sb_err'])

	tt_mag = 22.5 - 2.5 * np.log10( tt_sb )
	tt_mag_err = 2.5 * tt_err / ( np.log(10) * tt_sb )

	pre_tot_r.append( tt_r )
	pre_tot_mag.append( tt_mag )
	pre_tot_mag_err.append( tt_mag_err )

c_dat = pds.read_csv( pre_BG_path + 'photo-z_tot-BCG-star-Mass_dered_color_profile.csv' )
pre_dered_R, pre_dered_g2r, pre_dered_g2r_err = np.array( c_dat['R_kpc'] ), np.array( c_dat['g-r'] ), np.array( c_dat['g-r_err'] )
pre_dered_r2i, pre_dered_r2i_err = np.array( c_dat['r-i'] ), np.array( c_dat['r-i_err'] )
sm_pre_dered_g2r = signal.savgol_filter( pre_dered_g2r, 7, 3)
sm_pre_dered_r2i = signal.savgol_filter( pre_dered_r2i, 7, 3)



#... figs
color_s = [ 'r', 'g', 'darkred' ]
line_c = [ 'b', 'r'  ]
line_s = [ '--', '-' ]


fig = plt.figure( figsize = (5.8, 5.4) )
ax = fig.add_axes( [0.12, 0.32, 0.83, 0.63] )
sub_ax = fig.add_axes( [0.12, 0.11, 0.83, 0.21] )

for kk in range( 3 ):

	l1, = ax.plot( nbg_tot_r[kk], nbg_tot_mag[kk], ls = '-', color = color_s[kk], alpha = 0.75, label = '%s' % band[kk],)
	ax.fill_between( nbg_tot_r[kk], y1 = nbg_tot_mag[kk] - nbg_tot_mag_err[kk], 
					y2 = nbg_tot_mag[kk] + nbg_tot_mag_err[kk], color = color_s[kk], alpha = 0.15,)
	l2, = ax.plot( pre_tot_r[kk], pre_tot_mag[kk], ls = '--', color = color_s[kk], alpha = 0.75,)

	sub_ax.plot( pre_tot_r[kk], nbg_tot_mag[kk] - pre_tot_mag[kk], ls = '-', color = color_s[kk], alpha = 0.75,)

legend_2 = ax.legend( handles = [ l1, l2 ], labels = ['Now', 'Previous'], loc = 3, frameon = False, fontsize = 15,)
legend_20 = ax.legend( loc = 1, frameon = False, fontsize = 15,)
ax.add_artist( legend_2 )

ax.set_xlim( 1e0, 1e3)
ax.set_xscale('log')

ax.set_ylim( 19.5, 34 )
ax.invert_yaxis()
ax.set_ylabel('SB [mag / $arcsec^2$]', fontsize = 15,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xlabel('R [kpc]', fontsize = 15,)
sub_ax.set_xscale('log')

sub_ax.set_ylim( 0.1, -1.5 )
sub_ax.set_ylabel('$SB_{now} - SB_{previous}$', fontsize = 15,)
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)
ax.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/total_sample_SB_compare.png', dpi = 300)
plt.close()


fig = plt.figure( figsize = (5.8, 5.4) )
ax = fig.add_axes( [0.12, 0.32, 0.83, 0.63] )
sub_ax = fig.add_axes( [0.12, 0.11, 0.83, 0.21] )

ax.plot( p_obs_R, p_obs_SM, ls = '--', color = 'b', alpha = 0.5, label = 'Previous',)
ax.fill_between( p_obs_R, y1 = p_obs_SM - p_obs_SM_err, y2 = p_obs_SM + p_obs_SM_err, color = 'b', alpha = 0.12,)

ax.plot( obs_R, obs_SM, ls = '-', color = 'r', alpha = 0.5, label = 'Now',)
ax.fill_between( obs_R, y1 = obs_SM - obs_SM_err, y2 = obs_SM + obs_SM_err, color = 'r', alpha = 0.12,)

ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 6e3, 3e9 )
ax.set_yscale( 'log' )
ax.set_xlim( 1e0, 1e3 )
ax.set_xscale('log')
ax.set_ylabel('$\\Sigma_{\\ast} \; [M_{\\odot}]$', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

sub_ax.plot( p_obs_R, obs_SM / p_obs_SM, ls = '-', color = 'r',)
sub_ax.axhline( y = 1, ls = ':', color = 'k', alpha = 0.75,)
sub_ax.axhline( y = 0.8, ls = '-.', color = 'k', alpha = 0.75,)

sub_ax.set_xlim( ax.get_xlim() )
sub_ax.set_xlabel('R [kpc]', fontsize = 12)
sub_ax.set_xscale('log')
sub_ax.set_ylim( 0.55, 1.1 )
sub_ax.set_ylabel('$\\Sigma_{\\ast}^{now} / \\Sigma_{\\ast}^{previous}$', fontsize = 12)
sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)
ax.set_xticklabels( labels = [] )

plt.savefig('/home/xkchen/all_sample_SM_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.plot( dered_R, sm_dered_g2r, ls = '-', color = 'r', alpha = 0.75, label = 'Now')
ax.fill_between( dered_R, y1 = sm_dered_g2r - dered_g2r_err, y2 = sm_dered_g2r + dered_g2r_err, color = 'r', alpha = 0.12,)

ax.plot( pre_dered_R, sm_pre_dered_g2r, ls = '--', color = 'r', alpha = 0.75, label = 'Previous')
ax.fill_between( pre_dered_R, y1 = sm_pre_dered_g2r - pre_dered_g2r_err, 
				y2 = sm_pre_dered_g2r + pre_dered_g2r_err, color = 'r', ls = '--', alpha = 0.12,)

ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 0.70, 1.55 )

ax.set_xlim(1e0, 1e3)
ax.set_xscale('log')
ax.set_ylabel('g - r', fontsize = 12,)
ax.set_xlabel('R [Mpc]', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/total-sample_g2r_compare.png', dpi = 300)
plt.close()


fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])

ax.plot( dered_R, sm_dered_r2i, ls = '-', color = 'r', alpha = 0.75, label = 'Now')
ax.fill_between( dered_R, y1 = sm_dered_r2i - dered_r2i_err, y2 = sm_dered_r2i + dered_r2i_err, color = 'r', alpha = 0.12,)

ax.plot( pre_dered_R, sm_pre_dered_r2i, ls = '--', color = 'r', alpha = 0.75, label = 'Previous')
ax.fill_between( pre_dered_R, y1 = sm_pre_dered_r2i - pre_dered_r2i_err, 
				y2 = sm_pre_dered_r2i + pre_dered_r2i_err, color = 'r', ls = '--', alpha = 0.12,)

ax.legend( loc = 3, frameon = False,)
ax.set_ylim( 0.45, 0.95 )

ax.set_xlim(1e0, 1e3)
ax.set_xscale('log')
ax.set_ylabel('r - i', fontsize = 12,)
ax.set_xlabel('R [Mpc]', fontsize = 12,)
ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

plt.savefig('/home/xkchen/total-sample_r2i_compare.png', dpi = 300)
plt.close()

