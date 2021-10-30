import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func
from light_measure import light_measure_weit
from img_cat_param_match import match_func
from img_pre_selection import extra_match_func

# cosmology model
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
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec


### the total photo-z img (with img selection applied)
home = '/home/xkchen/mywork/ICL/data/'

### === ### mass_bin
lo_dat = pds.read_csv( home + 'BCG_stellar_mass_cat/low_star-Mass_cat.csv')
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_rich, lo_M_star = np.array(lo_dat.rich), np.array(lo_dat.lg_Mass)
idlx = (lo_z >= 0.2) & (lo_z <= 0.3)
print('low, N = ', np.sum(idlx) )
print('low, N = ', len(lo_ra) )

hi_dat = pds.read_csv( home + 'BCG_stellar_mass_cat/high_star-Mass_cat.csv')
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_rich, hi_M_star = np.array(hi_dat.rich), np.array(hi_dat.lg_Mass)
idhx = (hi_z >= 0.2) & (hi_z <= 0.3)
print('high, N = ', np.sum(idhx) )
print('high, N = ', len(hi_ra) )

### === ### age bin, g r i band common catalog
cat = pds.read_csv( home + 'cat_z_form/clslowz_z0.17-0.30_bc03_cat.csv' )
ra, dec, z = np.array(cat['ra']), np.array(cat['dec']), np.array(cat['z'])
rich, z_form = np.array(cat['lambda']), np.array(cat['z_form'])

#. mass unit : M_sun / h^2
lg_Mstar_z_phot = np.array( cat['lg_M*_photo_z'] )
origin_ID = np.arange( 0, len(z), )

## estimate difference of lookback time from z_formed to z_obs
lb_time_0 = Test_model.lookback_time( z ).value
lb_time_1 = Test_model.lookback_time( z_form ).value
age_time = lb_time_1 - lb_time_0

## divide line for age-bin sample
pre_N = 6
bins_edg = np.logspace( np.log10(19), np.log10(100), pre_N)
bins_edg = np.r_[ bins_edg, 200]

medi_age = []
cen_rich = 0.5 * (bins_edg[1:] + bins_edg[:-1])

for ii in range( pre_N ):

	id_lim = (rich >= bins_edg[ii] ) & ( rich < bins_edg[ii+1] )
	lim_age = age_time[ id_lim ]

	medi_age.append( np.median( lim_age ) )

medi_age = np.array( medi_age )

rich_x = np.logspace( np.log10(19), np.log10(200), 25)

## use a simple power law 
fit_F = np.polyfit( np.log10(cen_rich), np.log10( medi_age), deg = 2)

Pf = np.poly1d( fit_F )
fit_line = 10**( Pf( np.log10( rich_x) ) )

#...save this line
# keys = [ 'rich', 'age' ]
# values = [ rich_x, fit_line ]
# fill = dict( zip(keys, values) )
# data = pds.DataFrame( fill )
# data.to_csv( '/home/xkchen/tmp_run/data_files/figs/age-bin_fixed-rich_divid_line.csv' )


cat_lis_0 = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
out_path_0 ='/home/xkchen/mywork/ICL/data/BCG_stellar_mass_cat/photo_z_gri_common/'

cat_lis_1 = ['younger', 'older']
out_path_1 = '/home/xkchen/mywork/ICL/data/cat_z_form/age_bin_cat/gri_common_cat/'

sf_len = 5 ## at least 5
f2str = '%.' + '%df' % sf_len

"""
for kk in range( 0,1 ):
	## mass-bin sample
	sub_lo_dat = pds.read_csv( out_path_0 + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis_0[0], band[kk]),)
	sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
	sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])

	sub_lo_rich = lo_rich[sub_lo_origin_dex]
	sub_lo_Mstar = lo_M_star[sub_lo_origin_dex]


	sub_hi_dat = pds.read_csv( out_path_0 + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis_0[1], band[kk]),)
	sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
	sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])

	sub_hi_rich = hi_rich[sub_hi_origin_dex]
	sub_hi_Mstar = hi_M_star[sub_hi_origin_dex]

	## match to the age information
	out_ra = [ f2str % ll for ll in ra ]
	out_dec = [ f2str % ll for ll in dec]
	out_z = [ f2str % ll for ll in z ]

	psu_imgx, psu_imgy = np.ones( len(sub_lo_ra),), np.ones( len(sub_lo_dec),)
	lo_list_dex = extra_match_func( out_ra, out_dec, out_z, sub_lo_ra, sub_lo_dec, sub_lo_z, psu_imgx, psu_imgy, sf_len = sf_len,)[-1]

	sub_lo_age = age_time[ lo_list_dex ]

	psu_imgx, psu_imgy = np.ones( len(sub_hi_ra),), np.ones( len(sub_hi_dec),)
	hi_list_dex = extra_match_func( out_ra, out_dec, out_z, sub_hi_ra, sub_hi_dec, sub_hi_z, psu_imgx, psu_imgy, sf_len = sf_len,)[-1]

	sub_hi_age = age_time[ hi_list_dex ]

	#...save the age information into catalog files
	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	# values = [ sub_lo_ra, sub_lo_dec, sub_lo_z, rich[ lo_list_dex ], lg_Mstar_z_phot[lo_list_dex], age_time[lo_list_dex] ]
	# fill = dict( zip(keys, values) )
	# data = pds.DataFrame( fill )
	# data.to_csv( out_path_0 + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis_0[0] )

	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	# values = [ sub_hi_ra, sub_hi_dec, sub_hi_z, rich[ hi_list_dex ], lg_Mstar_z_phot[hi_list_dex], age_time[hi_list_dex] ]
	# fill = dict( zip(keys, values) )
	# data = pds.DataFrame( fill )
	# data.to_csv( out_path_0 + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis_0[1] )


	plt.figure()
	plt.title( 'sample richness check' )
	plt.hist( sub_lo_rich, bins = 50, density = True, histtype = 'step', color = 'b', alpha = 0.65, label = 'low $M^{BCG}_{\\ast}$',)
	plt.axvline( x = np.median(sub_lo_rich), ls = '-', color = 'b', alpha = 0.65, label = 'median')
	plt.axvline( x = np.mean(sub_lo_rich), ls = '--', color = 'b', alpha = 0.65, label = 'mean')

	plt.hist( sub_hi_rich, bins = 50, density = True, histtype = 'step', color = 'r', alpha = 0.65, label = 'high $M^{BCG}_{\\ast}$',)
	plt.axvline( x = np.median(sub_hi_rich), ls = '-', color = 'r', alpha = 0.65,)
	plt.axvline( x = np.mean(sub_hi_rich), ls = '--', color = 'r', alpha = 0.65,)

	tot_rich = np.r_[ lo_rich, hi_rich ]

	plt.hist( tot_rich, bins = 50, density = True, histtype = 'step', color = 'k', alpha = 0.5, label = 'total',)
	plt.axvline( x = np.median( tot_rich), ls = '-', color = 'k', alpha = 0.5,)
	plt.axvline( x = np.mean( tot_rich), ls = '--', color = 'k', alpha = 0.5,)

	plt.xscale( 'log' )
	plt.yscale( 'log' )
	plt.legend( loc = 1 )
	plt.xlabel( '$\\lambda$' )
	plt.ylabel( 'pdf' )
	plt.savefig('/home/xkchen/mass-bin_richness_check.png', dpi = 300)
	plt.close()


	y_lim_lo = np.min( [ sub_lo_Mstar.min(), sub_hi_Mstar.min() ] )
	y_lim_up = np.max( [ sub_lo_Mstar.max(), sub_hi_Mstar.max() ] )

	fig = plt.figure( )
	ax = fig.add_axes( [0.17, 0.12, 0.80, 0.80] )
	ax0 = fig.add_axes( [ 0.65, 0.26, 0.30, 0.25] )
	ax1 = fig.add_axes( [ 0.65, 0.24, 0.30, 0.02] )

	_point_rich = np.array([20, 30, 40, 50, 100, 200])
	line_divi = 0.446 * np.log10( _point_rich ) + 10.518

	ax.scatter( sub_lo_rich, sub_lo_Mstar, s = 15, c = sub_lo_age, cmap = 'bwr', alpha = 0.5, vmin = 1, vmax = 11,)
	ax.scatter( sub_hi_rich, sub_hi_Mstar, s = 15, c = sub_hi_age, cmap = 'bwr', alpha = 0.5, vmin = 1, vmax = 11,)
	ax.plot( _point_rich, line_divi, ls = '-', color = 'k', alpha = 0.5,)

	age_edgs = np.logspace( 0, np.log10(12), 50)
	ax0.hist( sub_lo_age, bins = age_edgs, density = True, histtype = 'step', color = 'b', alpha = 0.5, )
	ax0.hist( sub_hi_age, bins = age_edgs, density = True, histtype = 'step', color = 'r', alpha = 0.5, )
	ax0.set_xlim( 1, 11 )
	ax0.set_ylabel('pdf')

	cmap = mpl.cm.bwr
	norm = mpl.colors.Normalize( vmin = 1, vmax = 11 )

	c_ticks = np.array([1, 3, 5, 7, 9, 11])
	cbs = mpl.colorbar.ColorbarBase( ax = ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, orientation = 'horizontal',)
	cbs.set_label( 'age [Gyr]' )

	cmap.set_under('cyan')
	cbs.ax.set_xticklabels( labels = ['%.0f' % ll for ll in c_ticks ] )
	ax0.set_xticks( [] )


	ax.set_xlim( 20, 200 )
	ax.set_ylim( 10.0, 12.0 )

	ax.set_xscale( 'log' )
	ax.set_xlabel( '$ \\lambda $' , fontsize = 15,)
	ax.set_ylabel( '$ lgM^{BCG}_{\\ast} [M_{\\odot} / h^2] $', fontsize = 15)

	ax.text( 1e2, 11.50, s = 'high $M_{\\ast}^{BCG}$', fontsize = 15, rotation = 10,)
	ax.text( 1e2, 11.15, s = 'low $M_{\\ast}^{BCG}$', fontsize = 15, rotation = 10,)

	xtick_arr = [20, 30, 40, 50, 100, 200]
	tick_lis = [ '%d' % ll for ll in xtick_arr ]
	ax.set_xticks( xtick_arr, minor = True,)
	ax.set_xticklabels( labels = tick_lis, fontsize = 15, minor = True,)

	ax.set_xticks( [ 100 ] )
	ax.set_xticklabels( labels = ['100'], fontsize = 15 )

	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/mass-bin_sample.png', dpi = 300)
	plt.close()
"""

for kk in range( 0,1 ):
	## age-bin sample
	sub_lo_dat = pds.read_csv( out_path_1 + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis_1[0], band[kk]),)
	sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
	sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])

	sub_lo_rich = rich[sub_lo_origin_dex]

	sub_lo_Mstar = lg_Mstar_z_phot[sub_lo_origin_dex]

	sub_lo_age = age_time[sub_lo_origin_dex]


	sub_hi_dat = pds.read_csv( out_path_1 + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis_1[1], band[kk]),)
	sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
	sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])

	sub_hi_rich = rich[sub_hi_origin_dex]

	sub_hi_Mstar = lg_Mstar_z_phot[sub_hi_origin_dex]

	sub_hi_age = age_time[sub_hi_origin_dex ]

	#...save the age information into catalog files
	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	# values = [ sub_lo_ra, sub_lo_dec, sub_lo_z, sub_lo_rich, sub_lo_Mstar, sub_lo_age ]
	# fill = dict( zip(keys, values) )
	# data = pds.DataFrame( fill )
	# data.to_csv( out_path_1 + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis_1[0] )

	# keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	# values = [ sub_hi_ra, sub_hi_dec, sub_hi_z, sub_hi_rich, sub_hi_Mstar, sub_hi_age ]
	# fill = dict( zip(keys, values) )
	# data = pds.DataFrame( fill )
	# data.to_csv( out_path_1 + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis_1[1] )


	y_lim_lo = np.min( [ sub_lo_Mstar.min(), sub_hi_Mstar.min() ] )
	y_lim_up = np.max( [ sub_lo_Mstar.max(), sub_hi_Mstar.max() ] )

	plt.figure()
	plt.title( 'sample richness check' )
	plt.hist( sub_lo_rich, bins = 50, density = True, histtype = 'step', color = 'b', alpha = 0.65, label = 'low $M^{BCG}_{\\ast}$',)
	plt.axvline( x = np.median(sub_lo_rich), ls = '-', color = 'b', alpha = 0.65, label = 'median')
	plt.axvline( x = np.mean(sub_lo_rich), ls = '--', color = 'b', alpha = 0.65, label = 'mean')

	plt.hist( sub_hi_rich, bins = 50, density = True, histtype = 'step', color = 'r', alpha = 0.65, label = 'high $M^{BCG}_{\\ast}$',)
	plt.axvline( x = np.median(sub_hi_rich), ls = '-', color = 'r', alpha = 0.65,)
	plt.axvline( x = np.mean(sub_hi_rich), ls = '--', color = 'r', alpha = 0.65,)

	plt.hist( rich, bins = 50, density = True, histtype = 'step', color = 'k', alpha = 0.5, label = 'total',)
	plt.axvline( x = np.median( rich), ls = '-', color = 'k', alpha = 0.5,)
	plt.axvline( x = np.mean( rich), ls = '--', color = 'k', alpha = 0.5,)

	plt.xscale( 'log' )
	plt.yscale( 'log' )
	plt.legend( loc = 1 )
	plt.xlabel( '$\\lambda$' )
	plt.ylabel( 'pdf' )
	plt.savefig('/home/xkchen/age-bin_richness_check.png', dpi = 300)
	plt.close()


	fig = plt.figure( )
	ax = fig.add_axes( [0.12, 0.12, 0.80, 0.80] )
	ax0 = fig.add_axes( [ 0.59, 0.26, 0.30, 0.25] )
	ax1 = fig.add_axes( [ 0.59, 0.24, 0.30, 0.02] )

	ax.scatter( sub_lo_rich, sub_lo_age, s = 15, c = sub_lo_Mstar, cmap = 'bwr', alpha = 0.5, vmin = 10, vmax = 12,)
	ax.scatter( sub_hi_rich, sub_hi_age, s = 15, c = sub_hi_Mstar, cmap = 'bwr', alpha = 0.5, vmin = 10, vmax = 12,)

	mass_edgs = np.linspace( 10, 12, 50)
	ax0.hist( sub_lo_Mstar, bins = mass_edgs, density = True, histtype = 'step', color = 'b', alpha = 0.5, )
	ax0.hist( sub_hi_Mstar, bins = mass_edgs, density = True, histtype = 'step', color = 'r', alpha = 0.5, )
	ax0.set_xlim( 10, 12 )
	ax0.set_ylabel('pdf')

	# color bar
	cmap = mpl.cm.bwr
	# norm = mpl.colors.Normalize( vmin = y_lim_lo, vmax = y_lim_up )
	norm = mpl.colors.Normalize( vmin = 10, vmax = 12 )

	c_ticks = ax0.get_xticks()

	cbs = mpl.colorbar.ColorbarBase( ax = ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, orientation = 'horizontal',)
	cbs.set_label( '$ lgM^{BCG}_{\\ast} [M_{\\odot} / h^2] $' )

	cmap.set_under('cyan')
	cbs.ax.set_xticklabels( labels = ['%.1f' % ll for ll in c_ticks ] )
	ax0.set_xticks( [] )

	ax.plot( rich_x, fit_line, 'k-', alpha = 0.5,)
	ax.set_xlim( 20, 200 )
	ax.set_ylim( 1, 12)

	ax.set_ylabel('Age [Gyr]', fontsize = 15,)
	ax.set_xlabel('$\\lambda$', fontsize = 15,)
	ax.set_xscale( 'log' )
	ax.set_yscale( 'log' )

	x_ticks = np.array( [ 20, 30, 40, 50, 100, 200] )

	ax.set_xticks( x_ticks, minor = True,)
	ax.set_xticklabels( labels = [ '%d' % ll for ll in x_ticks ], minor = True, fontsize = 15,)

	ax.set_xticks( [ 100 ] )
	ax.set_xticklabels( labels = [ '100' ],fontsize = 15,)

	y_ticks = np.array( [1, 2, 3, 4, 5, 10] )
	ax.set_yticks( y_ticks, minor = True,)
	ax.set_yticklabels( labels = [ '%d' % ll for ll in y_ticks ], minor = True, fontsize = 15,)

	ax.set_yticks( [ 1, 10 ] )
	ax.set_yticklabels( labels = [ '1', '10'],fontsize = 15,)

	ax.text( 1e2, 1e1, s = 'older', fontsize = 15)
	ax.text( 1e2, 6e0, s = 'younger', fontsize = 15)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 12,)

	plt.savefig('/home/xkchen/age-bin_sample.png', dpi = 300)
	plt.close()
