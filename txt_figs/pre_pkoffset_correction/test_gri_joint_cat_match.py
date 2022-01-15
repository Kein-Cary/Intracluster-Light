## this file use for extra-catalog (not original catalog of the redMapper image info.)
## match and properties comparison
import glob
import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import subprocess as subpro
import astropy.constants as C

from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binned
from astropy import cosmology as apcy
from fig_out_module import zref_BCG_pos_func

## cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

## constant
rad2asec = U.rad.to(U.arcsec)

def extra_match_func(ra_list, dec_list, z_lis, cat_ra, cat_dec, cat_z, cat_imgx, cat_imgy, sf_len = 5,):
	"""
	cat_imgx, cat_imgy : BCG location in image frame
	cat_ra, cat_dec, cat_z : catalog information of image catalog
	ra_list, dec_list, z_lis : catalog information of which used to match to the image catalog
	"""
	lis_ra, lis_dec, lis_z = [], [], []
	lis_x, lis_y = [], []

	com_s = '%.' + '%df' % sf_len

	origin_dex = []

	for kk in range( len(cat_ra) ):

		identi = ( com_s % cat_ra[kk] in ra_list) * (com_s % cat_dec[kk] in dec_list)# * (com_s % cat_z[kk] in z_lis)

		if identi == True:

			## use the location of the source in catalog to make sure they are the same objects in different catalog
			ndex_0 = ra_list.index( com_s % cat_ra[kk] )
			ndex_1 = dec_list.index( com_s % cat_dec[kk] )

			if ndex_0 == ndex_1:
				lis_ra.append( cat_ra[kk] )
				lis_dec.append( cat_dec[kk] )
				lis_z.append( cat_z[kk] )
				lis_x.append( cat_imgx[kk] )
				lis_y.append( cat_imgy[kk] )

				## origin_dex record the location of objs in the origin catalog (not the image catalog),
				origin_dex.append( ndex_0 )
			else:
				continue
		else:
			continue

	match_ra = np.array( lis_ra )
	match_dec = np.array( lis_dec )
	match_z = np.array( lis_z )
	match_x = np.array( lis_x )
	match_y = np.array( lis_y )
	origin_dex = np.array( origin_dex )

	return match_ra, match_dec, match_z, match_x, match_y, origin_dex

def gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,):
	"""
	origin_ID : the catalog location of the matched sources
	"""
	r_dat = pds.read_csv( r_band_file )
	r_ra, r_dec, r_z = np.array( r_dat['ra'] ), np.array( r_dat['dec'] ), np.array( r_dat['z'] )
	r_imgx, r_imgy, r_origin_ID = np.array( r_dat['bcg_x'] ), np.array( r_dat['bcg_y'] ), np.array( r_dat['origin_ID'] )

	g_dat = pds.read_csv( g_band_file )
	g_ra, g_dec, g_z = np.array( g_dat['ra'] ), np.array( g_dat['dec'] ), np.array( g_dat['z'] )
	g_imgx, g_imgy, g_origin_ID = np.array( g_dat['bcg_x'] ), np.array( g_dat['bcg_y'] ), np.array( g_dat['origin_ID'] )

	i_dat = pds.read_csv( i_band_file )
	i_ra, i_dec, i_z = np.array( i_dat['ra'] ), np.array( i_dat['dec'] ), np.array( i_dat['z'] )
	i_imgx, i_imgy, i_origin_ID = np.array( i_dat['bcg_x'] ), np.array( i_dat['bcg_y'] ), np.array( i_dat['origin_ID'] )

	N_r, N_g, N_i = len(r_origin_ID), len(g_origin_ID), len(i_origin_ID)

	### common of r and g band
	com_id = []
	sub_lis_r = []
	sub_lis_g = []

	for ii in range( N_r ):

		id_dex = np.abs(r_origin_ID[ ii ] - g_origin_ID)
		id_order = id_dex == 0

		if np.sum( id_order ) == 1:
			get_id = np.where( id_dex == 0)[0][0]

			com_id.append( g_origin_ID[ get_id ] )
			sub_lis_g.append( get_id )
			sub_lis_r.append( ii )

	### save medi catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ r_ra[sub_lis_r], r_dec[sub_lis_r], r_z[sub_lis_r], r_imgx[sub_lis_r], r_imgy[sub_lis_r], r_origin_ID[sub_lis_r] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( medi_r_file )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ g_ra[sub_lis_g], g_dec[sub_lis_g], g_z[sub_lis_g], g_imgx[sub_lis_g], g_imgy[sub_lis_g], g_origin_ID[sub_lis_g] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( medi_g_file )


	### match with i band
	medi_r_dat = pds.read_csv( medi_r_file )
	medi_r_ra, medi_r_dec, medi_r_z = np.array( medi_r_dat['ra'] ), np.array( medi_r_dat['dec'] ), np.array( medi_r_dat['z'] )
	medi_r_imgx, medi_r_imgy, medi_r_origin_ID = np.array( medi_r_dat['bcg_x'] ), np.array( medi_r_dat['bcg_y'] ), np.array( medi_r_dat['origin_ID'] )

	medi_g_dat = pds.read_csv( medi_g_file )
	medi_g_ra, medi_g_dec, medi_g_z = np.array( medi_g_dat['ra'] ), np.array( medi_g_dat['dec'] ), np.array( medi_g_dat['z'] )
	medi_g_imgx, medi_g_imgy, medi_g_origin_ID = np.array( medi_g_dat['bcg_x'] ), np.array( medi_g_dat['bcg_y'] ), np.array( medi_g_dat['origin_ID'] )


	N_mid = len( com_id )

	com_id_1 = []
	sub_lis_r_1 = []

	sub_lis_i = []

	for ii in range( N_mid ):

		id_dex = np.abs( medi_r_origin_ID[ ii ] - i_origin_ID )
		id_order = id_dex == 0

		if np.sum( id_order ) == 1:

			get_id = np.where( id_dex == 0)[0][0]
			com_id_1.append( i_origin_ID[ get_id ] )
			sub_lis_i.append( get_id )
			sub_lis_r_1.append( ii )

	### save the final common catalog
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ i_ra[sub_lis_i], i_dec[sub_lis_i], i_z[sub_lis_i], i_imgx[sub_lis_i], i_imgy[sub_lis_i], i_origin_ID[sub_lis_i] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_i_file )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ medi_r_ra[sub_lis_r_1], medi_r_dec[sub_lis_r_1], medi_r_z[sub_lis_r_1], 
				medi_r_imgx[sub_lis_r_1], medi_r_imgy[sub_lis_r_1], medi_r_origin_ID[sub_lis_r_1] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_r_file )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
	values = [ medi_g_ra[sub_lis_r_1], medi_g_dec[sub_lis_r_1], medi_g_z[sub_lis_r_1], 
				medi_g_imgx[sub_lis_r_1], medi_g_imgy[sub_lis_r_1], medi_g_origin_ID[sub_lis_r_1] ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_g_file )

	return

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.coordinates import SkyCoord

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

sf_len = 5 ## at least 5
f2str = '%.' + '%df' % sf_len

home = '/home/xkchen/mywork/ICL/data/'
out_path = '/home/xkchen/jupyter/tmp_cats/'

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

cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

### g,r,i band sub-samples match
sub_path = '/home/xkchen/Downloads/z_form_cat_check/match_fixed_cat/mass_bin/'
'''
for ll in range( 2 ):

	r_band_file = sub_path + '%s_r-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
	g_band_file = sub_path + '%s_g-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
	i_band_file = sub_path + '%s_i-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]

	medi_r_file = out_path + '%s_r-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	medi_g_file = out_path + '%s_g-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]

	out_r_file = out_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	out_g_file = out_path + '%s_g-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	out_i_file = out_path + '%s_i-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]

	gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,)
'''

'''
for kk in range( 3 ):

	sub_lo_dat = pds.read_csv( out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk]),)
	sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
	sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])
	sub_lo_rich = lo_rich[sub_lo_origin_dex]
	sub_lo_Mstar = lo_M_star[sub_lo_origin_dex]

	cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk])
	out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[0], band[kk])
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)	


	sub_hi_dat = pds.read_csv( out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk]),)
	sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
	sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])
	sub_hi_rich = hi_rich[sub_hi_origin_dex]
	sub_hi_Mstar = hi_M_star[sub_hi_origin_dex]

	cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk])
	out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[1], band[kk])
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	## figs
	plt.figure()
	plt.hist( sub_lo_z, bins = 50, color = 'b', alpha = 0.5, density = True, label = 'low $M_{\\ast}$',)
	plt.axvline( x = np.mean(sub_lo_z), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
	plt.axvline( x = np.median(sub_lo_z), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

	plt.hist( sub_hi_z, bins = 50, color = 'r', alpha = 0.5, density = True, label = 'high $M_{\\ast}$',)
	plt.axvline( x = np.mean(sub_hi_z), ls = '--', color = 'r', alpha = 0.5, )
	plt.axvline( x = np.median(sub_hi_z), ls = '-', color = 'r', alpha = 0.5, )

	plt.legend( loc = 2)
	plt.xlabel('photometric redshift of BCGs')
	plt.ylabel('pdf')
	plt.savefig('/home/xkchen/figs/%s_band_mass_bin_obs-z_compare.png' % band[kk], dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( sub_lo_rich, bins = 50, color = 'b', alpha = 0.5, density = True, label = 'low $M_{\\ast}$',)
	plt.axvline( x = np.mean(sub_lo_rich), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
	plt.axvline( x = np.median(sub_lo_rich), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

	plt.hist( sub_hi_rich, bins = 50, color = 'r', alpha = 0.5, density = True, label = 'high $M_{\\ast}$',)
	plt.axvline( x = np.mean(sub_hi_rich), ls = '--', color = 'r', alpha = 0.5, )
	plt.axvline( x = np.median(sub_hi_rich), ls = '-', color = 'r', alpha = 0.5, )

	plt.yscale('log')
	plt.xscale('log')
	plt.legend( loc = 1)
	plt.xlabel('$\\lambda$')
	plt.ylabel('pdf')
	plt.savefig('/home/xkchen/figs/%s_band_mass_bin_rich_compare.png' % band[kk], dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( sub_lo_Mstar, bins = 50, color = 'b', alpha = 0.5, density = True, label = 'low $M_{\\ast}$',)
	plt.axvline( np.log10( np.mean(10**sub_lo_Mstar) ), ls = '--', color = 'b', alpha = 0.5, label = 'Mean')
	plt.axvline( np.log10( np.median(10**sub_lo_Mstar) ), ls = '-', color = 'b', alpha = 0.5, label = 'Median')

	plt.hist( sub_hi_Mstar, bins = 50, color = 'r', alpha = 0.5, density = True, label = 'high $M_{\\ast}$',)
	plt.axvline( np.log10( np.mean(10**sub_hi_Mstar) ), ls = '--', color = 'r', alpha = 0.5, )
	plt.axvline( np.log10( np.median(10**sub_hi_Mstar) ), ls = '-', color = 'r', alpha = 0.5, )

	# plt.yscale('log')
	# plt.xscale('log')
	plt.legend( loc = 2)
	plt.xlabel( '$ lg(M_{\\ast}) [M_{\\odot} / h]$' )
	plt.ylabel('pdf')
	plt.savefig('/home/xkchen/figs/%s_band_mass_bin_Mstar_compare.png' % band[kk], dpi = 300)
	plt.close()

raise
'''

for kk in range( 3 ):

	lo_dat = pds.read_csv( sub_path + '%s_%s-band_photo-z-match_BCG-pos_cat.csv' % (cat_lis[0], band[kk]),)
	lo_ra, lo_dec, lo_z = np.array( lo_dat['ra'] ), np.array( lo_dat['dec'] ), np.array( lo_dat['z'] )
	lo_imgx, lo_imgy, lo_origin_ID = np.array( lo_dat['bcg_x'] ), np.array( lo_dat['bcg_y'] ), np.array( lo_dat['origin_ID'] )
	lo_Mstar = lo_M_star[ lo_origin_ID ]

	hi_dat = pds.read_csv( sub_path + '%s_%s-band_photo-z-match_BCG-pos_cat.csv' % (cat_lis[1], band[kk]),)
	hi_ra, hi_dec, hi_z = np.array( hi_dat['ra'] ), np.array( hi_dat['dec'] ), np.array( hi_dat['z'] )
	hi_imgx, hi_imgy, hi_origin_ID = np.array( hi_dat['bcg_x'] ), np.array( hi_dat['bcg_y'] ), np.array( hi_dat['origin_ID'] )
	hi_Mstar = hi_M_star[ hi_origin_ID ]

	plt.figure(  )
	plt.title( '%s band $ M_{\\ast} $ compare' % band[kk] )

	plt.hist( lo_Mstar, bins = 50, density = True, color = 'b', alpha = 0.5, label = 'low $M_{\\ast}$',)
	plt.axvline( np.log10( np.mean(10**lo_Mstar) ), ls = '-', color = 'b', alpha = 0.5, label = 'Mean')
	plt.axvline( np.log10( np.median(10**lo_Mstar) ), ls = '--', color = 'b', alpha = 0.5, label = 'Median')

	plt.hist( hi_Mstar, bins = 50, density = True, color = 'r', alpha = 0.5, label = 'high $M_{\\ast}$',)
	plt.axvline( np.log10( np.mean(10**hi_Mstar) ), ls = '-', color = 'r', alpha = 0.5, )
	plt.axvline( np.log10( np.median(10**hi_Mstar) ), ls = '--', color = 'r', alpha = 0.5, )

	plt.xlabel( 'lg($M_{\\ast}$)' ) 
	plt.ylabel('pdf')
	plt.legend( loc = 2 )

	plt.savefig('/home/xkchen/figs/%s-band_match_photo-z_M-star_compare.png' % band[kk], dpi = 300)
	plt.close()

raise


### === ### age bin, g r i band common catalog
cat = pds.read_csv( home + 'cat_z_form/clslowz_z0.17-0.30_bc03_cat.csv' )
ra, dec, z = np.array(cat['ra']), np.array(cat['dec']), np.array(cat['z'])
rich, z_form = np.array(cat['lambda']), np.array(cat['z_form'])
lg_Mstar_z_phot = np.array( cat['lg_M*_photo_z'] )
origin_ID = np.arange( 0, len(z), )

## estimate difference of lookback time from z_formed to z_obs
lb_time_0 = Test_model.lookback_time( z ).value
lb_time_1 = Test_model.lookback_time( z_form ).value
age_time = lb_time_1 - lb_time_0


lo_dat = pds.read_csv( home + 'cat_z_form/clslowz_z0.17-0.30_bc03_younger_cat.csv' )
lo_ra, lo_dec, lo_z = np.array( lo_dat.ra), np.array( lo_dat.dec), np.array( lo_dat.z)
lo_rich, lo_z_form = np.array( lo_dat.rich), np.array( lo_dat.z_form)
lo_origin_ID = np.array( lo_dat.origin_ID )
lo_origin_ID = lo_origin_ID.astype( int )

idlx = (lo_z >= 0.2) & (lo_z <= 0.3)
print('low, N = ', np.sum(idlx) )
print('low, N = ', len(lo_ra) )

hi_dat = pds.read_csv( home + 'cat_z_form/clslowz_z0.17-0.30_bc03_older_cat.csv' )
hi_ra, hi_dec, hi_z = np.array( hi_dat.ra), np.array( hi_dat.dec), np.array( hi_dat.z)
hi_rich, hi_z_form = np.array( hi_dat.rich), np.array( hi_dat.z_form)
hi_origin_ID = np.array( hi_dat.origin_ID )
hi_origin_ID = hi_origin_ID.astype( int )

idhx = (hi_z >= 0.2) & (hi_z <= 0.3)
print('high, N = ', np.sum(idhx) )
print('high, N = ', len(hi_ra) )

cat_lis = ['younger', 'older']

### g,r,i band sub-samples match
sub_path = '/home/xkchen/Downloads/z_form_cat_check/match_fixed_cat/age_bin/'
'''
for ll in range( 2 ):

	r_band_file = sub_path + '%s_r-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
	g_band_file = sub_path + '%s_g-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
	i_band_file = sub_path + '%s_i-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]

	medi_r_file = out_path + '%s_r-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	medi_g_file = out_path + '%s_g-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]

	out_r_file = out_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	out_g_file = out_path + '%s_g-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
	out_i_file = out_path + '%s_i-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]

	gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,)

for kk in range( 3 ):

	sub_lo_dat = pds.read_csv( out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk]),)
	sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
	sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])
	sub_lo_rich = rich[sub_lo_origin_dex]
	sub_lo_Mstar = lg_Mstar_z_phot[sub_lo_origin_dex]

	cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk])
	out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[0], band[kk])
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)	


	sub_hi_dat = pds.read_csv( out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk]),)
	sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
	sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])
	sub_hi_rich = rich[sub_hi_origin_dex]
	sub_hi_Mstar = lg_Mstar_z_phot[sub_hi_origin_dex]

	cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk])
	out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[1], band[kk])
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	print( len( sub_lo_ra) )
	print( len( sub_hi_ra) )
	## figs
	plt.figure()
	plt.hist( sub_lo_z, bins = 50, color = 'b', alpha = 0.5, density = True, label = 'younger',)
	plt.axvline( x = np.mean(sub_lo_z), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
	plt.axvline( x = np.median(sub_lo_z), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

	plt.hist( sub_hi_z, bins = 50, color = 'r', alpha = 0.5, density = True, label = 'older',)
	plt.axvline( x = np.mean(sub_hi_z), ls = '--', color = 'r', alpha = 0.5, )
	plt.axvline( x = np.median(sub_hi_z), ls = '-', color = 'r', alpha = 0.5, )

	plt.legend( loc = 2)
	plt.xlabel('photometric redshift of BCGs')
	plt.ylabel('pdf')
	plt.savefig('/home/xkchen/figs/%s_band_age_bin_obs-z_compare.png' % band[kk], dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( sub_lo_rich, bins = 50, color = 'b', alpha = 0.5, density = True, label = 'younger',)
	plt.axvline( x = np.mean(sub_lo_rich), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
	plt.axvline( x = np.median(sub_lo_rich), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

	plt.hist( sub_hi_rich, bins = 50, color = 'r', alpha = 0.5, density = True, label = 'older',)
	plt.axvline( x = np.mean(sub_hi_rich), ls = '--', color = 'r', alpha = 0.5, )
	plt.axvline( x = np.median(sub_hi_rich), ls = '-', color = 'r', alpha = 0.5, )

	plt.yscale('log')
	plt.xscale('log')
	plt.legend( loc = 1)
	plt.xlabel('$\\lambda$')
	plt.ylabel('pdf')
	plt.savefig('/home/xkchen/figs/%s_band_age_bin_rich_compare.png' % band[kk], dpi = 300)
	plt.close()


	plt.figure()
	plt.hist( sub_lo_Mstar, bins = 50, color = 'b', alpha = 0.5, density = True, label = 'younger',)
	plt.axvline( np.log10( np.mean(10**sub_lo_Mstar) ), ls = '--', color = 'b', alpha = 0.5, label = 'Mean')
	plt.axvline( np.log10( np.median(10**sub_lo_Mstar) ), ls = '-', color = 'b', alpha = 0.5, label = 'Median')

	plt.hist( sub_hi_Mstar, bins = 50, color = 'r', alpha = 0.5, density = True, label = 'older',)
	plt.axvline( np.log10( np.mean(10**sub_hi_Mstar) ), ls = '--', color = 'r', alpha = 0.5, )
	plt.axvline( np.log10( np.median(10**sub_hi_Mstar) ), ls = '-', color = 'r', alpha = 0.5, )

	plt.legend( loc = 2)
	plt.xlabel( '$ lg(M_{\\ast}) [M_{\\odot} / h]$' )
	plt.ylabel('pdf')
	plt.savefig('/home/xkchen/figs/%s_band_age_bin_Mstar_compare.png' % band[kk], dpi = 300)
	plt.close()
'''

