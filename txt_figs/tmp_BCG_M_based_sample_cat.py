import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc
import scipy.signal as signal

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
from scipy import interpolate as interp
from scipy import optimize

import matplotlib as mpl
import matplotlib.pyplot as plt

from img_pre_selection import cat_match_func
from img_pre_selection import extra_match_func, gri_common_cat_func
from fig_out_module import zref_BCG_pos_func
from fig_out_module import cc_grid_img, grid_img

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##### constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

path = '/home/xkchen/mywork/ICL/data/'

### === ### origin mass-bin sample
cat = pds.read_csv( path + 'cat_z_form/clslowz_z0.17-0.30_bc03_cat.csv' )
ra, dec, z = np.array(cat['ra']), np.array(cat['dec']), np.array(cat['z'])
rich, z_form = np.array(cat['lambda']), np.array(cat['z_form'])
lg_Mstar = np.array( cat['lg_M*_photo_z'] )

lb_time_0 = Test_model.lookback_time( z ).value
lb_time_1 = Test_model.lookback_time( z_form ).value
age_time = lb_time_1 - lb_time_0

origin_ID = np.arange( 0, len(z), )

def rich_binned_func(M_bins, data_arr, out_file_hi, out_file_low, divid_f):

	high_z_form = np.array( [] )
	hi_map_rich, hi_map_age = np.array( [] ), np.array( [] )
	hi_map_ra, hi_map_dec, hi_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	hi_map_Mstar = np.array( [] )
	hi_map_ID = np.array( [] )

	low_z_form = np.array( [] )
	lo_map_rich, lo_map_age = np.array( [] ), np.array( [] )
	lo_map_ra, lo_map_dec, lo_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	lo_map_Mstar = np.array( [] )
	lo_map_ID = np.array( [] )

	bins_lgM = M_bins
	N_bin = len( M_bins )
	ra, dec, z, z_form, rich, age_time, lg_Mstar, origin_ID = data_arr[:]

	for ii in range( N_bin - 1):

		id_lim = (lg_Mstar >= bins_lgM[ii] ) & (lg_Mstar < bins_lgM[ii+1] )

		axis_lgM = np.log10( (10**bins_lgM[ii] + 10**bins_lgM[ii+1]) / 2 )
		thresh_rich = 10**( divid_f(axis_lgM) )

		lim_rich = rich[ id_lim ]
		lim_ra, lim_dec, lim_zc = ra[id_lim], dec[id_lim], z[id_lim]

		lim_z = z_form[ id_lim ]
		lim_age = age_time[ id_lim ]
		lim_Mstar = lg_Mstar[ id_lim ]
		ID_lim = origin_ID[ id_lim ]


		idvx_hi = lim_rich >= thresh_rich
		idvx_lo = lim_rich < thresh_rich

		high_z_form = np.r_[ high_z_form, lim_z[ idvx_hi ] ]

		hi_map_rich = np.r_[ hi_map_rich, lim_rich[ idvx_hi ]  ]

		hi_map_ra = np.r_[ hi_map_ra, lim_ra[ idvx_hi ] ]

		hi_map_dec = np.r_[ hi_map_dec, lim_dec[ idvx_hi ] ]

		hi_map_z = np.r_[ hi_map_z, lim_zc[ idvx_hi ] ]

		hi_map_age = np.r_[ hi_map_age, lim_age[ idvx_hi ] ]

		hi_map_Mstar = np.r_[ hi_map_Mstar, lim_Mstar[ idvx_hi ] ]

		hi_map_ID = np.r_[ hi_map_ID, ID_lim[ idvx_hi] ]


		low_z_form = np.r_[ low_z_form, lim_z[ idvx_lo ] ]

		lo_map_rich = np.r_[ lo_map_rich, lim_rich[ idvx_lo ] ]

		lo_map_ra = np.r_[ lo_map_ra, lim_ra[ idvx_lo ] ]

		lo_map_dec = np.r_[ lo_map_dec, lim_dec[ idvx_lo ] ]

		lo_map_z = np.r_[ lo_map_z, lim_zc[ idvx_lo ] ]

		lo_map_age = np.r_[ lo_map_age, lim_age[ idvx_lo ] ]

		lo_map_Mstar = np.r_[ lo_map_Mstar, lim_Mstar[ idvx_lo ] ]

		lo_map_ID = np.r_[ lo_map_ID, ID_lim[ idvx_lo ] ]

	## save the divided sample
	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'origin_ID']
	values = [ hi_map_ra, hi_map_dec, hi_map_z, high_z_form, hi_map_rich, hi_map_age, hi_map_ID ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file_hi )

	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'origin_ID']
	values = [ lo_map_ra, lo_map_dec, lo_map_z, low_z_form, lo_map_rich, lo_map_age, lo_map_ID ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame(fill)
	data.to_csv( out_file_low )

	# print( 'high', hi_map_rich.shape )
	# print( 'low', lo_map_rich.shape )

	return

def age_binned_func( M_bins, data_arr, out_file_hi, out_file_low, divid_f):

	high_z_form = np.array( [] )
	hi_map_rich, hi_map_age = np.array( [] ), np.array( [] )
	hi_map_ra, hi_map_dec, hi_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	hi_map_Mstar = np.array( [] )
	hi_map_ID = np.array( [] )

	low_z_form = np.array( [] )
	lo_map_rich, lo_map_age = np.array( [] ), np.array( [] )
	lo_map_ra, lo_map_dec, lo_map_z = np.array( [] ), np.array( [] ), np.array( [] )
	lo_map_Mstar = np.array( [] )
	lo_map_ID = np.array( [] )

	bins_lgM = M_bins
	N_bin = len( M_bins )

	ra, dec, z, z_form, rich, age_time, lg_Mstar, origin_ID = data_arr[:]

	for ii in range( N_bin - 1):

		id_lim = (lg_Mstar >= bins_lgM[ii] ) & (lg_Mstar < bins_lgM[ii+1] )

		axis_lgM = np.log10( (10**bins_lgM[ii] + 10**bins_lgM[ii+1]) / 2 )
		thresh_age = 10**divid_f( axis_lgM )

		lim_rich = rich[ id_lim ]
		lim_ra, lim_dec, lim_zc = ra[id_lim], dec[id_lim], z[id_lim]

		lim_z = z_form[ id_lim ]
		lim_age = age_time[ id_lim ]
		lim_Mstar = lg_Mstar[ id_lim ]
		ID_lim = origin_ID[ id_lim ]


		idvx_hi = lim_age >= thresh_age
		idvx_lo = lim_age < thresh_age

		high_z_form = np.r_[ high_z_form, lim_z[ idvx_hi ] ]

		hi_map_rich = np.r_[ hi_map_rich, lim_rich[ idvx_hi ]  ]

		hi_map_ra = np.r_[ hi_map_ra, lim_ra[ idvx_hi ] ]

		hi_map_dec = np.r_[ hi_map_dec, lim_dec[ idvx_hi ] ]

		hi_map_z = np.r_[ hi_map_z, lim_zc[ idvx_hi ] ]

		hi_map_age = np.r_[ hi_map_age, lim_age[ idvx_hi ] ]

		hi_map_Mstar = np.r_[ hi_map_Mstar, lim_Mstar[ idvx_hi ] ]

		hi_map_ID = np.r_[ hi_map_ID, ID_lim[ idvx_hi] ]


		low_z_form = np.r_[ low_z_form, lim_z[ idvx_lo ] ]

		lo_map_rich = np.r_[ lo_map_rich, lim_rich[ idvx_lo ] ]

		lo_map_ra = np.r_[ lo_map_ra, lim_ra[ idvx_lo ] ]

		lo_map_dec = np.r_[ lo_map_dec, lim_dec[ idvx_lo ] ]

		lo_map_z = np.r_[ lo_map_z, lim_zc[ idvx_lo ] ]

		lo_map_age = np.r_[ lo_map_age, lim_age[ idvx_lo ] ]

		lo_map_Mstar = np.r_[ lo_map_Mstar, lim_Mstar[ idvx_lo ] ]

		lo_map_ID = np.r_[ lo_map_ID, ID_lim[ idvx_lo ] ]

	## save the divided sample
	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'origin_ID']
	values = [ hi_map_ra, hi_map_dec, hi_map_z, high_z_form, hi_map_rich, hi_map_age, hi_map_ID ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file_hi )

	keys = ['ra', 'dec', 'z', 'z_form', 'rich', 'age', 'origin_ID']
	values = [ lo_map_ra, lo_map_dec, lo_map_z, low_z_form, lo_map_rich, lo_map_age, lo_map_ID ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame(fill)
	data.to_csv( out_file_low )

	return

### === ### sample divid
def rich_divid():

	pre_N = 6
	bins_edg = np.linspace( 10, 12, 6)

	medi_rich = []

	cen_lgM = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )

	for ii in range( pre_N - 1):

		id_lim = ( lg_Mstar >= bins_edg[ii] ) & ( lg_Mstar < bins_edg[ii+1] )
		lim_rich = rich[ id_lim ]

		medi_rich.append( np.median( lim_rich ) )

	medi_rich = np.array( medi_rich )

	lgM_x = np.linspace( 10, 12, 30 )

	## use a simple power law 
	fit_F = np.polyfit( cen_lgM, np.log10( medi_rich ), deg = 2)
	Pf = np.poly1d( fit_F )
	fit_line = Pf( lgM_x )

	#...save this line
	# keys = [ 'lgM', 'lg_rich' ]
	# values = [ lgM_x, fit_line ]
	# fill = dict( zip(keys, values) )
	# data = pds.DataFrame( fill )
	# data.to_csv( '/home/xkchen/tmp_run/data_files/figs/rich-bin_fixed-BCG-M_divid_line.csv' )


	plt.figure( )
	ax = plt.subplot(111)
	tf = ax.scatter( lg_Mstar, np.log10( rich ), s = 15, c = 'r', alpha = 0.5,)

	ax.set_ylim(1.30, 2.30)
	ax.set_xlim(10, 12)

	for jj in range( pre_N ):
		ax.axvline( x = bins_edg[jj], ls = '--', color = 'k', alpha = 0.5,)

	ax.plot( cen_lgM, np.log10( medi_rich ), 'ks', alpha = 0.5)
	ax.plot( lgM_x, fit_line, ls = '-', color = 'k',)

	ax.set_ylabel('$ lg \\lambda $')
	ax.set_xlabel('$ lg M_{\\ast} $')

	plt.savefig('/home/xkchen/M-rich_view.png', dpi = 300)
	plt.close()

	raise

	N_bin = 30
	bins_lgM = np.linspace( 10, 12, N_bin)

	data_arr = [ ra, dec, z, z_form, rich, age_time, lg_Mstar, origin_ID ]
	out_file_hi = '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_hi-rich_cat.csv'
	out_file_low = '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_low-rich_cat.csv'
	# rich_binned_func( bins_lgM, data_arr, out_file_hi, out_file_low, Pf)

	### === ### match to images
	load = '/home/xkchen/mywork/ICL/data/photo_cat/'
	out_path = '/home/xkchen/tmp_run/data_files/figs/'

	lo_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_low-rich_cat.csv' )
	lo_ra, lo_dec, lo_z = np.array( lo_dat.ra), np.array( lo_dat.dec), np.array( lo_dat.z)
	lo_rich, lo_z_form = np.array( lo_dat.rich), np.array( lo_dat.z_form)
	lo_age = np.array( lo_dat.age )

	lo_origin_ID = np.array( lo_dat.origin_ID )
	lo_origin_ID = lo_origin_ID.astype( int )

	idlx = (lo_z >= 0.2) & (lo_z <= 0.3)
	print('low, N = ', np.sum(idlx) )
	print('low, N = ', len(lo_ra) )

	hi_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_hi-rich_cat.csv' )
	hi_ra, hi_dec, hi_z = np.array( hi_dat.ra), np.array( hi_dat.dec), np.array( hi_dat.z)
	hi_rich, hi_z_form = np.array( hi_dat.rich), np.array( hi_dat.z_form)
	hi_age = np.array( hi_dat.age )

	hi_origin_ID = np.array( hi_dat.origin_ID )
	hi_origin_ID = hi_origin_ID.astype( int )

	idhx = (hi_z >= 0.2) & (hi_z <= 0.3)
	print('high, N = ', np.sum(idhx) )
	print('high, N = ', len(hi_ra) )

	## gri-band common catalog
	cat_lis = [ 'low-rich', 'hi-rich' ]
	fig_name = [ 'low-rich', 'hi-rich' ]

	"""
	sf_len = 5
	f2str = '%.' + '%df' % sf_len
	## match images
	for kk in range( 3 ):

		pdat = pds.read_csv( load + 'photo-z_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ kk ],)
		p_ra, p_dec, p_z = np.array(pdat['ra']), np.array(pdat['dec']), np.array(pdat['z'])
		bcg_x, bcg_y = np.array(pdat['bcg_x']), np.array(pdat['bcg_y'])

		out_ra = [ f2str % ll for ll in lo_ra]
		out_dec = [ f2str % ll for ll in lo_dec]
		out_z = [ f2str % ll for ll in lo_z ]

		# match_ra, match_dec, match_z, match_x, match_y, input_id = cat_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y, sf_len,)
		match_ra, match_dec, match_z, match_x, match_y, input_id = extra_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y,)

		print( '%s band, low' % band[kk], len(match_ra) )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
		values = [ match_ra, match_dec, match_z, match_x, match_y, lo_origin_ID[input_id] ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( out_path + 'low-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )

		out_ra = [ f2str % ll for ll in hi_ra]
		out_dec = [ f2str % ll for ll in hi_dec]
		out_z = [ f2str % ll for ll in hi_z ]

		# match_ra, match_dec, match_z, match_x, match_y, input_id = cat_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y, sf_len,)
		match_ra, match_dec, match_z, match_x, match_y, input_id = extra_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y,)
		print( '%s band, high' % band[kk], len(match_ra) )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
		values = [ match_ra, match_dec, match_z, match_x, match_y, hi_origin_ID[input_id] ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( out_path + 'hi-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )


		cat_file = out_path + 'low-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ]
		out_file = out_path + 'low-rich_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % band[ kk ]
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

		cat_file = out_path + 'hi-rich_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ]
		out_file = out_path + 'hi-rich_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % band[ kk ]
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	for ll in range( 2 ):

		r_band_file = out_path + '%s_r-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
		g_band_file = out_path + '%s_g-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
		i_band_file = out_path + '%s_i-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]

		medi_r_file = out_path + '%s_r-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]
		medi_g_file = out_path + '%s_g-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]

		out_r_file = out_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
		out_g_file = out_path + '%s_g-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
		out_i_file = out_path + '%s_i-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]

		gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,)

	for kk in range( 0,1 ):

		sub_lo_dat = pds.read_csv( out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk]),)
		sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
		sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])

		sub_lo_rich = rich[ sub_lo_origin_dex]
		sub_lo_Mstar = lg_Mstar[ sub_lo_origin_dex]
		sub_lo_age = age_time[ sub_lo_origin_dex]

		cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk])
		out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[0], band[kk])
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)	


		sub_hi_dat = pds.read_csv( out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk]),)
		sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
		sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])

		sub_hi_rich = rich[ sub_hi_origin_dex]
		sub_hi_Mstar = lg_Mstar[ sub_hi_origin_dex]
		sub_hi_age = age_time[ sub_hi_origin_dex]

		cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk])
		out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[1], band[kk])
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

		print( len( sub_lo_ra) )
		print( len( sub_hi_ra) )
		## figs
		plt.figure()
		plt.hist( sub_lo_age, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_age), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_age), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_age, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_age), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_age), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel('age [Gyr]')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_rich_bin_BCG-age_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_z, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_z), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_z), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_z, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_z), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_z), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel('photometric redshift of BCGs')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_rich_bin_obs-z_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_rich, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_rich), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_rich), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_rich, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_rich), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_rich), ls = '-', color = 'r', alpha = 0.5, )

		plt.yscale('log')
		plt.xscale('log')
		plt.legend( loc = 1)
		plt.xlabel('$\\lambda$')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_rich_bin_rich_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_Mstar, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( np.log10( np.mean(10**sub_lo_Mstar) ), ls = '--', color = 'b', alpha = 0.5, label = 'Mean')
		plt.axvline( np.log10( np.median(10**sub_lo_Mstar) ), ls = '-', color = 'b', alpha = 0.5, label = 'Median')

		plt.hist( sub_hi_Mstar, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( np.log10( np.mean(10**sub_hi_Mstar) ), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( np.log10( np.median(10**sub_hi_Mstar) ), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel( '$ lg(M_{\\ast}) [M_{\\odot} / h]$' )
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_rich_bin_Mstar_compare.png' % band[kk], dpi = 300)
		plt.close()
	"""
	sub_lo_dat = pds.read_csv( out_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0]),)
	sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
	sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])

	sub_lo_rich = rich[ sub_lo_origin_dex]
	sub_lo_Mstar = lg_Mstar[ sub_lo_origin_dex]
	sub_lo_age = age_time[ sub_lo_origin_dex]

	sub_hi_dat = pds.read_csv( out_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1]),)
	sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
	sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])

	sub_hi_rich = rich[ sub_hi_origin_dex]
	sub_hi_Mstar = lg_Mstar[ sub_hi_origin_dex]
	sub_hi_age = age_time[ sub_hi_origin_dex]

	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	values = [ sub_lo_ra, sub_lo_dec, sub_lo_z, sub_lo_rich, sub_lo_Mstar, sub_lo_age ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )
	data.to_csv( out_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )

	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	values = [ sub_hi_ra, sub_hi_dec, sub_hi_z, sub_hi_rich, sub_hi_Mstar, sub_hi_age ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )
	data.to_csv( out_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )


	fig = plt.figure( )
	ax = fig.add_axes( [0.12, 0.12, 0.80, 0.80] )
	ax0 = fig.add_axes( [ 0.23, 0.70, 0.28, 0.20] )
	ax1 = fig.add_axes( [ 0.23, 0.68, 0.28, 0.02] )

	tf = ax.scatter( lg_Mstar, np.log10( rich ), s = 15, c = age_time, cmap = 'bwr', alpha = 0.5, vmin = 1, vmax = 11,)

	age_edgs = np.logspace( 0, np.log10(12), 50)

	ax0.hist( sub_lo_age, bins = age_edgs, density = True, histtype = 'step', color = 'b', alpha = 0.5, )
	ax0.hist( sub_hi_age, bins = age_edgs, density = True, histtype = 'step', color = 'r', alpha = 0.5, )
	ax0.set_xlim( 1, 11 )
	ax0.set_ylabel('pdf')
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in',)

	cmap = mpl.cm.bwr
	norm = mpl.colors.Normalize( vmin = 1, vmax = 11 )

	c_ticks = np.array([1, 3, 5, 7, 9, 11])
	cbs = mpl.colorbar.ColorbarBase( ax = ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, orientation = 'horizontal',)
	cbs.set_label( 'age [Gyr]' )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in',)

	cmap.set_under('cyan')
	cbs.ax.set_xticklabels( labels = ['%.0f' % ll for ll in c_ticks ] )
	ax0.set_xticks( [] )

	ax.plot( lgM_x, fit_line, ls = '-', color = 'k',)

	ax.text( 11.75, 1.70, s = 'high $ \\lambda $', fontsize = 15, rotation = 30,)
	ax.text( 11.75, 1.55, s = 'low $ \\lambda $', fontsize = 15, rotation = 30,)

	ax.set_ylim( 1.30, 2.30 )
	ax.set_xlim( 10, 12 )

	xtick_arr = [10, 10.5, 11, 11.5, 12 ]
	tick_lis = [ '%.1f' % ll for ll in xtick_arr ]
	ax.set_xticks( xtick_arr,)
	ax.set_xticklabels( labels = tick_lis, fontsize = 15,)

	ax.set_ylabel('$ lg \\lambda $', fontsize = 15,)
	ax.set_xlabel('$ lgM^{BCG}_{\\ast} [M_{\\odot} / h^2] $', fontsize = 15,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/BCG-M-based_rich-bin.png', dpi = 300)
	plt.close()

	raise

### === ### sample divid
def age_divid():

	## divid_0
	# pre_N = 6 # 7
	# bins_edg = np.linspace( 10.5, 11.6, pre_N)

	# medi_age = []

	# cen_lgM = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )

	# for ii in range( pre_N - 1):

	# 	id_lim = ( lg_Mstar >= bins_edg[ii] ) & ( lg_Mstar < bins_edg[ii+1] )
	# 	lim_age = age_time[ id_lim ]

	# 	medi_age.append( np.median( lim_age ) )

	# medi_age = np.log10( np.array( medi_age ) )

	# lgM_x = np.linspace( 10, 12, 30 )
	# ## use a simple power law 
	# fit_F = np.polyfit( cen_lgM, medi_age, deg = 3)

	# Pf = np.poly1d( fit_F )
	# fit_line = Pf( lgM_x )

	## divid_1
	# idy_lim = np.log10( age_time ) >= 0.6

	# idx_lim = ( lg_Mstar >= 10.75) & ( lg_Mstar <= 11.75 )
	# id_lim = idx_lim & idy_lim

	# lim_ra, lim_dec, lim_z = ra[ id_lim], dec[ id_lim], z[ id_lim]
	# lim_rich, lim_z_form = rich[ id_lim], z_form[ id_lim]
	# lim_lgM, lim_age = lg_Mstar[ id_lim], age_time[ id_lim]
	# lim_order = origin_ID[ id_lim] 

	# pre_N = 6
	# bins_edg = np.linspace( 10.75, 11.75, pre_N)

	# medi_age = []

	# cen_lgM = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )

	# for ii in range( len(bins_edg) - 1):

	# 	idvx = ( lim_lgM >= bins_edg[ii] ) & ( lim_lgM < bins_edg[ii+1] )
	# 	sub_age = lim_age[ idvx ]

	# 	medi_age.append( np.median( sub_age ) )

	# medi_age = np.log10( np.array( medi_age ) )

	# fit_F = np.polyfit( cen_lgM, medi_age, deg = 1)
	# Pf = np.poly1d( fit_F )

	## divid_2 
	idy_lim = np.log10( age_time ) >= 0.4

	idx_lim = ( lg_Mstar >= 10.5) & ( lg_Mstar <= 11.75 )
	id_lim = idx_lim & idy_lim

	lim_ra, lim_dec, lim_z = ra[ id_lim], dec[ id_lim], z[ id_lim]
	lim_rich, lim_z_form = rich[ id_lim], z_form[ id_lim]
	lim_lgM, lim_age = lg_Mstar[ id_lim], age_time[ id_lim]
	lim_order = origin_ID[ id_lim] 

	pre_N = 7
	bins_edg = np.linspace( 10.5, 11.75, pre_N)

	medi_age = []

	cen_lgM = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )

	for ii in range( len(bins_edg) - 1):

		idvx = ( lim_lgM >= bins_edg[ii] ) & ( lim_lgM < bins_edg[ii+1] )
		sub_age = lim_age[ idvx ]

		medi_age.append( np.median( sub_age ) )

	medi_age = np.log10( np.array( medi_age ) )


	idx_lowM = (lg_Mstar >= 10) & (lg_Mstar <= 10.50)
	age_lowM = age_time[ idx_lowM ]
	medi_lowT = np.log10( np.median(age_lowM) )
	medi_age = np.r_[ medi_lowT, medi_age ]
	cen_lgM = np.r_[ np.log10(0.5 * (10**10 + 10**10.5) ), cen_lgM ]

	new_lgMx = np.linspace(10, 12, 30)

	Pf = interp.interp1d( cen_lgM, medi_age, kind = 'linear', fill_value = 'extrapolate',)
	fit_line = Pf( new_lgMx )

	smoot_line = signal.savgol_filter( fit_line, 3, 1 )
	Pf_cc = interp.interp1d( new_lgMx, smoot_line, kind = 'linear', fill_value = 'extrapolate',)

	#...save this line
	keys = [ 'lgM', 'lg_age' ]
	values = [ new_lgMx, smoot_line ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )
	data.to_csv( '/home/xkchen/tmp_run/data_files/figs/age-bin_fixed-BCG-M_divid_line.csv' )


	plt.figure( )
	ax = plt.subplot(111)

	tf = ax.scatter( lg_Mstar, np.log10( age_time ), s = 20, c = 'r', alpha = 0.5,)

	ax.set_xlim(10, 12)

	for jj in range( pre_N ):
		ax.axvline( x = bins_edg[jj], ls = '--', color = 'k', alpha = 0.5,)

	ax.plot( cen_lgM, medi_age, 'ks', alpha = 0.5)

	# ax.plot( new_lgMx, fit_line, ls = '-', color = 'k',)
	ax.plot( new_lgMx, smoot_line, ls = '-', color = 'k',)

	ax.set_ylabel('$ lg \\tau [Gyr] $')
	ax.set_xlabel('$ lg M_{\\ast} $')
	ax.set_ylim( 0, 1.1 )
	plt.savefig('/home/xkchen/M-age_view.png', dpi = 300)
	plt.close()


	N_bin = 30
	bins_lgM = np.linspace( 10, 12, N_bin)

	data_arr = [ ra, dec, z, z_form, rich, age_time, lg_Mstar, origin_ID ]
	out_file_hi = '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_hi-age_cat.csv'
	out_file_low = '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_low-age_cat.csv'

	# age_binned_func( bins_lgM, data_arr, out_file_hi, out_file_low, Pf)
	# age_binned_func( bins_lgM, data_arr, out_file_hi, out_file_low, Pf_cc)

	### === ### match to images
	load = '/home/xkchen/mywork/ICL/data/photo_cat/'
	out_path = '/home/xkchen/tmp_run/data_files/figs/'

	lo_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_low-age_cat.csv' )
	lo_ra, lo_dec, lo_z = np.array( lo_dat.ra), np.array( lo_dat.dec), np.array( lo_dat.z)
	lo_rich, lo_z_form = np.array( lo_dat.rich), np.array( lo_dat.z_form)
	lo_age = np.array( lo_dat.age )

	lo_origin_ID = np.array( lo_dat.origin_ID )
	lo_origin_ID = lo_origin_ID.astype( int )

	idlx = (lo_z >= 0.2) & (lo_z <= 0.3)
	print('low, N = ', np.sum(idlx) )
	print('low, N*= ', len(lo_ra) )

	hi_dat = pds.read_csv( '/home/xkchen/tmp_run/data_files/figs/clslowz_z0.17-0.30_bc03_hi-age_cat.csv' )
	hi_ra, hi_dec, hi_z = np.array( hi_dat.ra), np.array( hi_dat.dec), np.array( hi_dat.z)
	hi_rich, hi_z_form = np.array( hi_dat.rich), np.array( hi_dat.z_form)
	hi_age = np.array( hi_dat.age )

	hi_origin_ID = np.array( hi_dat.origin_ID )
	hi_origin_ID = hi_origin_ID.astype( int )

	idhx = (hi_z >= 0.2) & (hi_z <= 0.3)
	print('high, N = ', np.sum(idhx) )
	print('high, N*= ', len(hi_ra) )

	## gri-band common catalog
	cat_lis = [ 'low-age', 'hi-age' ]
	fig_name = [ 'younger', 'older' ]

	"""
	sf_len = 5
	f2str = '%.' + '%df' % sf_len
	## match images
	for kk in range( 3 ):

		pdat = pds.read_csv( load + 'photo-z_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ kk ],)
		p_ra, p_dec, p_z = np.array(pdat['ra']), np.array(pdat['dec']), np.array(pdat['z'])
		bcg_x, bcg_y = np.array(pdat['bcg_x']), np.array(pdat['bcg_y'])

		out_ra = [ f2str % ll for ll in lo_ra]
		out_dec = [ f2str % ll for ll in lo_dec]
		out_z = [ f2str % ll for ll in lo_z ]

		match_ra, match_dec, match_z, match_x, match_y, input_id = extra_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y,)

		print( '%s band, low' % band[kk], len(match_ra) )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
		values = [ match_ra, match_dec, match_z, match_x, match_y, lo_origin_ID[input_id] ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( out_path + 'low-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )

		out_ra = [ f2str % ll for ll in hi_ra]
		out_dec = [ f2str % ll for ll in hi_dec]
		out_z = [ f2str % ll for ll in hi_z ]

		match_ra, match_dec, match_z, match_x, match_y, input_id = extra_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y,)
		print( '%s band, high' % band[kk], len(match_ra) )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
		values = [ match_ra, match_dec, match_z, match_x, match_y, hi_origin_ID[input_id] ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( out_path + 'hi-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )


		cat_file = out_path + 'low-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ]
		out_file = out_path + 'low-age_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % band[ kk ]
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

		cat_file = out_path + 'hi-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ]
		out_file = out_path + 'hi-age_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % band[ kk ]
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	for ll in range( 2 ):

		r_band_file = out_path + '%s_r-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
		g_band_file = out_path + '%s_g-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
		i_band_file = out_path + '%s_i-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]

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
		
		sub_lo_rich = rich[ sub_lo_origin_dex]
		sub_lo_Mstar = lg_Mstar[ sub_lo_origin_dex]
		sub_lo_age = age_time[ sub_lo_origin_dex]

		cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk])
		out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[0], band[kk])
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)	


		sub_hi_dat = pds.read_csv( out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk]),)
		sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
		sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])

		sub_hi_rich = rich[ sub_hi_origin_dex]
		sub_hi_Mstar = lg_Mstar[ sub_hi_origin_dex]
		sub_hi_age = age_time[ sub_hi_origin_dex]

		cat_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk])
		out_file = out_path + '%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[1], band[kk])
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

		print( 'low', len( sub_lo_ra) )
		print( 'high', len( sub_hi_ra) )
		## figs
		plt.figure()
		plt.hist( sub_lo_age, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_age), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_age), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_age, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_age), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_age), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel('age [Gyr]')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_BCG-age_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_z, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_z), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_z), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_z, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_z), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_z), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel('photometric redshift of BCGs')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_obs-z_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_rich, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_rich), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_rich), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_rich, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_rich), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_rich), ls = '-', color = 'r', alpha = 0.5, )

		plt.yscale('log')
		plt.xscale('log')
		plt.legend( loc = 1)
		plt.xlabel('$\\lambda$')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_rich_compare.png' % band[kk], dpi = 300)
		plt.close()

		plt.figure()
		plt.hist( sub_lo_Mstar, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( np.log10( np.mean(10**sub_lo_Mstar) ), ls = '--', color = 'b', alpha = 0.5, label = 'Mean')
		plt.axvline( np.log10( np.median(10**sub_lo_Mstar) ), ls = '-', color = 'b', alpha = 0.5, label = 'Median')

		plt.hist( sub_hi_Mstar, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( np.log10( np.mean(10**sub_hi_Mstar) ), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( np.log10( np.median(10**sub_hi_Mstar) ), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel( '$ lg(M_{\\ast}) [M_{\\odot} / h]$' )
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_Mstar_compare.png' % band[kk], dpi = 300)
		plt.close()
	"""

	sub_lo_dat = pds.read_csv( out_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0]),)
	sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
	sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])

	sub_lo_rich = rich[ sub_lo_origin_dex]
	sub_lo_Mstar = lg_Mstar[ sub_lo_origin_dex]
	sub_lo_age = age_time[ sub_lo_origin_dex]

	sub_hi_dat = pds.read_csv( out_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1]),)
	sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
	sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])

	sub_hi_rich = rich[ sub_hi_origin_dex]
	sub_hi_Mstar = lg_Mstar[ sub_hi_origin_dex]
	sub_hi_age = age_time[ sub_hi_origin_dex]

	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	values = [ sub_lo_ra, sub_lo_dec, sub_lo_z, sub_lo_rich, sub_lo_Mstar, sub_lo_age ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )
	data.to_csv( out_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[0] )

	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'BCG_age']
	values = [ sub_hi_ra, sub_hi_dec, sub_hi_z, sub_hi_rich, sub_hi_Mstar, sub_hi_age ]
	fill = dict( zip(keys, values) )
	data = pds.DataFrame( fill )
	data.to_csv( out_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv' % cat_lis[1] )


	fig = plt.figure( )
	ax = fig.add_axes( [0.12, 0.12, 0.80, 0.80] )
	ax0 = fig.add_axes( [ 0.63, 0.25, 0.28, 0.25] )
	ax1 = fig.add_axes( [ 0.63, 0.23, 0.28, 0.02] )

	tf = ax.scatter( lg_Mstar, np.log10( age_time ), s = 15, c = np.log10( rich ), cmap = 'bwr', alpha = 0.5, vmin = 1.30, vmax = 2.30,)

	lgrich_edgs = np.linspace( 1.30, 2.30, 53)

	ax0.hist( np.log10( sub_lo_rich ), bins = lgrich_edgs, density = True, histtype = 'step', color = 'b', alpha = 0.5, )
	ax0.hist( np.log10( sub_hi_rich ), bins = lgrich_edgs, density = True, histtype = 'step', color = 'r', alpha = 0.5, )
	ax0.set_xlim( 1.3, 2.3 )
	ax0.set_yscale('log')
	ax0.set_ylabel('pdf')
	ax0.tick_params( axis = 'both', which = 'both', direction = 'in',)

	# color bar
	cmap = mpl.cm.bwr
	norm = mpl.colors.Normalize( vmin = 1.30, vmax = 2.30 )

	c_ticks = ax0.get_xticks()

	cbs = mpl.colorbar.ColorbarBase( ax = ax1, cmap = cmap, norm = norm, extend = 'neither', ticks = c_ticks, orientation = 'horizontal',)
	cbs.set_label( '$ lg\\lambda $' )
	ax1.tick_params( axis = 'both', which = 'both', direction = 'in',)

	cmap.set_under('cyan')
	cbs.ax.set_xticklabels( labels = ['%.1f' % ll for ll in c_ticks ] )
	ax0.set_xticks( [] )

	ax.plot( new_lgMx, smoot_line, ls = '-', color = 'k',)

	ax.set_ylabel('lg(age) [Gyr]', fontsize = 15,)
	ax.set_xlabel('$ lgM^{BCG}_{\\ast} [M_{\\odot} / h^2] $', fontsize = 15,)

	ax.set_xlim( 10, 12)
	ax.set_ylim( 0.0, 1.05)

	x_ticks = np.array( [ 10, 10.5, 11, 11.5, 12 ] )
	ax.set_xticks( x_ticks,)
	ax.set_xticklabels( labels = [ '%.1f' % ll for ll in x_ticks ], fontsize = 15,)

	ax.text( 10.15, 0.5, s = 'older', fontsize = 15, rotation = 30,)
	ax.text( 10.25, 0.2, s = 'younger', fontsize = 15, rotation = 30,)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in', labelsize = 15,)

	plt.savefig('/home/xkchen/BCG-M-based_age-bin.png', dpi = 300)
	plt.close()

	raise

def adjust_age_bin():
	## limed samples
	idy_lim = np.log10( age_time ) >= 0.6
	idx_lim = ( lg_Mstar >= 10.75) & ( lg_Mstar <= 11.75 )
	id_lim = idx_lim & idy_lim

	lim_ra, lim_dec, lim_z = ra[ id_lim], dec[ id_lim], z[ id_lim]
	lim_rich, lim_z_form = rich[ id_lim], z_form[ id_lim]
	lim_lgM, lim_age = lg_Mstar[ id_lim], age_time[ id_lim]
	lim_order = origin_ID[ id_lim] 

	pre_N = 6
	bins_edg = np.linspace( 10.75, 11.75, pre_N)

	medi_age = []

	cen_lgM = np.log10( 0.5 * (10**bins_edg[1:] + 10**bins_edg[:-1]) )

	for ii in range( len(bins_edg) - 1):

		idvx = ( lim_lgM >= bins_edg[ii] ) & ( lim_lgM < bins_edg[ii+1] )
		sub_age = lim_age[ idvx ]

		medi_age.append( np.median( sub_age ) )

	medi_age = np.log10( np.array( medi_age ) )

	## adjust
	lgM_x = np.linspace( 10.75, 11.75, 30)
	# fit_F = np.polyfit( cen_lgM, medi_age, deg = 1)
	fit_F = np.polyfit( cen_lgM, medi_age, deg = 2)

	Pf = np.poly1d( fit_F )
	fit_line = Pf( lgM_x )


	plt.figure( )
	ax = plt.subplot(111)
	tf = ax.scatter( lim_lgM, np.log10( lim_age ), s = 20, c = 'r', alpha = 0.5,)

	ax.set_xlim(10.75, 11.75)

	for jj in range( pre_N ):
		ax.axvline( x = bins_edg[jj], ls = '--', color = 'k', alpha = 0.5,)

	plt.plot( cen_lgM, medi_age, 'ks', alpha = 0.5)
	plt.plot( lgM_x, fit_line, ls = '-', color = 'k',)

	ax.set_ylabel('$ lg \\tau [Gyr] $')
	ax.set_xlabel('$ lg M_{\\ast} $')
	ax.set_ylim( 0.6, 1.1 )
	plt.savefig('/home/xkchen/M-age_view.png', dpi = 300)
	plt.close()


	N_bin = 30

	# bins_lgM = np.linspace( 10, 12, N_bin)
	# data_arr = [ ra, dec, z, z_form, rich, age_time, lg_Mstar, origin_ID ]
	bins_lgM = np.linspace( 10.75, 11.75, 30)
	data_arr = [ lim_ra, lim_dec, lim_z, lim_z_form, lim_rich, lim_age, lim_lgM, lim_order ]

	out_file_hi = '/home/xkchen/tmp_run/data_files/figs/limed_bc03_hi-age_cat.csv'
	out_file_low = '/home/xkchen/tmp_run/data_files/figs/limed_bc03_low-age_cat.csv'
	age_binned_func( bins_lgM, data_arr, out_file_hi, out_file_low, Pf)

	###
	load = '/home/xkchen/mywork/ICL/data/photo_cat/'
	out_path = '/home/xkchen/tmp_run/data_files/figs/'

	lo_dat = pds.read_csv( out_file_low )
	lo_ra, lo_dec, lo_z = np.array( lo_dat.ra), np.array( lo_dat.dec), np.array( lo_dat.z)
	lo_rich, lo_z_form = np.array( lo_dat.rich), np.array( lo_dat.z_form)
	lo_age = np.array( lo_dat.age )

	lo_origin_ID = np.array( lo_dat.origin_ID )
	lo_origin_ID = lo_origin_ID.astype( int )

	idlx = (lo_z >= 0.2) & (lo_z <= 0.3)

	print('low, N = *', np.sum(idlx) )
	print('low, N = ', len(lo_ra) )

	hi_dat = pds.read_csv( out_file_hi )
	hi_ra, hi_dec, hi_z = np.array( hi_dat.ra), np.array( hi_dat.dec), np.array( hi_dat.z)
	hi_rich, hi_z_form = np.array( hi_dat.rich), np.array( hi_dat.z_form)
	hi_age = np.array( hi_dat.age )

	hi_origin_ID = np.array( hi_dat.origin_ID )
	hi_origin_ID = hi_origin_ID.astype( int )

	idhx = (hi_z >= 0.2) & (hi_z <= 0.3)

	print('high, N = *', np.sum(idhx) )
	print('high, N = ', len(hi_ra) )


	sf_len = 5
	f2str = '%.' + '%df' % sf_len
	## match images
	for kk in range( 3 ):

		pdat = pds.read_csv( load + 'photo-z_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ kk ],)
		p_ra, p_dec, p_z = np.array(pdat['ra']), np.array(pdat['dec']), np.array(pdat['z'])
		bcg_x, bcg_y = np.array(pdat['bcg_x']), np.array(pdat['bcg_y'])

		out_ra = [ f2str % ll for ll in lo_ra]
		out_dec = [ f2str % ll for ll in lo_dec]
		out_z = [ f2str % ll for ll in lo_z ]

		match_ra, match_dec, match_z, match_x, match_y, input_id = extra_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y,)

		print( '%s band, low' % band[kk], len(match_ra) )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
		values = [ match_ra, match_dec, match_z, match_x, match_y, lo_origin_ID[input_id] ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( out_path + 'limed_low-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )

		out_ra = [ f2str % ll for ll in hi_ra]
		out_dec = [ f2str % ll for ll in hi_dec]
		out_z = [ f2str % ll for ll in hi_z ]

		match_ra, match_dec, match_z, match_x, match_y, input_id = extra_match_func(out_ra, out_dec, out_z, p_ra, p_dec, p_z, bcg_x, bcg_y,)
		print( '%s band, high' % band[kk], len(match_ra) )

		keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'origin_ID']
		values = [ match_ra, match_dec, match_z, match_x, match_y, hi_origin_ID[input_id] ]
		fill = dict(zip(keys, values))
		data = pds.DataFrame(fill)
		data.to_csv( out_path + 'limed_hi-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ] )


		cat_file = out_path + 'limed_low-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ]
		out_file = out_path + 'limed_low-age_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % band[ kk ]
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

		cat_file = out_path + 'limed_hi-age_%s-band_photo-z-match_BCG-pos_cat.csv' % band[ kk ]
		out_file = out_path + 'limed_hi-age_%s-band_photo-z-match_BCG-pos_cat_z-ref.csv' % band[ kk ]
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	## gri-band common catalog
	cat_lis = [ 'low-age', 'hi-age' ]
	fig_name = [ 'younger', 'older' ]

	for ll in range( 2 ):

		r_band_file = out_path + 'limed_%s_r-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
		g_band_file = out_path + 'limed_%s_g-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]
		i_band_file = out_path + 'limed_%s_i-band_photo-z-match_BCG-pos_cat.csv' % cat_lis[ll]

		medi_r_file = out_path + 'limed_%s_r-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]
		medi_g_file = out_path + 'limed_%s_g-band_photo-z-match_rg-common_BCG-pos_cat.csv' % cat_lis[ ll ]

		out_r_file = out_path + 'limed_%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
		out_g_file = out_path + 'limed_%s_g-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]
		out_i_file = out_path + 'limed_%s_i-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ ll ]

		gri_common_cat_func(r_band_file, g_band_file, i_band_file, medi_r_file, medi_g_file, out_r_file, out_g_file, out_i_file,)

	for kk in range( 3 ):

		sub_lo_dat = pds.read_csv( out_path + 'limed_%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk]),)
		sub_lo_ra, sub_lo_dec, sub_lo_z = np.array( sub_lo_dat['ra']), np.array( sub_lo_dat['dec']), np.array( sub_lo_dat['z'])
		sub_lo_origin_dex = np.array( sub_lo_dat['origin_ID'])

		sub_lo_rich = rich[ sub_lo_origin_dex]
		sub_lo_Mstar = lg_Mstar[ sub_lo_origin_dex]
		sub_lo_age = age_time[ sub_lo_origin_dex]

		cat_file = out_path + 'limed_%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[0], band[kk])
		out_file = out_path + 'limed_%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[0], band[kk])
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)	


		sub_hi_dat = pds.read_csv( out_path + 'limed_%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk]),)
		sub_hi_ra, sub_hi_dec, sub_hi_z = np.array( sub_hi_dat['ra']), np.array( sub_hi_dat['dec']), np.array( sub_hi_dat['z'])
		sub_hi_origin_dex = np.array( sub_hi_dat['origin_ID'])

		sub_hi_rich = rich[ sub_hi_origin_dex]
		sub_hi_Mstar = lg_Mstar[ sub_hi_origin_dex]
		sub_hi_age = age_time[ sub_hi_origin_dex]

		cat_file = out_path + 'limed_%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[1], band[kk])
		out_file = out_path + 'limed_%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat_z-ref.csv' % (cat_lis[1], band[kk])
		zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)


		print( 'low', len( sub_lo_ra) )
		print( 'high', len( sub_hi_ra) )

		## figs
		plt.figure()
		plt.hist( sub_lo_age, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_age), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_age), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_age, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_age), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_age), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel('age [Gyr]')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_BCG-age_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_z, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_z), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_z), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_z, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_z), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_z), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel('photometric redshift of BCGs')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_obs-z_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_rich, bins = 50, color = 'b', alpha = 0.5, density = False, label = fig_name[0],)
		plt.axvline( x = np.mean(sub_lo_rich), ls = '--', color = 'b', alpha = 0.5, label = 'mean',)
		plt.axvline( x = np.median(sub_lo_rich), ls = '-', color = 'b', alpha = 0.5, label = 'median',)

		plt.hist( sub_hi_rich, bins = 50, color = 'r', alpha = 0.5, density = False, label = fig_name[1],)
		plt.axvline( x = np.mean(sub_hi_rich), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( x = np.median(sub_hi_rich), ls = '-', color = 'r', alpha = 0.5, )

		plt.yscale('log')
		plt.xscale('log')
		plt.legend( loc = 1)
		plt.xlabel('$\\lambda$')
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_rich_compare.png' % band[kk], dpi = 300)
		plt.close()


		plt.figure()
		plt.hist( sub_lo_Mstar, bins = 50, color = 'b', alpha = 0.5, density = True, label = fig_name[0],)
		plt.axvline( np.log10( np.mean(10**sub_lo_Mstar) ), ls = '--', color = 'b', alpha = 0.5, label = 'Mean')
		plt.axvline( np.log10( np.median(10**sub_lo_Mstar) ), ls = '-', color = 'b', alpha = 0.5, label = 'Median')

		plt.hist( sub_hi_Mstar, bins = 50, color = 'r', alpha = 0.5, density = True, label = fig_name[1],)
		plt.axvline( np.log10( np.mean(10**sub_hi_Mstar) ), ls = '--', color = 'r', alpha = 0.5, )
		plt.axvline( np.log10( np.median(10**sub_hi_Mstar) ), ls = '-', color = 'r', alpha = 0.5, )

		plt.legend( loc = 2)
		plt.xlabel( '$ lg(M_{\\ast}) [M_{\\odot} / h]$' )
		plt.ylabel('pdf')
		plt.savefig('/home/xkchen/%s_band_age_bin_Mstar_compare.png' % band[kk], dpi = 300)
		plt.close()

	raise

###..
rich_divid()

# age_divid()

# adjust_age_bin()

