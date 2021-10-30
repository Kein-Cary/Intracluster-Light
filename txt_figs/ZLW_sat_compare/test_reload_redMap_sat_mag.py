"""
This file match the redMaPPer identify members
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

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
from scipy import optimize
from scipy import stats as sts
from scipy import signal
from scipy import interpolate as interp
from scipy import optimize
from scipy import integrate as integ

def sql_color( cat_ra, cat_dec, cat_ID, out_file ):

	import mechanize
	from io import StringIO

	url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

	Nz = len(cat_ra)

	record_s = []
	column_name = []

	for kk in range( Nz ):

		ra_g = cat_ra[kk]
		dec_g = cat_dec[kk]

		data_set = """
		SELECT
			p.objID, p.ra, p.dec, 
			p.modelMag_g, p.modelMag_r, p.modelMag_i,
			p.cModelMag_g, p.cModelMag_r, p.cModelMag_i,
			p.dered_g, p.dered_r, p.dered_i

		FROM PhotoObjAll as p

		WHERE
			p.objID = %d
		""" % cat_ID[kk]

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()
		#print(data_set)
		
		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		sql_s = str(response.get_data(), encoding = 'utf-8')

		record_s.append( sql_s.split()[2] )

		if kk == Nz - 1:
			column_name.append( sql_s.split()[1] )

	#. save data
	doc = open( out_file, 'w')
	print( column_name[0], file = doc)

	for kk in range( Nz ):
		print( record_s[kk], file = doc)
	doc.close()

	print( 'down!' )

	return

##... read catalog
cat_lis = [ 'low-age', 'hi-age' ]
"""
for ll in range( 1,2 ):

	tmp_ra, tmp_dec, tmp_z = np.array( [] ), np.array( [] ), np.array( [] )
	tmp_IDs = np.array( [], dtype = int )

	with h5py.File('/home/xkchen/figs/ZLW_cat_15/%s_clus-sat_record.h5' % cat_lis[ll], 'r') as f:
		keys = list( f.keys() )
		N_ks = len( keys )

		for jj in range( N_ks ):
			_sub_arr = f["/clust_%d/arr/" % jj][()]

			sub_ra, sub_dec, sub_z = _sub_arr[0], _sub_arr[1], _sub_arr[2]
			sub_IDs = f["/clust_%d/IDs/" % jj][()]
			
			#.. query with function
			# out_file = ('/home/xkchen/figs/ZLW_cat_15/redMaPPer_sat_mag_sql/' + 
			# 			'photo-z_r-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % ( cc_ra[jj], cc_dec[jj], cc_z[jj] ),)[0]
			# sql_color( sub_ra, sub_dec, sub_z, sub_IDs, out_file )

			tmp_ra = np.r_[ tmp_ra, sub_ra ]
			tmp_dec = np.r_[ tmp_dec, sub_dec ]
			tmp_z = np.r_[ tmp_z, sub_z ]
			tmp_IDs = np.r_[ tmp_IDs, sub_IDs ]

	# ordex_arr = np.arange( 0, len( tmp_ra ) )

	# tmp_arr = np.array( [ ordex_arr, tmp_ra, tmp_dec ] ).T
	# np.savetxt('/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/' + 
					'%s_redMaPPer-sat_galx_cat.dat' % cat_lis[ll], tmp_arr, fmt = ('%.0f', '%.5f', '%.5f'),)

	# n_step = 10000

	# for tt in range( 9 ):
	# 	if tt == 8:
	# 		da0 = np.int( tt * n_step )
	# 		da1 = len( tmp_ra )
	# 	else:
	# 		da0 = np.int( tt * n_step )
	# 		da1 = np.int( (tt + 1) * n_step )

	# 	tmp_arr = np.array( [ np.r_[0, ordex_arr[ da0: da1 ] ], np.r_[ 1, tmp_ra[ da0: da1 ] ], np.r_[ 2, tmp_dec[ da0: da1 ] ] ] ).T
	# 	np.savetxt('/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/' + 
					'%s_redMaPPer-sat_galx_cat_%d.dat' % (cat_lis[ll], tt), tmp_arr, fmt = ('%.0f', '%.5f', '%.5f'),)

raise
"""

"""
#.high t_age subsample catalog combine
tot_cat = np.loadtxt('/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/hi-age_redMaPPer-sat_galx_cat.dat')
tot_ra, tot_dec = tot_cat[:,1], tot_cat[:,2]
tot_dex = tot_cat[:,0]

tmp_ra, tmp_dec = np.zeros( len(tot_ra), ), np.zeros( len(tot_ra), )
tmp_objIDs = np.zeros( len(tot_ra), )

tmp_r_mod_mag, tmp_g_mod_mag, tmp_i_mod_mag = np.zeros( len(tot_ra),), np.zeros( len(tot_ra),), np.zeros( len(tot_ra),)
tmp_r_cmod_mag, tmp_g_cmod_mag, tmp_i_cmod_mag = np.zeros( len(tot_ra),), np.zeros( len(tot_ra),), np.zeros( len(tot_ra),)
tmp_dered_rmag, tmp_dered_gmag, tmp_dered_imag = np.zeros( len(tot_ra),), np.zeros( len(tot_ra),), np.zeros( len(tot_ra),)

tmp_dex = np.array( [] )

for tt in range( 9 ):

	dat = pds.read_csv( '/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/hi-age_galax_mag_%d.csv' % tt, skiprows = 1 )

	tmp_dex = np.r_[ tmp_dex, dat['col0'] ]
	print( len( dat['ra'] ) )

	_tt_dex = np.array( dat['col0'] )

	tmp_objIDs[ _tt_dex ] = np.array( dat['objID'] )
	tmp_r_mod_mag[ _tt_dex ] = np.array( dat['modelMag_r'] )
	tmp_g_mod_mag[ _tt_dex ] = np.array( dat['modelMag_g'] )
	tmp_i_mod_mag[ _tt_dex ] = np.array( dat['modelMag_i'] )

	tmp_ra[ _tt_dex ] = np.array( dat['ra'] ) 
	tmp_dec[ _tt_dex ] = np.array( dat['dec'] )

	tmp_r_cmod_mag[ _tt_dex ] = np.array( dat['cModelMag_r'] )
	tmp_g_cmod_mag[ _tt_dex ] = np.array( dat['cModelMag_g'] ) 
	tmp_i_cmod_mag[ _tt_dex ] = np.array( dat['cModelMag_i'] ) 
	
	tmp_dered_rmag[ _tt_dex ] = np.array( dat['dered_r'] ) 
	tmp_dered_gmag[ _tt_dex ] = np.array( dat['dered_g'] )
	tmp_dered_imag[ _tt_dex ] = np.array( dat['dered_i'] ) 

# plt.figure()
# plt.plot(tot_ra, tot_dec, 'ro', alpha = 0.5,)
# plt.plot(tmp_ra, tmp_dec, 'g*', alpha = 0.5,)
# plt.show()

#. missing match in total table squery
dda = list( set(tot_dex).difference( set(tmp_dex) ) )
dda = np.array( dda )
dda = dda.astype( int )

# lack_ra = tot_ra[ dda ]
# lack_dec = tot_dec[ dda ]
# lack_arr = np.array( [dda, lack_ra, lack_dec] ).T
# np.savetxt('/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/lack_cat.dat', lack_arr, fmt = ('%.0f', '%.5f', '%.5f'),)

# lack_IDs = tmp_IDs[ dda ]
# lack_IDs = np.array([1237678600236302444, 1237655370356228117, 1237658191073640454, 1237668297667706901, 
# 					1237663479796924524, 1237661356463685739, 1237678600230404136, 1237678578756157632, 
# 					1237651754543874119, 1237657067402625034, 1237651250440110087, 1237680306937462797])
# lack_file = '/home/xkchen/lack_cat.csv'
# sql_color( lack_ra, lack_dec, lack_IDs, lack_file )


#. combine all catalog
p_dat = pds.read_csv('/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/lack_cat.csv')

tmp_objIDs[ dda ] = np.array( p_dat['objID'] )
tmp_r_mod_mag[ dda ] = np.array( p_dat['modelMag_r'] )
tmp_g_mod_mag[ dda ] = np.array( p_dat['modelMag_g'] )
tmp_i_mod_mag[ dda ] = np.array( p_dat['modelMag_i'] )

tmp_ra[ dda ] = np.array( p_dat['ra'] ) 
tmp_dec[ dda ] = np.array( p_dat['dec'] )

tmp_r_cmod_mag[ dda ] = np.array( p_dat['cModelMag_r'] )
tmp_g_cmod_mag[ dda ] = np.array( p_dat['cModelMag_g'] ) 
tmp_i_cmod_mag[ dda ] = np.array( p_dat['cModelMag_i'] ) 

tmp_dered_rmag[ dda ] = np.array( p_dat['dered_r'] ) 
tmp_dered_gmag[ dda ] = np.array( p_dat['dered_g'] )
tmp_dered_imag[ dda ] = np.array( p_dat['dered_i'] )

keys = ['col0', 'objID', 'ra', 'dec', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'cModelMag_g', 
		'cModelMag_r', 'cModelMag_i', 'dered_g', 'dered_r', 'dered_i']
values = [ tot_dex, tmp_objIDs, tot_ra, tot_dec, tmp_g_mod_mag, tmp_r_mod_mag, tmp_i_mod_mag, 
		tmp_g_cmod_mag, tmp_r_cmod_mag, tmp_i_cmod_mag, tmp_dered_gmag, tmp_dered_rmag, tmp_dered_imag ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/redMaPPer-cat_hi-age_galx_mag.csv' )
"""

###... rematch member galaxies and clusters
cat_lis = [ 'low-age', 'hi-age' ]
mem_path = '/home/xkchen/figs/sat_cat_ZLW/redMap_mem_match/'

'''
for ll in range( 2 ):

	#. read cluster catalog
	dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_z_form/bcg_M_based_cat/age_bin/' + 
		'%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % cat_lis[ll] )
	cc_ra, cc_dec, cc_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	N_clus = len( cc_ra )

	#. read pre-match catalogs
	tot_ra, tot_dec, tot_z = np.array( [] ), np.array( [] ), np.array( [] )
	tot_IDs = np.array( [], dtype = int )
	tot_cen_dL = np.array( [] )
	tot_Pmem = np.array( [] )

	n_sat = []
	with h5py.File('/home/xkchen/figs/ZLW_cat_15/%s_clus-sat_record.h5' % cat_lis[ll], 'r') as f:
		keys = list( f.keys() )
		N_ks = len( keys )

		for jj in range( N_ks ):
			_sub_arr = f["/clust_%d/arr/" % jj][()]
			sub_IDs = f["/clust_%d/IDs/" % jj][()]

			tot_ra = np.r_[ tot_ra, _sub_arr[0] ]
			tot_dec = np.r_[ tot_dec, _sub_arr[1] ]
			tot_z = np.r_[ tot_z, _sub_arr[2] ]
			tot_cen_dL = np.r_[ tot_cen_dL, _sub_arr[3] ]
			tot_Pmem = np.r_[ tot_Pmem, _sub_arr[4] ]

			tot_IDs = np.r_[ tot_IDs, sub_IDs ]
			n_sat.append( len(sub_IDs) )

	n_sat = np.array( n_sat )
	d_num = np.cumsum( n_sat )
	d_num = np.r_[ 0, d_num ]

	#. read reload catalog
	p_dat = pds.read_csv('/home/xkchen/figs/sat_cat_ZLW/redMap_cat_reload/' + 'redMaPPer-cat_%s_galx_mag.csv' % cat_lis[ll] )

	tmp_objIDs = np.array( p_dat['objID'] )
	tmp_r_mod_mag = np.array( p_dat['modelMag_r'] )
	tmp_g_mod_mag = np.array( p_dat['modelMag_g'] )
	tmp_i_mod_mag = np.array( p_dat['modelMag_i'] )

	tmp_ra = np.array( p_dat['ra'] ) 
	tmp_dec = np.array( p_dat['dec'] )

	tmp_r_cmod_mag = np.array( p_dat['cModelMag_r'] )
	tmp_g_cmod_mag = np.array( p_dat['cModelMag_g'] ) 
	tmp_i_cmod_mag = np.array( p_dat['cModelMag_i'] ) 

	tmp_dered_rmag = np.array( p_dat['dered_r'] ) 
	tmp_dered_gmag = np.array( p_dat['dered_g'] )
	tmp_dered_imag = np.array( p_dat['dered_i'] )

	tmp_g2r = tmp_g_mod_mag - tmp_r_mod_mag
	tmp_g2i = tmp_g_mod_mag - tmp_i_mod_mag
	tmp_r2i = tmp_r_mod_mag - tmp_i_mod_mag

	tmp_dered_g2r = tmp_dered_gmag - tmp_dered_rmag
	tmp_dered_g2i = tmp_dered_gmag - tmp_dered_imag
	tmp_dered_r2i = tmp_dered_rmag - tmp_dered_imag

	#. save member cat with add color
	record_tree = h5py.File('/home/xkchen/figs/sat_cat_ZLW/reload_%s_clus-sat_record.h5' % cat_lis[ll], 'w')

	for jj in range( N_clus ):

		ra_g, dec_g, z_g = cc_ra[jj], cc_dec[jj], cc_z[jj]

		sub_cen_R = tot_cen_dL[ d_num[jj]: d_num[jj+1] ]
		sub_Pmem = tot_Pmem[ d_num[jj]: d_num[jj+1] ]
		sub_ra, sub_dec, sub_z = [ tot_ra[ d_num[jj]: d_num[jj+1] ], tot_dec[ d_num[jj]: d_num[jj+1] ], 
									tot_z[ d_num[jj]: d_num[jj+1] ] ]
		sub_obj_IDs = tot_IDs[ d_num[jj]: d_num[jj+1] ]

		sub_r_mag = tmp_r_mod_mag[ d_num[jj]: d_num[jj+1] ]
		sub_g_mag = tmp_g_mod_mag[ d_num[jj]: d_num[jj+1] ]
		sub_i_mag = tmp_i_mod_mag[ d_num[jj]: d_num[jj+1] ]

		sub_dered_rmag = tmp_dered_rmag[ d_num[jj]: d_num[jj+1] ]
		sub_dered_gmag = tmp_dered_gmag[ d_num[jj]: d_num[jj+1] ]
		sub_dered_imag = tmp_dered_imag[ d_num[jj]: d_num[jj+1] ]

		sub_r_mag_err = np.ones( n_sat[jj], ) * 0.5
		sub_g_mag_err = np.ones( n_sat[jj], ) * 0.5
		sub_i_mag_err = np.ones( n_sat[jj], ) * 0.5

		sub_g2r = sub_g_mag - sub_r_mag
		sub_g2i = sub_g_mag - sub_i_mag
		sub_r2i = sub_r_mag - sub_i_mag

		sub_dered_g2r = sub_dered_gmag - sub_dered_rmag
		sub_dered_g2i = sub_dered_gmag - sub_dered_imag
		sub_dered_r2i = sub_dered_rmag - sub_dered_imag

		keys = [ 'centric_R(Mpc/h)', 'r_mags', 'g_mags', 'i_mags', 'P_member', 'r_mag_err', 'g_mag_err', 'i_mag_err', 
					'dered_r_mags', 'dered_g_mags', 'dered_i_mags', 'ra', 'dec', 'z']
		values = [ sub_cen_R, sub_r_mag, sub_g_mag, sub_i_mag, sub_Pmem, sub_r_mag_err, sub_g_mag_err, sub_i_mag_err, 
					sub_dered_rmag, sub_dered_gmag, sub_dered_imag, sub_ra, sub_dec, sub_z ]
		fill = dict( zip( keys, values ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( mem_path + 'photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % ('r', ra_g, dec_g, z_g), )


		out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R, sub_Pmem, sub_g2r, sub_g2i, sub_r2i, 
								sub_dered_g2r, sub_dered_g2i, sub_dered_r2i ] )
		gk = record_tree.create_group( "clust_%d/" % jj )
		dk0 = gk.create_dataset( "arr", data = out_arr )
		dk1 = gk.create_dataset( "IDs", data = sub_obj_IDs )

	record_tree.close()

'''

###... rematch for richness and BCG stellar mass binned sample
mem_path = '/home/xkchen/figs/sat_cat_ZLW/redMap_mem_match/'
home = '/home/xkchen/mywork/ICL/data/'

###... richness binned sample
cat_lis = [ 'low-rich', 'hi-rich' ]
cat_file = home + 'cat_z_form/bcg_M_based_cat/rich_bin/%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv'

###... bcg mass binned sample
# cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# cat_file = home + 'BCG_stellar_mass_cat/photo_z_gri_common/%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv'

for ll in range( 2 ):

	dat = pds.read_csv( cat_file % cat_lis[ll] )
	cc_ra, cc_dec, cc_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	N_clus = len( cc_ra )

	record_tree = h5py.File('/home/xkchen/figs/sat_cat_ZLW/reload_%s_clus-sat_record.h5' % cat_lis[ll], 'w')

	for jj in range( N_clus ):

		ra_g, dec_g, z_g = cc_ra[jj], cc_dec[jj], cc_z[jj]

		sub_dat = pds.read_csv( mem_path + 'photo-z_%s-band_ra%.3f_dec%.3f_z%.3f_members_mag.csv' % ('r', ra_g, dec_g, z_g),)

		sub_cen_R = np.array( sub_dat['centric_R(Mpc/h)'] )
		sub_Pmem = np.array( sub_dat['P_member'] )
		sub_ra, sub_dec, sub_z = np.array( sub_dat['ra'] ), np.array( sub_dat['dec'] ), np.array( sub_dat['z'] )

		sub_r_mag = np.array( sub_dat['r_mags'] )
		sub_g_mag = np.array( sub_dat['g_mags'] )
		sub_i_mag = np.array( sub_dat['i_mags'] )

		sub_dered_rmag = np.array( sub_dat['dered_r_mags'] )
		sub_dered_gmag = np.array( sub_dat['dered_g_mags'] )
		sub_dered_imag = np.array( sub_dat['dered_i_mags'] )

		sub_g2r = sub_g_mag - sub_r_mag
		sub_g2i = sub_g_mag - sub_i_mag
		sub_r2i = sub_r_mag - sub_i_mag

		sub_dered_g2r = sub_dered_gmag - sub_dered_rmag
		sub_dered_g2i = sub_dered_gmag - sub_dered_imag
		sub_dered_r2i = sub_dered_rmag - sub_dered_imag

		out_arr = np.array( [ sub_ra, sub_dec, sub_z, sub_cen_R, sub_Pmem, sub_g2r, sub_g2i, sub_r2i, 
								sub_dered_g2r, sub_dered_g2i, sub_dered_r2i ] )
		gk = record_tree.create_group( "clust_%d/" % jj )
		dk0 = gk.create_dataset( "arr", data = out_arr )

	record_tree.close()

