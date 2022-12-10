import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import scipy.stats as sts
from io import StringIO
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

#.
from Mass_rich_radius import rich2R_Simet
from img_sat_fig_out_mode import zref_sat_pos_func


###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']

z_ref = 0.25
pixel = 0.396
a_ref = 1 / (z_ref + 1)


### === ### all those satellite have been applied frame-limit, and P_member = 0.8 cut, exclude BCGs
def mem_match_func( img_cat_file, mem_cat_file, out_sat_file ):

	##. cluster cat
	dat = pds.read_csv( img_cat_file )

	ref_ra, ref_dec, ref_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	ref_Rvir, ref_rich = np.array( dat['R_vir'] ), np.array( dat['rich'] )
	ref_clust_ID = np.array( dat['clust_ID'] )

	Ns = len( ref_ra )


	##. satellite samples
	s_dat = pds.read_csv( mem_cat_file )

	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec, p_zspec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] ), np.array( s_dat['z_spec'])
	R_sat, R_sat2Rv = np.array( s_dat['R_cen'] ), np.array( s_dat['Rcen/Rv'] )

	p_gr, p_ri, p_gi = np.array( s_dat['g-r'] ), np.array( s_dat['r-i'] ), np.array( s_dat['g-i'] )
	p_clus_ID = np.array( s_dat['clus_ID'] )
	p_clus_ID = p_clus_ID.astype( int )


	##. member find
	sat_ra, sat_dec, sat_z = np.array([]), np.array([]), np.array([])
	sat_Rcen, sat_gr, sat_ri, sat_gi = np.array([]), np.array([]), np.array([]), np.array([])

	sat_R2Rv = np.array([])
	sat_host_ID = np.array([])

	cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array([]), np.array([]), np.array([])

	##. match
	for pp in range( Ns ):

		id_px = p_clus_ID == ref_clust_ID[ pp ]

		cut_ra, cut_dec, cut_z = p_ra[ id_px ], p_dec[ id_px ], p_zspec[ id_px ]

		cut_R2Rv = R_sat2Rv[ id_px ]

		cut_gr, cut_ri, cut_gi = p_gr[ id_px ], p_ri[ id_px ], p_gi[ id_px ]
		cut_Rcen = R_sat[ id_px ]

		#. record array
		sat_ra = np.r_[ sat_ra, cut_ra ]
		sat_dec = np.r_[ sat_dec, cut_dec ]
		sat_z = np.r_[ sat_z, cut_z ]

		sat_Rcen = np.r_[ sat_Rcen, cut_Rcen ]
		sat_R2Rv = np.r_[ sat_R2Rv, cut_R2Rv ]

		sat_gr = np.r_[ sat_gr, cut_gr ]
		sat_ri = np.r_[ sat_ri, cut_ri ]
		sat_gi = np.r_[ sat_gi, cut_gi ]

		sat_host_ID = np.r_[ sat_host_ID, np.ones( len(cut_ra),) * ref_clust_ID[pp] ]

		cp_bcg_ra = np.r_[ cp_bcg_ra, np.ones( len(cut_ra),) * ref_ra[pp] ]
		cp_bcg_dec = np.r_[ cp_bcg_dec, np.ones( len(cut_ra),) * ref_dec[pp] ]
		cp_bcg_z = np.r_[ cp_bcg_z, np.ones( len(cut_ra),) * ref_z[pp] ]

	#. save member infor
	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'ra', 'dec', 'z_spec', 'Rcen/Rv', 'R_cen', 'g-r', 'r-i', 'g-i', 'clus_ID' ]
	values = [ cp_bcg_ra, cp_bcg_dec, cp_bcg_z, sat_ra, sat_dec, sat_z, sat_R2Rv, sat_Rcen, sat_gr, sat_ri, sat_gi, sat_host_ID ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_sat_file )

	return


### === cluster catalog
def cluster_binned():
	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
						'Extend-BCGM_rgi-common_cat.csv')

	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	rich, lg_Mh = np.array( dat['rich'] ), np.array( dat['lg_Mh'] )
	R_vir, clust_ID = np.array( dat['R_vir'] ), np.array( dat['clust_ID'] )


	##. rich binned
	bin_rich = [ 20, 30, 50, 210 ]

	tmp_rich, tmp_lgMh = [], []

	for kk in range( len(bin_rich) - 1 ):

		id_vx = ( rich >= bin_rich[ kk ] ) & ( rich <= bin_rich[ kk + 1 ] )
		
		sub_ra, sub_dec, sub_z = ra[ id_vx ], dec[ id_vx ], z[ id_vx ]
		sub_rich, sub_lg_Mh, sub_Rv = rich[ id_vx ], lg_Mh[ id_vx ], R_vir[ id_vx ]
		sub_clus_ID = clust_ID[ id_vx ]

		##.
		keys = [ 'ra', 'dec', 'z', 'clust_ID', 'rich', 'lg_Mh', 'R_vir' ]
		values = [ sub_ra, sub_dec, sub_z, sub_clus_ID, sub_rich, sub_lg_Mh, sub_Rv ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]), )

		tmp_rich.append( sub_rich )
		tmp_lgMh.append( sub_lg_Mh )

	medi_Mh_0 = np.log10( np.median( 10**( tmp_lgMh[0] ) ) )
	medi_Mh_1 = np.log10( np.median( 10**( tmp_lgMh[1] ) ) )
	medi_Mh_2 = np.log10( np.median( 10**( tmp_lgMh[2] ) ) )


	plt.figure()
	plt.hist( rich, bins = 55, density = False, histtype = 'step', color = 'r',)

	plt.axvline( x = 30, ls = ':', color = 'k', )
	plt.axvline( x = 50, ls = ':', color = 'k', )

	plt.text(20, 30, s = 'n=%d' % len(tmp_rich[0]) + '\n' + '$\\lg M_{h}$=%.2f' % medi_Mh_0,)

	plt.text(33, 30, s = 'n=%d' % len(tmp_rich[1]) + '\n' + '$\\lg M_{h}$=%.2f' % medi_Mh_1,)

	plt.text(80, 30, s = 'n=%d' % len(tmp_rich[2]) + '\n' + '$\\lg M_{h}$=%.2f' % medi_Mh_2,)

	plt.xlabel('$\\lambda$')
	plt.xscale('log')

	plt.ylabel('# of cluster')
	plt.yscale('log')

	plt.xlim( 19, 200 )

	plt.savefig('/home/xkchen/rich_binned_cluster.png', dpi = 300 )
	plt.close()

	return

def clust_member_match():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

	bin_rich = [ 20, 30, 50, 210 ]

	# ##. cluster match to satellites
	# for kk in range( len(bin_rich) - 1 ):

	# 	img_cat_file = cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1])
	# 	mem_cat_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv'
	# 	out_sat_file = cat_path + 'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1])
	# 	mem_match_func( img_cat_file, mem_cat_file, out_sat_file )
	

	### === tables for Background stacking
	##. cluster member match with satellites position
	for kk in range( len(bin_rich) - 1 ):

		dat = pds.read_csv(cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
		clus_IDs = np.array( dat['clus_ID'] )

		sub_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

		for tt in range( 3 ):

			band_str = band[ tt ]

			pat = pds.read_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/iMag_fix_Rbin/shufle_test/img_tract_cat/' + 
				'Extend-BCGM_rgi-common_frame-limit_exlu-BCG_Sat_%s-band_origin-img_position.csv' % band_str )

			kk_ra, kk_dec = np.array( pat['sat_ra'] ), np.array( pat['sat_dec'] )
			kk_bcg_x, kk_bcg_y = np.array( pat['bcg_x'] ), np.array( pat['bcg_y'] )
			kk_sat_x, kk_sat_y = np.array( pat['sat_x'] ), np.array( pat['sat_y'] )
			kk_PA = np.array( pat['sat_PA2bcg'] )

			kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

			#. match P_mem cut sample
			idx, sep, d3d = sub_coord.match_to_catalog_sky( kk_coord )
			id_lim = sep.value < 2.7e-4

			#. satellite information (for check)
			mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
			mp_bcg_x, mp_bcg_y = kk_bcg_x[ idx[ id_lim ] ], kk_bcg_y[ idx[ id_lim ] ]
			mp_sat_x, mp_sat_y = kk_sat_x[ idx[ id_lim ] ], kk_sat_y[ idx[ id_lim ] ]
			mp_sat_PA = kk_PA[ idx[ id_lim ] ]

			#. save
			keys = pat.columns[1:]
			mp_arr = [ bcg_ra, bcg_dec, bcg_z, mp_ra, mp_dec, mp_bcg_x, mp_bcg_y, 
										mp_sat_x, mp_sat_y, mp_sat_PA, clus_IDs ]
			fill = dict( zip( keys, mp_arr ) )
			out_data = pds.DataFrame( fill )
			out_data.to_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv' 
				% (bin_rich[kk], bin_rich[kk + 1], band_str),)


	##. cluster member match with the cutout information at z_obs and z-ref
	for kk in range( len(bin_rich) - 1 ):

		dat = pds.read_csv(cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
		clus_IDs = np.array( dat['clus_ID'] )

		sub_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )

		for tt in range( 3 ):

			band_str = band[ tt ]

			##. satellite location and cutout at z_obs
			dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
					'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band_str,)
			kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			kk_imgx, kk_imgy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )

			kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

			idx, sep, d3d = sub_coord.match_to_catalog_sky( kk_coord )
			id_lim = sep.value < 2.7e-4

			mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
			mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'cut_cx', 'cut_cy']
			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, mp_imgx, mp_imgy ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos.csv' 
				% (bin_rich[kk], bin_rich[kk + 1], band_str),)


			##. satellite location and cutout at z_ref
			dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
					'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[kk] )
			kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
			kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

			kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

			idx, sep, d3d = sub_coord.match_to_catalog_sky( kk_coord )
			id_lim = sep.value < 2.7e-4

			mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
			mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

			keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
			values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, mp_imgx, mp_imgy ]
			fill = dict( zip( keys, values ) )
			data = pds.DataFrame( fill )
			data.to_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_member_pos-zref.csv' 
				% (bin_rich[kk], bin_rich[kk + 1], band_str),)

	return


##.
def count_N_sat():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

	bin_rich = [ 20, 30, 50, 210 ]

	##. radius binned satellite
	sub_name = ['inner', 'middle', 'outer']
	##... 
	R_bins = [ 0, 200, 400 ]   ## kpc


	#... number count for the entire sample
	for kk in range( 3 ):
		
		##. entire all sample
		dat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)
		clus_IDs = np.array( dat['clust_ID'] )
		clus_IDs = clus_IDs.astype( int )

		N_w = len( clus_IDs )


		##. member table
		dat = pds.read_csv(cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

		bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
		cp_clus_IDs = np.array( dat['clus_ID'] )
		cp_clus_IDs = cp_clus_IDs.astype( int )

		N_g = np.zeros( len(bcg_ra),)

		for tt in range( N_w ):
			sub_IDs = clus_IDs[ tt ]

			id_vx = cp_clus_IDs == sub_IDs
			N_g[ id_vx ] = np.sum( id_vx )

		#. save N_g for BG_img stacking weight
		keys = ['ra', 'dec', 'z', 'N_g']
		values = [ bcg_ra, bcg_dec, bcg_z, N_g ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( cat_path + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat-Ng.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

	return


# cluster_binned()
# clust_member_match()
# count_N_sat()



### === figs
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'

bin_rich = [ 20, 30, 50, 210 ]
fig_name = ['low-$\\lambda$', 'medi-$\\lambda$', 'high-$\\lambda$']
sub_name = ['inner', 'middle', 'outer']

line_c = ['b', 'g', 'r']

#.
R_bins = [ 0, 0.2, 0.4 ]       ## scaled radius
R_phy = [ 0, 200, 400 ]       ## physical radius


fig = plt.figure( figsize = (12, 6),)
ax0 = plt.subplot( 121 )
ax1 = plt.subplot( 122 )

for kk in range( len(bin_rich) - 1 ):
	
	##.
	s_dat = pds.read_csv( cat_path + 
		'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )
	clus_IDs = np.array( s_dat['clus_ID'] )

	p_Rsat = np.array( s_dat['R_cen'] )
	p_R2Rv = np.array( s_dat['Rcen/Rv'] )

	p_Rsat = p_Rsat * 1e3 / h    ##. physical radius


	##.
	sub_N0 = p_R2Rv <= R_bins[1]
	sub_N1 = ( p_R2Rv > R_bins[1] ) & ( p_R2Rv <= R_bins[2] )
	sub_N2 = p_R2Rv >= R_bins[2]

	##.
	phy_sub0 = p_Rsat <= R_phy[1]
	phy_sub1 = ( p_Rsat > R_phy[1] ) & ( p_Rsat <= R_phy[2] )
	phy_sub2 = p_Rsat >= R_phy[2]

	print( '*' * 10 )
	print( np.sum(sub_N0), np.sum(sub_N1), np.sum(sub_N2) )
	print( np.sum(phy_sub0), np.sum(phy_sub1), np.sum(phy_sub2) )

	##.
	ax0.hist( p_Rsat, bins = 55, density = True, color = line_c[kk], histtype = 'step', label = fig_name[kk],)
	ax1.hist( p_R2Rv, bins = 55, density = True, color = line_c[kk], histtype = 'step', label = fig_name[kk],)

ax0.set_xlabel('$R_{sat} \;[kpc]$')
ax0.legend( loc = 1, frameon = False,)
ax0.axvline( x = 200, ls = ':', color = 'k',)
ax0.axvline( x = 400, ls = ':', color = 'k',)

ax1.set_xlabel('$R_{sat} / R_{200m}$')
ax1.axvline( x = 0.2, ls = ':', color = 'k',)
ax1.axvline( x = 0.4, ls = ':', color = 'k',)
ax1.legend( loc = 1, frameon = False,)

plt.savefig('/home/xkchen/sat_Rs_hist.png', dpi = 300)
plt.close()

