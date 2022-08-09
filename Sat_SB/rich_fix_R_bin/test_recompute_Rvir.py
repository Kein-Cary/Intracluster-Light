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


### === ### binned satellite based on scaled radius
##. cluster catalog
def rich_binned_clust_cat():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_binned/cat/'
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'

	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/' + 
						'Extend-BCGM_rgi-common_cat.csv')

	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	rich = np.array( dat['rich'] )
	clust_ID = np.array( dat['clust_ID'] )


	##. re-compute R200m, and M200m
	M_vir, R_vir = rich2R_Simet( z, rich )  ## M_sun, kpc
	M_vir, R_vir = M_vir * h, R_vir * h / 1e3       ## M_sun / h, Mpc / h
	lg_Mvir = np.log10( M_vir )


	##. re-save the table
	keys = ['ra', 'dec', 'z', 'rich', 'lg_Mh', 'R_vir', 'clust_ID']
	values = [ ra, dec, z, rich, lg_Mvir, R_vir, clust_ID ]
	fill = dict(zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'Extend-BCGM_rgi-common_cat.csv',)


	##. rich binned
	bin_rich = [ 20, 30, 50, 210 ]

	tmp_rich, tmp_lgMh = [], []

	for kk in range( 3 ):

		dat = pds.read_csv( cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]), )
		sub_ra, sub_dec, sub_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
		sub_rich = np.array( dat['rich'] )
		sub_IDs = np.array( dat['clust_ID'] )


		##. re-compute R200m, and M200m
		sub_M_vir, sub_R_vir = rich2R_Simet( sub_z, sub_rich )  ## M_sun, kpc
		sub_M_vir, sub_R_vir = sub_M_vir * h, sub_R_vir * h / 1e3       ## M_sun / h, Mpc / h
		sub_lg_Mvir = np.log10( sub_M_vir )

		##. re-save the table
		keys = [ 'ra', 'dec', 'z', 'clust_ID', 'rich', 'lg_Mh', 'R_vir' ]
		values = [ sub_ra, sub_dec, sub_z, sub_IDs, sub_rich, sub_lg_Mvir, sub_R_vir ]
		fill = dict(zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]), )

		tmp_rich.append( sub_rich )
		tmp_lgMh.append( sub_lg_Mvir )


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

##. rematch member 
def member_cat_refill_Rv():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/'
	out_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'

	##. entire satellite sample
	dat = pds.read_csv( out_path + 'Extend-BCGM_rgi-common_cat.csv' )
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	rich, R_vir, lg_Mh = np.array( dat['rich'] ), np.array( dat['R_vir'] ), np.array( dat['lg_Mh'] )

	clust_IDs = np.array( dat['clust_ID'] )
	clust_IDs = clust_IDs.astype( int )

	N_s = len( ra )

	files = [	'Extend-BCGM_rgi-common_member-cat.csv',
				'Extend-BCGM_rgi-common_frame-limit_member-cat.csv',
				'Extend-BCGM_rgi-common_frame-lim_exlu-BCG_member-cat.csv',
				'Extend-BCGM_rgi-common_frame-limit_Pm-cut_member-cat.csv',
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' ]

	N_files = len( files )

	for pp in range( N_files ):

		cat = pds.read_csv( cat_path + files[ pp ] )

		sub_bcg_ra, sub_bcg_dec, sub_bcg_z = np.array( cat['bcg_ra'] ), np.array( cat['bcg_dec'] ), np.array( cat['bcg_z'] )
		sub_ra, sub_dec, sub_z = np.array( cat['ra'] ), np.array( cat['dec'] ), np.array( cat['z_spec'] )

		sub_IDs = np.array( cat['clus_ID'] )

		sub_gr, sub_ri, sub_gi = np.array( cat['g-r'] ), np.array( cat['r-i'] ), np.array( cat['g-i'] )
		sub_Rcen = np.array( cat['R_cen'] )


		sub_R2Rv = np.zeros( len(sub_ra), )

		for nn in range( N_s ):

			nn_ID = clust_IDs[ nn ]

			id_gx = sub_IDs == nn_ID

			sub_R2Rv[ id_gx ] = sub_Rcen[ id_gx ] / R_vir[ nn ]

		##. save
		keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'ra', 'dec', 'z_spec', 'Rcen/Rv', 'R_cen', 'g-r', 'r-i', 'g-i', 'clus_ID' ]
		values = [ sub_bcg_ra, sub_bcg_dec, sub_bcg_z, sub_ra, sub_dec, sub_z, sub_R2Rv, sub_Rcen, sub_gr, sub_ri, sub_gi, sub_IDs ]
		fill = dict( zip( keys, values) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( out_path + files[ pp ] )

	return


def clust_member_match():

	cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'

	bin_rich = [ 20, 30, 50, 210 ]

	##. cluster match to satellites
	for kk in range( len(bin_rich) - 1 ):

		img_cat_file = cat_path + 'clust_rich_%d-%d_cat.csv' % ( bin_rich[kk], bin_rich[kk + 1])
		mem_cat_file = cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv'
		out_sat_file = cat_path + 'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1])
		mem_match_func( img_cat_file, mem_cat_file, out_sat_file )


	### === tables for Background stacking
	##. cluster member match with satellites position
	for kk in range( len(bin_rich) - 1 ):

		dat = pds.read_csv( cat_path + 
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


# rich_binned_clust_cat()
# member_cat_refill_Rv()
# clust_member_match()



### === figs
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'

bin_rich = [ 20, 30, 50, 210 ]

# Rs_lim = np.array( [0, 5e-2, 1e-1, 2e-1, 4e-1, 1] ) ### times R200m
Rs_lim = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m

tmp_Rcen, tmp_R2Rv = [], []

for kk in range( 3 ):

	dat = pds.read_csv(cat_path + 
			'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv' % ( bin_rich[kk], bin_rich[kk + 1]),)

	sub_ra, sub_dec, sub_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['bcg_z'] )
	sub_Rcen = np.array( dat['R_cen'] )
	sub_R2Rv = np.array( dat['Rcen/Rv'] )

	#.
	for nn in range( len(Rs_lim) - 1 ):
		
		sub_N0 = ( Rs_lim[ nn ] <= sub_R2Rv ) & ( Rs_lim[ nn + 1 ] > sub_R2Rv )
		print( np.sum( sub_N0 ) )

	print('*' * 20 )

	tmp_Rcen.append( sub_Rcen )
	tmp_R2Rv.append( sub_R2Rv )

#.
color_s = ['b', 'g', 'r']
line_name = ['$\\lambda \\leq 30$', '$30 \\leq \\lambda \\leq 50$', '$\\lambda \\geq 50$']

R_edgs = np.logspace( -3, 0, 55 )

fig = plt.figure()
ax = fig.add_axes( [0.10, 0.10, 0.80, 0.85] )

for kk in range( 3 ):

	ax.hist( tmp_R2Rv[ kk ], bins = R_edgs, density = True, histtype = 'step', 
			color = color_s[ kk ], label = line_name[ kk ],)

for kk in range( len(Rs_lim) - 1 ):

	ax.axvline( Rs_lim[ kk ], ls = ':', color = 'k',)

ax.set_xscale('log')
ax.set_xlim( 1e-3, 1e0 )
ax.set_xlabel('$R_{sat} / R_{200m}$')

ax.legend( loc = 2,)

plt.savefig('/home/xkchen/R_scale_hist.png', dpi = 300)
plt.close()

