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
	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'ra', 'dec', 'z_spec', 
			'Rsat/Rv', 'R_sat', 'g-r', 'r-i', 'g-i', 'clus_ID' ]
	values = [ cp_bcg_ra, cp_bcg_dec, cp_bcg_z, sat_ra, sat_dec, sat_z, 
			sat_R2Rv, sat_Rcen, sat_gr, sat_ri, sat_gi, sat_host_ID ]

	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_sat_file )

	return


### === cluster catalog
def cluster_binned():

	cat_path = '/home/xkchen/Pictures/BG_calib_SBs/fixR_bin/cat/'

	##. the halo mass and radius haven been update in '.../rich_rebin/'
	dat = pds.read_csv('/home/xkchen/Pictures/BG_calib_SBs/sat_cat_z02_03/' + 
						'Extend-BCGM_rgi-common_cat.csv')

	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	rich = np.array( dat['rich'] )
	clust_ID = np.array( dat['clust_ID'] )


	##. re-compute R200m, and M200m
	M_vir, R_vir = rich2R_Simet( z, rich )  ## M_sun, kpc
	M_vir, R_vir = M_vir * h, R_vir * h / 1e3       ## M_sun / h, Mpc / h
	lg_Mvir = np.log10( M_vir )


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


### === ..
cluster_binned()


