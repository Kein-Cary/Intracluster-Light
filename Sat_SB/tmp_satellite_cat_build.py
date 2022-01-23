"""
for given redMaPPer cluster, using this file to find the member galaxies and
query or match their properties (i.e. Mstar, color, model and cModel magnitudes, centric distance,
location on image frame and cut region)
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pds
import h5py

import mechanize
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


### === mem cat
def entire_sample_func():

	c_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')

	RA = np.array( c_dat.RA )
	DEC = np.array( c_dat.DEC )
	ID = np.array( c_dat.OBJID )

	rich = np.array( c_dat.LAMBDA )
	Z_photo = np.array( c_dat.Z_LAMBDA )
	Z_photo_err = np.array( c_dat.Z_LAMBDA_ERR )
	Z_spec = np.array( c_dat.Z_SPEC )

	clus_order = np.array( c_dat.ID )

	#. 0.1~z~0.33
	idx_lim = ( Z_photo >= 0.1 ) & ( Z_photo <= 0.33 )
	lim_ra, lim_dec, lim_z = RA[ idx_lim ], DEC[ idx_lim ], Z_photo[ idx_lim ]
	lim_ID = ID[ idx_lim ]

	lim_rich = rich[ idx_lim ]
	lim_zspec = Z_spec[ idx_lim ]
	lim_z_pho_err = Z_photo_err[ idx_lim ]
	lim_order = clus_order[ idx_lim ]


	#. approximation on radius (R200m)
	M_vir, R_vir = rich2R_Simet( lim_z, lim_rich )  ## M_sun, kpc
	M_vir, R_vir = M_vir * h, R_vir * h / 1e3       ## M_sun / h, Mpc / h
	lg_Mvir = np.log10( M_vir )


	###... save the cluster (BCG) information
	keys = ['ra', 'dec', 'z_pho', 'z_pho_err', 'z_spec', 'objID', 'rich', 'lg_M200m', 'R200m', 'clus_ID']
	values = [ lim_ra, lim_dec, lim_z, lim_z_pho_err, lim_zspec, lim_ID, lim_rich, lg_Mvir, R_vir, lim_order ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-pho_0.1-0.33_cat.csv',)


	# #.sql table
	# or_dex = np.arange( len(lim_ra) )
	# out_arr = [ or_dex, lim_ra, lim_dec ]

	# out_arr[0] = np.r_[ 0, out_arr[0] ]
	# out_arr[1] = np.r_[ 1, out_arr[1] ]
	# out_arr[2] = np.r_[ 2, out_arr[2] ]

	# put_arr = np.array( out_arr ).T
	# np.savetxt('/home/xkchen/tmp_zphot_0.1-0.33_sql-cat.txt', put_arr, fmt = ('%d', '%.8f', '%.8f'), )



	### === matched member for clusters above
	Nz = len( lim_z )

	m_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')
	m_ra, m_dec, m_z = np.array( m_dat.RA ), np.array( m_dat.DEC ), np.array( m_dat.Z_SPEC )

	clus_IDs = np.array( m_dat.ID )   ## the ID number of cluster in redMaPPer
	R_cen = np.array( m_dat.R )       ## Mpc / h
	P_mem = np.array( m_dat.P )
	m_objIDs = np.array( m_dat.OBJID )

	#. model mag for pre-color view
	m_mag_r = np.array( m_dat.MODEL_MAG_R )
	m_mag_g = np.array( m_dat.MODEL_MAG_G )
	m_mag_i = np.array( m_dat.MODEL_MAG_I )


	record_tree = h5py.File('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/' + 
							'redMaPPer_z-phot_0.1-0.33_cluster-sate_record.h5', 'w')

	for kk in range( Nz ):

		id_vx = clus_IDs == lim_order[ kk ]

		sub_ra, sub_dec, sub_z = m_ra[ id_vx ], m_dec[ id_vx ], m_z[ id_vx ]
		sub_Pm, sub_Rcen, sub_objIDs = P_mem[ id_vx ], R_cen[ id_vx ], m_objIDs[ id_vx ]
		sub_magr, sub_magg, sub_magi = m_mag_r[ id_vx ], m_mag_g[ id_vx ], m_mag_i[ id_vx ]

		##... list member galaxy table
		keys = ["ra", "dec", "z_spec", "Pm", "Rcen", "ObjID", "mod_mag_r", "mod_mag_g", "mod_mag_i"]
		out_arr = [ sub_ra, sub_dec, sub_z, sub_Pm, sub_Rcen, sub_objIDs, sub_magr, sub_magg, sub_magi]

		gk = record_tree.create_group( "clust_%d/" % lim_order[ kk ] )
		_tt_able = QTable( out_arr, names = keys,)
		dk0 = gk.create_dataset( "mem_table", data = _tt_able )

	record_tree.close()


	# #.sql table
	# or_dex = np.arange( len( cp_ra ) )
	# out_arr = [ or_dex, cp_ra, cp_dec ]

	# out_arr[0] = np.r_[ 0, out_arr[0] ]
	# out_arr[1] = np.r_[ 1, out_arr[1] ]
	# out_arr[2] = np.r_[ 2, out_arr[2] ]

	# put_arr = np.array( out_arr ).T
	# np.savetxt('/home/xkchen/tmp_zphot_0.1-0.33_sql-member-cat.txt', put_arr, fmt = ('%d', '%.8f', '%.8f'), )

	return

# entire_sample_func()


### === target sample match
def extra_cat_match_func( set_ra, set_dec, set_z, out_sat_file, out_img_file):
	"""
	ra, dec, z_set : BCGs or clusters need to find their member
	out_sat_file : '.h5' file or '.csv' file, record member in the order of input cluster table
	out_img_file : '.csv' file, record the image name of clusters sample (selected by z_photo only)
					also includes cluster properties (i.e. richness, R_vir)
	"""

	#. BCG cat
	dat = pds.read_csv('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-pho_0.1-0.33_cat.csv')
	
	ref_ra, ref_dec, ref_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z_pho'] )
	ref_lgMh, ref_Rvir, ref_rich = np.array( dat['lg_M200m'] ), np.array( dat['R200m'] ), np.array( dat['rich'] )
	ref_clust_ID = np.array( dat['clus_ID'] )

	ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg )
	set_coord = SkyCoord( ra = set_ra * U.deg, dec = set_dec * U.deg )


	idx, sep, d3d = set_coord.match_to_catalog_sky( ref_coord )
	id_lim = sep.value < 2.7e-4

	mp_ra, mp_dec, mp_z = ref_ra[ idx[ id_lim ] ], ref_dec[ idx[ id_lim ] ], ref_z[ idx[ id_lim ] ]
	mp_lgMh, mp_Rvir, mp_rich = ref_lgMh[ idx[ id_lim ] ], ref_Rvir[ idx[ id_lim ] ], ref_rich[ idx[ id_lim ] ]

	mp_clus_ID = ref_clust_ID[ idx[ id_lim ] ]

	#. save cluster properties and image information
	keys = [ 'ra', 'dec', 'z', 'rich', 'lg_Mh', 'R_vir', 'clust_ID' ]
	values = [ mp_ra, mp_dec, mp_z, mp_rich, mp_lgMh, mp_Rvir, mp_clus_ID ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_img_file )


	#. member match
	Ns = len( mp_ra )

	sat_ra, sat_dec, sat_z = np.array([]), np.array([]), np.array([])
	sat_Rcen, sat_gr, sat_ri, sat_gi = np.array([]), np.array([]), np.array([]), np.array([])
	sat_host_ID = np.array([])

	cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array([]), np.array([]), np.array([])

	for pp in range( Ns ):

		pre_dat = Table.read('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/' + 
							'redMaPPer_z-phot_0.1-0.33_cluster-sate_record.h5', path = 'clust_%d/mem_table' % mp_clus_ID[pp],)

		sub_ra, sub_dec, sub_z = np.array( pre_dat['ra'] ), np.array( pre_dat['dec'] ), np.array( pre_dat['z_spec'] )
		sub_Rcen = np.array( pre_dat['Rcen'] )

		sub_rmag = np.array( pre_dat['mod_mag_r'] )
		sub_gmag = np.array( pre_dat['mod_mag_g'] )
		sub_imag = np.array( pre_dat['mod_mag_i'] )

		sub_gr, sub_ri, sub_gi = sub_gmag - sub_rmag, sub_rmag - sub_imag, sub_gmag - sub_imag
		sub_R2Rv = sub_Rcen / mp_Rvir[ pp ]

		#. record array
		sat_ra = np.r_[ sat_ra, sub_ra ]
		sat_dec = np.r_[ sat_dec, sub_dec ]
		sat_z = np.r_[ sat_z, sub_z ]

		sat_Rcen = np.r_[ sat_Rcen, sub_R2Rv ]
		sat_gr = np.r_[ sat_gr, sub_gr ]
		sat_ri = np.r_[ sat_ri, sub_ri ]
		sat_gi = np.r_[ sat_gi, sub_gi ]

		sat_host_ID = np.r_[ sat_host_ID, np.ones( len(sub_ra),) * mp_clus_ID[pp] ]

		cp_bcg_ra = np.r_[ cp_bcg_ra, np.ones( len(sub_ra),) * mp_ra[pp] ]
		cp_bcg_dec = np.r_[ cp_bcg_dec, np.ones( len(sub_ra),) * mp_dec[pp] ]
		cp_bcg_z = np.r_[ cp_bcg_z, np.ones( len(sub_ra),) * mp_z[pp] ]

	#. save member infor
	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'ra', 'dec', 'z_spec', 'Rcen/Rv', 'g-r', 'r-i', 'g-i', 'clus_ID' ]
	values = [ cp_bcg_ra, cp_bcg_dec, cp_bcg_z, sat_ra, sat_dec, sat_z, sat_Rcen, sat_gr, sat_ri, sat_gi, sat_host_ID ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_sat_file,)

	return

def frame_limit_mem_match_func( img_cat_file, mem_cat_file, img_file, out_sat_file, Pm_cut = False):
	"""
	img_cat_file : catalog('.csv') includes cluster information, need to find frame limited satellites
	mem_cat_file : all member galaxies catalog corresponding to the image catalog
	"""

	#. img cat
	dat = pds.read_csv( img_cat_file )

	ref_ra, ref_dec, ref_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	ref_Rvir, ref_rich = np.array( dat['R_vir'] ), np.array( dat['rich'] )
	ref_clust_ID = np.array( dat['clust_ID'] )

	Ns = len( ref_ra )

	#. member find
	sat_ra, sat_dec, sat_z = np.array([]), np.array([]), np.array([])
	sat_Rcen, sat_gr, sat_ri, sat_gi = np.array([]), np.array([]), np.array([]), np.array([])

	sat_R2Rv = np.array([])
	sat_host_ID = np.array([])

	cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array([]), np.array([]), np.array([])

	for pp in range( Ns ):

		#. member cat. load
		pre_dat = Table.read( mem_cat_file, path = 'clust_%d/mem_table' % ref_clust_ID[pp],)

		sub_ra, sub_dec, sub_z = np.array( pre_dat['ra'] ), np.array( pre_dat['dec'] ), np.array( pre_dat['z_spec'] )
		
		sub_Rcen = np.array( pre_dat['Rcen'] )
		sub_Pm = np.array( pre_dat['Pm'] )

		sub_rmag = np.array( pre_dat['mod_mag_r'] )
		sub_gmag = np.array( pre_dat['mod_mag_g'] )
		sub_imag = np.array( pre_dat['mod_mag_i'] )

		sub_gr, sub_ri, sub_gi = sub_gmag - sub_rmag, sub_rmag - sub_imag, sub_gmag - sub_imag
		sub_R2Rv = sub_Rcen / ref_Rvir[ pp ]


		#. img frame load
		ra_g, dec_g, z_g = ref_ra[ pp ], ref_dec[ pp ], ref_z[ pp ]

		img_dat = fits.open( img_file % ('r', ra_g, dec_g, z_g), )
		Header = img_dat[0].header
		img_arr = img_dat[0].data

		wcs_lis = awc.WCS( Header )
		x_pos, y_pos = wcs_lis.all_world2pix( sub_ra, sub_dec, 0)


		id_x0 = ( x_pos >= 0 ) & ( x_pos <= 2047 )
		id_y0 = ( y_pos >= 0 ) & ( y_pos <= 1488 )

		if Pm_cut == False:
			id_lim = id_x0 & id_y0

		else:
			id_Pmem = sub_Pm >= 0.8   ## member probability cut
			id_lim = ( id_x0 & id_y0 ) & id_Pmem


		cut_ra, cut_dec, cut_z = sub_ra[ id_lim ], sub_dec[ id_lim ], sub_z[ id_lim ]
		cut_R2Rv = sub_R2Rv[ id_lim ]

		cut_gr, cut_ri, cut_gi = sub_gr[ id_lim ], sub_ri[ id_lim ], sub_gi[ id_lim ]
		cut_Rcen = sub_Rcen[ id_lim ]


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


### === ### catalog match
def catalog_build():

	dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
							'low_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	ra_0, dec_0, z_0 = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	dat = pds.read_csv('/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/BCG_M_bin/' + 
							'high_BCG_star-Mass_photo-z-match_rgi-common_cat_params.csv')
	ra_1, dec_1, z_1 = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

	ra = np.r_[ ra_0, ra_1 ]
	dec = np.r_[ dec_0, dec_1 ]
	z = np.r_[ z_0, z_1 ]


	##... all member match
	out_sat_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_member-cat.csv'
	out_img_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv'

	# extra_cat_match_func( ra, dec, z, out_sat_file, out_img_file)   ## member match


	##... image frame limited satellite sample (how many member located in current image catalog)
	img_cat_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv'
	img_file = '/media/xkchen/My Passport/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	mem_cat_file = '/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_cluster-sate_record.h5'
	out_sat_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv'

	# frame_limit_mem_match_func( img_cat_file, mem_cat_file, img_file, out_sat_file )


	##... image frame limited + P_mem cut
	img_cat_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv'
	img_file = '/media/xkchen/My Passport/data/SDSS/photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	mem_cat_file = '/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_cluster-sate_record.h5'
	out_sat_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_Pm-cut_member-cat.csv'

	Pm_cut = True
	# frame_limit_mem_match_func( img_cat_file, mem_cat_file, img_file, out_sat_file, Pm_cut = Pm_cut )

	return

# catalog_build()


"""
### === 
##.. for entire member sample, need to exclude BCG
##.. (since BCG is the highest P_cen member)

# dat_1 = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_Pm-cut_member-cat.csv')
dat_1 = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-limit_member-cat.csv')

s_ra, s_dec = np.array( dat_1['ra'] ), np.array( dat_1['dec'] )

s_coord = SkyCoord( ra = s_ra * U.deg, dec = s_dec * U.deg )

keys = dat_1.columns[1:]
N_ks = len( keys )

tmp_arr = []

for ll in range( N_ks ):

	tmp_arr.append( np.array( dat_1['%s' % keys[ll] ] ) )


dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
ref_ra, ref_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

Ns = len( ref_ra )

tmp_order = []

for kk in range( Ns ):

	kk_coord = SkyCoord( ra = ref_ra[kk] * U.deg, dec = ref_dec[kk] * U.deg )

	idx, d2d, d3d = kk_coord.match_to_catalog_sky( s_coord )

	daa = np.array( [ idx ] )

	tmp_order.append( daa[0] )


#.. exclude BCGs
exlu_arr = []

for ll in range( N_ks ):

	sub_arr = np.delete( tmp_arr[ll], tmp_order )

	exlu_arr.append( sub_arr )

fill = dict( zip( keys, exlu_arr ) )
data = pds.DataFrame( fill )
# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_exlu-BCG_member-cat.csv')

"""


### === ### satellites view and division
"""
s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

p_Rsat = np.array( s_dat['R_cen'] )
p_R2Rv = np.array( s_dat['Rcen/Rv'] )
clus_IDs = np.array( s_dat['clus_ID'] )


## divide by scaled R
# R_cut = 0.191 # np.median( p_Rcen )
# id_vx = p_R2Rv <= R_cut

# #. inner part
# out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx ], bcg_dec[ id_vx ], bcg_z[ id_vx ]
# out_s_ra, out_s_dec = p_ra[ id_vx ], p_dec[ id_vx ]
# out_Rsat = p_Rsat[ id_vx ]
# out_R2Rv = p_R2Rv[ id_vx ]
# out_clus_ID = clus_IDs[ id_vx ]

# keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID'] 
# values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID]
# fill = dict( zip( keys, values ) )
# data = pds.DataFrame( fill )
# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_cat.csv')


# #. outer part
# out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx == False ], bcg_dec[ id_vx == False ], bcg_z[ id_vx == False ]
# out_s_ra, out_s_dec = p_ra[ id_vx == False ], p_dec[ id_vx == False ]
# out_Rsat = p_Rsat[ id_vx == False ]
# out_R2Rv = p_R2Rv[ id_vx == False ]
# out_clus_ID = clus_IDs[ id_vx == False ]

# keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID']
# values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID ]
# fill = dict( zip( keys, values ) )
# data = pds.DataFrame( fill )
# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_cat.csv')


## divide by R
R_cut = 0.213
id_vx = p_Rsat <= R_cut

#. inner part
out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx ], bcg_dec[ id_vx ], bcg_z[ id_vx ]
out_s_ra, out_s_dec = p_ra[ id_vx ], p_dec[ id_vx ]
out_Rsat = p_Rsat[ id_vx ]
out_R2Rv = p_R2Rv[ id_vx ]
out_clus_ID = clus_IDs[ id_vx ]

keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID' ] 
values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_cat.csv')


#. outer part
out_c_ra, out_c_dec, out_c_z = bcg_ra[ id_vx == False ], bcg_dec[ id_vx == False ], bcg_z[ id_vx == False ]
out_s_ra, out_s_dec = p_ra[ id_vx == False ], p_dec[ id_vx == False ]
out_Rsat = p_Rsat[ id_vx == False ]
out_R2Rv = p_R2Rv[ id_vx == False ]
out_clus_ID = clus_IDs[ id_vx == False ]

keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'R_sat', 'R2Rv', 'clus_ID' ]
values = [ out_c_ra, out_c_dec, out_c_z, out_s_ra, out_s_dec, out_Rsat, out_R2Rv, out_clus_ID ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_cat.csv')

"""


### === satellite position record and match
##... need to go through the image catalog and record the satellite location firstly

def divi_match():

	band = ['r', 'g', 'i']

	##... position at z_ref
	# for kk in range( 3 ):

	# 	band_str = band[ kk ]

	# 	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
	# 						'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos-compare.csv' % band_str )

	# 	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )
	# 	sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

	# 	mx, my = np.array( dat['mx'] ), np.array( dat['my'] )
	# 	pk_x, pk_y = np.array( dat['peak_x'] ), np.array( dat['peak_y'] )

	# 	_off_cx, _off_cy = mx - 1, my - 1    #. position adjust


	# 	# off_R = np.sqrt( (mx - pk_x)**2 + (my - pk_y)**2 )

	# 	# plt.figure()
	# 	# plt.hist( off_R, bins = np.linspace(0, 5, 100), density = True, histtype = 'step',)
	# 	# plt.axvline( x = np.median( off_R ), ls = '--', label = 'median',)
	# 	# plt.axvline( x = np.mean( off_R ), ls = '-', label = 'mean',)
	# 	# plt.xlabel('$\\Delta_{cen-peak} \; [pixels]$')
	# 	# plt.savefig('/home/xkchen/%s-band_position_compare.png' % band_str, dpi = 300)
	# 	# plt.close()

	# 	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'cut_cx', 'cut_cy']
	# 	values = [ bcg_ra, bcg_dec, bcg_z, sat_ra, sat_dec, _off_cx, _off_cy ]
	# 	fill = dict( zip( keys, values) )
	# 	out_data = pds.DataFrame( fill )
	# 	out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
	# 					'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band_str,)

	# 	cat_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band_str
	# 	out_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band_str

	# 	z_ref = 0.25
	# 	pix_size = 0.396

	# 	zref_sat_pos_func( cat_file, z_ref, out_file, pix_size )


	s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')
	bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )

	##... divided by scaled radius
	# s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_cat.csv')
	# s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_cat.csv')
	# bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	# p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )


	##... divided by physic-R
	# s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_cat.csv')
	# s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_cat.csv')
	# bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
	# p_ra, p_dec = np.array( s_dat['sat_ra'] ), np.array( s_dat['sat_dec'] )


	pre_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

	for kk in range( 3 ):

		dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos.csv' % band[ kk ])
		kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		kk_imgx, kk_imgy = np.array( dat['cut_cx'] ), np.array( dat['cut_cy'] )

		kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

		idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
		id_lim = sep.value < 2.7e-4

		mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]
		mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'cut_cx', 'cut_cy']
		values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_member_%s-band_pos.csv' % band[kk] )

		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_%s-band_pos.csv' % band[kk] )
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_%s-band_pos.csv' % band[kk] )
		
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_%s-band_pos.csv' % band[kk] )
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_%s-band_pos.csv' % band[kk] )

		#. z-ref position
		dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[ kk ])
		kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
		kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

		kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

		idx, sep, d3d = pre_coord.match_to_catalog_sky( kk_coord )
		id_lim = sep.value < 2.7e-4

		mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	
		mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

		keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
		values = [ bcg_ra, bcg_dec, bcg_z, p_ra, p_dec, mp_imgx, mp_imgy ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_member_%s-band_pos_z-ref.csv' % band[kk] )

		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_inner-mem_%s-band_pos_z-ref.csv' % band[kk] )
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_outer-mem_%s-band_pos_z-ref.csv' % band[kk] )

		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_inner-mem_%s-band_pos_z-ref.csv' % band[kk] )
		# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/Extend-BCGM_rgi-common_frame-lim_Pm-cut_R-phy_outer-mem_%s-band_pos_z-ref.csv' % band[kk] )

	return

divi_match()

raise


##... record the member or richness of cluster sample (for Ng_weit stacking)
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

rich = np.array( dat['rich'] )
clus_IDs = np.array( dat['clust_ID'] )
R_vir = np.array( dat['R_vir'] )


tmp_Ng, tmp_Ng_80 = [], []

#. Ng record for subsamples division
tt_Ng_80_in, tt_Ng_80_out = [], []

N_ss = len( ra )

for kk in range( N_ss ):

	pre_dat = Table.read('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/' + 
						'redMaPPer_z-phot_0.1-0.33_cluster-sate_record.h5', path = 'clust_%d/mem_table' % clus_IDs[ kk ],)

	sub_Pm = np.array( pre_dat['Pm'] )
	sub_ra = np.array( pre_dat['ra'] )
	sub_Rsat = np.array( pre_dat['Rcen'] )

	_kk_N0 = len( sub_ra )

	id_Px = sub_Pm >= 0.8
	_kk_N1 = np.sum( id_Px )

	#. radius division
	kk_R2Rv = sub_Rsat[ id_Px ] / R_vir[ kk ]

	id_Rx = kk_R2Rv <= 0.191

	tt_Ng_80_in.append( np.sum( id_Rx ) )
	tt_Ng_80_out.append( np.sum( id_Rx == False ) )

	tmp_Ng.append( _kk_N0 )
	tmp_Ng_80.append( _kk_N1 )

# keys = ['ra', 'dec', 'z', 'Ng', 'Ng_80', 'rich']
# values = [ ra, dec, z, np.array( tmp_Ng ), np.array(tmp_Ng_80), rich ]
# fill = dict( zip( keys, values ) )
# data = pds.DataFrame( fill )
# data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat_Ng.csv')

##... 
keys = ['ra', 'dec', 'z', 'Ng_80']
values = [ ra, dec, z, np.array( tt_Ng_80_in) ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat_inner-mem_Ng.csv')

keys = ['ra', 'dec', 'z', 'Ng_80']
values = [ ra, dec, z, np.array( tt_Ng_80_out ) ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat_outer-mem_Ng.csv')

