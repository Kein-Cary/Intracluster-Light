"""
take 2000 satellite image to test
SB(r), compare with SDSS profMean
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
import scipy.interpolate as interp

from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from astropy.table import Table, QTable

from img_sat_pros_stack import single_img_SB_func
from img_sat_pros_stack import aveg_SB_func

from img_sat_fig_out_mode import zref_sat_pos_func
from img_sat_fig_out_mode import arr_jack_func

###... cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2arcsec = U.rad.to(U.arcsec)

band = ['r', 'g', 'i']
pixel = 0.396
z_ref = 0.25


### === 
def sat_pros_sql(ra_set, dec_set, ID_set, out_file):

	url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

	Nz = len( ra_set )

	for kk in range( Nz ):

		ra_g = ra_set[kk]
		dec_g = dec_set[kk]

		data_set = """
		SELECT
			pro.objID, pro.bin, pro.band, pro.profMean, pro.profErr
		FROM PhotoProfile AS pro
		WHERE
			pro.objID = %d
			AND pro.bin BETWEEN 0 AND 15
		""" % ID_set[kk]

		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()

		#print(data_set)
		
		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()

		s = str(response.get_data(), encoding = 'utf-8')
		doc = open( out_file % (ra_g, dec_g), 'w')
		print(s, file = doc)
		doc.close()

		print( 'kk = ', kk )

	return


### === 
"""
s_dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

bcg_ra, bcg_dec, bcg_z = np.array( s_dat['bcg_ra'] ), np.array( s_dat['bcg_dec'] ), np.array( s_dat['bcg_z'] )
p_ra, p_dec = np.array( s_dat['ra'] ), np.array( s_dat['dec'] )
clus_ID = np.array( s_dat['clus_ID'] )
R_sat = np.array( s_dat['R_cen'] )

N_ss = len( p_ra )
N_tt = 3000

rnd_seed = np.random.seed( 1 )
tt0 = np.random.choice( N_ss, N_tt, replace = False )

lim_sra, lim_sdec = p_ra[ tt0 ], p_dec[ tt0 ]
lim_clus_ID = clus_ID[ tt0 ]
lim_R_sat = R_sat[ tt0 ]

lim_coord = SkyCoord( ra = lim_sra * U.deg, dec = lim_sdec * U.deg )


### === satellite with objID
dat = fits.open( '/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/redMaPPer_z-phot_0.1-0.33_member_params.fit')

ref_ra, ref_dec, ref_IDs = dat[1].data['ra'], dat[1].data['dec'], dat[1].data['objID']

ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg )


idx, sep, d3d = lim_coord.match_to_catalog_sky( ref_coord )
id_lim = sep.value < 2.7e-4

mp_ra, mp_dec, mp_IDs = ref_ra[ idx[ id_lim ] ], ref_dec[ idx[ id_lim ] ], ref_IDs[ idx[ id_lim ] ]
mp_host_ID = lim_clus_ID[ id_lim ]
mp_R_sat = lim_R_sat[ id_lim ]

keys = [ 'ra', 'dec', 'obj_IDs', 'clus_ID', 'R_sat' ]
values=  [ mp_ra, mp_dec, mp_IDs, mp_host_ID, mp_R_sat ]

fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_cat.csv',)

raise

### === SDSS cat. sql and average SB(r) profile estimate 
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_cat.csv',)

ra, dec, IDs = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['obj_IDs'] )
out_file = '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean/ra%.4f_dec%.4f_SDSS_prof.txt'

sub_ra, sub_dec, sub_IDs = ra, dec, IDs
sat_pros_sql( sub_ra, sub_dec, sub_IDs, out_file )

"""



### === map to SDSS catalog information
##... corresponding BCG catalog
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_cat.csv')
tt_IDs = np.array( dat['clus_ID'] )

net_IDs = list( set( tt_IDs ) )
net_IDs = np.array( net_IDs ).astype( int )

cat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_cat_z02_03/Extend-BCGM_rgi-common_cat.csv')
c_ra, c_dec, c_z = np.array( cat['ra'] ), np.array( cat['dec'] ), np.array( cat['z'] )
ref_IDs = np.array( cat['clust_ID'] )

raise

N_c = len( net_IDs )

tmp_ra, tmp_dec, tmp_z = [], [], []
tmp_cID = []
tmp_Ng = []

for pp in range( N_c ):

	id_vx = ref_IDs == net_IDs[ pp ]

	tmp_ra.append( c_ra[ id_vx ][0] )
	tmp_dec.append( c_dec[ id_vx ][0] )
	tmp_z.append( c_z[ id_vx ][0] )
	tmp_cID.append( ref_IDs[ id_vx ][0] )

	id_nx = tt_IDs == net_IDs[ pp ]
	tmp_Ng.append( np.sum( id_nx ) )


keys = ['ra', 'dec', 'z', 'clus_ID', 'selected_N']
values = [ np.array( tmp_ra ), np.array( tmp_dec ), np.array( tmp_z ), np.array( tmp_cID ), np.array( tmp_Ng ) ]
fill = dict( zip( keys, values) )
out_data = pds.DataFrame( fill )
out_data.to_csv( '/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_BCG-cat.csv',)




##... satellite position cat. match
dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_cat.csv',)
sat_ra, sat_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

sat_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg)

for kk in range( 3 ):

	#. z-ref position
	dat = pds.read_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/pos_cat/' + 
			'Extend-BCGM_rgi-common_frame-limit_member_%s-band_pos_z-ref.csv' % band[ kk ])

	bcg_ra, bcg_dec, bcg_z = np.array( dat['bcg_ra'] ), np.array( dat['bcg_dec'] ), np.array( dat['bcg_z'] )

	kk_ra, kk_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )
	kk_imgx, kk_imgy = np.array( dat['sat_x'] ), np.array( dat['sat_y'] )

	kk_coord = SkyCoord( ra = kk_ra * U.deg, dec = kk_dec * U.deg )

	idx, sep, d3d = sat_coord.match_to_catalog_sky( kk_coord )
	id_lim = sep.value < 2.7e-4

	mp_bcg_ra, mp_bcg_dec, mp_bcg_z = bcg_ra[ idx[ id_lim ] ], bcg_dec[ idx[ id_lim ] ], bcg_z[ idx[ id_lim ] ]

	mp_ra, mp_dec = kk_ra[ idx[ id_lim ] ], kk_dec[ idx[ id_lim ] ]	

	mp_imgx, mp_imgy = kk_imgx[ idx[ id_lim ] ], kk_imgy[ idx[ id_lim ] ]

	print( len(mp_ra) )

	keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'sat_ra', 'sat_dec', 'sat_x', 'sat_y']
	values = [ mp_bcg_ra, mp_bcg_dec, mp_bcg_z, mp_ra, mp_dec, mp_imgx, mp_imgy ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv('/home/xkchen/figs/extend_bcgM_cat_Sat/sat_SB_check/SDSS_profMean_check_cat_%s-band_pos_z-ref.csv' % band[kk] )


