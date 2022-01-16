import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pds
from io import StringIO
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

#. dust map with the recalibration by Schlafly & Finkbeiner (2011)
import sfdmap
E_map = sfdmap.SFDMap('/home/xkchen/module/dust_map/sfddata_maskin')
from extinction_redden import A_wave
Rv = 3.1


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
Mag_sun = [ 4.65, 5.11, 4.53 ]
z_ref = 0.25


### === ### dust map query (on BCG positions)
# dat = pds.read_csv('/home/xkchen/Extend_BCGM_bin_cat.csv')
# ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

# pos_deg = SkyCoord( ra, dec, unit = 'deg')
# p_EBV = E_map.ebv( pos_deg )
# A_v = Rv * p_EBV

# Al_r = A_wave( L_wave[ 0 ], Rv) * A_v
# Al_g = A_wave( L_wave[ 1 ], Rv) * A_v
# Al_i = A_wave( L_wave[ 2 ], Rv) * A_v

# keys = [ 'ra', 'dec', 'z', 'E_bv', 'Al_r', 'Al_g', 'Al_i' ]
# values = [ ra, dec, z, p_EBV, Al_r, Al_g, Al_i ]
# fill = dict( zip( keys, values) )
# out_data = pds.DataFrame( fill )
# out_data.to_csv( '/home/xkchen/Extend_NCGM_bin_dust_value.csv')

def samp_params_map():

	##... subsamples dust match
	path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/'

	dat = pds.read_csv( path + 'Extend_NCGM_bin_dust_value.csv')
	ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	Al_g, Al_r, Al_i = np.array( dat['Al_g'] ), np.array( dat['Al_r'] ), np.array( dat['Al_i'] )

	ref_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )

	cc_dat = pds.read_csv( path + 'Extend_BCGM_bin_cat.csv')

	lg_Mstar = np.array( cc_dat['lg_Mbcg'] )  ## in units of M_sun / h^2
	rich = np.array( cc_dat['rich'] )
	age = np.array( cc_dat['age'] )


	##.. subsamples
	# cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
	# cat_file = path + 'BCG_M_bin/%s_r-band_photo-z-match_rgi-common_BCG_cat.csv'
	# out_file = path + 'BCG_M_bin/%s_photo-z-match_rgi-common_cat_params.csv'

	# cat_lis = [ 'low-rich', 'hi-rich' ]
	# cat_file = path + 'rich_bin_fixed_bcgM/%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv'
	# out_file = path + 'rich_bin_fixed_bcgM/%s_photo-z-match_rgi-common_cat_params.csv'

	# cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
	# cat_file = path + 'BCG_M_bin/%s_gri-common_P-cen_lim_cat.csv'
	# out_file = path + 'BCG_M_bin/%s_gri-common_P-cen_lim_cat_params.csv'

	# cat_lis = ['low-lgM20', 'hi-lgM20']
	# aper_path = '/home/xkchen/figs/extend_bcgM_cat/aperM_bin/'
	# cat_file = aper_path + '%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv'
	# out_file = aper_path + '%s_r-band_photo-z-match_rgi-common_cat_params.csv'


	#.. re-binned richness subsamples
	cat_lis = [ 'low-rich', 'hi-rich' ]
	cat_file = path + 're_bin_rich_bin/%s_r-band_photo-z-match_rgi-common_BCG-pos_cat.csv'
	out_file = path + 're_bin_rich_bin/%s_photo-z-match_rgi-common_cat_params.csv'


	for ii in range( 2 ):

		pat = pds.read_csv( cat_file % cat_lis[ii],)
		p_ra, p_dec, p_z = np.array( pat['ra'] ), np.array( pat['dec'] ), np.array( pat['z'] )

		sub_coord = SkyCoord( ra = p_ra * U.deg, dec = p_dec * U.deg )

		idx, sep, d3d = sub_coord.match_to_catalog_sky( ref_coord )
		id_lim = sep.value < 2.7e-4

		mp_Alg, mp_Alr, mp_Ali = Al_g[ idx[ id_lim ] ], Al_r[ idx[ id_lim ] ], Al_i[ idx[ id_lim ] ]
		mp_rich, mp_lgMstar = rich[ idx[ id_lim ] ], lg_Mstar[ idx[ id_lim ] ]
		mp_age = age[ idx[ id_lim ] ]

		#. save
		keys = ['ra', 'dec', 'z', 'rich', 'lg_Mstar', 'age', 'Al_r', 'Al_g', 'Al_i']
		values = [ p_ra, p_dec, p_z, mp_rich, mp_lgMstar, mp_age, mp_Alr, mp_Alg, mp_Ali ]
		fill = dict( zip( keys, values ) )
		data = pds.DataFrame( fill )
		data.to_csv( out_file % cat_lis[ii],)


samp_params_map()

raise

### === ### P_cen query and deredden color query
path = '/home/xkchen/mywork/ICL/data/BCG_Mstar_extend_cat/'

dat = pds.read_csv( path + 'Extend_BCGM_bin_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

lg_Mstar = np.array( dat['lg_Mbcg'] )  # mass unit : M_sun / h^2

tt_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg )


sdss_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
goal_data = fits.getdata( sdss_file )

RA = np.array( goal_data.RA )
DEC = np.array( goal_data.DEC )
Z_photo = np.array( goal_data.Z_LAMBDA )
ID = np.array( goal_data.OBJID )

P_cen = np.array( goal_data.P_CEN )


#. deredden magnitude
r_mag_bcgs = np.array(goal_data.MODEL_MAG_R)
g_mag_bcgs = np.array(goal_data.MODEL_MAG_G)
i_mag_bcgs = np.array(goal_data.MODEL_MAG_I)

model_g2r = g_mag_bcgs - r_mag_bcgs
model_g2i = g_mag_bcgs - i_mag_bcgs
model_r2i = r_mag_bcgs - i_mag_bcgs


idx_lim = ( Z_photo >= 0.2 ) & ( Z_photo <= 0.3 )

lim_ra, lim_dec, lim_z = RA[ idx_lim ], DEC[ idx_lim ], Z_photo[ idx_lim ]

lim_IDs = ID[ idx_lim ]
lim_Pcen = P_cen[idx_lim]

lim_g_mag, lim_r_mag, lim_i_mag = g_mag_bcgs[ idx_lim ], r_mag_bcgs[ idx_lim ], i_mag_bcgs[ idx_lim ]

lim_g2r, lim_g2i, lim_r2i = model_g2r[ idx_lim ], model_g2i[ idx_lim ], model_r2i[ idx_lim ]

tot_coord = SkyCoord( ra = lim_ra * U.deg, dec = lim_dec * U.deg )


#... match P_cen and deredden color
idx, sep, d3d = tt_coord.match_to_catalog_sky( tot_coord )
id_lim = sep.value < 2.7e-4

mp_Pcen_0 = lim_Pcen[:,0][ idx[ id_lim ] ]
mp_Pcen_1 = lim_Pcen[:,1][ idx[ id_lim ] ]
mp_Pcen_2 = lim_Pcen[:,2][ idx[ id_lim ] ]
mp_Pcen_3 = lim_Pcen[:,3][ idx[ id_lim ] ]
mp_Pcen_4 = lim_Pcen[:,4][ idx[ id_lim ] ]

mp_mod_mag_g = lim_g_mag[ idx[ id_lim ] ]
mp_mod_mag_r = lim_r_mag[ idx[ id_lim ] ]
mp_mod_mag_i = lim_i_mag[ idx[ id_lim ] ]

mp_dered_g2r, mp_dered_g2i, mp_dered_r2i = lim_g2r[ idx[ id_lim ] ], lim_g2i[ idx[ id_lim ] ], lim_r2i[ idx[ id_lim ] ]


keys = ['ra', 'dec', 'z', 'P_cen0', 'P_cen1', 'P_cen2', 'P_cen3', 'P_cen4',
		'mode_g_mag', 'mode_r_mag', 'mode_i_mag', 'dered_g-r', 'dered_g-i', 'dered_r-i', 'lg_Mstar']

values = [ ra, dec, z, mp_Pcen_0, mp_Pcen_1, mp_Pcen_2, mp_Pcen_3, mp_Pcen_4,
		mp_mod_mag_g, mp_mod_mag_r, mp_mod_mag_i, mp_dered_g2r, mp_dered_g2i, mp_dered_r2i, lg_Mstar ]

fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv( path + 'Extend_BCGM_bin_dered-mag_Pcen_cat.csv')


### ... subsample Pcen cut match
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']

Pcen_lim = 0.85

tmp_Pcen = []

for kk in range( 2 ):

	k_dat = pds.read_csv( path + 'BCG_M_bin/%s_r-band_photo-z-match_rgi-common_BCG_cat.csv' % cat_lis[kk],)
	sub_ra, sub_dec, sub_z = np.array( k_dat['ra'] ), np.array( k_dat['dec'] ), np.array( k_dat['z'] )

	kk_coord = SkyCoord( ra = sub_ra * U.deg, dec = sub_dec * U.deg )

	ordex, d2d, d3d = kk_coord.match_to_catalog_sky( tot_coord )
	id_lim = d2d.value < 2.7e-4

	kk_Pcen_0 = lim_Pcen[:,0][ ordex[ id_lim ] ]
	kk_Pcen_1 = lim_Pcen[:,1][ ordex[ id_lim ] ]
	kk_Pcen_2 = lim_Pcen[:,2][ ordex[ id_lim ] ]
	kk_Pcen_3 = lim_Pcen[:,3][ ordex[ id_lim ] ]
	kk_Pcen_4 = lim_Pcen[:,4][ ordex[ id_lim ] ]

	kk_P_cen = [ kk_Pcen_0, kk_Pcen_1, kk_Pcen_2, kk_Pcen_3, kk_Pcen_4 ]


	#. save the properties of subsamples
	keys = ['ra', 'dec', 'z', 'P_cen_0', 'P_cen_1', 'P_cen_2', 'P_cen_3', 'P_cen_4' ]
	values = [ sub_ra, sub_dec, sub_z, kk_Pcen_0, kk_Pcen_1, kk_Pcen_2, kk_Pcen_3, kk_Pcen_4 ]

	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( path + 'BCG_M_bin/%s_gri-common_Pcen_cat.csv' % cat_lis[kk], )


	#. save the P_cen selected samples
	id_Px = kk_Pcen_0 >= Pcen_lim

	print( np.sum(id_Px) / len( sub_ra ) )
	bin_ra, bin_dec, bin_z = sub_ra[ id_Px ], sub_dec[ id_Px ], sub_z[ id_Px ]	

	keys = ['ra', 'dec', 'z']
	values = [ bin_ra, bin_dec, bin_z ]
	fill = dict( zip( keys, values ) )
	data = pds.DataFrame( fill )
	data.to_csv( path + 'BCG_M_bin/%s_gri-common_P-cen_lim_cat.csv' % cat_lis[kk], )

	tmp_Pcen.append( kk_Pcen_0 )


