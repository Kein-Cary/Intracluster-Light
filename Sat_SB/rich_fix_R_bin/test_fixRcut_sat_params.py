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


### === ### query table

##. member galaxy information catalog~( absolute magnitude)
pat = fits.open('/home/xkchen/figs/extend_Zphoto_cat/zphot_01_033_cat/' + 
				'redMaPPer_z-phot_0.1-0.33_member_params.fit')

cp_table = pat[1].data
cp_ra, cp_dec = cp_table['ra'], cp_table['dec']
cp_z, cp_zErr = cp_table['z'], cp_table['zErr']

cp_cmag_u = cp_table['cModelMag_u']
cp_cmag_g = cp_table['cModelMag_g']
cp_cmag_r = cp_table['cModelMag_r']
cp_cmag_i = cp_table['cModelMag_i']
cp_cmag_z = cp_table['cModelMag_z']

cp_coord = SkyCoord( ra = cp_ra * U.deg, dec = cp_dec * U.deg )


##. redMaPPer member catalog
cat = fits.open('/home/xkchen/mywork/ICL/data/redmapper/' + 
				'redmapper_dr8_public_v6.3_members.fits')

sat_table = cat[1].data

host_ID = sat_table['ID']  
sat_ra, sat_dec = sat_table['RA'], sat_table['DEC']

sat_objID = sat_table['OBJID']
sat_objID = sat_objID.astype( int )

#.( these mag are deredden-applied )
sat_mag_r = sat_table['MODEL_MAG_R']
sat_mag_g = sat_table['MODEL_MAG_G']
sat_mag_i = sat_table['MODEL_MAG_I']
sat_mag_u = sat_table['MODEL_MAG_U']
sat_mag_z = sat_table['MODEL_MAG_Z']

sat_coord = SkyCoord( ra = sat_ra * U.deg, dec = sat_dec * U.deg )



### === ### over all Pm-cut galaxy params
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'

dat = pds.read_csv( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_member-cat.csv')

sub_ra, sub_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
sub_coord = SkyCoord( ra = sub_ra * U.deg, dec = sub_dec * U.deg )

#.
id_x0, d2d, d3d = sub_coord.match_to_catalog_sky( cp_coord )
id_lim = d2d.value < 2.7e-4

lim_cmag_u = cp_cmag_u[ id_x0[id_lim] ]
lim_cmag_g = cp_cmag_g[ id_x0[id_lim] ]
lim_cmag_r = cp_cmag_r[ id_x0[id_lim] ]
lim_cmag_i = cp_cmag_i[ id_x0[id_lim] ]
lim_cmag_z = cp_cmag_z[ id_x0[id_lim] ]

lim_z = cp_z[ id_x0[id_lim] ]
lim_zErr = cp_zErr[ id_x0[id_lim] ]

#.
id_x1, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord )
id_lim = d2d.value < 2.7e-4

lim_mag_u = sat_mag_u[ id_x1[id_lim] ]
lim_mag_g = sat_mag_g[ id_x1[id_lim] ]
lim_mag_r = sat_mag_r[ id_x1[id_lim] ]
lim_mag_i = sat_mag_i[ id_x1[id_lim] ]
lim_mag_z = sat_mag_z[ id_x1[id_lim] ]

lim_IDs = sat_objID[ id_x1[id_lim] ]

#.
keys = list( dat.columns[1:] )
N_ks = len( keys )

tmp_arr = []

for dd in range( N_ks ):
	tmp_arr.append( np.array( dat[ keys[ dd ] ] ) )

##.
keys = keys + [	'objID', 'z', 'zErr', 
				'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
				'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z']

lim_arr = [ lim_IDs, lim_z, lim_zErr, 
			lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, 
			lim_mag_u, lim_mag_g, lim_mag_r, lim_mag_i, lim_mag_z ]

for dd in range( len( lim_arr ) ):
	tmp_arr.append( lim_arr[ dd ] )

tab_file = Table( tmp_arr, names = keys )
tab_file.write( cat_path + 'Extend-BCGM_rgi-common_frame-lim_Pm-cut_exlu-BCG_mem_params.fits', overwrite = True )

print('Done!')

# raise


### === ### subsamples

##. subsamples catalog 
cat_path = '/home/xkchen/figs/extend_bcgM_cat_Sat/rich_R_rebin/cat/'

#.
bin_rich = [ 20, 30, 50, 210 ]

# ##. fixed R for all richness subsample
# R_str = 'phy'

# # R_bins = np.array( [ 0, 300, 400, 550, 5000] )     ##. kpc
# R_bins = np.array( [ 0, 150, 300, 400, 550, 5000] )


##. average shuffle test
R_str = 'scale'

# R_bins = np.array( [0, 0.126, 0.24, 0.40, 0.56, 1] )   ### times R200m
R_bins = np.array( [0, 1e-1, 2e-1, 3e-1, 4.5e-1, 1] )   ### times R200m, for rich + sR bin


#.
for kk in range( 3 ):

	for nn in range( len( R_bins ) - 1 ):

		if R_str == 'phy':

			dat = pds.read_csv( cat_path + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_cat.csv' % 
				( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

		if R_str == 'scale':

			dat = pds.read_csv( cat_path + 
				'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_cat.csv' % 
				( bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]),)

		sub_ra, sub_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		sub_coord = SkyCoord( ra = sub_ra * U.deg, dec = sub_dec * U.deg )

		#.
		id_x0, d2d, d3d = sub_coord.match_to_catalog_sky( cp_coord )
		id_lim = d2d.value < 2.7e-4

		lim_cmag_u = cp_cmag_u[ id_x0[id_lim] ]
		lim_cmag_g = cp_cmag_g[ id_x0[id_lim] ]
		lim_cmag_r = cp_cmag_r[ id_x0[id_lim] ]
		lim_cmag_i = cp_cmag_i[ id_x0[id_lim] ]
		lim_cmag_z = cp_cmag_z[ id_x0[id_lim] ]

		lim_z = cp_z[ id_x0[id_lim] ]
		lim_zErr = cp_zErr[ id_x0[id_lim] ]

		#.
		id_x1, d2d, d3d = sub_coord.match_to_catalog_sky( sat_coord )
		id_lim = d2d.value < 2.7e-4

		lim_mag_u = sat_mag_u[ id_x1[id_lim] ]
		lim_mag_g = sat_mag_g[ id_x1[id_lim] ]
		lim_mag_r = sat_mag_r[ id_x1[id_lim] ]
		lim_mag_i = sat_mag_i[ id_x1[id_lim] ]
		lim_mag_z = sat_mag_z[ id_x1[id_lim] ]

		lim_IDs = sat_objID[ id_x1[id_lim] ]

		#.
		keys = list( dat.columns[1:] )
		N_ks = len( keys )

		tmp_arr = []

		for dd in range( N_ks ):

			tmp_arr.append( np.array( dat[ keys[ dd ] ] ) )

		##.
		keys = keys + [	'objID', 'z', 'zErr', 
						'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i', 'cModelMag_z', 
						'modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z']

		lim_arr = [ lim_IDs, lim_z, lim_zErr, 
					lim_cmag_u, lim_cmag_g, lim_cmag_r, lim_cmag_i, lim_cmag_z, 
					lim_mag_u, lim_mag_g, lim_mag_r, lim_mag_i, lim_mag_z ]

		for dd in range( len( lim_arr ) ):

			tmp_arr.append( lim_arr[ dd ] )

		##. save
		if R_str == 'phy':

			tab_file = Table( tmp_arr, names = keys )
			tab_file.write( cat_path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_phyR_%d-%dkpc_mem_params.fits' % 
					(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]), overwrite = True )

		if R_str == 'scale':

			tab_file = Table( tmp_arr, names = keys )
			tab_file.write( cat_path + 
					'Extend-BCGM_rgi-common_frame-lim_Pm-cut_rich_%d-%d_%.2f-%.2fR200m_mem_params.fits' % 
					(bin_rich[kk], bin_rich[kk + 1], R_bins[nn], R_bins[nn + 1]), overwrite = True )

print('Done!')

