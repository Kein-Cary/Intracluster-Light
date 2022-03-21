"""
for given random cluster in redMaPPer, using this file to find the member galaxies and
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

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

#.
from Mass_rich_radius import rich2R_Simet
from img_sat_fig_out_mode import zref_sat_pos_func


### === cosmology model
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


def frame_limit_mem_match_func( img_cat_file, galx_cat_file, img_file, band_lis, out_galx_file):
	"""
	img_cat_file : catalog('.csv') includes cluster information, need to find frame limited satellites
	galx_cat_file : all galaxies in the sky coverage of SDSS
    img_file : images corresponding to the random cluster catalog
    out_galx_file : image + galaxy matched catalog
	"""

	#. random cluster catalog image cat
	dat = pds.read_csv( img_cat_file )
	ref_ra, ref_dec, ref_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
	ref_rich = np.array( dat['rich'] )

	#. order the random cluster
	Ns = len( ref_ra )

	list_ord = np.arange( Ns )
	ref_clust_ID = list_ord.astype( int )


	#. 'member' match~( here may not be member, but the galaxy within a given image frame)
	sat_ra, sat_dec, sat_z = np.array([]), np.array([]), np.array([])
	sat_gr, sat_ri, sat_gi = np.array([]), np.array([]), np.array([])
	sat_objID = np.array([])

	sat_host_ID = np.array([])  ##. here the ID is the order for cluster match
	cp_bcg_ra, cp_bcg_dec, cp_bcg_z = np.array([]), np.array([]), np.array([])


	for pp in range( Ns ):

		#. galaxy cat. load
		galx_dat = fits.open( galx_cat_file )
		pre_dat = galx_dat[1].data

		sub_ra, sub_dec, sub_z = np.array( pre_dat['ra'] ), np.array( pre_dat['dec'] ), np.array( pre_dat['z_spec'] )

		sub_objID = np.array( pre_dat['objID'] )
		sub_rmag = np.array( pre_dat['model_r'] )
		sub_gmag = np.array( pre_dat['model_g'] )
		sub_imag = np.array( pre_dat['model_i'] )

		sub_gr, sub_ri, sub_gi = sub_gmag - sub_rmag, sub_rmag - sub_imag, sub_gmag - sub_imag


		#. img frame load
		ra_g, dec_g, z_g = ref_ra[ pp ], ref_dec[ pp ], ref_z[ pp ]

		img_dat = fits.open( img_file % (band_lis, ra_g, dec_g, z_g), )
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
		cut_gr, cut_ri, cut_gi = sub_gr[ id_lim ], sub_ri[ id_lim ], sub_gi[ id_lim ]
		cut_objID = sub_objID[ id_lim ]


		#. record array
		sat_ra = np.r_[ sat_ra, cut_ra ]
		sat_dec = np.r_[ sat_dec, cut_dec ]
		sat_z = np.r_[ sat_z, cut_z ]

		sat_gr = np.r_[ sat_gr, cut_gr ]
		sat_ri = np.r_[ sat_ri, cut_ri ]
		sat_gi = np.r_[ sat_gi, cut_gi ]

		sat_objID = np.r_[ sat_objID, cut_objID ]

		sat_host_ID = np.r_[ sat_host_ID, np.ones( len(cut_ra),) * ref_clust_ID[pp] ]
		cp_bcg_ra = np.r_[ cp_bcg_ra, np.ones( len(cut_ra),) * ref_ra[pp] ]
		cp_bcg_dec = np.r_[ cp_bcg_dec, np.ones( len(cut_ra),) * ref_dec[pp] ]
		cp_bcg_z = np.r_[ cp_bcg_z, np.ones( len(cut_ra),) * ref_z[pp] ]

	#. save member infor
	keys = [ 'bcg_ra', 'bcg_dec', 'bcg_z', 'ra', 'dec', 'z_spec', 'g-r', 'r-i', 'g-i', 'clus_ID' ]
	values = [ cp_bcg_ra, cp_bcg_dec, cp_bcg_z, sat_ra, sat_dec, sat_z, sat_gr, sat_ri, sat_gi, sat_host_ID ]
	fill = dict( zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_galx_file )

	return


def img_member_reselect():
	"""
	to count galaxies belong to a same image frame~(based on the order given in match processing)
	"""

	return


### === data load
"""
##.. read the table of all galaxy and save as fits file
c_path = '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/'

with h5py.File( c_path + 
	'redmapper_notclean_objid_ugriz_BLENDED_or_notNODEBLEND_ALL_i_err_nocollision_nocenterpost.hdf5', 'r') as f:
	pat = np.array( f['Galaxy'] )

keys = ['ra', 'dec', 'z_spec', 'cmodel_i', 'cmodelerr_i', 'objID', 
		'model_g', 'model_r', 'model_i', 'model_u', 'model_z', 
		'modelerr_g', 'modelerr_r', 'modelerr_i', 'modelerr_u', 'modelerr_z']

ra_arr, dec_arr, ID_arr = np.array( pat['RA'] ), np.array( pat['DEC'] ), np.array( pat['objid'] )
cModmag_i, cModMagErr_i = np.array( pat['cmodel_i'] ), np.array( pat['cmodelerr_i'] )

modMag_r, madMagErr_r = np.array( pat['model_r'] ), np.array( pat['modelerr_r'] )
modMag_g, madMagErr_g = np.array( pat['model_g'] ), np.array( pat['modelerr_g'] )
modMag_i, madMagErr_i = np.array( pat['model_i'] ), np.array( pat['modelerr_i'] )
modMag_u, madMagErr_u = np.array( pat['model_u'] ), np.array( pat['modelerr_u'] )
modMag_z, madMagErr_z = np.array( pat['model_z'] ), np.array( pat['modelerr_z'] )

##. assume z_spec = 0.25~( will match real z_photo from SDSS later)
z_arr = np.ones( len( modMag_g ), ) * 0.25

colm_arr = [ ra_arr, dec_arr, z_arr, cModmag_i, cModMagErr_i, ID_arr,
			modMag_g, modMag_r, modMag_i, modMag_u, modMag_z, 
			madMagErr_g, madMagErr_r, madMagErr_i, madMagErr_u, madMagErr_z ]

##. save 
tab_file = Table( colm_arr, names = keys )
tab_file.write( c_path + 
	'redmapper_notclean_objid_ugriz_BLENDED_or_notNODEBLEND_ALL_i_err_nocollision_nocenterpost.fits', overwrite = True)

"""


##.. the total selected image in random catalog~(have applied image selection)
c_path = '/home/xkchen/data/SDSS/field_galx_redMap/galx_cat/'
rand_path = '/home/xkchen/fig_tmp/random_cat/2_28/'
img_path = '/home/xkchen/data/SDSS/redMap_random/'
out_path = '/home/xkchen/data/SDSS/field_galx_redMap/redMap_rand_match_cat/'

# c_path = '/home/xkchen/figs/field_sat_redMap/galx_cat/'
# rand_path = '/home/xkchen/mywork/ICL/data/cat_random/match_2_28/'
# img_path = '/media/xkchen/My Passport/data/SDSS/redMap_random/'


for tt in range( rank, rank + 1):

    band_str = band[ tt ]

    ##. selected random image catalog
    img_cat_file = rand_path + 'random_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band_str

    ##. galaxy over SDSS sky-coverage~( with the same selection as redMaPPer)
    galx_cat_file = ( c_path + 
        'redmapper_notclean_objid_ugriz_BLENDED_or_notNODEBLEND_ALL_i_err_nocollision_nocenterpost.fits', )[0]

    img_file = img_path + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
    out_galx_file = out_path + 'SDSS_redMaPPer_random_%s-band_match_field_galax_cat.csv' % band_str

    frame_limit_mem_match_func( img_cat_file, galx_cat_file, img_file, band_str, out_galx_file )

    print( '%s band done!' % band_str )

