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
from img_sat_shuffle_map import no_alin_shufl_func
from img_sat_shuffle_map import frame_align_shufll_func
from img_sat_shuffle_map import BCG_align_shufl_func

#.
import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


### === ### cosmology model
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



### === ### shuffle satellite and but fix_Rsat only~(no alignment)
"""
cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/BG_imgs_nomock/shufl_cat/'

#.
bin_rich = [ 20, 30, 50, 210 ]

N_shufl = 60

#.
for pp in range( rank, rank + 1 ):

	for dd in range( 3 ):

		band_str = band[ dd ]

		for tt in range( 3 ):

			#.
			clust_cat = cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1])

			#.
			mem_cat = ( cat_path + 
					'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv'
					% (bin_rich[tt], bin_rich[tt + 1], band_str), )[0]

			#.
			out_file = ( out_path + 'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv' % 
							(bin_rich[tt], bin_rich[tt + 1], band_str, pp), )[0]

			no_alin_shufl_func( clust_cat, mem_cat, out_file )

raise
"""


### === ### shuffle satellite fixed the alignment with the image frame
"""
bin_rich = [ 20, 30, 50, 210 ]

cat_path = '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/shufl_img_wBCG/shufl_cat/'

N_shufl = 60

for kk in range( rank, rank + 1 ):

	for dd in range( 3 ):

		band_str = band[ dd ]

		for tt in range( 3 ):

			#.
			clust_cat = cat_path + 'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1])

			#.
			mem_cat = ( cat_path + 
						'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv'
						% (bin_rich[tt], bin_rich[tt + 1], band_str), )[0]

			##. id_fixRs = True
			out_file = ( out_path + 
						'clust_rich_%d-%d_%s-band_sat_fixRs-shuffle-%d_position.csv'
						% (bin_rich[tt], bin_rich[tt + 1], band_str, kk), )[0]

			id_fixRs = True
			frame_align_shufll_func(  clust_cat, mem_cat, out_file, id_fixRs = id_fixRs)


			##. id_fixRs = False
			out_file = ( out_path + 
						'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv'
						% (bin_rich[tt], bin_rich[tt + 1], band_str, kk), )[0]

			frame_align_shufll_func(  clust_cat, mem_cat, out_file )

raise
"""


### === ### shuffle satellite fixed the alignment with the major axis of BCG
"""
##. Position Angle record
recd_path = '/home/xkchen/data/SDSS/member_files/shufl_woBCG_wPA/PA_cat/'

bin_rich = [ 20, 30, 50, 210 ]

#.
for kk in range( 3 ):

	band_str = band[ kk ]

	##.
	cat_1 = pds.read_csv( '/home/xkchen/data/SDSS/member_files/shufl_woBCG_wPA/PA_cat/' + 
							'BCG_located-params_%s-band.csv' % band_str,)
	ra_1, dec_1, z_1 = np.array( cat_1['ra'] ), np.array( cat_1['dec'] ), np.array( cat_1['z'] )

	IDs_1 = np.array( cat_1['clus_ID'] )
	IDs_1 = IDs_1.astype( int )

	#. -90 ~ 90, in unit of deg
	PA_1 = np.array( cat_1['PA'] )

	#.
	for tt in range( 3 ):

		##. cluster catalog
		cat = pds.read_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/' + 
						'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1]),)

		set_IDs = np.array( cat['clust_ID'] )
		set_IDs = set_IDs.astype( int )

		N_clus = len( set_IDs )


		##. memmber catalog
		dat = pds.read_csv( '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/' + 
				'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_origin-img_position.csv' % 
				(bin_rich[tt], bin_rich[tt + 1], band_str ),)

		#.
		keys = list( dat.columns[1:] )

		N_ks = len( keys )

		tmp_arr = []

		#.
		for nn in range( N_ks ):

			tmp_arr.append( np.array( dat[ keys[ nn ] ] ) )

		#.
		sat_ra, sat_dec = np.array( dat['sat_ra'] ), np.array( dat['sat_dec'] )

		sub_IDs = np.array( dat['clus_ID'] )
		sub_IDs = sub_IDs.astype( int )

		N_sat = len( sat_ra )

		##. BCG PA mapping
		tmp_bcg_PA = np.zeros( N_sat,)

		for nn in range( N_clus ):

			id_vx = sub_IDs == set_IDs[ nn ]
			id_ux = IDs_1 == set_IDs[ nn ]

			if np.sum( id_vx ) > 0:
				tmp_bcg_PA[ id_vx ] = np.ones( np.sum(id_vx),) * PA_1[ id_ux ][0]

			else:
				continue

		##. in unit of rad
		tmp_bcg_PA = tmp_bcg_PA * np.pi / 180

		keys.append( 'BCG_PA' )
		tmp_arr.append( tmp_bcg_PA )

		fill = dict( zip( keys, tmp_arr ) )
		out_data = pds.DataFrame( fill )
		out_data.to_csv( recd_path + 
					'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_orin-img-pos_with-BCG-PA.csv' % 
					(bin_rich[tt], bin_rich[tt + 1], band_str),)

raise
"""


### ... 
cat_path = '/home/xkchen/data/SDSS/member_files/shufl_woBCG_wPA/PA_cat/'
out_path = '/home/xkchen/data/SDSS/member_files/shufl_woBCG_wPA/shufl_cat/'

bin_rich = [ 20, 30, 50, 210 ]

N_shufl = 60

for kk in range( rank, rank + 1):

	for dd in range( 3 ):

		band_str = band[ dd ]

		for tt in range( 3 ):

			#.
			clust_cat = ( '/home/xkchen/fig_tmp/Extend_Mbcg_richbin_sat_cat/' + 
					'clust_rich_%d-%d_cat.csv' % (bin_rich[tt], bin_rich[tt + 1]), )[0]

			#.
			mem_cat = ( cat_path + 
					'clust_rich_%d-%d_rgi-common_frame-lim_Pm-cut_exlu-BCG_sat_%s-band_orin-img-pos_with-BCG-PA.csv'
					% (bin_rich[tt], bin_rich[tt + 1], band_str), )[0]

			##.
			out_file = ( out_path + 'clust_rich_%d-%d_%s-band_sat_fixRs-shuffle-%d_position.csv'
					% (bin_rich[tt], bin_rich[tt + 1], band_str, kk), )[0]

			id_fixRs = True
			BCG_align_shufl_func( clust_cat, mem_cat, out_file, id_fixRs = id_fixRs)


			##.
			out_file = ( out_path + 'clust_rich_%d-%d_%s-band_sat-shuffle-%d_position.csv'
					% (bin_rich[tt], bin_rich[tt + 1], band_str, kk), )[0]

			BCG_align_shufl_func( clust_cat, mem_cat, out_file )

