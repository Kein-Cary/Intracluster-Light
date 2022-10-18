"""
compare measurement of galaxies located in overlap region of image frames
"""
import sys 
sys.path.append('/home/xkchen/tool/Conda/Tools/normitems')

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpathes
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import h5py
import random
import numpy as np
import pandas as pds

import scipy.stats as scists
import astropy.stats as astrosts

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc

from scipy import interpolate as interp
from astropy import cosmology as apcy
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates import Angle

#.
from tqdm import tqdm
import subprocess as subpro
import glob

#.
from img_segmap import array_mask_func
from img_segmap import simp_finder_func
from light_measure import light_measure_Z0_weit


##. cosmology
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##. constants
# u,g,r,i,z == 0,1,2,3,4
band_id = [ 0, 1, 2, 3, 4 ]
band = ['u', 'g', 'r', 'i', 'z']

#.
band_str = 'r'
pixel = 0.396

rad2arcsec = U.rad.to(U.arcsec)



### === ### cluster catalog
#.
dat = pds.read_csv('/home/xkchen/figs_cp/SDSS_img_load/cat_file/' + 
				'redMaPPer_z-phot_0.1-0.33_clust_sql_size.csv',)

bcg_ra, bcg_dec, bcg_z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

sub_IDs = np.array( dat['clus_ID'] )
sub_IDs = sub_IDs.astype( int )

N_g = len( bcg_ra )



### === ### BCG modelling and compare
out_path = '/home/xkchen/figs_cp/SDSS_img_load/sat_SB_compare/obj_file/'
SB_path = '/home/xkchen/figs_cp/SDSS_img_load/sat_SB_compare/SB_pros/'

##.
"""
m_dat = fits.getdata('/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_members.fits')

clus_IDs = np.array( m_dat.ID )   ## the ID number of cluster in redMaPPer
P_mem = np.array( m_dat.P )
p_ra, p_dec = np.array( m_dat.RA ), np.array( m_dat.DEC )

for nn in range( 4, 5 ):

	##. combined image
	data = fits.open('/home/xkchen/figs_cp/SDSS_img_load/drizzle_frame_r-band_test.fits')
	comb_img = data[1].data
	wcs_lis = awc.WCS( data[1].header )

	##. member location
	id_vx = clus_IDs == sub_IDs[ nn ]
	mp_s_ra, mp_s_dec = p_ra[ id_vx ], p_dec[ id_vx ]
	mp_Pm = P_mem[ id_vx ]

	sx_1, sy_1 = wcs_lis.all_world2pix( mp_s_ra, mp_s_dec, 0 )

	##. SB of satellite in different image frame
	Ns = np.sum( id_vx )

	#.save the member catalog
	keys = ['ra', 'dec', 'sx', 'sy', 'Pm']
	values = [mp_s_ra, mp_s_dec, sx_1, sy_1, mp_Pm]

	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( out_path + 'tmp_sat-cat_list.csv',)


	##. individual image
	frame_files = glob.glob('/home/xkchen/figs_cp/SDSS_img_load/frame*.fits.bz2')
	Nf = len( frame_files )

	#.
	# for pp in range( Ns ):
	for pp in range( 10 ):

		ra_x, dec_x = mp_s_ra[ pp ], mp_s_dec[ pp ]

		#.
		n_rin = 25

		tmp_R = np.zeros( (Nf, n_rin),)
		tmp_SB = np.zeros( (Nf, n_rin),)
		tmp_SB_err = np.zeros( (Nf, n_rin),)

		for dd in range( Nf ):

			img_data = fits.open( frame_files[ dd ] )

			img_arr = img_data[0].data
			wcs_str = awc.WCS( img_data[0].header )

			dd_x, dd_y = wcs_str.all_world2pix( ra_x, dec_x, 0 )

			identi_0 = ( dd_x >= 0 ) & ( dd_x <= 2047 )
			identi_1 = ( dd_y >= 0 ) & ( dd_y <= 1488 )
			identi = identi_0 & identi_1

			Rp_bins = np.logspace(0, 2, n_rin)

			if identi:

				Npx, Npy = img_arr.shape[1], img_arr.shape[0]
				weit_arr = np.ones( (Npy, Npx), dtype = np.float32 )

				SB_list = light_measure_Z0_weit( img_arr, weit_arr, pixel, dd_x, dd_y, Rp_bins)
				Intns, Angl_r, Intns_err, N_pix, nsum_ratio = SB_list[:]

				id_nul = N_pix < 1.

				dd_r = Angl_r + 0.
				dd_sb = Intns + 0.
				dd_err = Intns_err + 0.

				dd_r[ id_nul ] = np.nan
				dd_sb[ id_nul ] = np.nan
				dd_err[ id_nul ] = np.nan

			else:

				dd_r = np.ones( n_rin,) * np.nan
				dd_sb = np.ones( n_rin,) * np.nan
				dd_err = np.ones( n_rin,) * np.nan

			#.
			tmp_R[ dd,:] = dd_r
			tmp_SB[ dd,:] = dd_sb
			tmp_SB_err[ dd,:] = dd_err

		##. save
		keys = ['R', 'SB', 'sb_err']
		values = [tmp_R, tmp_SB, tmp_SB_err]

		tab_file = Table( values, names = keys )
		tab_file.write( SB_path + 'tmp_sat-SB_ra%.5f-dec%.5f_on-orin-frame.fits' % (ra_x, dec_x), 
						overwrite = True )

raise
"""


##. SB profiles on the drizzle co-added image
"""
data = fits.open('/home/xkchen/figs_cp/SDSS_img_load/drizzle_frame_r-band_test.fits')
comb_img = data[1].data

wcs_lis = awc.WCS( data[1].header )

#.
dat = pds.read_csv( out_path + 'tmp_sat-cat_list.csv',)
s_ra, s_dec = np.array( dat['ra'] ), np.array( dat['dec'] )
sat_x, sat_y = np.array( dat['sx'] ), np.array( dat['sy'] )

N_s = len( s_ra )

# for dd in range( N_s ):
for dd in range( 10 ):

	ra_x, dec_x = s_ra[ dd ], s_dec[ dd ]

	#.
	n_rin = 25

	dd_x, dd_y = wcs_lis.all_world2pix( ra_x, dec_x, 0 )

	Rp_bins = np.logspace(0, 2, n_rin)

	#.
	R_cut = 100

	nx, ny = np.int( dd_x), np.int( dd_y )
	off_x, off_y = dd_x - nx, dd_y - ny

	cut_img = comb_img[ ny-R_cut: ny+R_cut, nx-R_cut: nx+R_cut ]
	cut_weit = np.ones( (cut_img.shape[0], cut_img.shape[1]), dtype = np.float32)

	cx, cy = 100 + off_x, 100 + off_y

	#.
	SB_list = light_measure_Z0_weit( cut_img, cut_weit, pixel, cx, cy, Rp_bins)
	Intns, Angl_r, Intns_err, N_pix, nsum_ratio = SB_list[:]

	id_nul = N_pix < 1.

	dd_r = Angl_r + 0.
	dd_sb = Intns + 0.
	dd_err = Intns_err + 0.

	dd_r[ id_nul ] = np.nan
	dd_sb[ id_nul ] = np.nan
	dd_err[ id_nul ] = np.nan

	#.
	keys = ['R', 'SB', 'sb_err']
	values = [ dd_r, dd_sb, dd_err ]
	fill = dict( zip( keys, values ) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( SB_path + 'tmp_sat-SB_ra%.5f-dec%.5f_co-add-frame.csv' % (ra_x, dec_x),)

raise
"""


##... figs
dat = pds.read_csv( out_path + 'tmp_sat-cat_list.csv',)
s_ra, s_dec = np.array( dat['ra'] ), np.array( dat['dec'] )

N_s = len( s_ra )

for dd in range( 10 ):

	ra_x, dec_x = s_ra[ dd ], s_dec[ dd ]

	#.
	pat = pds.read_csv( SB_path + 'tmp_sat-SB_ra%.5f-dec%.5f_co-add-frame.csv' % (ra_x, dec_x),)

	tt_r = np.array( pat['R'] )
	tt_sb = np.array( pat['SB'] )
	tt_err = np.array( pat['sb_err'] )

	id_nul = np.isnan( tt_sb )
	
	dt_r = tt_r[ id_nul == False ]
	dt_sb = tt_sb[ id_nul == False ]
	tmp_F = interp.interp1d( dt_r, dt_sb, kind = 'linear', fill_value = 'extrapolate')

	#.
	sb_data = fits.open( SB_path + 'tmp_sat-SB_ra%.5f-dec%.5f_on-orin-frame.fits' % (ra_x, dec_x),)

	R_arrs = sb_data[1].data['R']
	SB_arrs = sb_data[1].data['SB']
	err_arrs = sb_data[1].data['sb_err']

	pda = SB_arrs[:,-2]

	N_ps = len( pda )

	id_nan = np.isnan( pda )
	n_dex = np.where(id_nan == False )[0]

	#.
	fig = plt.figure()
	ax = fig.add_axes([0.12, 0.32, 0.80, 0.63])
	sub_ax = fig.add_axes([0.12, 0.11, 0.80, 0.21])

	ax.plot( tt_r, tt_sb, 'k-', alpha = 0.75, label = 'co-add')

	did = 0

	for mm in ( n_dex ):

		did += 1

		ax.plot( R_arrs[mm,:], SB_arrs[mm,:], ls = '--', color = mpl.cm.rainbow( did / len(n_dex) ),
				alpha = 0.5,)

		sub_ax.plot( R_arrs[mm,:], SB_arrs[mm,:] / tmp_F( R_arrs[mm,:] ), ls = '--', 
				color = mpl.cm.rainbow( did / len(n_dex) ), alpha = 0.5,)

	ax.legend( loc = 3, frameon = False,)
	ax.set_xlabel('R [arcsec]')
	ax.set_xscale('log')

	ax.set_ylabel('$ \\mu \; [nanomaggy \, / \, arcsec^{2}] $')
	ax.set_yscale('log')

	sub_ax.set_xlim( ax.get_xlim() )
	sub_ax.set_xlabel('R [arcsec]')
	sub_ax.set_xscale('log')

	sub_ax.set_ylabel('$\\mu \, / \, \\mu_{co{-}add}$')

	sub_ax.yaxis.set_minor_locator( ticker.AutoMinorLocator() )
	sub_ax.tick_params( axis = 'both', which = 'both', direction = 'in',)
	ax.tick_params( axis = 'both', which = 'both', direction = 'in',)
	ax.set_xticklabels( labels = [] )

	plt.savefig('/home/xkchen/Sat_SB_check_ra%.5f-dec%.5f.png' % (ra_x, dec_x), dpi = 300)
	plt.close()

