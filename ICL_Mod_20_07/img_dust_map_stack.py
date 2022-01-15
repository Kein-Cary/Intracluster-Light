"""
SFD 1998, with the correction of Schlafly and Finkbeiner, 2011
"""
import h5py
import numpy as np
import pandas as pds

import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import subprocess as subpro
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord

#. dust map with the recalibration by Schlafly & Finkbeiner (2011)
import sfdmap
E_map = sfdmap.SFDMap('/home/xkchen/module/dust_map/sfddata_maskin')

#. dust map of SFD 1998
# from dustmaps.sfd import SFDQuery
# sfd = SFDQuery()

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

#.pipe process
from img_resample import resamp_func
from BCG_SB_pro_stack import BCG_SB_pros_func, single_img_SB_func
from fig_out_module import arr_jack_func
from light_measure import light_measure_weit
from img_jack_stack import jack_main_func, zref_lim_SB_adjust_func
from img_jack_stack import SB_pros_func
from img_jack_stack import aveg_stack_img
from img_stack import stack_func
from img_edg_cut_stack import cut_stack_func

###=== func.s and constants

#. parameter for Galactic
Rv = 3.1

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.25

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

def single_map_func( band_str, ra_lis, dec_lis, z_lis, img_file, out_file ):

	l_x0 = np.linspace(0, 2047, 2048)
	l_y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array( np.meshgrid( l_x0, l_y0 ) )

	Ns = len( z_lis )

	for jj in range( Ns ):

		ra_g, dec_g, z_g = ra_lis[jj], dec_lis[jj], z_lis[jj]

		img_data = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)

		img_arr = img_data[0].data
		header = img_data[0].header
		wcs_lis = awc.WCS( header )

		ra_img, dec_img = wcs_lis.all_pix2world( img_grid[0,:], img_grid[1,:], 0 )
		pos_img = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')

		p_EBV = E_map.ebv( pos_img )
		A_v = Rv * p_EBV
		A_l = A_wave( l_wave[ kk ], Rv) * A_v

		## save the extinction correct data
		hdu = fits.PrimaryHDU()
		hdu.data = A_l
		hdu.header = header
		hdu.writeto( out_file % ( band_str, ra_g, dec_g, z_g), overwrite = True,)

	return

###===### catalog read 
import matplotlib as plt
import matplotlib.pyplot as plt

home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'
out_path = '/home/xkchen/data/SDSS/dust_map/'


## fixed BCG Mstar samples
# cat_lis = [ 'low-rich', 'hi-rich' ]
# dat_file = load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_BCG-pos_cat.csv'

cat_lis = [ 'low-age', 'hi-age' ]
dat_file = load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_BCG-pos_cat.csv'

## fixed richness samples
# cat_lis = [ 'younger', 'older' ]
# dat_file = load + 'z_formed_cat/%s_%s-band_photo-z-match_BCG-pos_cat.csv'

# cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']
# dat_file = load + 'photo_z_cat/%s_%s-band_photo-z-match_BCG-pos_cat.csv'


#. build dust map of single images
for kk in range( 3 ):

	band_info = band[ kk ]

	for ll in range( 2 ):

		d_cat = pds.read_csv( dat_file % ( cat_lis[ll], band_info ),)
		ra, dec, z = np.array( d_cat['ra']), np.array( d_cat['dec']), np.array( d_cat['z'])

		N_clus = len( z )

		m, n = divmod( N_clus, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		set_z, set_ra, set_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]

		img_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
		out_file = home + 'dust_map/map_imgs/dust_map-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
		single_map_func( band_info, set_ra, set_dec, set_z, img_file, out_file )

