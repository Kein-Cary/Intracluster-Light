import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from scipy import optimize
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy.stats import binned_statistic as binned
from fig_out_module import cc_grid_img, grid_img

from img_stack import stack_func
from img_edg_cut_stack import cut_stack_func
from light_measure import light_measure_weit

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

##### constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()


### === ###
def simple_match(ra_lis, dec_lis, z_lis, ref_file, id_choose = False,):

	ref_dat = pds.read_csv( ref_file )
	tt_ra, tt_dec, tt_z = np.array(ref_dat.ra), np.array(ref_dat.dec), np.array(ref_dat.z)

	dd_ra, dd_dec, dd_z = [], [], []
	order_lis = []

	for kk in range( len(tt_z) ):
		identi = ('%.3f' % tt_ra[kk] in ra_lis) * ('%.3f' % tt_dec[kk] in dec_lis) # * ('%.3f' % tt_z[kk] in z_lis)

		if id_choose == True:
			if identi == True:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

			else:
				continue
		else:
			if identi == True:
				continue
			else:
				dd_ra.append( tt_ra[kk])
				dd_dec.append( tt_dec[kk])
				dd_z.append( tt_z[kk])
				order_lis.append( kk )

	dd_ra = np.array( dd_ra)
	dd_dec = np.array( dd_dec)
	dd_z = np.array( dd_z)
	order_lis = np.array( order_lis )

	return order_lis


### === ### data load
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'


d_file = home + 'photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'
mask_file = home + 'photo_files/pos_offset_correct_imgs/mask_img/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
orin_file = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'


##... catalog
band_str = band[0]

##... except catalog
if band_str == 'r':
	out_ra = [ '164.740', '141.265', ]
	out_dec = [ '11.637', '11.376', ]
	out_z = [ '0.298', '0.288', ]

if band_str == 'g':
	out_ra = [ '206.511', '141.265', '236.438', ]
	out_dec = [ '38.731', '11.376', '1.767', ]
	out_z = [ '0.295', '0.288', '0.272', ]

"""
##... sub-sample of over all stack-catalog
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

lo_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[0], band_str),)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

if band_str != 'i':
	ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[0], band_str)
	order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

	lo_ra, lo_dec, lo_z = lo_ra[order_lis], lo_dec[order_lis], lo_z[order_lis]
	lo_imgx, lo_imgy = lo_imgx[order_lis], lo_imgy[order_lis]

hi_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[1], band_str),)
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

if band_str != 'i':
	ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[1], band_str)
	order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

	hi_ra, hi_dec, hi_z = hi_ra[order_lis], hi_dec[order_lis], hi_z[order_lis]
	hi_imgx, hi_imgy = hi_imgx[order_lis], hi_imgy[order_lis]

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]


### subsample division
N_bin = 30

zN = len( z )
id_arr = np.arange( zN )

id_group = id_arr % N_bin

lis_ra, lis_dec, lis_z = [], [], []
lis_x, lis_y = [], []

for nn in range( N_bin ):

	id_xbin = np.where( id_group == nn )[0]

	lis_ra.append( ra[ id_xbin ] )
	lis_dec.append( dec[ id_xbin ] )
	lis_z.append( z[ id_xbin ] )
	lis_x.append( clus_x[ id_xbin ] )
	lis_y.append( clus_y[ id_xbin ] )

for kk in range( 28, 29 ):

	sub_ra, sub_dec, sub_z = lis_ra[ kk ], lis_dec[ kk ], lis_z[ kk ] 
	
	Ns = len( sub_ra )
	print( Ns )

"""

##.. extra-500 catalog test
pat = np.loadtxt('/home/xkchen/extra-500_sub-10.txt')
sub_ra, sub_dec, sub_z = pat[0], pat[1], pat[2]
Ns = len( sub_ra )

for kk in range( 1 ):

	for ll in range( Ns ):

		ra_g, dec_g, z_g = sub_ra[ ll ], sub_dec[ ll ], sub_z[ ll ]

		data = fits.open( mask_file % (band_str, ra_g, dec_g, z_g), )
		mask_img = data[0].data
		Header = data[0].header

		wcs_lis = awc.WCS( Header )
		cx, cy = wcs_lis.all_world2pix( ra_g, dec_g, 0 )

		ori_data = fits.open( orin_file % (band_str, ra_g, dec_g, z_g),)
		ori_img = ori_data[0].data

		#.
		dx = cx - 0
		dy = cy - 0
		dR = np.sqrt( dx**2 + dy**2 )

		if dR > 350:

			fig = plt.figure( figsize = (13.12, 4.8) )
			ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
			ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

			ax0.set_title('cluster ra%.3f dec%.3f z%.3f' % (ra_g, dec_g, z_g),)
			ax0.plot( cx, cy, 'ro', alpha = 0.5, markersize = 2,)
			ax0.imshow( ori_img, origin = 'lower', cmap = 'Greys', vmin = -1e-3, vmax = 1e0, 
						norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)

			dl0 = 250
			dl1 = 450
			Box = Rectangle( xy = (cx - dl0, cy - dl0), width = dl0, height = dl0, fill = False, 
								ec = 'r', ls = '--', linewidth = 1, alpha = 0.75)
			ax0.add_patch( Box )

			Box = Rectangle( xy = (cx - dl1, cy - dl1), width = dl1, height = dl1, fill = False, 
								ec = 'r', ls = '-', linewidth = 1, alpha = 0.75)
			ax0.add_patch( Box )


			ax1.imshow( mask_img, origin = 'lower', cmap = 'Greys', vmin = -1e-3, vmax = 1e0, 
						norm = mpl.colors.SymLogNorm( linthresh = 0.001, linscale = 0.1, base = 10),)
			ax1.plot( cx, cy, 'ro', alpha = 0.5, markersize = 2,)

			Box = Rectangle( xy = (cx - dl0, cy - dl0), width = dl0, height = dl0, fill = False, 
								ec = 'r', ls = '--', linewidth = 1, alpha = 0.75)
			ax1.add_patch( Box )

			Box = Rectangle( xy = (cx - dl1, cy - dl1), width = dl1, height = dl1, fill = False, 
								ec = 'r', ls = '--', linewidth = 1, alpha = 0.75)
			ax1.add_patch( Box )

			plt.savefig('/home/xkchen/figs/clus_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
			plt.close()

		else:
			continue


