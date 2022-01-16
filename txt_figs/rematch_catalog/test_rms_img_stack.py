import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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


### img future check
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

d_file = home + 'photo_files/pos_offset_correct_imgs/resamp_img/photo-z_resamp_%s_ra%.3f_dec%.3f_z%.3f.fits'


### === ### rms imags calculation
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

id_cen = 0
band_str = band[ 0 ]


## except images
if band_str == 'r':
	out_ra = [ '164.740', '141.265', ]
	out_dec = [ '11.637', '11.376',  ]
	out_z = [ '0.298', '0.288',      ]

if band_str == 'g':
	out_ra = [ '206.511', '141.265', '236.438', ]
	out_dec = [ '38.731', '11.376', '1.767', ]
	out_z = [ '0.295', '0.288', '0.272', ]


##... extra ~450 images
"""
#. BCG position at z_ref
ref_dat = pds.read_csv( load + 'Extend_Mbcg_cat/' + 
						'high_BCG_star-Mass_%s-band_photo-z-match_pk-offset_cat_z-ref.csv' % band_str )

ref_ra_0, ref_dec_0, ref_z_0 = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] ), np.array( ref_dat['z'] )
ref_bcgx_0, ref_bcgy_0 = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )


ref_dat = pds.read_csv( load + 'Extend_Mbcg_cat/' + 
						'low_BCG_star-Mass_%s-band_photo-z-match_pk-offset_cat_z-ref.csv' % band_str )

ref_ra_1, ref_dec_1, ref_z_1 = np.array( ref_dat['ra'] ), np.array( ref_dat['dec'] ), np.array( ref_dat['z'] )
ref_bcgx_1, ref_bcgy_1 = np.array( ref_dat['bcg_x'] ), np.array( ref_dat['bcg_y'] )


ref_ra = np.r_[ ref_ra_0, ref_ra_1 ]
ref_dec = np.r_[ ref_dec_0, ref_dec_1 ]
ref_z = np.r_[ ref_z_0, ref_z_1 ]
ref_bcgx = np.r_[ ref_bcgx_0, ref_bcgx_1 ]
ref_bcgy = np.r_[ ref_bcgy_0, ref_bcgy_1 ]

ref_coord = SkyCoord( ra = ref_ra * U.deg, dec = ref_dec * U.deg,)


#. different catalog
dat = pds.read_csv('/home/xkchen/gri_diff_cat.csv')
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )

sub_coord = SkyCoord( ra = ra * U.deg, dec = dec * U.deg,)

idx, sep, d3d = sub_coord.match_to_catalog_sky( ref_coord )
id_lim = sep.value < 2.7e-4

print('matched Ng = ', np.sum(id_lim) )

mp_ra, mp_dec, mp_z = ra[ id_lim ], dec[ id_lim ], z[ id_lim ]
mp_bcg_x, mp_bcg_y = ref_bcgx[ idx[ id_lim ] ], ref_bcgy[ idx[ id_lim ] ]

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
values = [ mp_ra, mp_dec, mp_z, mp_bcg_x, mp_bcg_y ]
fill = dict( zip( keys, values ) )
data = pds.DataFrame( fill )
data.to_csv( '/home/xkchen/rgi-differ-cat_%s-band_pk-offset_BCG-pos_z-ref.csv' % band_str,)
"""

"""
dat = pds.read_csv('/home/xkchen/rgi-differ-cat_%s-band_pk-offset_BCG-pos_z-ref.csv' % band_str,)
ra, dec, z = np.array( dat['ra'] ), np.array( dat['dec'] ), np.array( dat['z'] )
bcg_x, bcg_y = np.array( dat['bcg_x'] ), np.array( dat['bcg_y'] )

Ns = len( ra )
id_lis = np.arange( Ns ).astype( int )

m, n = divmod( Ns, cpus )
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

mp_z, mp_ra, mp_dec = z[N_sub0 : N_sub1], ra[N_sub0 : N_sub1], dec[N_sub0 : N_sub1]
mp_bcg_x, mp_bcg_y = bcg_x[N_sub0 : N_sub1], bcg_y[N_sub0 : N_sub1]

out_file = '/home/xkchen/figs/rgi-differ-cat_%s-band_stack_test_img_%d-rank.h5' % (band_str, rank)
out_cont = '/home/xkchen/figs/rgi-differ-cat_%s-band_stack_test_pix-cont_%d-rank.h5' % (band_str, rank)
out_rms = '/home/xkchen/figs/rgi-differ-cat_%s-band_stack_test_rms_%d-rank.h5' % (band_str, rank)

cut_stack_func(	d_file, out_file, mp_z, mp_ra, mp_dec, band_str, mp_bcg_x, mp_bcg_y, id_cen, N_edg = 1, rms_file = out_rms, 
				pix_con_file = out_cont, id_mean = 0 )

"""


# pat = np.loadtxt('/home/xkchen/extra-500_sub-10.txt')
# p_ra, p_dec, p_z = pat[0], pat[1], pat[2]

if rank == 0:
	pat = np.loadtxt('/home/xkchen/sub-28_half_10.txt')

if rank == 1:
	pat = np.loadtxt('/home/xkchen/sub-28_half_11.txt')

p_ra, p_dec, p_z = pat[0], pat[1], pat[2]

p_ra_lis = ['%.3f' % pp for pp in p_ra ]
p_dec_lis = ['%.3f' % pp for pp in p_dec ]
p_z_lis = ['%.3f' % pp for pp in p_z ]


out_ra = out_ra + p_ra_lis
out_dec = out_dec + p_dec_lis
out_z = out_z + p_z_lis


"""
### === subsamples image
for ll in range( 2 ):

	dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	if band_str != 'i':
		ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[ll], band_str)
		order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

		ra, dec, z = ra[order_lis], dec[order_lis], z[order_lis]
		clus_x, clus_y = clus_x[order_lis], clus_y[order_lis]

	print('N_sample = ', len(ra),)
	print('band = %s' % band_str,)

	out_file = '/home/xkchen/figs/photo-z_match_%s_%s-band_stack_test_img.h5' % ( cat_lis[ll], band_str )
	out_cont = '/home/xkchen/figs/photo-z_match_%s_%s-band_stack_test_pix-cont.h5' % ( cat_lis[ll], band_str )
	out_rms = '/home/xkchen/figs/photo-z_match_%s_%s-band_stack_test_rms.h5' % ( cat_lis[ll], band_str )
	cut_stack_func(	d_file, out_file, z, ra, dec, band_str, clus_x, clus_y, id_cen, N_edg = 1, rms_file = out_rms, 
					pix_con_file = out_cont, id_mean = 0 )

print('subsamples!')
"""


### === entire cluster sample
cat_lis = [ 'low_BCG_star-Mass', 'high_BCG_star-Mass']

lo_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[0], band_str),)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

print( len(lo_ra) )

if band_str != 'i':
	ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[0], band_str)
	order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

	lo_ra, lo_dec, lo_z = lo_ra[order_lis], lo_dec[order_lis], lo_z[order_lis]
	lo_imgx, lo_imgy = lo_imgx[order_lis], lo_imgy[order_lis]

print( len(lo_ra) )


hi_dat = pds.read_csv( load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[1], band_str),)
hi_ra, hi_dec, hi_z = np.array( hi_dat.ra ), np.array( hi_dat.dec ), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array( hi_dat.bcg_x ), np.array( hi_dat.bcg_y )

print( len(hi_ra) )

if band_str != 'i':
	ref_file = load + 'Extend_Mbcg_cat/%s_%s-band_photo-z-match_rgi-common_pk-offset_BCG-pos_z-ref.csv' % (cat_lis[1], band_str)
	order_lis = simple_match( out_ra, out_dec, out_z, ref_file,)

	hi_ra, hi_dec, hi_z = hi_ra[order_lis], hi_dec[order_lis], hi_z[order_lis]
	hi_imgx, hi_imgy = hi_imgx[order_lis], hi_imgy[order_lis]

print( len(hi_ra) )


ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

imgx = np.r_[ lo_imgx, hi_imgx ]
imgy = np.r_[ lo_imgy, hi_imgy ]


# out_file = '/home/xkchen/figs/photo-z_match_tot-BCG-star-Mass_%s-band_stack_test_img.h5' % band_str
# out_cont = '/home/xkchen/figs/photo-z_match_tot-BCG-star-Mass_%s-band_stack_test_pix-cont.h5' % band_str
# out_rms = '/home/xkchen/figs/photo-z_match_tot-BCG-star-Mass_%s-band_stack_test_rms.h5' % band_str

# cut_stack_func(d_file, out_file, z, ra, dec, band_str, imgx, imgy, id_cen, N_edg = 1, rms_file = out_rms, 
# 					pix_con_file = out_cont, id_mean = 0 )

out_file = '/home/xkchen/figs/tot-BCG-star-Mass_%s-band_stack_test_img_no-half-1%d.h5' % (band_str, rank)
out_cont = '/home/xkchen/figs/tot-BCG-star-Mass_%s-band_stack_test_pix-cont_no-half-1%d.h5' % (band_str, rank)
out_rms = '/home/xkchen/figs/tot-BCG-star-Mass_%s-band_stack_test_rms_no-half-1%d.h5' % (band_str, rank)

cut_stack_func(	d_file, out_file, z, ra, dec, band_str, imgx, imgy, id_cen, N_edg = 1, rms_file = out_rms, 
				pix_con_file = out_cont, id_mean = 0 )


raise


##... 28th subsamples
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
	lis_x.append( imgx[ id_xbin ] )
	lis_y.append( imgy[ id_xbin ] )

tp_ra, tp_dec, tp_z = np.array([]), np.array([]), np.array([])
tp_imgx, tp_imgy = np.array([]), np.array([])

for pp in range( N_bin ):

	if pp == 28:

		# continue

		tp_ra = np.r_[ tp_ra, lis_ra[ pp ] ]
		tp_dec = np.r_[ tp_dec, lis_dec[ pp ] ]
		tp_z = np.r_[ tp_z, lis_z[ pp ] ]
		tp_imgx = np.r_[ tp_imgx, lis_x[ pp ] ]
		tp_imgy = np.r_[ tp_imgy, lis_y[ pp ] ]

	else:

		# tp_ra = np.r_[ tp_ra, lis_ra[ pp ] ]
		# tp_dec = np.r_[ tp_dec, lis_dec[ pp ] ]
		# tp_z = np.r_[ tp_z, lis_z[ pp ] ]
		# tp_imgx = np.r_[ tp_imgx, lis_x[ pp ] ]
		# tp_imgy = np.r_[ tp_imgy, lis_y[ pp ] ]

		continue

Nt = np.int( len(tp_ra) / 2 )

# cc_ra_0, cc_dec_0, cc_z_0 = tp_ra[:Nt], tp_dec[:Nt], tp_z[:Nt]
# out_arr = np.array([ cc_ra_0, cc_dec_0, cc_z_0 ])
# np.savetxt('/home/xkchen/sub-28_half_0.txt', out_arr,)

# cc_ra_1, cc_dec_1, cc_z_1 = tp_ra[Nt:], tp_dec[Nt:], tp_z[Nt:]
# out_arr = np.array([ cc_ra_1, cc_dec_1, cc_z_1 ])
# np.savetxt('/home/xkchen/sub-28_half_1.txt', out_arr,)

if rank == 0:
	out_file = '/home/xkchen/figs/sub-28_%s-band_stack_test_img.h5' % band_str
	out_cont = '/home/xkchen/figs/sub-28_%s-band_stack_test_pix-cont.h5' % band_str
	out_rms = '/home/xkchen/figs/sub-28_%s-band_stack_test_rms.h5' % band_str
	cut_stack_func(	d_file, out_file, tp_z, tp_ra, tp_dec, band_str, tp_imgx, tp_imgy, id_cen, N_edg = 1, rms_file = out_rms, 
					pix_con_file = out_cont, id_mean = 0 )

if rank == 1:
	out_file = '/home/xkchen/figs/sub-28_%s-band_stack_test_img_half-0.h5' % band_str
	out_cont = '/home/xkchen/figs/sub-28_%s-band_stack_test_pix-cont_half-0.h5' % band_str
	out_rms = '/home/xkchen/figs/sub-28_%s-band_stack_test_rms_half-0.h5' % band_str

	cut_stack_func(	d_file, out_file, tp_z[:Nt], tp_ra[:Nt], tp_dec[:Nt], band_str, tp_imgx[:Nt], tp_imgy[:Nt], id_cen, 
					N_edg = 1, rms_file = out_rms, pix_con_file = out_cont, id_mean = 0 )

if rank == 2:
	out_file = '/home/xkchen/figs/sub-28_%s-band_stack_test_img_half-1.h5' % band_str
	out_cont = '/home/xkchen/figs/sub-28_%s-band_stack_test_pix-cont_half-1.h5' % band_str
	out_rms = '/home/xkchen/figs/sub-28_%s-band_stack_test_rms_half-1.h5' % band_str

	cut_stack_func(	d_file, out_file, tp_z[Nt:], tp_ra[Nt:], tp_dec[Nt:], band_str, tp_imgx[Nt:], tp_imgy[Nt:], id_cen, 
					N_edg = 1, rms_file = out_rms, pix_con_file = out_cont, id_mean = 0 )

