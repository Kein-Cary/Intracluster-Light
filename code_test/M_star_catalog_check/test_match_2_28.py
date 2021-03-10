import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from scipy import signal
from scipy import interpolate as interp
from astropy.coordinates import SkyCoord

from img_pre_selection import cat_match_func
from fig_out_module import cc_grid_img, grid_img
from light_measure import jack_SB_func
from fig_out_module import zref_BCG_pos_func
from BCG_SB_pro_stack import BCG_SB_pros_func

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
# constant
rad2asec = U.rad.to(U.arcsec)
pixel = 0.396
band = ['r', 'g', 'i',]
mag_add = np.array([0, 0, 0])

z_ref = 0.25

#****************************#
load = '/home/xkchen/mywork/ICL/data/'

lo_dat = pds.read_csv(load + 'BCG_stellar_mass_cat/low_star-Mass_cat.csv')
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_rich, lo_M_star = np.array(lo_dat.rich), np.array(lo_dat.lg_Mass)

C_lo = SkyCoord(ra = lo_ra * U.degree, dec = lo_dec * U.degree)

hi_dat = pds.read_csv(load + 'BCG_stellar_mass_cat/high_star-Mass_cat.csv')
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_rich, hi_M_star = np.array(hi_dat.rich), np.array(hi_dat.lg_Mass)

C_hi = SkyCoord(ra = hi_ra * U.degree, dec = hi_dec * U.degree)

### match to final selected imgs
for ll in range( 3 ):

	dat = pds.read_csv( load + 
		'cat_select/match_2_28/cluster_%s-band_tot_remain_cat_set_200-grid_6.0-sigma.csv' % band[ll],)
	ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
	clus_x, clus_y = np.array(dat['bcg_x']), np.array(dat['bcg_y'])

	'''
	com_cat = SkyCoord(ra = ra * U.degree, dec = dec * U.degree)

	print( 'band, %s' % band[ll] )

	idx, d2d, d3d = C_lo.match_to_catalog_sky( com_cat )
	
	idmx = d2d.value <= 1e-3
	print( len(ra[idx][idmx] ) )

	idx, d2d, d3d = C_hi.match_to_catalog_sky( com_cat )

	idmx = d2d.value <= 1e-3
	print( len(ra[idx][idmx] ) )
	'''


	sf_len = 3
	f2str = '%.' + '%df' % sf_len

	## low mass sample
	out_ra = [ f2str % ll for ll in lo_ra]
	out_dec = [ f2str % ll for ll in lo_dec]
	out_z = [ f2str % ll for ll in lo_z]

	lis_ra, lis_dec, lis_z, lis_x, lis_y = cat_match_func( out_ra, out_dec, out_z, ra, dec, z, clus_x, clus_y, sf_len, id_choice = True,)

	print( 'band, %s' % band[ll] )
	print( len(lis_z) )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y',]
	values = [lis_ra, lis_dec, lis_z, lis_x, lis_y]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( 'low_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll],)

	## high mass sample
	out_ra_1 = [ f2str % ll for ll in hi_ra]
	out_dec_1 = [ f2str % ll for ll in hi_dec]
	out_z_1 = [ f2str % ll for ll in hi_z]

	lis_ra_1, lis_dec_1, lis_z_1, lis_x_1, lis_y_1 = cat_match_func( out_ra_1, out_dec_1, out_z_1, ra, dec, z, clus_x, clus_y, sf_len, id_choice = True,)

	print( len(lis_z_1) )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y',]
	values = [ lis_ra_1, lis_dec_1, lis_z_1, lis_x_1, lis_y_1 ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)

	data.to_csv( 'high_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll],)


	cat_file = 'low_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll]
	out_file = 'low_BCG_star-Mass_%s-band_BCG-pos_cat_z-ref.csv' % band[ll]
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)

	cat_file = 'high_BCG_star-Mass_%s-band_BCG-pos_cat.csv' % band[ll]
	out_file = 'high_BCG_star-Mass_%s-band_BCG-pos_cat_z-ref.csv' % band[ll]
	zref_BCG_pos_func(cat_file, z_ref, out_file, pixel)



