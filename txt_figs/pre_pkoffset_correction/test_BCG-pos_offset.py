import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
import astropy.io.ascii as asc
import astropy.wcs as awc

from astropy.coordinates import SkyCoord
from scipy import interpolate as interp
from astropy import cosmology as apcy

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

#**********# gravity
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

band = ['r', 'g', 'i']
cat_lis = ['low_BCG_star-Mass', 'high_BCG_star-Mass']
# BCG position comparison
band_str = '%s' % np.array( band[N_sub0 : N_sub1] )[0]
#band_str = band[ kk ]

print('band is %s' % band_str )

gal_file = home + 'source_detect_cat/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
bcg_proti = home + 'BCG_photometric/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt'
img_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

for ll in range( 2 ):

	dat = pds.read_csv(load + 'img_cat/%s_%s-band_BCG-pos_cat.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	Ns = len( z )

	r_tmp_dR = []
	g_tmp_dR = []
	i_tmp_dR = []

	r_eff_R = []
	g_eff_R = []
	i_eff_R = []

	for mm in range( Ns ):

		ra_g, dec_g, z_g = ra[mm], dec[mm], z[mm]

		r_data = fits.open( img_file % ('r', ra_g, dec_g, z_g),)
		r_wcs = awc.WCS(r_data[0].header)
		r_cx, r_cy = r_wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		g_data = fits.open( img_file % ('g', ra_g, dec_g, z_g),)
		g_wcs = awc.WCS(g_data[0].header)
		g_cx, g_cy = g_wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		i_data = fits.open( img_file % ('i', ra_g, dec_g, z_g),)
		i_wcs = awc.WCS(i_data[0].header)
		i_cx, i_cy = i_wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		source_lis = asc.read( gal_file % (band_str, ra_g, dec_g, z_g),)
		cx = np.array(source_lis['X_IMAGE'])
		cy = np.array(source_lis['Y_IMAGE'])

		if band_str == 'r':
			dR = np.sqrt( (cx - r_cx)**2 + (cy - r_cy)**2)
			idx = dR == dR.min()

		if band_str == 'g':
			dR = np.sqrt( (cx - g_cx)**2 + (cy - g_cy)**2)
			idx = dR == dR.min()

		if band_str == 'i':
			dR = np.sqrt( (cx - i_cx)**2 + (cy - i_cy)**2)
			idx = dR == dR.min()

		cen_cx, cen_cy = cx[idx][0], cy[idx][0]

		r_off_dr = np.sqrt( (cen_cx - r_cx)**2 + (cen_cy - r_cy)**2 )
		g_off_dr = np.sqrt( (cen_cx - g_cx)**2 + (cen_cy - g_cy)**2 )
		i_off_dr = np.sqrt( (cen_cx - i_cx)**2 + (cen_cy - i_cy)**2 )

		r_tmp_dR.append( r_off_dr )
		g_tmp_dR.append( g_off_dr )
		i_tmp_dR.append( i_off_dr )

		BCG_photo_cat = pds.read_csv( bcg_proti % (z_g, ra_g, dec_g), skiprows = 1)
		r_Reff = np.array(BCG_photo_cat['deVRad_r'])[0]
		g_Reff = np.array(BCG_photo_cat['deVRad_g'])[0]
		i_Reff = np.array(BCG_photo_cat['deVRad_i'])[0]
		r_Reff, g_Reff, i_Reff = r_Reff / pixel, g_Reff / pixel, i_Reff / pixel

		r_eff_R.append( r_Reff )
		g_eff_R.append( g_Reff )
		i_eff_R.append( i_Reff )

	r_tmp_dR = np.array( r_tmp_dR )
	g_tmp_dR = np.array( g_tmp_dR )
	i_tmp_dR = np.array( i_tmp_dR )

	r_eff_R = np.array( r_eff_R )
	g_eff_R = np.array( g_eff_R )
	i_eff_R = np.array( i_eff_R )

	keys = ['ra', 'dec', 'z', 'r_off_D', 'g_off_D', 'i_off_D', 'r_Reff', 'g_Reff', 'i_Reff']
	values = [ra, dec, z, r_tmp_dR, g_tmp_dR, i_tmp_dR, r_eff_R, g_eff_R, i_eff_R]
	fill = dict(zip( keys, values) )
	data = pds.DataFrame(fill)
	data.to_csv( '/home/xkchen/%s_%s-band_off-D_BCG-eff-R.csv' % (cat_lis[ll], band_str),)

	print( '%s band finished !' % band_str )

