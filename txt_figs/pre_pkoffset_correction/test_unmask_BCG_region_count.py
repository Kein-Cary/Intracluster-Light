import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from scipy import optimize
from scipy import ndimage

from img_random_SB_fit import random_SB_fit_func, clust_SB_fit_func, cc_rand_sb_func
from img_BG_sub_SB_measure import BG_sub_sb_func

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
band = ['r', 'g', 'i']
psf_FWHM = [ 1.56, 1.67, 1.50 ] # arcsec

### === ### photo-z sample
home = '/home/xkchen/data/SDSS/'
load = '/home/xkchen/fig_tmp/'

origin_files = home + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
mask_files = home + 'photo_files/mask_imgs/photo-z_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
resamp_files = home + 'photo_files/resample_imgs/photo-z_resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
source_cat = home + 'photo_files/detect_source_cat/photo-z_img_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'

cat_lis = [ 'low-age', 'hi-age' ]
fig_name = [ 'younger', 'older' ]

band_str = band[ rank ]

for ll in range( 2 ):

	dat = pds.read_csv( load + 'bcg_M_simi_cat/%s_%s-band_photo-z-match_rgi-common_BCG-pos_cat.csv' % (cat_lis[ll], band_str),)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Nz = len( z )

	set_ra, set_dec, set_z = [], [], []
	set_x, set_y = [], []

	for tt in range( Nz ):

		ra_g, dec_g, z_g = ra[tt], dec[tt], z[tt]
		cen_x, cen_y = clus_x[tt], clus_y[tt]

		data_o = fits.open( origin_files % (band_str, ra_g, dec_g, z_g),)
		img_o = data_o[0].data

		data_m = fits.open( mask_files % (band_str, ra_g, dec_g, z_g),)
		img_m = data_m[0].data

		Da_z = Test_model.angular_diameter_distance( z_g ).value
		L_pix = Da_z * 10**3 * pixel / rad2asec
		R1Mpc = 1e3 / L_pix
		R200kpc = 2e2 / L_pix

		## check BCG region
		xn, yn = np.int(cen_x), np.int(cen_y)
		cen_region = img_m[ yn - np.int(100 / L_pix): yn + np.int(100 / L_pix) + 1, xn - np.int(100 / L_pix): xn + np.int(100 / L_pix) + 1]
		N_pix = cen_region.shape[0] * cen_region.shape[1]
		idnn = np.isnan( cen_region )

		eta = np.sum( idnn ) / N_pix

		if eta >= 0.30: 
			continue

		else:

			set_ra.append( ra_g )
			set_dec.append( dec_g )
			set_z.append( z_g )

			set_x.append( cen_x )
			set_y.append( cen_y )

	set_ra, set_dec, set_z = np.array( set_ra ), np.array( set_dec ), np.array( set_z )
	set_x, set_y = np.array( set_x ), np.array( set_y )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ set_ra, set_dec, set_z, set_x, set_y ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame( fill )
	out_data.to_csv( '/home/xkchen/%s_%s-band_BCG_region_unmask_70.csv' % (cat_lis[ll], band_str),)

print( 'down !' )
