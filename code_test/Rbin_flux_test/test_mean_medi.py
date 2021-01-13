import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import scipy.interpolate as interp

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i']

load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'

##### the mean and median flux of total sample imgs (before resampling)
## cluster imgs
lo_dat = pds.read_csv(load + 'img_cat/low_BCG_star-Mass_r-band_remain_cat.csv',)
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

hi_dat = pds.read_csv(load + 'img_cat/high_BCG_star-Mass_r-band_remain_cat.csv',)
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]

tmp_Npix, tmp_mf, tmp_mid_f = [], [], []
tot_pix_f = []

for kk in range( len(z) ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

	data = fits.open( home +
		'tmp_stack/cluster/cluster_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % (ra_g, dec_g, z_g),)
	img = data[0].data

	aveg_f = np.nanmean( img )
	mid_f = np.nanmedian( img )
	idnn = np.isnan( img )
	idvx = idnn == False

	tmp_Npix.append( np.sum( idvx ) )
	tmp_mf.append( aveg_f )
	tmp_mid_f.append( mid_f )

	tot_pix_f.append( img[ idvx ] )

tmp_Npix = np.array( tmp_Npix )
tmp_mf = np.array( tmp_mf )
tmp_mid_f = np.array( tmp_mid_f )

tot_pix_f = np.hstack( tot_pix_f )
tot_pix_median_f = np.median( tot_pix_f )
medi_arr = np.ones(len(z), dtype = np.float32) * tot_pix_median_f

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'N_eff', 'aveg_flux', 'medi_f', 'global_medi',]
values = [ ra, dec, z, clus_x, clus_y, tmp_Npix, tmp_mf, tmp_mid_f, medi_arr ]
fill = dict( zip(keys,values) )
bin_data = pds.DataFrame(fill)
bin_data.to_csv('/home/xkchen/cluster_r-band_tot_remain_img_aveg-flux.csv')

print( 'cluster img, tot_pix_median_f = ', tot_pix_median_f)
print('finished cluster imgs !')


## random imgs
dat = pds.read_csv(load + 'random_cat/12_21/random_r-band_tot_remain_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

tmp_Npix, tmp_mf, tmp_mid_f = [], [], []
tot_pix_f = []

for kk in range( len(z) ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

	data = fits.open(home + 'tmp_stack/random/random_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % (ra_g, dec_g, z_g),)
	img = data[0].data

	aveg_f = np.nanmean( img )
	mid_f = np.nanmedian( img )
	idnn = np.isnan( img )
	idvx = idnn == False

	tmp_Npix.append( np.sum( idvx ) )
	tmp_mf.append( aveg_f )
	tmp_mid_f.append( mid_f )

	tot_pix_f.append( img[ idvx ] )

tmp_Npix = np.array( tmp_Npix )
tmp_mf = np.array( tmp_mf )
tmp_mid_f = np.array( tmp_mid_f )

tot_pix_f = np.hstack( tot_pix_f )
tot_pix_median_f = np.median( tot_pix_f )
medi_arr = np.ones(len(z), dtype = np.float32) * tot_pix_median_f

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'N_eff', 'aveg_flux', 'medi_f', 'global_medi',]
values = [ ra, dec, z, clus_x, clus_y, tmp_Npix, tmp_mf, tmp_mid_f, medi_arr ]
fill = dict( zip(keys,values) )
bin_data = pds.DataFrame(fill)
bin_data.to_csv('/home/xkchen/random_r-band_tot_remain_img_aveg-flux.csv')

print( 'tot_pix_median_f = ', tot_pix_median_f )
print( 'finished random imgs !' )

raise

############ after resampling
## global mean and medain
lo_dat = pds.read_csv(load + 'img_cat/low_BCG_star-Mass_r-band_remain_cat_resamp_BCG-pos.csv')
lo_ra, lo_dec, lo_z = np.array(lo_dat.ra), np.array(lo_dat.dec), np.array(lo_dat.z)
lo_imgx, lo_imgy = np.array(lo_dat.bcg_x), np.array(lo_dat.bcg_y)

hi_dat = pds.read_csv(load + 'img_cat/high_BCG_star-Mass_r-band_remain_cat_resamp_BCG-pos.csv')
hi_ra, hi_dec, hi_z = np.array(hi_dat.ra), np.array(hi_dat.dec), np.array(hi_dat.z)
hi_imgx, hi_imgy = np.array(hi_dat.bcg_x), np.array(hi_dat.bcg_y)

ra = np.r_[ lo_ra, hi_ra ]
dec = np.r_[ lo_dec, hi_dec ]
z = np.r_[ lo_z, hi_z ]
Ns = len(z)

clus_x = np.r_[ lo_imgx, hi_imgx ]
clus_y = np.r_[ lo_imgy, hi_imgy ]

print('N_sample = ', len(ra),)
print('band = r')

tmp_Npix, tmp_mf, tmp_mid_f = [], [], []
tot_pix_f = []

for kk in range( Ns ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

	data = fits.open( home + 'tmp_stack/pix_resample/resamp-r-ra%.3f-dec%.3f-redshift%.3f.fits' % (ra_g, dec_g, z_g),)
	img = data[0].data

	aveg_f = np.nanmean( img )
	mid_f = np.nanmedian( img )
	idnn = np.isnan( img )
	idvx = idnn == False

	tmp_Npix.append( np.sum( idvx ) )
	tmp_mf.append( aveg_f )
	tmp_mid_f.append( mid_f )

	tot_pix_f.append( img[ idvx ] )

tmp_Npix = np.array( tmp_Npix )
tmp_mf = np.array( tmp_mf )
tmp_mid_f = np.array( tmp_mid_f )

tot_pix_f = np.hstack( tot_pix_f )
tot_pix_median_f = np.median( tot_pix_f )
medi_arr = np.ones(len(z), dtype = np.float32) * tot_pix_median_f

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'N_eff', 'aveg_flux', 'medi_f', 'global_medi',]
values = [ ra, dec, z, clus_x, clus_y, tmp_Npix, tmp_mf, tmp_mid_f, medi_arr ]
fill = dict( zip(keys,values) )
bin_data = pds.DataFrame(fill)
bin_data.to_csv('/home/xkchen/cluster_r-band_tot_remain_img_resamp_aveg-flux.csv')

print('record sample img flux of cluster !')


dat = pds.read_csv( load + 'random_cat/12_21/random_r-band_tot_remain_zref_BCG-pos.csv' )
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
clus_x, clus_y = np.array(dat['bcg_x']), np.array(dat['bcg_y'])

Ns = len(ra)

tmp_Npix, tmp_mf, tmp_mid_f = [], [], []
tot_pix_f = []

for kk in range( Ns ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

	data = fits.open( home + 'tmp_stack/pix_resample/random_resamp-r-ra%.3f-dec%.3f-redshift%.3f.fits' % (ra_g, dec_g, z_g),)
	img = data[0].data

	aveg_f = np.nanmean( img )
	mid_f = np.nanmedian( img )
	idnn = np.isnan( img )
	idvx = idnn == False

	tmp_Npix.append( np.sum( idvx ) )
	tmp_mf.append( aveg_f )
	tmp_mid_f.append( mid_f )

	tot_pix_f.append( img[ idvx ] )

tmp_Npix = np.array( tmp_Npix )
tmp_mf = np.array( tmp_mf )
tmp_mid_f = np.array( tmp_mid_f )

tot_pix_f = np.hstack( tot_pix_f )
tot_pix_median_f = np.median( tot_pix_f )
medi_arr = np.ones(len(z), dtype = np.float32) * tot_pix_median_f

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'N_eff', 'aveg_flux', 'medi_f', 'global_medi',]
values = [ ra, dec, z, clus_x, clus_y, tmp_Npix, tmp_mf, tmp_mid_f, medi_arr ]
fill = dict( zip(keys,values) )
bin_data = pds.DataFrame(fill)
bin_data.to_csv('/home/xkchen/random_r-band_tot_remain_img_resamp_aveg-flux.csv')

print('record sample img flux of random !')

