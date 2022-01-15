import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy import ndimage
from reproject import reproject_exact

import astropy.io.ascii as asc
import subprocess as subpro

from img_jack_stack import jack_main_func
from light_measure import jack_SB_func
from light_measure import cc_grid_img, grid_img
from light_measure import light_measure_Z0_weit, light_measure_rn_Z0_weit

##### constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

##### cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i', 'u', 'z']
#########
def source_mask(img_file, gal_cat):

	data = fits.open(img_file)
	img = data[0].data

	source = asc.read(gal_cat)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])

	Kron = 16
	a = Kron * A
	b = Kron * B

	tot_cx = cx
	tot_cy = cy
	tot_a = a
	tot_b = b
	tot_theta = theta
	tot_Numb = Numb

	mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
	ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
	oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
	basic_coord = np.array(np.meshgrid(ox, oy))
	major = tot_a / 2
	minor = tot_b / 2
	senior = np.sqrt(major**2 - minor**2)

	for k in range(tot_Numb):
		xc = tot_cx[k]
		yc = tot_cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = tot_theta[k] * np.pi/180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

		df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
		df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
		fr = df1**2 / lr**2 + df2**2 / sr**2
		jx = fr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float32)
		iv[iu] = np.nan
		mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img

	return mask_img

def decals_sdss_match_func(set_ra, set_dec, set_z, decals_file, sdss_file, out_file):

	Ns = len(set_z)

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

		### decals imgs
		desi_data = fits.open( decals_file % (ra_g, dec_g, z_g),)
		Head_0 = desi_data[0].header
		desi_img = desi_data[0].data

		### sdss imgs
		sdss_data = fits.open( sdss_file % (ra_g, dec_g, z_g),)
		Head_1 = sdss_data[0].header

		relign_img = reproject_exact(desi_data, Head_1,)[0]

		### save the reproject imgs
		hdu = fits.PrimaryHDU()
		hdu.data = relign_img
		hdu.header = Head_1
		hdu.writeto( out_file % (ra_g, dec_g, z_g), overwrite = True)

	return

def alt_decals_sdss_match_func(set_ra, set_dec, set_z, decals_file, sdss_file, out_file):

	Ns = len(set_z)

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

		### decals imgs
		desi_data = fits.open( decals_file % (ra_g, dec_g, z_g),)
		Head_0 = desi_data[0].header
		desi_img = desi_data[0].data

		### sdss imgs
		sdss_data = fits.open( sdss_file % (ra_g, dec_g, z_g),)
		Head_1 = sdss_data[0].header

		relign_img = reproject_exact(desi_data[1], Head_1,)[0]
		### flux unit conversion
		relign_img = relign_img * 10**(-3)

		### save the reproject imgs
		hdu = fits.PrimaryHDU()
		hdu.data = relign_img
		hdu.header = Head_1
		hdu.writeto( out_file % (ra_g, dec_g, z_g), overwrite = True)

	return

def sdss_mask_func(set_ra, set_dec, set_z, decals_file, sdss_mask_file, out_file,):

	Ns = len(set_z)

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

		### decals imgs (reprojected)
		desi_data = fits.open( decals_file % (ra_g, dec_g, z_g),)
		Head_0 = desi_data[0].header
		desi_img = desi_data[0].data

		### sdss mask array
		mask_data = fits.open( sdss_mask_file % (ra_g, dec_g, z_g),)
		mask_img = mask_data[0].data
		Head_1 = mask_data[0].header
		id_nan = np.isnan(mask_img)

		desi_mask_img = desi_img.copy()
		desi_mask_img[id_nan] = np.nan

		### save the masked imgs
		hdu = fits.PrimaryHDU()
		hdu.data = desi_mask_img
		hdu.header = Head_1
		hdu.writeto( out_file % (ra_g, dec_g, z_g), overwrite = True)

	return

def combine_mask_func(set_ra, set_dec, set_z, decals_file, sdss_mask_file, out_source_file, out_file,):

	Ns = len(set_z)

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]
		### decals imgs (reprojected)
		desi_data = fits.open( decals_file % (ra_g, dec_g, z_g),)
		Head_0 = desi_data[0].header
		desi_img = desi_data[0].data

		### sdss mask array
		mask_data = fits.open( sdss_mask_file % (ra_g, dec_g, z_g),)
		mask_img = mask_data[0].data
		Head_1 = mask_data[0].header

		param_A = 'default_mask_A.sex'
		out_cat = 'default_mask_A.param'

		out_load_A = out_source_file % (ra_g, dec_g, z_g)
		file_source = decals_file % (ra_g, dec_g, z_g)

		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_load_A, out_cat)
		a = subpro.Popen(cmd, shell = True)
		a.wait()

		remain_img = source_mask(file_source, out_load_A)
		id_nan = np.isnan(mask_img)
		remain_img[id_nan] = np.nan

		### save the masked imgs
		hdu = fits.PrimaryHDU()
		hdu.data = remain_img
		hdu.header = Head_1
		hdu.writeto( out_file % (ra_g, dec_g, z_g), overwrite = True)

	return

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

##### part 2
dat = pds.read_csv('/home/xkchen/fig_tmp/AB_bass_stacked_block_match.csv')
ra, dec, z = np.array(dat['bcg_ra']), np.array(dat['bcg_dec']), np.array(dat['bcg_z'])

zN = len( z )
m, n = divmod(zN, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

### sdss region match
decals_file = '/home/xkchen/fig_tmp/BASS/clust-mask_bass_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_file = '/home/xkchen/data/SDSS/wget_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

out_file = '/home/xkchen/fig_tmp/BASS_cut/New-bass-cut_r_ra%.3f_dec%.3f_z%.3f.fits'

decals_sdss_match_func(ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1], z[N_sub0 :N_sub1], decals_file, sdss_file, out_file,)

commd.Barrier()

print('finished!')
