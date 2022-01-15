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
		cen_x = np.int( Head_0['CRPIX1'] )
		cen_y = np.int( Head_0['CRPIX2'] )
		coord_0 = awc.WCS(Head_0)

		### sdss imgs
		sdss_data = fits.open( sdss_file % (ra_g, dec_g, z_g),)
		Head_1 = sdss_data[0].header
		sdss_img = sdss_data[0].data
		CPx = np.int( Head_1['CRPIX1'] )
		CPy = np.int( Head_1['CRPIX2'] )
		coord_1 = awc.WCS(Head_1)

		relign_img = reproject_exact(desi_data, Head_1,)[0]

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

dat = pds.read_csv('/home/xkchen/mywork/ICL/code/A_250_img_cat.csv')
ra, dec, z = np.array(dat.bcg_ra), np.array(dat.bcg_dec), np.array(dat.bcg_z)
ref_ra, ref_dec = np.array(dat.ref_ra), np.array(dat.ref_dec)
Ns = len(z)

home = '/media/xkchen/My Passport/data/'
load = '/home/xkchen/mywork/ICL/data/'
'''
### sdss region match
decals_file = home + 'BASS/A_250/desi_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_file = load + 'sdss_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
out_file = home + 'BASS/A_250_to_SDSS/decals_cut_r_ra%.3f_dec%.3f_z%.3f.fits'

decals_sdss_match_func(ra, dec, z, decals_file, sdss_file, out_file)
'''
print('to here')
'''
### apply sdss-like mask
decals_file = home + 'BASS/A_250_to_SDSS/decals_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_mask_file = home + 'SDSS/tmp_stack/cluster/cluster_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
out_file = home + 'BASS/A_250_mask/ap_sdss_mask_r_ra%.3f_dec%.3f_z%.3f.fits'

sdss_mask_func(ra, dec, z, decals_file, sdss_mask_file, out_file,)
'''
print('part 1 finished!')
'''
### combine mask (sdss-like + source detection)
decals_file = load + 'tmp_img/A_250_to_SDSS/decals_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_mask_file = home + 'SDSS/tmp_stack/cluster/cluster_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

out_source_file = load + 'source_find/decals_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
out_file = home + 'BASS/A_250_mask/comb_mask_r_ra%.3f_dec%.3f_z%.3f.fits'

combine_mask_func(ra, dec, z, decals_file, sdss_mask_file, out_source_file, out_file,)
'''
print('part 2 finished!')


from light_measure import cc_grid_img
load = '/media/xkchen/My Passport/data/SDSS/'

jack_SB_file = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_SB-pro_sdss-mask.h5'
jack_img = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_img_sdss-mask.h5'
jack_cont_arr = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_pix-cont_sdss-mask.h5'
'''
jack_SB_file = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test/jack_test/decals_A-250_BCG-stack_Mean_jack_pix-cont.h5'
'''
with h5py.File( jack_img, 'r') as f:
	tt_img = np.array(f['a'])
img_block, grd_pix = cc_grid_img(tt_img, 100, 100)[:2]

with h5py.File( jack_SB_file, 'r') as f:
	c_r_arr = np.array(f['r'])
	c_sb_arr = np.array(f['sb'])
	c_sb_err = np.array(f['sb_err'])

with h5py.File(load + '20_10_test/jack_test/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 'r') as f:
	com_r = np.array(f['r'])
	com_sb = np.array(f['sb'])
	com_sb_err = np.array(f['sb_err'])

fig = plt.figure( figsize = (13.12, 4.8) )
ax1 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
ax2 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

#tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -1e-2, vmax = 1e-2,)
cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01,)
cb.formatter.set_powerlimits((0,0))

ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = 'DECaLS',)
ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

ax2.plot(com_r, com_sb, ls = '-', color = 'b', alpha = 0.8, label = 'SDSS',)
ax2.fill_between(com_r, y1 = com_sb - com_sb_err, y2 = com_sb + com_sb_err, color = 'b', alpha = 0.2,)

ax2.set_ylim(1e-4, 3e-2)
ax2.set_yscale('log')
#ax2.set_ylim(-2e-3, 7e-3)

ax2.set_xlim(1e1, 1e3)
ax2.set_xlabel('R [arcsec]')
ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax2.set_xscale('log')
ax2.legend(loc = 1, frameon = False, fontsize = 8)
ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
ax2.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('decals_grid_2D_SB.png', dpi = 300)
plt.close()
