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

home = '/media/xkchen/My Passport/data/'
load = '/home/xkchen/mywork/ICL/data/'

### part 1 : DECaLS imgs
'''
dat = pds.read_csv(load + 'BASS_cat/BASS_test/A_250_img_cat.csv')
ra, dec, z = np.array(dat.bcg_ra), np.array(dat.bcg_dec), np.array(dat.bcg_z)

# sdss region match
decals_file = home + 'BASS/A_250/desi_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_file = load + 'sdss_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
out_file = home + 'BASS/A_250_to_SDSS/decals_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
decals_sdss_match_func(ra, dec, z, decals_file, sdss_file, out_file)

# apply sdss-like mask
decals_file = home + 'BASS/A_250_to_SDSS/decals_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_mask_file = home + 'SDSS/tmp_stack/cluster/cluster_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
out_file = home + 'BASS/A_250_mask/ap_sdss_mask_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_mask_func(ra, dec, z, decals_file, sdss_mask_file, out_file,)

# combine mask (sdss-like + source detection)
decals_file = load + 'tmp_img/A_250_to_SDSS/decals_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
sdss_mask_file = home + 'SDSS/tmp_stack/cluster/cluster_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

out_source_file = load + 'source_find/decals_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
out_file = home + 'BASS/A_250_mask/comb_mask_r_ra%.3f_dec%.3f_z%.3f.fits'
combine_mask_func(ra, dec, z, decals_file, sdss_mask_file, out_source_file, out_file,)
'''

##### part 2 (BASS imgs)
dat = pds.read_csv(load + 'BASS_cat/BASS_test/AB_bass_stacked_block_match.csv')
ra, dec, z = np.array(dat['bcg_ra']), np.array(dat['bcg_dec']), np.array(dat['bcg_z'])
'''
# sdss region match
#decals_file = load + 'tmp_img/AB_BASS_sdss-like_block/bass_r_ra%.3f_dec%.3f_z%.3f.fits'
decals_file = load + 'tmp_img/AB_BASS_block/BASS_r_ra%.3f_dec%.3f_z%.3f.fits.fz'

sdss_file = load + 'sdss_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

#out_file = load + 'tmp_img/AB_BASS_to_SDSS/bass_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
#decals_sdss_match_func(ra, dec, z, decals_file, sdss_file, out_file,)

out_file = load + 'tmp_img/AB_BASS_to_SDSS/Pub_bass_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
alt_decals_sdss_match_func(ra, dec, z, decals_file, sdss_file, out_file,)
'''

'''
### apply sdss-like mask
#decals_file = load + 'tmp_img/AB_BASS_to_SDSS/bass_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
#decals_file = load + 'tmp_img/AB_BASS_to_SDSS/Pub_bass_cut_r_ra%.3f_dec%.3f_z%.3f.fits'
decals_file = load + 'tmp_img/AB_BASS_to_SDSS/New-bass-cut_r_ra%.3f_dec%.3f_z%.3f.fits'

sdss_mask_file = home + 'SDSS/tmp_stack/cluster/cluster_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

#out_file = load + 'tmp_img/AB_BASS_mask/ap_sdss_mask_r_ra%.3f_dec%.3f_z%.3f.fits'
#out_file = load + 'tmp_img/AB_BASS_mask/ap_sdss_mask_Pub-bass_r_ra%.3f_dec%.3f_z%.3f.fits'
out_file = load + 'tmp_img/AB_BASS_mask/ap_sdss_mask_New-bass_r_ra%.3f_dec%.3f_z%.3f.fits'

sdss_mask_func(ra, dec, z, decals_file, sdss_mask_file, out_file,)
'''

#*****************************#
#### stacking imgs
home = '/media/xkchen/My Passport/data/'
load = '/media/xkchen/My Passport/data/SDSS/'

dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-250_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

sub_img = load + '20_10_test_jack/decals_A-250_BCG-stack_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test_jack/decals_A-250_BCG-stack_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test_jack/decals_A-250_BCG-stack_sub-%d_SB-pro.h5'

id_cen = 0
n_rbins = 110
N_bin = 30

'''
## also mask out those faint objs. in DECaLS imgs (DECaLS imgs)
d_file = '/media/xkchen/My Passport/data/BASS/A_250_mask/comb_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/decals_A-250_BCG-stack_jack-sub-%d_img.h5'
J_sub_pix_cont = load + '20_10_test_jack/decals_A-250_BCG-stack_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + '20_10_test_jack/decals_A-250_BCG-stack_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_img.h5'
jack_cont_arr = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_pix-cont.h5'
'''

'''
## just applied SDSS-like mask (DECaLS imgs)
d_file = '/media/xkchen/My Passport/data/BASS/A_250_mask/ap_sdss_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/decals_A-250_BCG-stack_jack-sub-%d_img_sdss-mask.h5'
J_sub_pix_cont = load + '20_10_test_jack/decals_A-250_BCG-stack_jack-sub-%d_pix-cont_sdss-mask.h5'
J_sub_sb = load + '20_10_test_jack/decals_A-250_BCG-stack_jack-sub-%d_SB-pro_sdss-mask.h5'

jack_SB_file = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_SB-pro_sdss-mask.h5'
jack_img = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_img_sdss-mask.h5'
jack_cont_arr = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_pix-cont_sdss-mask.h5'

jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

## use BASS coverage region img only ( DECaLS imgs, just applied SDSS-like mask)
idux = dec > 30
set_ra, set_dec, set_z = ra[idux], dec[idux], z[idux]
set_imgx, set_imgy = clus_x[idux], clus_y[idux]

'''
N_bin = 28

d_file = '/media/xkchen/My Passport/data/BASS/A_250_mask/ap_sdss_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/bass_A-250_match_jack-sub-%d_img_sdss-mask.h5'
J_sub_pix_cont = load + '20_10_test_jack/bass_A-250_match_jack-sub-%d_pix-cont_sdss-mask.h5'
J_sub_sb = load + '20_10_test_jack/bass_A-250_match_jack-sub-%d_SB-pro_sdss-mask.h5'

jack_SB_file = load + '20_10_test_jack/bass_A-250_match_Mean_jack_SB-pro_sdss-mask.h5'
jack_img = load + '20_10_test_jack/bass_A-250_match_Mean_jack_img_sdss-mask.h5'
jack_cont_arr = load + '20_10_test_jack/bass_A-250_match_Mean_jack_pix-cont_sdss-mask.h5'

jack_main_func(id_cen, N_bin, n_rbins, set_ra, set_dec, set_z, set_imgx, set_imgy, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

'''
## also stack the SDSS imgs for the same image sample
d_file = load + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

id_cen = 0
n_rbins = 110
N_bin = 21

J_sub_img = load + '20_10_test_jack/A250_jack-sub-%d_img_bass-region.h5'
J_sub_pix_cont = load + '20_10_test_jack/A250_jack-sub-%d_pix-cont_bass-region.h5'
J_sub_sb = load + '20_10_test_jack/A250_jack-sub-%d_SB-pro_bass-region.h5'

jack_SB_file = load + '20_10_test_jack/A250_Mean_jack_SB-pro_bass-region.h5'
jack_img = load + '20_10_test_jack/A250_Mean_jack_img_bass-region.h5'
jack_cont_arr = load + '20_10_test_jack/A250_Mean_jack_pix-cont_bass-region.h5'

jack_main_func(id_cen, N_bin, n_rbins, set_ra, set_dec, set_z, set_imgx, set_imgy, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

### BASS project imgs only (Zou et al.)
#......bad imgs
out_ra = ['116.883', '131.875', '182.190', '207.489', '208.942', 
		'132.835', '188.916', '194.616', '201.012', '209.715', 
		'219.728', '149.187',]

out_dec = ['33.761', '59.527',  '57.131',  '30.248',  '30.954', 
		'56.599', '41.652',  '65.359',  '55.779',  '33.910', 
		'53.601', '49.513',]

### Public BASS imgs
#......bad imgs
P_out_ra = ['116.883', '131.875', '182.190', '207.489', '208.942', 
		'132.835', '188.916', '194.616', '201.012', '209.715', 
		'219.728', '149.187', '152.225', '179.313', '182.963', 
		'205.126', '246.684', ]

P_out_dec = ['33.761', '59.527',  '57.131',  '30.248',  '30.954', 
		'56.599', '41.652',  '65.359',  '55.779',  '33.910', 
		'53.601', '49.513',  '36.975',  '33.611',  '48.820', 
		'41.260', '47.118',  ]

lis_ra, lis_dec, lis_z = [], [], []
lis_imgx, lis_imgy = [], []

p_lis_ra, p_lis_dec, p_lis_z = [], [], []
p_lis_imgx, p_lis_imgy = [], []

for kk in range( len(set_z) ):
	identi = ('%.3f' % set_ra[kk] in out_ra) * ('%.3f' % set_dec[kk] in out_dec)

	if identi == False:
		lis_ra.append( set_ra[kk] )
		lis_dec.append( set_dec[kk] )
		lis_z.append( set_z[kk] )
		lis_imgx.append( set_imgx[kk] )
		lis_imgy.append( set_imgy[kk] )
	else:
		continue

lis_ra = np.array(lis_ra )
lis_dec = np.array(lis_dec )
lis_z = np.array(lis_z )
lis_imgx = np.array(lis_imgx )
lis_imgy = np.array(lis_imgy )

for kk in range( len(set_z) ):
	identi = ('%.3f' % set_ra[kk] in P_out_ra) * ('%.3f' % set_dec[kk] in P_out_dec)

	if identi == False:
		p_lis_ra.append( set_ra[kk] )
		p_lis_dec.append( set_dec[kk] )
		p_lis_z.append( set_z[kk] )
		p_lis_imgx.append( set_imgx[kk] )
		p_lis_imgy.append( set_imgy[kk] )
	else:
		continue

p_lis_ra = np.array( p_lis_ra )
p_lis_dec = np.array( p_lis_dec )
p_lis_z = np.array( p_lis_z )
p_lis_imgx = np.array( p_lis_imgx )
p_lis_imgy = np.array( p_lis_imgy )

sub_img = load + '20_10_test_jack/decals_A-250_BCG-stack_sub-%d_img.h5'
sub_pix_cont = load + '20_10_test_jack/decals_A-250_BCG-stack_sub-%d_pix-cont.h5'
sub_sb = load + '20_10_test_jack/decals_A-250_BCG-stack_sub-%d_SB-pro.h5'

'''
## Zou et al. imgs
id_cen = 0
n_rbins = 110
N_bin = 25

d_file = '/home/xkchen/mywork/ICL/data/tmp_img/AB_BASS_mask/ap_sdss_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/Zou_bass_A250_jack-sub-%d_img_sdss-mask.h5'
J_sub_pix_cont = load + '20_10_test_jack/Zou_bass_A250_jack-sub-%d_pix-cont_sdss-mask.h5'
J_sub_sb = load + '20_10_test_jack/Zou_bass_A250_jack-sub-%d_SB-pro_sdss-mask.h5'

jack_SB_file = load + '20_10_test_jack/Zou_bass_A250_Mean_jack_SB-pro_sdss-mask.h5'
jack_img = load + '20_10_test_jack/Zou_bass_A250_Mean_jack_img_sdss-mask.h5'
jack_cont_arr = load + '20_10_test_jack/Zou_bass_A250_Mean_jack_pix-cont_sdss-mask.h5'

jack_main_func(id_cen, N_bin, n_rbins, lis_ra, lis_dec, lis_z, lis_imgx, lis_imgy, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

'''
## Zou et al. imgs (cluster-mask for sky estimation)
id_cen = 0
n_rbins = 110
N_bin = 25

d_file = '/home/xkchen/mywork/ICL/data/tmp_img/AB_BASS_mask/ap_sdss_mask_New-bass_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/New_Zou-bass_A250_jack-sub-%d_img_sdss-mask.h5'
J_sub_pix_cont = load + '20_10_test_jack/New_Zou-bass_A250_jack-sub-%d_pix-cont_sdss-mask.h5'
J_sub_sb = load + '20_10_test_jack/New_Zou-bass_A250_jack-sub-%d_SB-pro_sdss-mask.h5'

jack_SB_file = load + '20_10_test_jack/New_Zou-bass_A250_Mean_jack_SB-pro_sdss-mask.h5'
jack_img = load + '20_10_test_jack/New_Zou-bass_A250_Mean_jack_img_sdss-mask.h5'
jack_cont_arr = load + '20_10_test_jack/New_Zou-bass_A250_Mean_jack_pix-cont_sdss-mask.h5'

jack_main_func(id_cen, N_bin, n_rbins, lis_ra, lis_dec, lis_z, lis_imgx, lis_imgy, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

'''
## Public BASS imgs
id_cen = 0
n_rbins = 110
N_bin = 24

d_file = '/home/xkchen/mywork/ICL/data/tmp_img/AB_BASS_mask/ap_sdss_mask_Pub-bass_%s_ra%.3f_dec%.3f_z%.3f.fits'

J_sub_img = load + '20_10_test_jack/Pub_bass_A250_jack-sub-%d_img_sdss-mask.h5'
J_sub_pix_cont = load + '20_10_test_jack/Pub_bass_A250_jack-sub-%d_pix-cont_sdss-mask.h5'
J_sub_sb = load + '20_10_test_jack/Pub_bass_A250_jack-sub-%d_SB-pro_sdss-mask.h5'

jack_SB_file = load + '20_10_test_jack/Pub_bass_A250_Mean_jack_SB-pro_sdss-mask.h5'
jack_img = load + '20_10_test_jack/Pub_bass_A250_Mean_jack_img_sdss-mask.h5'
jack_cont_arr = load + '20_10_test_jack/Pub_bass_A250_Mean_jack_pix-cont_sdss-mask.h5'

jack_main_func(id_cen, N_bin, n_rbins, p_lis_ra, p_lis_dec, p_lis_z, p_lis_imgx, p_lis_imgy, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)
'''

#************************#
##### part 3
from light_measure import cc_grid_img
load = '/media/xkchen/My Passport/data/SDSS/'

'''
## combine mask case
jack_SB_file = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_SB-pro.h5'
jack_img = load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_img.h5'
'''
img_lis = [	load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_img_sdss-mask.h5', 
			load + '20_10_test_jack/bass_A-250_match_Mean_jack_img_sdss-mask.h5', 
			load + '20_10_test_jack/Zou_bass_A250_Mean_jack_img_sdss-mask.h5',
			load + '20_10_test_jack/Pub_bass_A250_Mean_jack_img_sdss-mask.h5',
			load + '20_10_test_jack/New_Zou-bass_A250_Mean_jack_img_sdss-mask.h5']

pro_lis = [ load + '20_10_test_jack/decals_A-250_BCG-stack_Mean_jack_SB-pro_sdss-mask.h5', 
			load + '20_10_test_jack/bass_A-250_match_Mean_jack_SB-pro_sdss-mask.h5', 
			load + '20_10_test_jack/Zou_bass_A250_Mean_jack_SB-pro_sdss-mask.h5',
			load + '20_10_test_jack/Pub_bass_A250_Mean_jack_SB-pro_sdss-mask.h5',
			load + '20_10_test_jack/New_Zou-bass_A250_Mean_jack_SB-pro_sdss-mask.h5']

line_name = ['DECaLS', 'DECaLS [BASS only]', 'BASS [Zou et al.]', 'BASS [Public]', 'BASS [new sky estimation]']
line_C = ['b', 'g', 'r', 'm', 'c']

jack_SB_file = load + '20_10_test_jack/A250_Mean_jack_SB-pro_bass-region.h5'
jack_img = load + '20_10_test_jack/A250_Mean_jack_img_bass-region.h5'

## SDSS result ('com_')
#with h5py.File(load + '20_10_test_jack/clust_BCG-stack_Mean_jack_img_30-FWHM-ov2.h5', 'r') as f:
with h5py.File(load + '20_10_test_jack/A250_Mean_jack_img_bass-region.h5', 'r') as f:
	com_img = np.array(f['a'])

#with h5py.File(load + '20_10_test_jack/clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2.h5', 'r') as f:
with h5py.File(load + '20_10_test_jack/A250_Mean_jack_SB-pro_bass-region.h5', 'r') as f:
	com_r = np.array(f['r'])
	com_sb = np.array(f['sb'])
	com_sb_err = np.array(f['sb_err'])

lis_R, lis_sb, lis_sb_err = [], [], []

for mm in range( 5 ):

	with h5py.File( pro_lis[mm], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	lis_R.append(c_r_arr)
	lis_sb.append(c_sb_arr)
	lis_sb_err.append(c_sb_err)

	with h5py.File( img_lis[mm], 'r') as f:
		tt_img = np.array(f['a'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	xn, yn = np.int(tt_img.shape[1] / 2), np.int(tt_img.shape[0] / 2)
	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	cen_x, cen_y = xn - x_low, yn - y_low

	img_block = cc_grid_img(dpt_img, 100, 100)[0]

	part_img = com_img[y_low: y_up+1, x_low: x_up + 1]
	com_block = cc_grid_img(part_img, 100, 100)[0]

	diffi_img = part_img - dpt_img
	diff_block = cc_grid_img(diffi_img, 100, 100)[0]

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.02, 0.09, 0.28, 0.85])
	ax1 = fig.add_axes([0.35, 0.09, 0.28, 0.85])
	ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.85])

	ax0.set_title( line_name[mm] )
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e-2, vmax = 2e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.set_title('SDSS img')
	tg = ax1.imshow(com_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e-2, vmax = 2e-2,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax2.set_title('SDSS img minus %s' % line_name[mm] )
	tg = ax2.imshow(diff_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -2e-2, vmax = 2e-2,)
	cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	plt.savefig('comapre_stacking_img_%d.png' % mm, dpi = 300)
	plt.close()

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax1 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax2 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

	ax1.set_title( line_name[mm] )
	tg = ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -1e-2, vmax = 1e-2,)
	clust = Circle(xy = (cen_x // 100, cen_y // 100), radius = (100 / pixel) / 100, fill = False, ec = 'b', 
		ls = '-', linewidth = 0.75, alpha = 0.5,)
	ax1.add_patch(clust)
	clust = Circle(xy = (cen_x // 100, cen_y // 100), radius = (50 / pixel) / 100, fill = False, ec = 'b', 
		ls = '--', linewidth = 0.75, alpha = 0.5,)
	ax1.add_patch(clust)

	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01,)
	cb.formatter.set_powerlimits((0,0))

	ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = line_name[mm] )
	ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

	idux = c_sb_arr < 0
	idvx = c_r_arr >= 20
	idx = idux * idvx
	ops_r = c_r_arr[idx]
	ops_sb = np.abs(c_sb_arr[idx])
	ops_err = c_sb_err[idx]
	ax2.plot(ops_r, ops_sb, ls = '--', color = 'r', alpha = 0.8,)
	ax2.fill_between(ops_r, y1 = ops_sb - ops_err, y2 = ops_sb + ops_err, color = 'r', alpha = 0.2,)

	ax2.plot(com_r, com_sb, ls = '-', color = 'b', alpha = 0.8, label = 'SDSS',)
	ax2.fill_between(com_r, y1 = com_sb - com_sb_err, y2 = com_sb + com_sb_err, color = 'b', alpha = 0.2,)

	ax2.axvline(x = 50, ls = '--', color = 'b', alpha = 0.5,)
	ax2.axvline(x = 100, ls = '-', color = 'b', alpha = 0.5,)

	ax2.set_ylim(1e-4, 3e-2)
	ax2.set_yscale('log')

	ax2.set_xlim(1e1, 1e3)
	ax2.set_xlabel('R [arcsec]')
	ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax2.set_xscale('log')
	ax2.legend(loc = 1, frameon = False, fontsize = 8)
	ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
	ax2.get_yaxis().set_minor_formatter(tick_form)

	plt.savefig('grid_2D_and_SB_compare_%d.png' % mm, dpi = 300)
	plt.close()

plt.figure()
ax = plt.subplot(111)

for mm in ( 2,4 ):

	if mm == 3:
		ax.plot(lis_R[mm], lis_sb[mm], ls = '-', color = line_C[mm], alpha = 0.8, label = line_name[mm],)
		ax.fill_between(lis_R[mm], y1 = lis_sb[mm] - lis_sb_err[mm], y2 = lis_sb[mm] + lis_sb_err[mm], color = line_C[mm], alpha = 0.2,)

		idux = lis_sb[mm] < 0
		idvx = lis_R[mm] >= 20
		idx = idux * idvx
		ops_r = lis_R[mm][idx]
		ops_sb = np.abs(lis_sb[mm][idx])
		ops_err = lis_sb_err[mm][idx]

		#ax.plot(ops_r, ops_sb, ls = '--', color = line_C[mm], alpha = 0.8,)
		#ax.fill_between(ops_r, y1 = ops_sb - ops_err, y2 = ops_sb + ops_err, color = line_C[mm], alpha = 0.2,)
	else:
		ax.plot(lis_R[mm], lis_sb[mm], ls = '-', color = line_C[mm], alpha = 0.8, label = line_name[mm],)
		ax.fill_between(lis_R[mm], y1 = lis_sb[mm] - lis_sb_err[mm], y2 = lis_sb[mm] + lis_sb_err[mm], color = line_C[mm], alpha = 0.2,)

	if mm == 2:
		idsb = 8e-4
		devi_sb = lis_sb[mm] - idsb
		ax.axhline(y =  idsb, ls = '-.', color = line_C[mm], alpha = 0.5,)
		ax.plot(lis_R[mm], devi_sb, ls = ':', color = line_C[mm], alpha = 0.8,)
		#ax.fill_between(lis_R[mm], y1 = devi_sb - lis_sb_err[mm], y2 = devi_sb + lis_sb_err[mm], color = line_C[mm], alpha = 0.2,)

ax.plot(com_r, com_sb, ls = '-', color = 'k', alpha = 0.8, label = 'SDSS',)
ax.fill_between(com_r, y1 = com_sb - com_sb_err, y2 = com_sb + com_sb_err, color = 'k', alpha = 0.2,)

idr = np.abs( com_r - 600 )
idrx = np.where( idr == idr.min() )[0]
idsb = com_sb[idrx ]
devi_sb = com_sb - idsb

ax.axhline(y =  idsb, ls = '-.', color = 'k', alpha = 0.5, label = 'Background [SDSS]',)
ax.plot(com_r, devi_sb, ls = ':', color = 'k', alpha = 0.8,)
ax.fill_between(com_r, y1 = devi_sb - com_sb_err, y2 = devi_sb + com_sb_err, color = 'k', alpha = 0.2,)

ax.set_ylim(1e-4, 3e-2)
ax.set_yscale('log')

ax.set_xlim(1e1, 1e3)
ax.set_xlabel('R [arcsec]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 1, frameon = False, fontsize = 8)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
ax.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('DECaLS_SDSS_SB_compare.png', dpi = 300)
plt.close()


