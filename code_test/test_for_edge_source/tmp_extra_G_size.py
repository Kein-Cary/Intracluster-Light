import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import subprocess as subpro
import astropy.constants as C

from groups import groups_find_func
from astropy import cosmology as apcy
from matplotlib.patches import Circle, Ellipse, Rectangle

### cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
### masking source
def source_mask(img_file, gal_cat, star_cat):

	data = fits.open(img_file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)

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

	## stars
	cat = pds.read_csv(star_cat, skiprows = 1)
	set_ra = np.array(cat['ra'])
	set_dec = np.array(cat['dec'])
	set_mag = np.array(cat['r'])
	OBJ = np.array(cat['type'])
	xt = cat['Column1']
	x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)

	set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]
	sub_A0 = lr_iso[ic] * 30
	sub_B0 = sr_iso[ic] * 30
	sub_chi0 = set_chi[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in xt]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]
	sub_A2 = lr_iso[ipx] * 75
	sub_B2 = sr_iso[ipx] * 75
	sub_chi2 = set_chi[ipx]

	comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
	comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
	Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
	Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
	phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]

	tot_cx = np.r_[cx, comx]
	tot_cy = np.r_[cy, comy]
	tot_a = np.r_[a, Lr]
	tot_b = np.r_[b, Sr]
	tot_theta = np.r_[theta, phi]
	tot_Numb = Numb + len(comx)

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

###############
band = ['r', 'g', 'i']
home = '/media/xkchen/My Passport/data/SDSS/'
load = '/home/xkchen/mywork/ICL/data/tmp_img/'

dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/clust-1000-select_remain_test.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_clus-1000-match_remain_test.csv')
#set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

set_ra = np.array([116.27008994, 237.98442339,  43.74691031, 328.70772309, 244.80394366, 223.64620176])
set_dec = np.array([33.21568573,  3.87253616,  1.21179466,  0.93070107, 27.85103334, 16.89659645])
set_z = np.array([0.21967424, 0.20158531, 0.23486698, 0.21077134, 0.22226553, 0.28242844])
'''
set_ra = np.array([1.83130964e+02, 1.25202514e+02, 1.71582876e+02, 2.29694374e+02, 1.20735569e+02, 1.16523656e+02, 
					1.18342186e+02, 2.21388906e+02, 3.59911976e+02, 2.64962976e-02, 1.19110406e+02])
set_dec = np.array([41.17222915, 32.16383756,  6.62684376,  5.22180692,  5.94082885, 40.42252864, 37.85882266, 
					5.77266257, 13.69215375, 34.28287086, 30.81077071])
set_z = np.array([0.20266347, 0.20379362, 0.21334173, 0.2720618 , 0.27837378, 0.2986178 , 0.29753768, 0.22983527,
					0.20360596, 0.22218521, 0.27797976])
'''
compare_lis = []

for kk in range( len(set_z) ):

	ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

	file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)
	#file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)

	data = fits.open(file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)
	xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	hdu = fits.PrimaryHDU()
	hdu.data = img
	hdu.header = head
	hdu.writeto('test_t.fits', overwrite = True)

	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'

	out_load_A = 'test_t.cat'
	file_source = 'test_t.fits'

	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_load_A, out_cat)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	## diffuse light region identify
	star_cat = '/home/xkchen/mywork/ICL/data/star_dr12_reload/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
	#star_cat = home + 'random_cat/star_cat/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)

	mask_1st = source_mask(file, out_load_A, star_cat)

	## detected sources
	source = asc.read(out_load_A)
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])

	Kron = 16
	a = Kron * A
	b = Kron * B

	## extra-G cat.
	gal_dat = pds.read_csv(load + 'source_find/clus_photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_g, dec_g, z_g),)
	#gal_dat = pds.read_csv(load + 'source_find/photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_g, dec_g, z_g),)
	gcat_x, gcat_y = np.array(gal_dat.imgx), np.array(gal_dat.imgy)

	r_adj = np.array([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
	lis_eta = []

	for nn in range( len(r_adj) ):
		gcat_a = np.ones( len(gcat_x) ) * np.nanmean( a ) * r_adj[nn]
		gcat_b = np.ones( len(gcat_x) ) * np.nanmean( a ) * r_adj[nn] # use a circle to mask
		gcat_chi = np.ones( len(gcat_x) ) * 0.

		mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float32)
		ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
		oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
		basic_coord = np.array(np.meshgrid(ox, oy))
		major = gcat_a / 2
		minor = gcat_b / 2
		senior = np.sqrt(major**2 - minor**2)
		N_sumb = len(gcat_x)

		for k in range( N_sumb ):
			xc = gcat_x[k]
			yc = gcat_y[k]

			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = gcat_chi[k] * np.pi/180

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

		mask_2nd = mask_path * mask_1st

		## count the edge pixels
		grdx = np.linspace(0, img.shape[1] - 1, img.shape[1])
		grdy = np.linspace(0, img.shape[0] - 1, img.shape[0])
		grd = np.array( np.meshgrid(grdx, grdy) )
		grd = grd.astype(np.int)

		idvx = (grd[0] < 300) | (grd[0] > img.shape[1] - 300)
		idvy = (grd[1] < 300) | (grd[1] > img.shape[0] - 300)
		idv = idvx | idvy

		edg_flux_0 = mask_1st[idv]
		id_nn = np.isnan(edg_flux_0)
		edg_cont_0 = len(edg_flux_0[id_nn == False])

		edg_flux_1 = mask_2nd[idv]
		id_nn = np.isnan(edg_flux_1)
		edg_cont_1 = len(edg_flux_1[id_nn == False])

		eta = (edg_cont_0 - edg_cont_1) / edg_cont_0
		lis_eta.append( eta )
		'''
		plt.figure( figsize = (12,6) )
		ax0 = plt.subplot(121)
		ax1 = plt.subplot(122)

		ax0.set_title('previous mask [ra%.3f dec%.3f z%.3f]' % (ra_g, dec_g, z_g), )
		ax0.imshow(mask_1st, origin = 'lower', cmap = 'seismic', vmin = -0.02, vmax = 0.02, )

		ax1.set_title('previous mask + extral galaxy catalog [$\\Delta_{pix}$ = %.3f]' % eta,)
		ax1.imshow(mask_2nd, origin = 'lower', cmap = 'seismic', vmin = -0.02, vmax = 0.02, )

		plt.tight_layout()
		#plt.savefig('clus_ra%.3f_dec%.3f_z%.3f_radj-%.1f.png' % (ra_g, dec_g, z_g, r_adj[nn]), dpi = 300)
		plt.savefig('random_ra%.3f_dec%.3f_z%.3f_radj-%.1f.png' % (ra_g, dec_g, z_g, r_adj[nn]), dpi = 300)
		plt.close()
		'''
	compare_lis.append(lis_eta)

plt.figure()
#plt.title('cluster edge extra galaxy mask test')
plt.title('random edge extra galaxy mask test')
for nn in range( len(set_z) ):
	plt.plot(r_adj, compare_lis[nn], color = mpl.cm.rainbow(nn / len(set_z)), ls = '-', alpha = 0.5)
	plt.plot(r_adj, compare_lis[nn], 'o', color = mpl.cm.rainbow(nn / len(set_z)), alpha = 0.5, )

plt.xlabel('$R_{mask} / \\bar{a} $')
plt.ylabel('$(N_{0} - N)/ N_{0}$ [edgs-pixel number change]')
plt.grid('both', alpha = 0.25)
#plt.savefig('clust_r-adj_test.png', dpi = 300)
plt.savefig('random_r-adj_test.png', dpi = 300)
plt.close()

raise
