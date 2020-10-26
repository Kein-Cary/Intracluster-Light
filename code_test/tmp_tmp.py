import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import skimage
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

pixel, z_ref = 0.396, 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0 / Da_ref)*rad2asec
Rpp = Angu_ref / pixel
####################################
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
	x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1) ##??

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
	sub_A0 = lr_iso[ic] * 1# 30
	sub_B0 = sr_iso[ic] * 1# 30
	sub_chi0 = set_chi[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in xt]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]
	sub_A2 = lr_iso[ipx] * 1# 75
	sub_B2 = sr_iso[ipx] * 1# 75
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

def cc_grid_img(img_data, N_stepx, N_stepy):

	binx = img_data.shape[1] // N_stepx ## bin number along 2 axis
	biny = img_data.shape[0] // N_stepy

	beyon_x = img_data.shape[1] - binx * N_stepx ## for edge pixels divid
	beyon_y = img_data.shape[0] - biny * N_stepy

	odd_x = np.ceil(beyon_x / binx)
	odd_y = np.ceil(beyon_y / biny)

	n_odd_x = beyon_x // odd_x
	n_odd_y = beyon_y // odd_y

	d_odd_x = beyon_x - odd_x * n_odd_x
	d_odd_y = beyon_y - odd_y * n_odd_y

	# get the bin width
	wid_x = np.zeros(binx, dtype = np.float32)
	wid_y = np.zeros(biny, dtype = np.float32)
	for kk in range(binx):
		if kk == n_odd_x :
			wid_x[kk] = N_stepx + d_odd_x
		elif kk < n_odd_x :
			wid_x[kk] = N_stepx + odd_x
		else:
			wid_x[kk] = N_stepx

	for kk in range(biny):
		if kk == n_odd_y :
			wid_y[kk] = N_stepy + d_odd_y
		elif kk < n_odd_y :
			wid_y[kk] = N_stepy + odd_y
		else:
			wid_y[kk] = N_stepy

	# get the bin edge
	lx = np.zeros(binx + 1, dtype = np.int32)
	ly = np.zeros(biny + 1, dtype = np.int32)
	for kk in range(binx):
		lx[kk + 1] = lx[kk] + wid_x[kk]
	for kk in range(biny):
		ly[kk + 1] = ly[kk] + wid_y[kk]

	patch_mean = np.zeros( (biny, binx), dtype = np.float )
	patch_pix = np.zeros( (biny, binx), dtype = np.float )
	patch_S0 = np.zeros( (biny, binx), dtype = np.float )
	patch_Var = np.zeros( (biny, binx), dtype = np.float )
	for nn in range( biny ):
		for tt in range( binx ):

			sub_flux = img_data[ly[nn]: ly[nn + 1], lx[tt]: lx[tt + 1] ]
			id_nn = np.isnan(sub_flux)

			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )
			patch_S0[nn,tt] = (ly[nn + 1] - ly[nn]) * (lx[tt + 1] - lx[tt])

	return patch_mean, patch_Var, patch_pix, patch_S0

band = ['r', 'g', 'i']
home = '/media/xkchen/My Passport/data/SDSS/'

### (mu, sigma) based on single img
#dat = pds.read_csv('result/test_1000-to-250_rule-out_cat_3.5-sigma.csv')
#dat = pds.read_csv('result/test_1000-to-459_remain_cat_3.5-sigma.csv')
### (mu, sigma) based on sample imgs
#dat = pds.read_csv('result/alt_test_1000-to-250_rule-out_cat_3.5-sigma.csv')
#dat = pds.read_csv('result/alt_test_1000-to-193_remain_cat_3.5-sigma.csv')

#dat = pds.read_csv('result/CC_test_1000-to-250_rule-out_cat_3.5-sigma.csv')
dat = pds.read_csv('result/CC_test_1000-to-459_remain_cat_3.5-sigma.csv')

### (mode_mu, mode_sigma) based on sample imgs
#dat = pds.read_csv('result/MM_test_1000-to-250_rule-out_cat_3.5-sigma.csv')
#dat = pds.read_csv('result/MM_test_1000-to-459_remain_cat_3.5-sigma.csv')
lis_ra, lis_dec, lis_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

for kk in range( 10, 20 ):#len(lis_z) ):

	ra_g, dec_g, z_g = lis_ra[kk], lis_dec[kk], lis_z[kk]

	file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)
	#file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)

	data = fits.open(file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)
	xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
	'''
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
		star_cat = '/home/xkchen/mywork/ICL/data/corrected_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
		#star_cat = '/home/xkchen/mywork/ICL/data/corrected_star_cat/random/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)

		remain_img = source_mask(file, out_load_A, star_cat)
	'''
	res_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % ('r', ra_g, dec_g, z_g)
	#res_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % ('r', ra_g, dec_g, z_g)
	res_data = fits.open(res_file)
	remain_img = res_data[0].data

	ca0, ca1 = np.int(img.shape[0] / 2), np.int(img.shape[1] / 2)
	cen_D = 500
	flux_cen = remain_img[ca0 - cen_D: ca0 + cen_D, ca1 - cen_D: ca1 + cen_D]

	N_step = 200

	cen_lx = np.arange(0, 1100, N_step)
	cen_ly = np.arange(0, 1100, N_step)
	nl0, nl1 = len(cen_ly), len(cen_lx)

	sub_pock_pix = np.zeros((nl0 - 1, nl1 - 1), dtype = np.float)
	sub_pock_flux = np.zeros((nl0 - 1, nl1 - 1), dtype = np.float)
	for nn in range(nl0 - 1):
		for tt in range(nl1 - 1):
			sub_flux = flux_cen[ cen_ly[nn]: cen_ly[nn+1], cen_lx[tt]: cen_lx[tt+1] ]
			id_nn = np.isnan(sub_flux)
			sub_pock_flux[nn,tt] = np.nanmean(sub_flux)
			sub_pock_pix[nn,tt] = len(sub_flux[id_nn == False])

	## mu, sigma of center region
	id_Nzero = sub_pock_pix > 100
	#mu = np.nanmean( sub_pock_flux[id_Nzero] )
	#sigm = np.nanstd( sub_pock_flux[id_Nzero] )

	##?? use sample mean and scatter
	#samp_dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/img_3100_mean_sigm.csv')
	samp_dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/img_test-1000_mean_sigm.csv')
	img_mu, img_sigma = np.array(samp_dat.cen_mu), np.array(samp_dat.cen_sigma)
	mu = np.mean(img_mu) # 3 * np.median(img_mu) - 2 * np.mean(img_mu) # 
	sigm = np.mean(img_sigma) # 3 * np.median(img_sigma) - 2 * np.mean(img_sigma) # 

	## mu, and deviation for sub-patches
	ly = np.arange(0, img.shape[0], N_step)
	ly = np.r_[ly, img.shape[0] - N_step, img.shape[0] ]
	lx = np.arange(0, img.shape[1], N_step)
	lx = np.r_[lx, img.shape[1] - N_step, img.shape[1] ]

	lx = np.delete(lx, -1)
	lx = np.delete(lx, -2)
	ly = np.delete(ly, -1)
	ly = np.delete(ly, -2)

	patch_mean = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_pix = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	for nn in range( len(ly) ):
		for tt in range( len(lx) ):

			sub_flux = remain_img[ly[nn]: ly[nn] + N_step, lx[tt]: lx[tt] + N_step]
			id_nn = np.isnan(sub_flux)
			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )

	id_zeros = patch_pix == 0.
	patch_pix[id_zeros] = np.nan
	patch_mean[id_zeros] = np.nan
	over_sb = (patch_mean - mu) / sigm

	##### img selection
	lim_sb = 3.5
	### first select
	identi = over_sb > lim_sb

	if np.sum(identi) < 1:
		plt.figure( figsize = (12, 6) )
		ax = plt.subplot(121)
		bx = plt.subplot(122)

		ax.set_title('sources ra%.3f dec%.3f z%.3f [pix-%d]' % (ra_g, dec_g, z_g, N_step) )
		ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

		clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.45,)
		ax.add_patch(clust)
		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])

		idmask = np.isnan(over_sb)
		over_sb[idmask] = 100
		bx.set_title('$ [\\mu_{patch} - \\mu_{center}] / \\sigma_{center} $')
		tf = bx.imshow(over_sb, origin = 'lower', cmap = 'seismic', vmin = -5, vmax = 5, )
		plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, )
		tf.cmap.set_under('cyan')
		tf.cmap.set_over('yellow')

		for mm in range(over_sb.shape[1]):
			for nn in range(over_sb.shape[0]):
				bx.text(mm, nn, s = '%.2f' % over_sb[nn, mm], ha = 'center', va = 'center', color = 'k', fontsize = 6, alpha = 0.5)
		plt.tight_layout()
		plt.savefig('norm-img_ra%.3f_dec%.3f_z%.3f_pix-%d.png' % (ra_g, dec_g, z_g, N_step), dpi = 300)
		plt.close()

	else:
		### lighter blocks find
		copy_arr = over_sb.copy()
		idnn = np.isnan(over_sb)
		copy_arr[idnn] = 100

		source_n, coord_x, coord_y = groups_find_func(copy_arr, lim_sb)

		lo_xs = lx[ [np.min( ll ) for ll in coord_x] ]
		hi_xs = lx[ [np.max( ll ) for ll in coord_x] ]
		lo_ys = ly[ [np.min( ll ) for ll in coord_y] ]
		hi_ys = ly[ [np.max( ll ) for ll in coord_y] ]
		### mainly focus on regions which close to edges
		idux = (lo_xs <= 500) | (2000 - hi_xs <= 500)
		iduy = (lo_ys <= 500) | (1400 - hi_ys <= 500)
		idu = idux | iduy

		### select groups with block number larger or equal to 3
		idv_s = (np.array(source_n) >= 3)
		id_pat_s = idu & idv_s

		if np.sum(id_pat_s) < 1:

			plt.figure( figsize = (12, 6) )
			ax = plt.subplot(121)
			bx = plt.subplot(122)

			ax.set_title('sources ra%.3f dec%.3f z%.3f [pix-%d]' % (ra_g, dec_g, z_g, N_step) )
			ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

			clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.45,)
			ax.add_patch(clust)
			ax.set_xlim(0, img.shape[1])
			ax.set_ylim(0, img.shape[0])

			idmask = np.isnan(over_sb)
			over_sb[idmask] = 100
			bx.set_title('$ [\\mu_{patch} - \\mu_{center}] / \\sigma_{center} $')
			tf = bx.imshow(over_sb, origin = 'lower', cmap = 'seismic', vmin = -5, vmax = 5, )
			plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, )
			tf.cmap.set_under('cyan')
			tf.cmap.set_over('yellow')
			for mm in range(over_sb.shape[1]):
				for nn in range(over_sb.shape[0]):
					bx.text(mm, nn, s = '%.2f' % over_sb[nn, mm], ha = 'center', va = 'center', color = 'k', fontsize = 6, alpha = 0.5)
			plt.tight_layout()
			plt.savefig('norm-img_ra%.3f_dec%.3f_z%.3f_pix-%d.png' % (ra_g, dec_g, z_g, N_step), dpi = 300)
			plt.close()

		else:
			id_vs_pri = (np.array(source_n) <= 5)
			id_pat_s = idu & (idv_s & id_vs_pri)

			id_True = np.where(id_pat_s == True)[0]
			loop_N = np.sum(id_pat_s)
			pur_N = np.zeros(loop_N, dtype = np.int)
			pur_mask = np.zeros(loop_N, dtype = np.int)
			pur_outlier = np.zeros(loop_N, dtype = np.int)

			for ll in range( loop_N ):
				id_group = id_True[ll]
				tot_pont = source_n[ id_group ]
				tmp_arr = copy_arr[ coord_y[ id_group ], coord_x[ id_group ] ]
				id_out = tmp_arr == 100.
				id_2_bright = tmp_arr > 8.0 # 9.5

				pur_N[ll] = tot_pont - np.sum(id_out)
				pur_mask[ll] = np.sum(id_out)
				pur_outlier[ll] = np.sum(id_2_bright) * ( np.sum(id_out) == 0)

			## at least 1 blocks have mean value above the lim_sb,(except mask region)
			#idnum = ( (pur_N >= 1) & (pur_mask >= 1) ) | (pur_outlier >= 1)
			idnum = pur_N >= 1 ## broadly case

			if np.sum(idnum) >= 1:
				plt.figure( figsize = (12, 6) )
				ax = plt.subplot(121)
				bx = plt.subplot(122)

				ax.set_title('sources ra%.3f dec%.3f z%.3f [pix-%d]' % (ra_g, dec_g, z_g, N_step) )
				ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

				clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.45,)
				ax.add_patch(clust)

				n_lock = np.sum(idnum)
				id_lock = np.where(idnum == True)[0]
				for qq in range( n_lock ):
					tmp_id = id_True[ id_lock[qq] ]
					sub_edgx = np.array( lx[ coord_x[ tmp_id ] ] )
					sub_edgy = np.array( ly[ coord_y[ tmp_id ] ] )

					for mm in range( source_n[ tmp_id ] ):

						a0, a1 = sub_edgx[mm], sub_edgx[mm] + N_step
						b0, b1 = sub_edgy[mm], sub_edgy[mm] + N_step
						region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, 
							ec = mpl.cm.rainbow( qq / 4), linewidth = 1, alpha = 0.5,)
						ax.add_patch(region)

				ax.set_xlim(0, img.shape[1])
				ax.set_ylim(0, img.shape[0])

				idmask = np.isnan(over_sb)
				over_sb[idmask] = 100
				bx.set_title('$ [\\mu_{patch} - \\mu_{center}] / \\sigma_{center} $')
				tf = bx.imshow(over_sb, origin = 'lower', cmap = 'seismic', vmin = -5, vmax = 5, )
				plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, )
				tf.cmap.set_under('cyan')
				tf.cmap.set_over('yellow')

				for mm in range(over_sb.shape[1]):
					for nn in range(over_sb.shape[0]):
						bx.text(mm, nn, s = '%.2f' % over_sb[nn, mm], ha = 'center', va = 'center', color = 'k', fontsize = 6, alpha = 0.5)
				plt.tight_layout()
				plt.savefig('bad-img_ra%.3f_dec%.3f_z%.3f_pix-%d.png' % (ra_g, dec_g, z_g, N_step), dpi = 300)
				plt.close()

			else:
				### search for larger groups
				idv = np.array(source_n) >= 5 # each group include 5 patches at least
				id_pat = idu & idv

				if np.sum(id_pat) < 1:
					plt.figure( figsize = (12, 6) )
					ax = plt.subplot(121)
					bx = plt.subplot(122)

					ax.set_title('sources ra%.3f dec%.3f z%.3f [pix-%d]' % (ra_g, dec_g, z_g, N_step) )
					ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

					clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.45,)
					ax.add_patch(clust)
					ax.set_xlim(0, img.shape[1])
					ax.set_ylim(0, img.shape[0])

					idmask = np.isnan(over_sb)
					over_sb[idmask] = 100
					bx.set_title('$ [\\mu_{patch} - \\mu_{center}] / \\sigma_{center} $')
					tf = bx.imshow(over_sb, origin = 'lower', cmap = 'seismic', vmin = -5, vmax = 5, )
					plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, )
					tf.cmap.set_under('cyan')
					tf.cmap.set_over('yellow')
					for mm in range(over_sb.shape[1]):
						for nn in range(over_sb.shape[0]):
							bx.text(mm, nn, s = '%.2f' % over_sb[nn, mm], ha = 'center', va = 'center', color = 'k', fontsize = 6, alpha = 0.5)
					plt.tight_layout()
					plt.savefig('norm-img_ra%.3f_dec%.3f_z%.3f_pix-%d.png' % (ra_g, dec_g, z_g, N_step), dpi = 300)
					plt.close()

				else:

					id_True = np.where(id_pat == True)[0]
					loop_N = np.sum(id_pat)
					pur_N = np.zeros(loop_N, dtype = np.int)
					for ll in range( loop_N ):
						id_group = id_True[ll]
						tot_pont = source_n[ id_group ]
						tmp_arr = copy_arr[ coord_y[ id_group ], coord_x[ id_group ] ]
						id_out = tmp_arr == 100.
						pur_N[ll] = tot_pont - np.sum(id_out)

					idnum = pur_N >= 2 ## at least 2 blocks have mean value above the lim_sb,(except mask region)

					if np.sum(idnum) < 1:
						plt.figure( figsize = (12, 6) )
						ax = plt.subplot(121)
						bx = plt.subplot(122)

						ax.set_title('sources ra%.3f dec%.3f z%.3f [pix-%d]' % (ra_g, dec_g, z_g, N_step) )
						ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

						clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.45,)
						ax.add_patch(clust)
						ax.set_xlim(0, img.shape[1])
						ax.set_ylim(0, img.shape[0])

						idmask = np.isnan(over_sb)
						over_sb[idmask] = 100
						bx.set_title('$ [\\mu_{patch} - \\mu_{center}] / \\sigma_{center} $')
						tf = bx.imshow(over_sb, origin = 'lower', cmap = 'seismic', vmin = -5, vmax = 5, )
						plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, )
						tf.cmap.set_under('cyan')
						tf.cmap.set_over('yellow')
						for mm in range(over_sb.shape[1]):
							for nn in range(over_sb.shape[0]):
								bx.text(mm, nn, s = '%.2f' % over_sb[nn, mm], ha = 'center', va = 'center', color = 'k', fontsize = 6, alpha = 0.5)
						plt.tight_layout()
						plt.savefig('norm-img_ra%.3f_dec%.3f_z%.3f_pix-%d.png' % (ra_g, dec_g, z_g, N_step), dpi = 300)
						plt.close()

					else:
						plt.figure( figsize = (12, 6) )
						ax = plt.subplot(121)
						bx = plt.subplot(122)

						ax.set_title('sources ra%.3f dec%.3f z%.3f [pix-%d]' % (ra_g, dec_g, z_g, N_step) )
						ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

						clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.45,)
						ax.add_patch(clust)

						n_lock = np.sum(idnum)
						id_lock = np.where(idnum == True)[0]
						for qq in range( n_lock ):
							tmp_id = id_True[ id_lock[qq] ]
							sub_edgx = np.array( lx[ coord_x[ tmp_id ] ] )
							sub_edgy = np.array( ly[ coord_y[ tmp_id ] ] )

							for mm in range( source_n[ tmp_id ] ):

								a0, a1 = sub_edgx[mm], sub_edgx[mm] + N_step
								b0, b1 = sub_edgy[mm], sub_edgy[mm] + N_step
								region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, 
									ec = mpl.cm.rainbow( qq / 4), linewidth = 1, alpha = 0.5,)
								ax.add_patch(region)

						ax.set_xlim(0, img.shape[1])
						ax.set_ylim(0, img.shape[0])

						idmask = np.isnan(over_sb)
						over_sb[idmask] = 100
						bx.set_title('$ [\\mu_{patch} - \\mu_{center}] / \\sigma_{center} $')
						tf = bx.imshow(over_sb, origin = 'lower', cmap = 'seismic', vmin = -5, vmax = 5, )
						plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, )
						tf.cmap.set_under('cyan')
						tf.cmap.set_over('yellow')

						for mm in range(over_sb.shape[1]):
							for nn in range(over_sb.shape[0]):
								bx.text(mm, nn, s = '%.2f' % over_sb[nn, mm], ha = 'center', va = 'center', color = 'k', fontsize = 6, alpha = 0.5)
						plt.tight_layout()
						plt.savefig('bad-img_ra%.3f_dec%.3f_z%.3f_pix-%d.png' % (ra_g, dec_g, z_g, N_step), dpi = 300)
						plt.close()

