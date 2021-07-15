import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

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
from astropy import cosmology as apcy

from groups import groups_find_func
from fig_out_module import cc_grid_img, grid_img
import time

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
def source_mask(img_file, cen_x, cen_y, gal_cat, star_cat, bcg_file = None,):

	data = fits.open(img_file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)

	source = asc.read(gal_cat)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE'])
	cy = np.array(source['Y_IMAGE'])
	p_type = np.array(source['CLASS_STAR'])

	Kron = 16 # 6
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
	ipx = idx

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

	tdr = np.sqrt((cen_x - cx)**2 + (cen_y - cy)**2)
	id_cen = tdr == np.min(tdr)

	if bcg_file is not None:
		BCG_photo_cat = pds.read_csv( bcg_file, skiprows = 1)
		r_Reff = np.array(BCG_photo_cat['deVRad_r'])[0]
		g_Reff = np.array(BCG_photo_cat['deVRad_g'])[0]
		i_Reff = np.array(BCG_photo_cat['deVRad_i'])[0]


	plt.figure()
	ax = plt.subplot(111)
	ax.imshow(img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm(),)
	'''
	# BCG identify
	for mm in range( Numb ):
		ellips = Ellipse(xy = (cx[mm], cy[mm]), width = a[mm], height = b[mm], angle = theta[mm], fill = False, 
			ec = 'c', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(ellips)
		#ax.plot(cx, cy, 'o', color = 'g', alpha = 0.05,)

	clust = Circle(xy = (cen_x, cen_y), radius = r_Reff / pixel, fill = False, ec = 'r', ls = '-', linewidth = 1, alpha = 0.5,)
	ax.add_patch(clust)

	clust = Circle(xy = (cen_x, cen_y), radius = g_Reff / pixel, fill = False, ec = 'g', ls = '--', linewidth = 1, alpha = 0.5,)
	ax.add_patch(clust)

	clust = Circle(xy = (cen_x, cen_y), radius = i_Reff / pixel, fill = False, ec = 'b', ls = ':', linewidth = 1, alpha = 0.5,)
	ax.add_patch(clust)

	ax.scatter(cx[id_cen], cy[id_cen], s = 25, marker = 'X', edgecolors = 'k', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)
	ax.scatter(cen_x, cen_y, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)

	ax.set_xlim(cen_x - 50, cen_x + 50)
	ax.set_ylim(cen_y - 50, cen_y + 50)	
	'''
	# stars
	#for mm in range( len(sub_x0) ):
	#	ellips = Ellipse(xy = (sub_x0[mm], sub_y0[mm]), width = sub_A0[mm] / 30, height = sub_B0[mm] / 30, angle = sub_chi0[mm],
	#	fill = False, ec = 'r', ls = '-', linewidth = 1, alpha = 0.5,)
	#	ax.add_patch( ellips )

	ax.scatter(sub_x0, sub_y0, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)

	ax.scatter(sub_x2, sub_y2, s = 25, marker = 's', edgecolors = 'r', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)

	#ax.scatter(x, y, s = 25, marker = 'o', edgecolors = 'g', facecolors = 'none', linewidth = 0.75, alpha = 0.5,)

	ax.set_xlim( 0, 2048)
	ax.set_ylim( 0, 1489)

	plt.savefig('source_located.png', dpi = 300)
	plt.close()

	raise

	return mask_img

band = ['r', 'g', 'i']
home = '/media/xkchen/My Passport/data/SDSS/'
load = '/home/xkchen/mywork/ICL/'

for kk in range( 30 ):#Nz ):

	#ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

	ra_g, dec_g, z_g = 184.1834886175924, 16.266165158249873, 0.2729847431182861
	#ra_g, dec_g, z_g = 153.27341918416187,7.850910990300251,0.2759900987148285

	#file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('i', ra_g, dec_g, z_g)
	file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)

	data = fits.open(file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)
	xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	hdu = fits.PrimaryHDU()
	hdu.data = img
	hdu.header = head
	hdu.writeto('test_t.fits', overwrite = True)

	#param_A = 'default_mask_A_g.sex'
	param_A = 'default_mask_A.sex'

	out_cat = 'default_mask_A.param'

	out_load_A = 'test_t.cat'
	file_source = 'test_t.fits'

	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_load_A, out_cat)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	## diffuse light region identify
	#star_cat = '/home/xkchen/mywork/ICL/data/corrected_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
	#bcg_info = '/media/xkchen/My Passport/data/SDSS/BCG_photometric/BCG_photo_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)

	#star_cat = '/home/xkchen/mywork/ICL/data/corrected_star_cat/random/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
	star_cat = '/home/xkchen/mywork/ICL/data/tmp_img/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
	#star_cat = '/home/xkchen/source_SQL_Z%.3f_ra%.3f_dec%.3f.csv' % (z_g, ra_g, dec_g)

	remain_img = source_mask(file, xn, yn, out_load_A, star_cat,)

	raise

	res_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % ('r', ra_g, dec_g, z_g)
	#res_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % ('r', ra_g, dec_g, z_g)
	res_data = fits.open(res_file)
	remain_img = res_data[0].data

	ca0, ca1 = np.int(img.shape[0] / 2), np.int(img.shape[1] / 2)
	cen_D = 500
	flux_cen = remain_img[ca0 - cen_D: ca0 + cen_D, ca1 - cen_D: ca1 + cen_D]

	N_step = 200

	sub_pock_flux, sub_pock_pix = grid_img(flux_cen, N_step, N_step)[:2]
	'''
	id_Nzero = sub_pock_pix > 100
	mu = np.nanmean( sub_pock_flux[id_Nzero] )
	sigm = np.nanstd( sub_pock_flux[id_Nzero] )
	'''

	#samp_dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/img_test-1000_mean_sigm.csv')
	samp_dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/img_A-250_mean_sigm.csv')
	img_mu, img_sigma = np.array(samp_dat.img_mu), np.array(samp_dat.img_sigma)
	mu = 3 * np.median(img_mu) - 2 * np.mean(img_mu) # np.mean(img_mu) # 
	sigm = 3 * np.median(img_sigma) - 2 * np.mean(img_sigma) # np.mean(img_sigma) # 

	## mu, and deviation for sub-patches
	patch_grd = cc_grid_img(remain_img, N_step, N_step)
	patch_mean = patch_grd[0]
	patch_pix = patch_grd[1]
	lx, ly = patch_grd[-2], patch_grd[-1]

	id_zeros = patch_pix == 0.
	patch_pix[id_zeros] = np.nan
	patch_mean[id_zeros] = np.nan
	over_sb = (patch_mean - mu) / sigm

	##### img selection
	lim_sb = 4
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

		bx.set_title('$ [\\mu_{patch} - Mode(\\mu)] / Mode(\\sigma) $')
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
			bx.set_title('$ [\\mu_{patch} - Mode(\\mu)] / Mode(\\sigma) $')
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

				bx.set_title('$ [\\mu_{patch} - Mode(\\mu)] / Mode(\\sigma) $')
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

					bx.set_title('$ [\\mu_{patch} - Mode(\\mu)] / Mode(\\sigma) $')
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

						bx.set_title('$ [\\mu_{patch} - Mode(\\mu)] / Mode(\\sigma) $')
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

						bx.set_title('$ [\\mu_{patch} - Mode(\\mu)] / Mode(\\sigma) $')
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

