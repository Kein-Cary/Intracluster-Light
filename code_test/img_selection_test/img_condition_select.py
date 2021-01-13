import matplotlib as mpl
#mpl.use('Agg')
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

from astropy import cosmology as apcy
from matplotlib.patches import Circle, Ellipse
from scipy.stats import binned_statistic as binned
from scipy.optimize import curve_fit
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

band = ['r', 'g', 'i']
home = '/media/xkchen/My Passport/data/SDSS/'
load = '/home/xkchen/mywork/ICL/data/tmp_img/'

####################

#dat = pds.read_csv('/home/xkchen/mywork/ICL/r_band_sky_catalog.csv')
dat = pds.read_csv('/home/xkchen/mywork/ICL/rand_r_band_catalog.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

### trouble images
lis_ra = np.array([231.20355251, 150.89168502,  26.22923052, 191.24271753])
lis_dec = np.array([11.30799735, 42.54243936, 14.20254774, 61.31027512])
lis_z = np.array([0.27726924, 0.2639949 , 0.26970404, 0.230333])

Nt = 100
## 1: train sample; 5, 4, 3, 2, 6: test sample
np.random.seed( 3 )
tt0 = np.random.choice(len(z), size = Nt, replace = False)

'''
### choose 1000 for SB profile test
Nt = 1000
np.random.seed( 0 )  ## 0 for stacking test
tt0 = np.random.choice(len(z), size = Nt, replace = False)
'''

#set_z, set_ra, set_dec = np.r_[z[tt0], lis_z], np.r_[ra[tt0], lis_ra], np.r_[dec[tt0], lis_dec] ## cluster
set_z, set_ra, set_dec = z[tt0], ra[tt0], dec[tt0]

S_crit_0 = 18000  ## big sources
S_crit_1 = 13500  ## median sources
S_crit_2 = 4500   ## small sources
e_crit = 0.85
rho_crit = 6.0e-4
cen_crit = 100    ## avoid choose BCG as a 'bad-feature'

bad_ra, bad_dec, bad_z, bad_bcgx, bad_bcgy = [], [], [], [], []
norm_ra, norm_dec, norm_z, norm_bcgx, norm_bcgy = [], [], [], [], []

for kk in range( len(set_z) ):

	ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

	#file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)
	file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)

	data = fits.open(file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)
	xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	Da_g = Test_model.angular_diameter_distance(z_g).value

	hdu = fits.PrimaryHDU()
	hdu.data = img
	hdu.header = head
	hdu.writeto('test.fits', overwrite = True)

	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'

	#out_load_A = load + 'source_find/mask_ra%.3f_dec%.3f_z%.3f_band-%s.cat' % (ra_g, dec_g, z_g, 'r')
	#out_load_A = load + 'source_find/rand_mask_ra%.3f_dec%.3f_z%.3f_band-%s.cat' % (ra_g, dec_g, z_g, 'r')

	out_load_A = 'test.cat'
	file_source = 'test.fits'

	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_load_A, out_cat)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_A)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])

	### test the concentration of sources
	Area_img = np.array(source['ISOAREAF_IMAGE'])
	S_l0 = np.array(source['ISO0'])
	S_l2 = np.array(source['ISO2'])
	S_l4 = np.array(source['ISO4'])
	S_l6 = np.array(source['ISO6'])
	S_com = np.pi * A * B

	R_l0 = np.sqrt(S_l0 / np.pi)
	R_l2 = np.sqrt(S_l2 / np.pi)
	R_l4 = np.sqrt(S_l4 / np.pi)
	R_l6 = np.sqrt(S_l6 / np.pi)

	ellipty = 1 - B / A

	Kron = 16
	a = Kron * A
	b = Kron * B

	## img identify quantity
	binx = np.arange(0, 2048, 200)
	binx = np.r_[binx, 1848, 2048]
	biny = np.arange(0, 1489, 200)
	biny = np.r_[biny, 1289, 1489]
	grdx = len(binx)
	grdy = len(biny)

	N_densi = np.zeros( (grdy - 1, grdx - 1), dtype = np.float )
	S_block = np.zeros( (grdy - 1, grdx - 1), dtype = np.float )
	## 2D-hist of the source density
	for nn in range( grdy-1 ):
		for tt in range( grdx-1 ):

			if nn == grdy - 3:
				nn += 1
			if tt == grdx - 3:
				tt += 1

			idy = (cy >= biny[nn]) & (cy <= biny[nn + 1])
			idx = (cx >= binx[tt]) & (cx <= binx[tt + 1])
			idm = idy & idx
			N_densi[nn, tt] = np.sum( idm )
			S_block[nn, tt] = (biny[nn + 1] - biny[nn]) * (binx[tt+1] - binx[tt])

	densi = N_densi / S_block

	##### too big source close to cluster
	## about 1Mpc / h to the cluster center
	R_pix = ( (1 / h) * rad2asec / Da_g )
	R_pix = R_pix / pixel

	R_cen = np.sqrt( (cx - xn)**2 + (cy - yn)**2 )
	id_R = (R_cen > cen_crit) & (R_cen <= R_pix)
	id_S = (Area_img >= S_crit_1)
	id_N = np.sum(id_R & id_S)

	## high density + high ellipticity + close to edge
	idu = densi >= rho_crit
	iy, ix = np.where(idu == True)
	n_region, xedg_region, yedg_region = [], [], []
	for nn in range( np.sum(idu) ):
		## just for edge-close sources
		da0 = (ix[nn] == 0) | (ix[nn] == densi.shape[1] - 1)
		da1 = (iy[nn] == 0) | (iy[nn] == densi.shape[0] - 1)

		if (da0 | da1):
			xlo, xhi = binx[ ix[nn] ], binx[ ix[nn] + 1 ]
			ylo, yhi = biny[ iy[nn] ], biny[ iy[nn] + 1 ]
			idx_s = (cx >= xlo) & (cx <= xhi)
			idy_s = (cy >= ylo) & (cy <= yhi)

			sub_ellipty = ellipty[ idx_s & idy_s ]
			sub_size = Area_img[ idx_s & idy_s ]
			id_e = sub_ellipty >= e_crit
			id_s = sub_size >= 50

			n_region.append( np.sum( id_e & id_s ) )
			xedg_region.append( (xlo, xhi) )
			yedg_region.append( (ylo, yhi) )
		else:
			continue

	n_region = np.array(n_region)

	## edges + small sources (at least) / big source
	id_xout = (cx <= 250) | (cx >= img.shape[1] - 250)
	id_yout = (cy <= 250) | (cy >= img.shape[0] - 250)
	id_out = id_xout | id_yout
	sub_ellipty = ellipty[id_out]
	sub_S = Area_img[id_out]
	sub_px = cx[id_out]
	sub_py = cy[id_out]
	sub_A = A[id_out]
	sub_B = B[id_out]
	###: small s_lim + high ellipticity
	id_e = sub_ellipty >= e_crit
	id_s_lo = sub_S >= 50
	id_B = sub_B > 2
	id_g0 = id_e & id_s_lo & id_B
	id_s = sub_S > S_crit_2
	###: high area (pixel^2) sources
	id_s_hi = sub_S > S_crit_0

	###: special source with long major-axis and high ellipticity
	id_e_hi = sub_ellipty >= 0.90
	id_subA = sub_A >= 35
	id_s_medi = sub_S >= 150
	id_extrem = id_e_hi & id_subA & id_s_medi

	n_block, xedg_block, yedg_block = [], [], []

	if (np.sum(id_s) > 0) & (np.sum(id_g0) > 0):
		idv = np.where(id_g0 == True)[0]
		idu = np.where(id_s == True)[0]
		for tt in range( np.sum(id_g0) ):

			xi, yi = sub_px[ idv[tt] ], sub_py[ idv[tt] ]
			block_xi, block_yi = xi // 200, yi // 200
			if xi >= 1848:
				block_xi = binx.shape[0] - 2
			if yi >= 1289:
				block_yi = biny.shape[0] - 2

			for ll in range( np.sum(id_s) ):

				xj, yj = sub_px[ idu[ll] ], sub_py[ idu[ll] ]
				block_xj, block_yj = xj // 200, yj // 200
				if xj >= 1848:
					block_xj = binx.shape[0] - 2
				if yj >= 1289:
					block_yj = biny.shape[0] - 2

				d_nb0 = np.abs( (block_xj - block_xi) ) <= 1.
				d_nb1 = np.abs( (block_yj - block_yi) ) <= 1.

				if (d_nb0 & d_nb1):

					n_block.append( np.sum(d_nb0 & d_nb1) )
					xlo, xhi = binx[ block_xi ], binx[ block_xi + 1]
					ylo, yhi = biny[ block_yi ], biny[ block_yi + 1]
					xedg_block.append( (xlo, xhi) )
					yedg_block.append( (ylo, yhi) )

				else:
					continue

	if np.sum(id_s_hi) > 0:
		idv = np.where(id_s_hi == True)[0]
		for tt in range( np.sum(id_s_hi) ):

			n_block.append( 1 )
			xi, yi = sub_px[ idv[tt] ], sub_py[ idv[tt] ]

			xlo, xhi = binx[ np.int(xi // 200) ], binx[ np.int(xi // 200) + 1]
			ylo, yhi = biny[ np.int(yi // 200) ], biny[ np.int(yi // 200) + 1]

			if xi >= 1848:
				xlo, xhi = binx[-2], binx[-1]
			if yi >= 1289:
				ylo, yhi = biny[-2], biny[-1]

			xedg_block.append( (xlo, xhi) )
			yedg_block.append( (ylo, yhi) )

	if np.sum(id_extrem) > 0:
		idv = np.where(id_extrem == True)[0]
		for tt in range( np.sum(id_extrem) ):

			n_block.append( 1 )
			xi, yi = sub_px[ idv[tt] ], sub_py[ idv[tt] ]

			xlo, xhi = binx[ np.int(xi // 200) ], binx[ np.int(xi // 200) + 1]
			ylo, yhi = biny[ np.int(yi // 200) ], biny[ np.int(yi // 200) + 1]

			if xi >= 1848:
				xlo, xhi = binx[-2], binx[-1]
			if yi >= 1289:
				ylo, yhi = biny[-2], biny[-1]

			xedg_block.append( (xlo, xhi) )
			yedg_block.append( (ylo, yhi) )

	n_block = np.array(n_block)

	## too big sources (no matter where it is)
	max_S_lim = 2.7e4
	id_xin = (cx >= 250) & (cx <= img.shape[1] - 250)
	id_yin = (cy >= 250) & (cy <= img.shape[0] - 250)
	id_in = id_xin & id_yin
	sub_S = Area_img[id_in]
	id_big = sub_S >= max_S_lim

	'''
	if np.sum(id_big) > 0:
		bad_ra.append(ra_g)
		bad_dec.append(dec_g)
		bad_z.append(z_g)
		bad_bcgx.append(xn)
		bad_bcgy.append(yn)

	elif id_N > 0:
		bad_ra.append(ra_g)
		bad_dec.append(dec_g)
		bad_z.append(z_g)
		bad_bcgx.append(xn)
		bad_bcgy.append(yn)

	elif np.sum(n_region) > 0:
		bad_ra.append(ra_g)
		bad_dec.append(dec_g)
		bad_z.append(z_g)
		bad_bcgx.append(xn)
		bad_bcgy.append(yn)

	elif np.sum(n_block) > 0:
		bad_ra.append(ra_g)
		bad_dec.append(dec_g)
		bad_z.append(z_g)
		bad_bcgx.append(xn)
		bad_bcgy.append(yn)

	else:
		norm_ra.append(ra_g)
		norm_dec.append(dec_g)
		norm_z.append(z_g)
		norm_bcgx.append(xn)
		norm_bcgy.append(yn)

	'''
	densi = np.delete(densi, -2, axis = 0)
	densi = np.delete(densi, -2, axis = 1)

	if np.sum(id_big) > 0: ## too big source

		plt.figure( figsize = (12, 6) )
		ax = plt.subplot(121)
		bx = plt.subplot(122)
		ax.set_title('sources ra%.3f dec%.3f z%.3f [max = %.1f]' % (ra_g, dec_g, z_g, Area_img.max() ) )
		ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

		for mm in range( Numb ):
			ellips = Ellipse(xy = (cx[mm], cy[mm]), width = a[mm], height = b[mm], angle = theta[mm], fill = False, 
				ec = 'g', ls = '-', linewidth = 1, alpha = 0.45,)
			ax.add_patch(ellips)

		idxm = np.where(id_big == True)[0]
		for mm in range( np.sum(id_big) ):

			ellips = Ellipse(xy = (cx[ idxm[mm] ], cy[ idxm[mm] ]), width = a[ idxm[mm] ], height = b[ idxm[mm] ], 
				angle = theta[ idxm[mm] ], fill = False, ec = 'm', ls = '-', linewidth = 1, alpha = 0.5,)
			ax.add_patch(ellips)

		clust = Circle(xy = (xn, yn), radius = R_pix, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(clust)

		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])

		bx.set_title('Number density')
		tf = bx.imshow(densi, origin = 'lower', cmap = 'rainbow', )
		plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, label = '$ \\rho_{N} [N / pix^2] $')
		plt.tight_layout()
		plt.savefig('bad-img_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
		plt.close()

	elif id_N > 0: ## median size source

		plt.figure( figsize = (12, 6) )
		ax = plt.subplot(121)
		bx = plt.subplot(122)
		ax.set_title('sources ra%.3f dec%.3f z%.3f [max = %.1f]' % (ra_g, dec_g, z_g, Area_img.max() ) )
		ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

		for mm in range( Numb ):
			ellips = Ellipse(xy = (cx[mm], cy[mm]), width = a[mm], height = b[mm], angle = theta[mm], fill = False, 
				ec = 'g', ls = '-', linewidth = 1, alpha = 0.45,)
			ax.add_patch(ellips)

		idv = Area_img == Area_img.max()
		ellips = Ellipse(xy = (cx[idv], cy[idv]), width = a[idv], height = b[idv], angle = theta[idv], fill = False, 
			ec = 'r', ls = '-', linewidth = 1, alpha = 0.45,)
		ax.add_patch(ellips)

		idxm = np.where(id_N == True)
		for mm in range( np.sum(id_N) ):
			ellips = Ellipse(xy = (cx[ idxm[mm] ], cy[ idxm[mm] ]), width = a[ idxm[mm] ], height = b[ idxm[mm] ], 
				angle = theta[ idxm[mm] ], fill = False, ec = 'm', ls = '-', linewidth = 1, alpha = 0.5,)
			ax.add_patch(ellips)

		clust = Circle(xy = (xn, yn), radius = R_pix, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(clust)

		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])

		bx.set_title('Number density')
		tf = bx.imshow(densi, origin = 'lower', cmap = 'rainbow', )
		plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, label = '$ \\rho_{N} [N / pix^2] $')
		plt.tight_layout()
		plt.savefig('bad-img_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
		plt.close()

	elif np.sum(n_region) > 0: ## edges with high rho_n, and ellipticity

		plt.figure( figsize = (12, 6) )
		ax = plt.subplot(121)
		bx = plt.subplot(122)
		ax.set_title('sources ra%.3f dec%.3f z%.3f [max = %.1f]' % (ra_g, dec_g, z_g, Area_img.max() ) )
		ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

		for mm in range( Numb ):
			ellips = Ellipse(xy = (cx[mm], cy[mm]), width = a[mm], height = b[mm], angle = theta[mm], fill = False, 
				ec = 'g', ls = '-', linewidth = 1, alpha = 0.45,)
			ax.add_patch(ellips)

		clust = Circle(xy = (xn, yn), radius = R_pix, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(clust)

		idm = n_region > 0
		idxm = np.where(idm == True)[0]

		for mm in range( np.sum(idm) ):
			a0, a1 = xedg_region[ idxm[mm] ][0], xedg_region[ idxm[mm] ][1]
			b0, b1 = yedg_region[ idxm[mm] ][0], yedg_region[ idxm[mm] ][1]
			region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, ec = 'm', linewidth = 1, alpha = 0.5)
			ax.add_patch(region)

		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])

		bx.set_title('Number density')
		tf = bx.imshow(densi, origin = 'lower', cmap = 'rainbow', )
		plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, label = '$ \\rho_{N} [N / pix^2] $')
		plt.tight_layout()
		plt.savefig('bad-img_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
		plt.close()

	elif np.sum(n_block) > 0: ## edge source + high ellipticity

		plt.figure( figsize = (12, 6) )
		ax = plt.subplot(121)
		bx = plt.subplot(122)
		ax.set_title('sources ra%.3f dec%.3f z%.3f [max = %.1f]' % (ra_g, dec_g, z_g, Area_img.max() ) )
		ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

		for mm in range( Numb ):
			ellips = Ellipse(xy = (cx[mm], cy[mm]), width = a[mm], height = b[mm], angle = theta[mm], fill = False, 
				ec = 'g', ls = '-', linewidth = 1, alpha = 0.45,)
			ax.add_patch(ellips)

		clust = Circle(xy = (xn, yn), radius = R_pix, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(clust)

		idm = n_block > 0
		idxm = np.where(idm == True)[0]

		for mm in range( np.sum(idm) ):
			a0, a1 = xedg_block[ idxm[mm] ][0], xedg_block[ idxm[mm] ][1]
			b0, b1 = yedg_block[ idxm[mm] ][0], yedg_block[ idxm[mm] ][1]
			region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, ec = 'm', linewidth = 1, alpha = 0.5)
			ax.add_patch(region)

		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])

		bx.set_title('Number density' )
		tf = bx.imshow(densi, origin = 'lower', cmap = 'rainbow', )
		plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, label = '$ \\rho_{N} [N / pix^2] $')
		plt.tight_layout()
		plt.savefig('bad-img_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
		plt.close()

	else:
		plt.figure( figsize = (12, 6) )
		ax = plt.subplot(121)
		bx = plt.subplot(122)
		ax.set_title('sources ra%.3f dec%.3f z%.3f [max = %.1f]' % (ra_g, dec_g, z_g, Area_img.max() ) )
		ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())

		for mm in range( Numb ):
			ellips = Ellipse(xy = (cx[mm], cy[mm]), width = a[mm], height = b[mm], angle = theta[mm], fill = False, 
				ec = 'g', ls = '-', linewidth = 1, alpha = 0.45,)
			ax.add_patch(ellips)

		clust = Circle(xy = (xn, yn), radius = R_pix, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(clust)

		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])

		bx.set_title('Number density' )
		tf = bx.imshow(densi, origin = 'lower', cmap = 'rainbow', )
		plt.colorbar(tf, ax = bx, fraction = 0.035, pad = 0.01, label = '$ \\rho_{N} [N / pix^2] $')
		plt.tight_layout()
		plt.savefig('norm-img_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
		plt.close()

raise
"""
x_ra = np.array( bad_ra )
x_dec = np.array( bad_dec )
x_z = np.array( bad_z )
x_xn = np.array( bad_bcgx )
x_yn = np.array( bad_bcgy )
keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
values = [x_ra, x_dec, x_z, x_xn, x_yn]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
#data.to_csv('cluster_bad_sample.csv')
data.to_csv('random_bad_sample.csv')

x_ra = np.array( norm_ra )
x_dec = np.array( norm_dec )
x_z = np.array( norm_z )
x_xn = np.array( norm_bcgx )
x_yn = np.array( norm_bcgy )
keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
values = [x_ra, x_dec, x_z, x_xn, x_yn]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
#data.to_csv('cluster_norm_sample.csv')
data.to_csv('random_norm_sample.csv')
"""
