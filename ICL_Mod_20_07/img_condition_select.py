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

### cosmology model
rad2asec = U.rad.to( U.arcsec )
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

def img_condition_select_func(band, set_ra, set_dec, set_z, data_file, cat_file, rule_out_file, remain_file, pixel = 0.396):
	"""
	band : observation band
	set_ra, set_dec, set_z : smaples need to find out-liers
	data_file : the observational imgs (before applying masking)
	{out-put file : XXX.csv
	rule_out_file : the file name of out-put catalogue for exclude imgs (not include in stacking)
	remain_file : the file name of out-put catalogue for stacking imgs (stacking imgs) }
	pixel : pixel scale in unit of arcsecond (default is 0.396, SDSS case)
	cat_file : the source catalog based on SExTractor calculation
	"""
	max_S_lim = 2.7e4
	S_crit_0 = 18000  ## big sources
	#S_crit_0 = 13700  ## median sources
	S_crit_2 = 4500   ## small sources
	e_crit = 0.85
	rho_crit = 6.0e-4
	cen_crit = 100    ## avoid choose BCG as a 'bad-feature'

	bad_ra, bad_dec, bad_z, bad_bcgx, bad_bcgy = [], [], [], [], []
	norm_ra, norm_dec, norm_z, norm_bcgx, norm_bcgy = [], [], [], [], []

	for kk in range( len(set_z) ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

		data = fits.open(data_file % (band, ra_g, dec_g, z_g) )
		img = data[0].data
		head = data[0].header
		wcs_lis = awc.WCS(head)
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		Da_g = Test_model.angular_diameter_distance(z_g).value

		source = asc.read(cat_file % (band, ra_g, dec_g, z_g), )
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE'])
		cy = np.array(source['Y_IMAGE'])
		p_type = np.array(source['CLASS_STAR'])

		### test the concentration of sources
		Area_img = np.array(source['ISOAREAF_IMAGE'])
		ellipty = 1 - B / A

		Kron = 16
		a = Kron * A
		b = Kron * B

		## img identify quantity
		N_step = 200

		biny = np.arange(0, img.shape[0], N_step)
		biny = np.r_[biny, img.shape[0] - N_step, img.shape[0] ]
		binx = np.arange(0, img.shape[1], N_step)
		binx = np.r_[binx, img.shape[1] - N_step, img.shape[1] ]

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

		densi = np.delete(densi, -2, axis = 0)
		densi = np.delete(densi, -2, axis = 1)
		binx = np.delete(binx, -3)
		binx = np.delete(binx, -1)
		biny = np.delete(biny, -3)
		biny = np.delete(biny, -1)

		##### too big source close to cluster
		## about 1Mpc / h to the cluster center
		R_pix = ( (1 / h) * rad2asec / Da_g )
		R_pix = R_pix / pixel

		R_cen = np.sqrt( (cx - xn)**2 + (cy - yn)**2 )
		id_R = (R_cen > cen_crit) & (R_cen <= R_pix)
		id_S = (Area_img >= S_crit_0)
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
				xlo, xhi = binx[ ix[nn] ], binx[ ix[nn] ] + N_step
				ylo, yhi = biny[ iy[nn] ], biny[ iy[nn] ] + N_step
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
		id_xout = (cx <= 400) | (cx >= img.shape[1] - 400)
		id_yout = (cy <= 400) | (cy >= img.shape[0] - 400)
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
				block_xi, block_yi = np.int(xi // 200), np.int(yi // 200)

				if xi >= 1848:
					block_xi = binx.shape[0] - 1
				if yi >= 1289:
					block_yi = biny.shape[0] - 1

				for ll in range( np.sum(id_s) ):

					xj, yj = sub_px[ idu[ll] ], sub_py[ idu[ll] ]
					block_xj, block_yj = np.int(xj // 200), np.int(yj // 200)
					if xj >= 1848:
						block_xj = binx.shape[0] - 1
					if yj >= 1289:
						block_yj = biny.shape[0] - 1

					d_nb0 = np.abs( (block_xj - block_xi) ) <= 1.
					d_nb1 = np.abs( (block_yj - block_yi) ) <= 1.

					if (d_nb0 & d_nb1):

						n_block.append( np.sum(d_nb0 & d_nb1) )
						xlo, xhi = binx[ block_xi ], binx[ block_xi ] + N_step
						ylo, yhi = biny[ block_yi ], biny[ block_yi ] + N_step
						xedg_block.append( (xlo, xhi) )
						yedg_block.append( (ylo, yhi) )

					else:
						continue

		if np.sum(id_s_hi) > 0:
			idv = np.where(id_s_hi == True)[0]
			for tt in range( np.sum(id_s_hi) ):

				n_block.append( 1 )
				xi, yi = sub_px[ idv[tt] ], sub_py[ idv[tt] ]

				xlo, xhi = binx[ np.int(xi // 200) ], binx[ np.int(xi // 200) ] + N_step
				ylo, yhi = biny[ np.int(yi // 200) ], biny[ np.int(yi // 200) ] + N_step

				if xi >= 1848:
					xlo, xhi = binx[-1], binx[-1] + N_step
				if yi >= 1289:
					ylo, yhi = biny[-1], biny[-1] + N_step

				xedg_block.append( (xlo, xhi) )
				yedg_block.append( (ylo, yhi) )

		if np.sum(id_extrem) > 0:
			idv = np.where(id_extrem == True)[0]
			for tt in range( np.sum(id_extrem) ):

				n_block.append( 1 )
				xi, yi = sub_px[ idv[tt] ], sub_py[ idv[tt] ]

				xlo, xhi = binx[ np.int(xi // 200) ], binx[ np.int(xi // 200) ] + N_step
				ylo, yhi = biny[ np.int(yi // 200) ], biny[ np.int(yi // 200) ] + N_step

				if xi >= 1848:
					xlo, xhi = binx[-1], binx[-1] + N_step
				if yi >= 1289:
					ylo, yhi = biny[-1], biny[-1] + N_step

				xedg_block.append( (xlo, xhi) )
				yedg_block.append( (ylo, yhi) )

		n_block = np.array(n_block)

		## too big sources (no matter where it is)
		id_xin = (cx >= 400) & (cx <= img.shape[1] - 400)
		id_yin = (cy >= 400) & (cy <= img.shape[0] - 400)
		id_in = id_xin & id_yin
		sub_S = Area_img[id_in]
		id_big = sub_S >= max_S_lim

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

	x_ra = np.array( bad_ra )
	x_dec = np.array( bad_dec )
	x_z = np.array( bad_z )
	x_xn = np.array( bad_bcgx )
	x_yn = np.array( bad_bcgy )
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [x_ra, x_dec, x_z, x_xn, x_yn]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( rule_out_file )

	x_ra = np.array( norm_ra )
	x_dec = np.array( norm_dec )
	x_z = np.array( norm_z )
	x_xn = np.array( norm_bcgx )
	x_yn = np.array( norm_bcgy )
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [x_ra, x_dec, x_z, x_xn, x_yn]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( remain_file )

def main():

	import time

	band = ['r', 'g', 'i']
	home = '/media/xkchen/My Passport/data/SDSS/'

	t0 = time.time()

	dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_select/r_band_sky_catalog.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	cat_file = '/home/xkchen/mywork/ICL/data/source_find/cluster_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	data_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	rule_file = 'cluster_tot-r-band_bad-img_cat.csv'
	remain_file = 'cluster_tot-r-band_norm-img_cat.csv'
	img_condition_select_func(band[0], set_ra, set_dec, set_z, data_file, cat_file, rule_file, remain_file,)

	t1 = time.time() - t0
	print(t1)

	dat = pds.read_csv('/home/xkchen/mywork/ICL/data/cat_select/rand_r_band_catalog.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	cat_file = '/home/xkchen/mywork/ICL/data/source_find/random_%s-band_mask_ra%.3f_dec%.3f_z%.3f.cat'
	data_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'

	rule_file = 'random_tot-r-band_bad-img_cat.csv'
	remain_file = 'random_tot-r-band_norm-img_cat.csv'
	img_condition_select_func(band[0], set_ra, set_dec, set_z, data_file, cat_file, rule_file, remain_file,)

	raise

if __name__ == "__main__":
	main()

