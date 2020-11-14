import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import astropy.constants as C
## from pipeline
from groups import groups_find_func
from light_measure import cc_grid_img, grid_img

def diffuse_identi_func(band, set_ra, set_dec, set_z, data_file, rule_out_file, remain_file, thres_S0, thres_S1, sigm_lim, 
	mu_sigm_file, id_single = True, id_mode = False,):
	"""
	band : observation band
	set_ra, set_dec, set_z : smaples need to find out-liers
	data_file : the observational imgs,(format: 'XXXX/XXX/XXX.XXX') and those imgs
				have been masked
	{out-put file : XXX.csv
	rule_out_file : the file name of out-put catalogue for exclude imgs (not include in stacking)
	remain_file : the file name of out-put catalogue for stacking imgs (stacking imgs) }

	thres_S0, thres_S1, sigm_lim : condition for selection, thres_S0, thres_S1 are block number limit, 
	and sigm_lim is brightness limit.
	id_single : bool, True : select imgs based on single img (mu_cen, sigma_cen), default is True,
					  False : select imgs based on sample imgs Mode(mu_cen, sigma_cen), or mean(mu_cen, sigma_cen)
	( mode(x) = 3 * median(x) - 2 * mean(x) )
	mu_sigm_file : file name which saved the (mu_cen, sigma_cen) of given img sample ('/XXX/XXX/XXX.csv')

	id_single : True -- select imgs based on single img 2D flux histogram;
				False -- select imgs based on average img 2D flux histogram of given img sample
	id_mode : for selecting imgs based on img sample properties,
	"""
	bad_ra, bad_dec, bad_z, bad_bcgx, bad_bcgy = [], [], [], [], []
	norm_ra, norm_dec, norm_z, norm_bcgx, norm_bcgy = [], [], [], [], []

	for kk in range( len(set_z) ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

		file = data_file % (band, ra_g, dec_g, z_g)
		data = fits.open( file )
		img = data[0].data
		head = data[0].header
		wcs_lis = awc.WCS(head)
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		remain_img = img.copy()

		ca0, ca1 = np.int(img.shape[0] / 2), np.int(img.shape[1] / 2)
		cen_D = 500
		flux_cen = remain_img[ca0 - cen_D: ca0 + cen_D, ca1 - cen_D: ca1 + cen_D]

		N_step = 200
		sub_pock_flux, sub_pock_pix = grid_img(flux_cen, N_step, N_step)[:2]

		## mu, sigma of center region
		if id_single == True:
			id_Nzero = sub_pock_pix > 100
			mu = np.nanmean( sub_pock_flux[id_Nzero] )
			sigm = np.nanstd( sub_pock_flux[id_Nzero] )
		else:
			## use sample mean and scatter
			samp_dat = pds.read_csv(mu_sigm_file)
			img_mu, img_sigma = np.array(samp_dat['img_mu']), np.array(samp_dat['img_sigma'])

			if id_mode == True:
				mu = 3 * np.median(img_mu) - 2 * np.mean(img_mu)
				sigm = 3 * np.median(img_sigma) - 2 * np.mean(img_sigma)
			else:
				mu = np.mean(img_mu)
				sigm = np.mean(img_sigma)

		patch_grd = cc_grid_img(remain_img, N_step, N_step)
		patch_mean = patch_grd[0]
		patch_pix = patch_grd[1]
		lx, ly = patch_grd[-2], patch_grd[-1]

		id_zeros = patch_pix == 0.
		patch_pix[id_zeros] = np.nan
		patch_mean[id_zeros] = np.nan
		over_sb = (patch_mean - mu) / sigm

		##### img selection
		lim_sb = sigm_lim

		### first select
		identi = over_sb > lim_sb

		if np.sum(identi) < 1:
			norm_ra.append(ra_g)
			norm_dec.append(dec_g)
			norm_z.append(z_g)
			norm_bcgx.append(xn)
			norm_bcgy.append(yn)

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
			idu = idux | iduy ## search for blocks around image edges
			#idu = True ## search for blocks in total image frame

			### select groups with block number larger or equal to 3
			idv_s = (np.array(source_n) >= thres_S0)
			id_pat_s = idu & idv_s

			if np.sum(id_pat_s) < 1:
				norm_ra.append(ra_g)
				norm_dec.append(dec_g)
				norm_z.append(z_g)
				norm_bcgx.append(xn)
				norm_bcgy.append(yn)

			else:
				id_vs_pri = (np.array(source_n) <= thres_S1)
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
					id_2_bright = tmp_arr > 8.0 # 9.5 ## groups must have a 'to bright region'

					pur_N[ll] = tot_pont - np.sum(id_out)
					pur_mask[ll] = np.sum(id_out)

					pur_outlier[ll] = np.sum(id_2_bright) * ( np.sum(id_out) == 0)

				## at least 2 blocks have mean value above the lim_sb and close to a big mask region
				idnum = ( (pur_N >= 1) & (pur_mask >= 1) ) | (pur_outlier >= 1)

				if np.sum(idnum) >= 1:
					bad_ra.append(ra_g)
					bad_dec.append(dec_g)
					bad_z.append(z_g)
					bad_bcgx.append(xn)
					bad_bcgy.append(yn)

				else:
					## search for larger groups
					# each group include 5 patches at least
					idv = np.array(source_n) >= thres_S1
					id_pat = idu & idv

					if np.sum(id_pat) < 1:
						norm_ra.append(ra_g)
						norm_dec.append(dec_g)
						norm_z.append(z_g)
						norm_bcgx.append(xn)
						norm_bcgy.append(yn)

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
							norm_ra.append(ra_g)
							norm_dec.append(dec_g)
							norm_z.append(z_g)
							norm_bcgx.append(xn)
							norm_bcgy.append(yn)

						else:
							bad_ra.append(ra_g)
							bad_dec.append(dec_g)
							bad_z.append(z_g)
							bad_bcgx.append(xn)
							bad_bcgy.append(yn)
	### 'bad' imgs
	x_ra = np.array( bad_ra )
	x_dec = np.array( bad_dec )
	x_z = np.array( bad_z )
	x_xn = np.array( bad_bcgx )
	x_yn = np.array( bad_bcgy )
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [x_ra, x_dec, x_z, x_xn, x_yn]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(rule_out_file)
	### normal imgs
	x_ra = np.array( norm_ra )
	x_dec = np.array( norm_dec )
	x_z = np.array( norm_z )
	x_xn = np.array( norm_bcgx )
	x_yn = np.array( norm_bcgy )
	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [x_ra, x_dec, x_z, x_xn, x_yn]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(remain_file)

	return

def main():

	import time

	from mpi4py import MPI
	commd = MPI.COMM_WORLD
	rank = commd.Get_rank()
	cpus = commd.Get_size()

	band = ['r', 'g', 'i']

	home = '/media/xkchen/My Passport/data/SDSS/'
	#home = '/home/xkchen/data/SDSS/'

	thres_S0, thres_S1 = 3, 5
	sigma = np.array([3.5, 4, 4.5, 5, 5.5, 6,]) ## sigma as limit
	'''
	thres_S0 = np.array([2, 3, 4]) ## block number as limit
	thres_S1 = 5
	sigma = 3.5 # 4
	'''
	m, n = divmod( len(sigma), cpus)
	#m, n = divmod( len(thres_S0), cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	##### cluster imgs
	dat = pds.read_csv(home + 'selection/tmp/cluster_tot-r-band_norm-img_cat.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	d_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

	## sigma as limit
	rule_file = home + 'selection/tmp/tot_clust_rule-out_cat_%.1f-sigma.csv' % (sigma[N_sub0: N_sub1][0])
	remain_file = home + 'selection/tmp/tot_clust_remain_cat_%.1f-sigma.csv' % (sigma[N_sub0: N_sub1][0])
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, 
		thres_S1, sigma[N_sub0: N_sub1][0],)
	'''
	## block number as limit
	rule_file = home + 'selection/tmp/tot_clust_rule-out_cat_%d-patch_%.1f-sigm.csv' % (thres_S0[N_sub0: N_sub1][0], sigma)
	remain_file = home + 'selection/tmp/tot_clust_remain_cat_%d-patch_%.1f-sigm.csv' % (thres_S0[N_sub0: N_sub1][0], sigma)
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0[N_sub0: N_sub1][0], thres_S1, sigma,)
	'''
	print('cluster finished!')
	"""
	##### random imgs
	dat = pds.read_csv(home + 'selection/tmp/random_tot-r-band_norm-img_cat.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	d_file = home + 'tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'	
	''''
	## sigma as limit
	rule_file = home + 'selection/tmp/tot_random_rule-out_cat_%.1f-sigma.csv' % (sigma[N_sub0: N_sub1][0])
	remain_file = home + 'selection/tmp/tot_random_remain_cat_%.1f-sigma.csv' % (sigma[N_sub0: N_sub1][0])
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0, 
		thres_S1, sigma[N_sub0: N_sub1][0],)
	'''
	## block number as limit
	rule_file = home + 'selection/tmp/tot_random_rule-out_cat_%d-patch_4-sigm.csv' % (thres_S0[N_sub0: N_sub1][0] )
	remain_file = home + 'selection/tmp/tot_random_remain_cat_%d-patch_4-sigm.csv' % (thres_S0[N_sub0: N_sub1][0] )
	diffuse_identi_func(band[0], set_ra, set_dec, set_z, d_file, rule_file, remain_file, thres_S0[N_sub0: N_sub1][0], thres_S1, sigma,)

	print('random finished!')
	"""

if __name__ == "__main__":
	main()
