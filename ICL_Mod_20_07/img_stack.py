import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

def stack_func(d_file, out_file, z_set, ra_set, dec_set, band, img_x, img_y, id_cen):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img (include file-name structure:'/xxx/xxx/xxx.xxx')
	img_x, img_y : BCG position (in image coordinate)
	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center	
	"""

	stack_N = len(z_set)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float32) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)

	for jj in range(stack_N):

		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]
		xn, yn = img_x[jj], img_y[jj]

		file = d_file % (band, ra_g, dec_g, z_g)
		data_A = fits.open( file )
		img_A = data_A[0].data
		head = data_A[0].header

		if id_cen == 0:
			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + img_A.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + img_A.shape[1])

		if id_cen == 1:
			rnx, rny = np.int(img_A.shape[1] / 2), np.int(img_A.shape[0] / 2) ## image center
			la0 = np.int(y0 - rny)
			la1 = np.int(y0 - rny + img_A.shape[0])
			lb0 = np.int(x0 - rnx)
			lb1 = np.int(x0 - rnx + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan
	
	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	with h5py.File(out_file, 'w') as f:
		f['a'] = np.array(stack_img)

	return

def main():

	band = 'r'

	## random img
	dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_norm_sample.csv')
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	dat_out = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_bad_sample.csv')
	bad_ra, bad_dec, bad_z = np.array(dat_out.ra), np.array(dat_out.dec), np.array(dat_out.z)
	bad_bcgx, bad_bcgy = np.array(dat_out.bcg_x), np.array(dat_out.bcg_y)

	#ra, dec, z = np.r_[ra, bad_ra], np.r_[dec, bad_dec], np.r_[z, bad_z]
	#clus_x, clus_y = np.r_[clus_x, bad_bcgx], np.r_[clus_y, bad_bcgy]

	id_cen = 1

	d_file = '/media/xkchen/My Passport/data/SDSS/re_mask/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
	out_file = '/home/xkchen/Downloads/test_imgs/random/random_select_center-stack_test.h5'
	stack_func(d_file, out_file, z, ra, dec, band, clus_x, clus_y, id_cen)

	## cluster img
	dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/cluster_norm_sample.csv')
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	dat_out = pds.read_csv('/home/xkchen/Downloads/test_imgs/cluster_bad_sample.csv')
	bad_ra, bad_dec, bad_z = np.array(dat_out.ra), np.array(dat_out.dec), np.array(dat_out.z)
	bad_bcgx, bad_bcgy = np.array(dat_out.bcg_x), np.array(dat_out.bcg_y)

	#ra, dec, z = np.r_[ra, bad_ra], np.r_[dec, bad_dec], np.r_[z, bad_z]
	#clus_x, clus_y = np.r_[clus_x, bad_bcgx], np.r_[clus_y, bad_bcgy]

	id_cen = 1
	d_file = '/media/xkchen/My Passport/data/SDSS/re_mask/cluster/mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'
	out_file = '/home/xkchen/Downloads/test_imgs/cluster/cluster_select_center-stack_test.h5'
	stack_func(d_file, out_file, z, ra, dec, band, clus_x, clus_y, id_cen)

if __name__ == "__main__":
	main()
