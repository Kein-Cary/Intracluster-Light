import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

def sky_stack_func(d_file, out_file, z_set, ra_set, dec_set, band, img_x, img_y, id_cen, id_mean):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img (include file-name structure:'/xxx/xxx/xxx.xxx')
	img_x, img_y : BCG position (in image coordinate)
	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center
	for sky imgs, id_cen can be 2 -- means 'random center stacking'
	id_mean : 0, 1, 2.  0 - img_add = img; 
	1 - img_add = img - np.mean(img); 2 - img_add = img - np.median(img)
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

		if id_cen == 2:
			rnx, rny = np.random.choice(img_A.shape[1], 1, replace = False), np.random.choice(img_A.shape[0], 1, replace = False)
			la0 = np.int(y0 - rny)
			la1 = np.int(y0 - rny + img_A.shape[0])
			lb0 = np.int(x0 - rnx)
			lb1 = np.int(x0 - rnx + img_A.shape[1])

		if id_mean == 0:
			img_add = img_A - 0.
		if id_mean == 1:
			img_add = img_A - np.nanmean(img_A)
		if id_mean == 2:
			img_add = img_A - np.nanmedian(img_A)

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_add[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_add[idv]
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

	dat = pds.read_csv('/home/xkchen/mywork/ICL/r_band_sky_catalog.csv')
	ra, dec, z = dat.ra, dat.dec, dat.z
	Nz = 10
	set_ra, set_dec, set_z = ra[:10], dec[:10], z[:10]

	d_file = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/sky-%s-band-ra%.3f-dec%.3f-z%.3f.fits'
	out_file = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/test_sky_stack.h5'
	band = 'r'
	id_mean = 0
	id_cen = 2
	clus_x = np.ones(10, dtype = np.float32)
	clus_y = np.ones(10, dtype = np.float32)
	sky_stack_func(d_file, out_file, set_z, set_ra, set_dec, band, clus_x, clus_y, id_cen, id_mean)

if __name__ == "__main__":
	main()
