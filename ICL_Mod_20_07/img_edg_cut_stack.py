import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

def cut_stack_func(d_file, out_file, z_set, ra_set, dec_set, band, img_x, img_y, id_cen, N_edg, rms_file = None, pix_con_file = None):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img (include file-name structure:'/xxx/xxx/xxx.xxx')
	img_x, img_y : BCG position (in image coordinate)
	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center
	rms_file : stacking img pixel variance 
	pix_con_file : the pixel counts in each stacking img pixel
	N_edg : the width of the edge region, pixels in this region will be set as 
			'no flux' contribution pixels (ie. set as np.nan)
	"""
	stack_N = len(z_set)

	if id_cen == 0:
		x0, y0 = 2427, 1765
		Nx = np.linspace(0, 4854, 4855)
		Ny = np.linspace(0, 3530, 3531)

		sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float32) * np.nan
		p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		pix_f2 = np.zeros((len(Ny), len(Nx)), dtype = np.float32)

	if id_cen == 1:
		x0, y0 = 1024, 744
		Nx = np.linspace(0, 2047, 2048)
		Ny = np.linspace(0, 1488, 1489)

		sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float32) * np.nan
		p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		pix_f2 = np.zeros((len(Ny), len(Nx)), dtype = np.float32)

	for jj in range(stack_N):

		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		dev_05_x = img_x[jj] - np.int(img_x[jj])
		dev_05_y = img_y[jj] - np.int(img_y[jj])
		if dev_05_x > 0.5:
			xn = np.int(img_x[jj]) + 1
		else:
			xn = np.int(img_x[jj])

		if dev_05_y > 0.5:
			yn = np.int(img_y[jj]) + 1
		else:
			yn = np.int(img_y[jj])

		file = d_file % (band, ra_g, dec_g, z_g)
		data_A = fits.open( file )
		img_A = data_A[0].data
		head = data_A[0].header

		## mask the edge region with N_edg
		img_A[:N_edg, :] = np.nan
		img_A[-N_edg:, :] = np.nan
		img_A[:, :N_edg] = np.nan
		img_A[:, -N_edg:] = np.nan

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

		## tmp array for rms
		pix_f2[la0: la1, lb0: lb1][idv] = pix_f2[la0: la1, lb0: lb1][idv] + img_A[idv]**2

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

	## pixel rms
	pix_f2[id_zero] = np.nan
	rms_arr = pix_f2 / p_count_A - stack_img**2
	rms_arr = np.sqrt(rms_arr)

	if rms_file != None:
		with h5py.File(rms_file, 'w') as f:
			f['a'] = np.array(rms_arr)

	if pix_con_file != None:
		with h5py.File(pix_con_file, 'w') as f:
			f['a'] = np.array(p_count_A)

	return

