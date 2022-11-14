import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits


##. partial cut stacking image
def part_cut_func( img_arr, bcg_x, bcg_y, bcg_PA, sat_x, sat_y, id_half = 'half', sat_PA = None, sat_ar = None ):
	"""
	img_arr : image array to be stacked
	bcg_x, bcg_y : the position of centeral galaxy
	bcg_PA : centeral galaxy position angle~(relative to the longer side of image frame -- 2048 pixels)
	sat_x, sat_y : the position of satellite

	----------------------------------------
	id_half : string type, 'half', 'half-V', 'angle', by default is 'half'
			'angle' -- pixels align the direction of target satellite will be masked, the angle size is determin by
						distance to the associated BCG and source detection size of target objects
			'half' -- along the major-axis of BCG, all pixels on the same side of the target object will be removed
			'half-V' -- again, all pixels on the same side of the target object will be removed, but the line is 
						perpendicular to the line of target object and BCg central point

	sat_PA : the position angle of the target satellite~(relative to the longer side of image frame -- 2048 pixels)
	sat_ar : the semi-major axis of satellite (in unit of pixel), given by source-extractor
	"""

	copy_img = img_arr.copy()

	### === half_BCG cut
	if id_half == 'half':

		k_k1 = np.tan( bcg_PA )
		k_b1 = bcg_y - k_k1 * bcg_x

		tag_dy = sat_y - ( k_k1 * sat_x + k_b1 )

		#.
		tt_Ny, tt_Nx = img_arr.shape
		xpt = np.linspace( 0, tt_Nx - 1, tt_Nx )
		ypt = np.linspace( 0, tt_Ny - 1, tt_Ny )

		grid_nx, grid_ny = np.meshgrid( xpt, ypt )

		#.
		delt_y = grid_ny - ( k_k1 * grid_nx + k_b1 )
		sign_y = tag_dy * delt_y

		id_sign = sign_y > 0.
		copy_img[ id_sign ] = np.nan

	### === partical cut
	if id_half == 'angle':

		##. lines align major-axis of BCG
		k1 = np.tan( bcg_PA )
		b1 = bcg_y - k1 * bcg_x

		##. lines align major-axis of satellite
		sat_chi = sat_PA
		cen_off_x = sat_ar * np.cos( sat_chi )

		#.
		k2 = np.tan( sat_chi )
		b2 = sat_y - k2 * sat_x

		lx_0 = sat_x - cen_off_x
		lx_1 = sat_x + cen_off_x

		ly_0 = k2 * lx_0 + b2
		ly_1 = k2 * lx_1 + b2

		#. lines link to BCG center
		kl_0 = ( ly_0 - bcg_y ) / ( lx_0 - bcg_x )
		bl_0 = ly_0 - lx_0 * kl_0

		kl_1 = ( ly_1 - bcg_y ) / ( lx_1 - bcg_x )
		bl_1 = ly_1 - lx_1 * kl_1


		##. identify the relative position of points to the major-axis
		tt_Ny, tt_Nx = img_arr.shape
		xpt = np.linspace( 0, tt_Nx - 1, tt_Nx )
		ypt = np.linspace( 0, tt_Ny - 1, tt_Ny )

		grid_nx, grid_ny = np.meshgrid( xpt, ypt )

		##.
		dy = sat_y - ( k1 * sat_x + b1 )
		delt_y = grid_ny - ( k1 * grid_nx + b1 )
		sign_y = dy * delt_y
		id_sign = sign_y > 0.


		dy_0 = sat_y - ( kl_0 * sat_x + bl_0 )
		delt_y0 = grid_ny - ( kl_0 * grid_nx + bl_0 )
		sign_y0 = dy_0 * delt_y0
		id_sign_0 = sign_y0 > 0.


		dy_1 = sat_y - ( kl_1 * sat_x + bl_1 )
		delt_y1 = grid_ny - ( kl_1 * grid_nx + bl_1 )
		sign_y1 = dy_1 * delt_y1
		id_sign_1 = sign_y1 > 0.

		#.
		id_msx = id_sign * id_sign_0 * id_sign_1
		copy_img[ id_msx ] = np.nan

	##...
	if id_half == 'half-V':

		##. lines align major-axis of BCG
		k1 = np.tan( bcg_PA )
		b1 = bcg_y - k1 * bcg_x

		##. line link satellite and BCG
		k2 = (sat_y - bcg_y) / (sat_x - bcg_x)
		b2 = sat_y - sat_x * k2
		# l2 = k2 * xx_1 + b2

		##. perpendocular to l2
		k3 = -1/k2
		b3 = bcg_y - k3 * bcg_x

		##. identify the relative position of points to the major-axis
		tt_Ny, tt_Nx = img_arr.shape
		xpt = np.linspace( 0, tt_Nx - 1, tt_Nx )
		ypt = np.linspace( 0, tt_Ny - 1, tt_Ny )

		grid_nx, grid_ny = np.meshgrid( xpt, ypt )

		#.
		dy = sat_y - ( k3 * sat_x + b3 )

		delt_y = grid_ny - ( k3 * grid_nx + b3 )
		sign_y = dy * delt_y

		id_sign = sign_y > 0.

		copy_img = img_arr.copy()
		copy_img[ id_sign ] = np.nan

	return copy_img


##.
def cut_stack_func( d_file, out_file, z_set, ra_set, dec_set, band, img_x, img_y, id_cen, N_edg, 
					bcg_PA, tag_x, tag_y, id_half = 'half', 
					sat_PA = None, sat_ar = None, rms_file = None, pix_con_file = None, id_mean = 0 ):
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

	for sky img case :

	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center
	for sky imgs, id_cen can be 2 -- means 'random center stacking'
	for id_cen = 2 case, img_x, img_y is random position of the img frame coordinate, they may be BCGs' location,
	maybe not.(in this case, need to creat a list of "fake BCG position")

	id_mean : 0, 1, 2.  0 - img_add = img; 
	1 - img_add = img - np.mean(img); 2 - img_add = img - np.median(img); Default is id_mean = 0

	--------------------------------
	bcg_PA : the position angle of BCG relative to the longer side of image frame~(2048 pixels)
	tag_x, tag_y : direction identification of the target points (satellite) for partial image cut.

	--------------------------------
	sat_PA : the position angle of the target satellite~(relative to the longer side of image frame -- 2048 pixels)
	sat_ar : the semi-major axis of satellite (in unit of pixel), given by source-extractor

	id_half : string type, 'half', 'half-V', 'angle', by default is 'half'
			'angle' -- pixels align the direction of target satellite will be masked, the angle size is determin by
						distance to the associated BCG and source detection size of target objects
			'half' -- along the major-axis of BCG, all pixels on the same side of the target object will be removed
			'half-V' -- again, all pixels on the same side of the target object will be removed, but the line is 
						perpendicular to the line of target object and BCg central point

	"""

	stack_N = len(z_set)

	if id_cen == 1:
		x0, y0 = 1024, 744
		Nx = np.linspace(0, 2047, 2048)
		Ny = np.linspace(0, 1488, 1489)

		sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float32) * np.nan
		p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		pix_f2 = np.zeros((len(Ny), len(Nx)), dtype = np.float32)

	if id_cen != 1:
		x0, y0 = 2427, 1765
		Nx = np.linspace(0, 4854, 4855)
		Ny = np.linspace(0, 3530, 3531)

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

		##. mask the edge region with N_edg
		img_A[:N_edg, :] = np.nan
		img_A[-N_edg:, :] = np.nan
		img_A[:, :N_edg] = np.nan
		img_A[:, -N_edg:] = np.nan

		##. patrical cut image
		if id_half == 'angle':
			pcut_img = part_cut_func( img_A, img_x[jj], img_y[jj], bcg_PA[jj], tag_x[jj], tag_y[jj], 
									id_half = id_half, sat_PA = sat_PA[jj], sat_ar = sat_ar[jj] )

		else:
			pcut_img = part_cut_func( img_A, img_x[jj], img_y[jj], bcg_PA[jj], tag_x[jj], tag_y[jj] )

		#.
		img_A = pcut_img + 0.

		##. centering
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

		## for sky img stacking
		if id_cen == 2:
			rnx, rny = img_x[jj], img_y[jj]
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

		## tmp array for rms
		pix_f2[la0: la1, lb0: lb1][idv] = pix_f2[la0: la1, lb0: lb1][idv] + img_add[idv]**2

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


##.
def stack_func( d_file, out_file, z_set, ra_set, dec_set, band, img_x, img_y, id_cen, 
				bcg_PA, tag_x, tag_y, id_half = 'half', 
				sat_PA = None, sat_ar = None, rms_file = None, pix_con_file = None, id_mean = 0 ):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img (include file-name structure:'/xxx/xxx/xxx.xxx')
	img_x, img_y : BCG position (in image coordinate)
	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center
	rms_file : stacking img pixel variance 
	pix_con_file : the pixel counts in each stacking img pixel

	for sky img case :

	id_cen : 0 - stacking by centering on BCGs, 1 - stacking by centering on img center
	for sky imgs, id_cen can be 2 -- means 'random center stacking'
	for id_cen = 2 case, img_x, img_y is random position of the img frame coordinate, they may be BCGs' location,
	maybe not.(in this case, need to creat a list of "fake BCG position")

	id_mean : 0, 1, 2.  0 - img_add = img; 
	1 - img_add = img - np.mean(img); 2 - img_add = img - np.median(img); Default is id_mean = 0

	--------------------------------
	bcg_PA : the position angle of BCG relative to the longer side of image frame~(2048 pixels)
	tag_x, tag_y : direction identification of the target points (satellite) for partial image cut.

	--------------------------------
	sat_PA : the position angle of the target satellite~(relative to the longer side of image frame -- 2048 pixels)
	sat_ar : the semi-major axis of satellite (in unit of pixel), given by source-extractor

	id_half : string type, 'half', 'half-V', 'angle', by default is 'half'
			'angle' -- pixels align the direction of target satellite will be masked, the angle size is determin by
						distance to the associated BCG and source detection size of target objects
			'half' -- along the major-axis of BCG, all pixels on the same side of the target object will be removed
			'half-V' -- again, all pixels on the same side of the target object will be removed, but the line is 
						perpendicular to the line of target object and BCg central point

	"""

	stack_N = len(z_set)

	if id_cen == 1:
		x0, y0 = 1024, 744
		Nx = np.linspace(0, 2047, 2048)
		Ny = np.linspace(0, 1488, 1489)

		sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float32) * np.nan
		p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float32)
		pix_f2 = np.zeros((len(Ny), len(Nx)), dtype = np.float32)

	if id_cen != 1:
		x0, y0 = 2427, 1765
		Nx = np.linspace(0, 4854, 4855)
		Ny = np.linspace(0, 3530, 3531)

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

		##. patrical cut image
		if id_half == 'angle':
			pcut_img = part_cut_func( img_A, img_x[jj], img_y[jj], bcg_PA[jj], tag_x[jj], tag_y[jj], 
									id_half = id_half, sat_PA = sat_PA[jj], sat_ar = sat_ar[jj] )

		else:
			pcut_img = part_cut_func( img_A, img_x[jj], img_y[jj], bcg_PA[jj], tag_x[jj], tag_y[jj] )

		#.
		img_A = pcut_img + 0.

		##. centering
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

		## for sky img stacking
		if id_cen == 2:
			rnx, rny = img_x[jj], img_y[jj]
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

		## tmp array for rms
		pix_f2[la0: la1, lb0: lb1][idv] = pix_f2[la0: la1, lb0: lb1][idv] + img_add[idv]**2

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

