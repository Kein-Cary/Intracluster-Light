import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from numba import vectorize
# constant
vc = C.c.to(U.km/U.s).value
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass
kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
Lsun2erg_s = U.L_sun.to(U.erg/U.s)
rad2arcsec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

# band information of SDSS
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

### covariance & correlarion matrix
def cov_MX_func(radius, pros, id_jack = True,):

	flux_array = np.array(pros)
	r_array = np.array(radius)
	Nt = len(flux_array)

	r_min, r_max = 0, np.nanmax(r_array)
	for ll in range(Nt):
		idnn = np.isnan( flux_array[ll] )
		lim_in_r = r_array[ll][ idnn == False]
		r_min = np.max([ r_min, np.min(lim_in_r) ])
		r_max = np.min([ r_max, np.max(lim_in_r) ])

	SB_value = []
	R_value = []
	for ll in range(Nt):
		idux = (r_array[ll] >= r_min) & (r_array[ll] <= r_max)
		set_flux = flux_array[ll][ idux ]
		set_r = r_array[ll][ idux ]
		SB_value.append( set_flux )
		R_value.append( set_r )

	SB_value = np.array(SB_value)
	R_value = np.array(R_value)
	R_mean = np.nanmean(R_value, axis = 0)

	mean_lit = np.nanmean(SB_value, axis = 0)
	std_lit = np.nanstd(SB_value, axis = 0)
	nx, ny = SB_value.shape[1], SB_value.shape[0]

	cov_tt = np.zeros((nx, nx), dtype = np.float)
	cor_tt = np.zeros((nx, nx), dtype = np.float)

	for qq in range(nx):
		for tt in range(nx):
			cov_tt[qq, tt] = np.sum( (SB_value[:,qq] - mean_lit[qq]) * (SB_value[:,tt] - mean_lit[tt]) ) / ny

	for qq in range(nx):
		for tt in range(nx):
			cor_tt[qq, tt] = cov_tt[qq, tt] / (std_lit[qq] * std_lit[tt])
	if id_jack == True:
		cov_MX = cov_tt * (ny - 1.) ## jackknife factor
	else:
		cov_MX = cov_tt * 1.
	cor_MX = cor_tt * 1.

	return R_mean, cov_MX, cor_MX

### dimming effect correction
def flux_recal(data, z0, zref):
	"""
	this function is used to rescale the pixel flux of sample images to reference redshift
	"""
	obs = data
	z0 = z0
	z1 = zref
	Da0 = Test_model.angular_diameter_distance(z0).value
	Da1 = Test_model.angular_diameter_distance(z1).value
	flux = obs * (1 + z0)**4 * Da0**2 / ((1 + z1)**4 * Da1**2)
	return flux

### jackknife SB 
def jack_SB_func(SB_array, R_array, band_id, N_sample,):
	"""
	stacking profile based on surface brightness,
	SB_array : list of surface brightness profile, in unit of " nanomaggies / arcsec^2 "
	SB_array, R_array : list type
	band_id : int type, for SB correction
	N_sample : number of sub-samples
	"""
	dx_r = np.array(R_array)
	dy_sb = np.array(SB_array)

	n_r = dx_r.shape[1]
	Len = np.zeros( n_r, dtype = np.float32)
	for nn in range( n_r ):
		tmp_I = dy_sb[:,nn]
		idnn = np.isnan(tmp_I)
		Len[nn] = N_sample - np.sum(idnn)

	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_SB = np.nanmean(dy_sb, axis = 0)
	std_Stack_SB = np.nanstd(dy_sb, axis = 0)

	### only calculate r bins in which sub-sample number larger than one
	id_one = Len > 1
	Stack_R = Stack_R[ id_one ]
	Stack_SB = Stack_SB[ id_one ]
	std_Stack_SB = std_Stack_SB[ id_one ]
	N_img = Len[ id_one ]
	jk_Stack_err = np.sqrt(N_img - 1) * std_Stack_SB

	### limit the radius bin contribution at least 1/3 * N_sample
	id_min = N_img >= np.int(N_sample / 3)
	lim_r = Stack_R[id_min]
	lim_R = np.nanmax(lim_r)

	## change flux to magnitude
	jk_Stack_SB = 22.5 - 2.5 * np.log10(Stack_SB) + mag_add[band_id]
	dSB0 = 22.5 - 2.5 * np.log10(Stack_SB + jk_Stack_err) + mag_add[band_id]
	dSB1 = 22.5 - 2.5 * np.log10(Stack_SB - jk_Stack_err) + mag_add[band_id]
	err0 = jk_Stack_SB - dSB0
	err1 = dSB1 - jk_Stack_SB
	id_nan = np.isnan(jk_Stack_SB)
	jk_Stack_SB, jk_Stack_R = jk_Stack_SB[id_nan == False], Stack_R[id_nan == False]
	jk_Stack_err0, jk_Stack_err1 = err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	jk_Stack_err1[idx_nan] = 100.

	return jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err, lim_R

### img grid
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

	return patch_mean, patch_pix, patch_Var, patch_S0, lx, ly

def grid_img(img_data, N_stepx, N_stepy):

	ly = np.arange(0, img_data.shape[0], N_stepy)
	ly = np.r_[ly, img_data.shape[0] - N_stepy, img_data.shape[0] ]
	lx = np.arange(0, img_data.shape[1], N_stepx)
	lx = np.r_[lx, img_data.shape[1] - N_stepx, img_data.shape[1] ]

	lx = np.delete(lx, -1)
	lx = np.delete(lx, -2)
	ly = np.delete(ly, -1)
	ly = np.delete(ly, -2)

	patch_mean = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_pix = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_Var = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	for nn in range( len(ly) ):
		for tt in range( len(lx) ):

			sub_flux = img_data[ly[nn]: ly[nn] + N_stepy, lx[tt]: lx[tt] + N_stepx]
			id_nn = np.isnan(sub_flux)
			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )

	return patch_mean, patch_pix, patch_Var

### surface brightness profile measurement (weight version)
###		[set weit_data as ones array for no weight case]
def light_measure_rn_Z0_weit(data, weit_data, pix_size, cx, cy, R_low, R_up):
	"""
	use for measuring surface brightness(SB) profile in angle coordinate,
		directly measure SB profile from observation img.
	data : the image use to measure SB profile
	pix_size : pixel size, in unit of "arcsec"
	cx, cy : the central position of objs in the image frame
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	R_low, R_up : the lower and uper limitation for given radius bin, in unit of pixel number
	"""
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)
	idu = (dr >= R_low) & (dr <= R_up)

	theta = np.arctan2((pix_id[1,:] - cy), (pix_id[0,:] - cx))
	chi = theta * 180 / np.pi

	samp_chi = chi[idu]
	samp_flux = data[idu]
	weit_arr = weit_data[idu]
	Intns = np.nansum( samp_flux * weit_arr ) / np.nansum( weit_arr )

	id_nn = np.isnan(samp_flux)
	N_pix = np.sum( id_nn == False )
	nsum_ratio = np.nansum(weit_arr) / np.sum( id_nn == False )

	cdr = R_up - R_low
	d_phi = ( cdr / (0.5 * (R_low + R_up) ) ) * 180 / np.pi
	N_phi = np.int(360 / d_phi) + 1
	phi = np.linspace(0, 360, N_phi)
	phi = phi - 180.

	tmpf = []
	for tt in range(len(phi) - 1):
		idv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt + 1])

		set_samp = samp_flux[idv]
		set_weit = weit_arr[idv]

		ttf = np.nansum(set_samp * set_weit) / np.nansum( set_weit )
		tmpf.append(ttf)

	# rms of flux
	tmpf = np.array(tmpf)
	id_inf = np.isnan(tmpf)
	tmpf[id_inf] = np.nan
	id_zero = tmpf == 0
	tmpf[id_zero] = np.nan

	id_nan = np.isnan(tmpf)
	id_fals = id_nan == False
	Tmpf = tmpf[id_fals]

	RMS = np.std(Tmpf)
	if len(Tmpf) > 1:
		Intns_err = RMS / np.sqrt(len(Tmpf) - 1)
	else:
		Intns_err = RMS

	Angl_r = (0.5 * (R_low + R_up) ) * pix_size

	return Intns, Angl_r, Intns_err, N_pix, nsum_ratio

def light_measure_Z0_weit(data, weit_data, pix_size, cx, cy, R_bins):
	"""
	use for measuring surface brightness(SB) profile in angle coordinate,
		directly measure SB profile from observation img.
	data : the image use to measure SB profile
	pix_size : pixel size, in unit of "arcsec"
	cx, cy : the central position of objs in the image frame
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	R_bins : radius bin edges for SB measurement, in unit of pixel
	"""
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	theta = np.arctan2((pix_id[1,:] - cy), (pix_id[0,:] - cx))
	chi = theta * 180 / np.pi
	# radius in unit of pixel number
	rbin = R_bins

	intens = np.zeros(len(rbin), dtype = np.float)
	intens_err = np.zeros(len(rbin), dtype = np.float)
	Angl_r = np.zeros(len(rbin), dtype = np.float)
	N_pix = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio = np.zeros(len(rbin), dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)

	for k in range(len(rbin) - 1):
		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		N_phi = np.int(360 / d_phi) + 1
		phi = np.linspace(0, 360, N_phi)
		phi = phi - 180

		ir = (dr >= rbin[k]) * (dr < rbin[k + 1])

		bool_sum = np.sum(ir)

		r_iner = rbin[k]
		r_out = rbin[k + 1]

		if bool_sum == 0:
			Angl_r[k] = 0.5 * (r_iner + r_out) * pix_size

		else:
			weit_arr = weit_data[ir]
			samp_flux = data[ir]
			samp_chi = chi[ir]

			tot_flux = np.nansum(samp_flux * weit_arr) / np.nansum(weit_arr)
			idnn = np.isnan( samp_flux )
			N_pix[k] = np.sum( idnn == False )
			nsum_ratio[k] = np.nansum(weit_arr) / np.sum( idnn == False )

			intens[k] = tot_flux
			Angl_r[k] = 0.5 * (r_iner + r_out) * pix_size

			tmpf = []
			for tt in range(len(phi) - 1):
				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				set_samp = samp_flux[iv]
				set_weit = weit_arr[iv]

				ttf = np.nansum(set_samp * set_weit) / np.nansum(set_weit)
				tmpf.append(ttf)

			# rms of flux
			tmpf = np.array(tmpf)
			id_inf = np.isnan(tmpf)
			tmpf[id_inf] = np.nan
			id_zero = tmpf == 0
			tmpf[id_zero] = np.nan
			id_nan = np.isnan(tmpf)
			id_fals = id_nan == False
			Tmpf = tmpf[id_fals]

			#RMS = np.sqrt( np.sum(Tmpf**2) / len(Tmpf) )
			RMS = np.std(Tmpf)
			if len(Tmpf) > 1:
				intens_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_err[k] = RMS

	idzo = N_pix < 1

	Intns = intens.copy()
	Intns[idzo] = 0.
	Intns_err = intens_err.copy()
	Intns_err[idzo] = 0.
	nsum_ratio[idzo] = 0.

	return Intns, Angl_r, Intns_err, N_pix, nsum_ratio

def light_measure_rn_weit(data, weit_data, pix_size, cx, cy, z0, R_low, R_up):
	"""
	use to get the surface brightness for given radius
	data : data used to measure brightness (2D-array)
	R_low, R_up : the low_limit and up_limit of the given radius (in unit of "kpc")
	cx, cy : the center location / the reference point of the radius
	pix_size : the pixel size in unit of arcsec
	z0 : the redshift of the data
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	"""
	Da0 = Test_model.angular_diameter_distance(z0).value
	R_pix_low = (R_low * 1e-3 * rad2arcsec / Da0) / pix_size
	R_pix_up = (R_up * 1e-3 * rad2arcsec / Da0) / pix_size

	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)
	idu = (dr >= R_pix_low) & (dr <= R_pix_up)

	theta = np.arctan2((pix_id[1,:] - cy), (pix_id[0,:] - cx))
	chi = theta * 180 / np.pi

	samp_chi = chi[idu]
	samp_flux = data[idu]
	weit_arr = weit_data[idu]
	Intns = np.nansum( samp_flux * weit_arr ) / np.nansum( weit_arr )

	id_nn = np.isnan(samp_flux)
	N_pix = np.sum( id_nn == False )
	nsum_ratio = np.nansum(weit_arr) / np.sum( id_nn == False )

	cdr = R_up - R_low
	d_phi = ( cdr / (0.5 * (R_low + R_up) ) ) * 180 / np.pi
	N_phi = np.int(360 / d_phi) + 1
	phi = np.linspace(0, 360, N_phi)
	phi = phi - 180.

	tmpf = []
	for tt in range(len(phi) - 1):
		idv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt + 1])

		set_samp = samp_flux[idv]
		set_weit = weit_arr[idv]

		ttf = np.nansum(set_samp * set_weit) / np.nansum( set_weit )
		tmpf.append(ttf)

	# rms of flux
	tmpf = np.array(tmpf)
	id_inf = np.isnan(tmpf)
	tmpf[id_inf] = np.nan
	id_zero = tmpf == 0
	tmpf[id_zero] = np.nan

	id_nan = np.isnan(tmpf)
	id_fals = id_nan == False
	Tmpf = tmpf[id_fals]

	RMS = np.std(Tmpf)
	if len(Tmpf) > 1:
		Intns_err = RMS / np.sqrt(len(Tmpf) - 1)
	else:
		Intns_err = RMS

	Intns_r = (0.5 * (R_low + R_up) )

	return Intns, Intns_r, Intns_err, N_pix, nsum_ratio

def light_measure_weit(data, weit_data, pix_size, cx, cy, z0, R_bins,):
	"""
	data: data used to measure (2D-array)
	Nbin: number of bins will devide
	R_bins : radius bin edges for SB measurement, in unit of pixels
	cx, cy: cluster central position in image frame (in inuit pixel)
	pix_size: pixel size
	z : the redshift of data
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	"""
	Da0 = Test_model.angular_diameter_distance(z0).value ## in unit 'Mpc'
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	theta = np.arctan2((pix_id[1,:]-cy), (pix_id[0,:]-cx))
	chi = theta * 180 / np.pi

	rbin = R_bins # have been divided bins, in unit of pixels
	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc

	intens = np.zeros(len(rbin), dtype = np.float)
	intens_r = np.zeros(len(rbin), dtype = np.float)
	intens_err = np.zeros(len(rbin), dtype = np.float)

	N_pix = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio = np.zeros(len(rbin), dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)

	for k in range(len(rbin) - 1):
		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		N_phi = np.int(360 / d_phi) + 1
		phi = np.linspace(0, 360, N_phi)
		phi = phi - 180

		ir = (dr >= rbin[k]) & (dr < rbin[k + 1])
		bool_sum = np.sum(ir)

		r_iner = set_r[k] ## useing radius in unit of kpc
		r_out = set_r[k + 1]

		if bool_sum == 0:
			intens_r[k] = 0.5 * (r_iner + r_out) # in unit of kpc

		else:
			weit_arr = weit_data[ir]

			samp_flux = data[ir]
			samp_chi = chi[ir]
			tot_flux = np.nansum(samp_flux * weit_arr) / np.nansum(weit_arr)

			idnn = np.isnan( samp_flux )
			N_pix[k] = np.sum( idnn == False )
			nsum_ratio[k] = np.nansum(weit_arr) / np.sum( idnn == False )			

			intens[k] = tot_flux
			intens_r[k] = 0.5 * (r_iner + r_out) # in unit of kpc

			tmpf = []
			for tt in range(len(phi) - 1):

				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				set_samp = samp_flux[iv]
				set_weit = weit_arr[iv]

				ttf = np.nansum(set_samp * set_weit) / np.nansum(set_weit)
				tmpf.append(ttf)

			# rms of flux
			tmpf = np.array(tmpf)
			id_inf = np.isnan(tmpf)
			tmpf[id_inf] = np.nan
			id_zero = tmpf == 0
			tmpf[id_zero] = np.nan

			id_nan = np.isnan(tmpf)
			id_fals = id_nan == False
			Tmpf = tmpf[id_fals]
			#RMS = np.sqrt( np.sum(Tmpf**2) / len(Tmpf) )
			RMS = np.std(Tmpf)
			if len(Tmpf) > 1:
				intens_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_err[k] = RMS

	idzo = N_pix < 1

	Intns = intens.copy()
	Intns[idzo] = 0.
	Intns_err = intens_err.copy()
	Intns_err[idzo] = 0.

	Intns_r = intens_r.copy()
	nsum_ratio[idzo] = 0.

	return Intns, Intns_r, Intns_err, N_pix, nsum_ratio

### surface mass density (based on NFW)
@vectorize
def sigmamc(r, Mc, c):
	"""
	r : radius at which calculate the 2d density, in unit kpc (r != 0)
	"""
	c = c
	R = r
	M = 10**Mc
	rho_c = (kpc2m/Msun2kg) * (3*H0**2)/(8*np.pi*G)
	r200_c = (3*M/(4*np.pi*rho_c*200))**(1/3)
	rs = r200_c / c
	# next similar variables are for comoving coordinate, with simble "_c"
	rho_0 = M / ((np.log(1+c)-c/(1+c))*4*np.pi*rs**3)
	f0_c = 2*rho_0*rs # use for test
	x = R/rs
	if x < 1: 
		f1 = np.sqrt(1-x**2)
		f2 = np.sqrt((1-x)/(1+x))
		f3 = x**2-1
		sigma_c = f0_c*(1-2*np.arctanh(f2)/f1)/f3
	elif x == 1:
		sigma_c = f0_c/3
	else:
		f1 = np.sqrt(x**2-1)
		f2 = np.sqrt((x-1)/(1+x))
		f3 = x**2-1
		sigma_c = f0_c*(1-2*np.arctan(f2)/f1)/f3
	return sigma_c

def sigmam(r, Mc, z, c):

	Qc = kpc2m / Msun2kg # recrect parameter for rho_c
	Z = z
	M = 10**Mc
	R = r
	Ez = np.sqrt(Omega_m*(1+Z)**3+Omega_k*(1+Z)**2+Omega_lambda)
	Hz = H0*Ez
	rhoc = Qc*(3*Hz**2)/(8*np.pi*G) # in unit Msun/kpc^3
	Deltac = (200/3)*(c**3/(np.log(1+c)-c/(c+1))) 
	r200 = (3*M/(4*np.pi*rhoc*200))**(1/3) # in unit kpc
	rs = r200/c
	f0 = 2*Deltac*rhoc*rs
	x = R/r200
	if x < 1: 
		f1 = np.sqrt(1-x**2)
		f2 = np.sqrt((1-x)/(1+x))
		f3 = x**2-1
		sigma = f0*(1-2*np.arctanh(f2)/f1)/f3
	elif x == 1:
		sigma = f0/3
	else:
		f1 = np.sqrt(x**2-1)
		f2 = np.sqrt((x-1)/(1+x))
		f3 = x**2-1
		sigma = f0*(1-2*np.arctan(f2)/f1)/f3
	return sigma

def main():

	#light_measure()
	sigmamc(100, 15, 5)
	#rho2d = sigmam(100, 15, 0, 5)

if __name__ == '__main__':
	main()
