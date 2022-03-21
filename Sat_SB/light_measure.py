import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy

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

### cumulative flux & surface brightness 'overdensity'
def cumula_flux_func(angl_r, bin_fdens,):
	"""
	angl_r : the radius outer edgs, in unit of arcsec
	bin_fdens : the average surface brightness between two radius edges
	"""
	N_bin = len( angl_r )
	flux_arr = np.zeros( N_bin, dtype = np.float32)

	for kk in range( N_bin ):

		if kk == 0:
			cfi = np.pi * angl_r[kk]**2 * bin_fdens[kk]
			flux_arr[kk] = cfi + flux_arr[kk]

		else:
			tps = kk + 0
			cfi = 0
			while tps > 0:
				cfi = cfi + np.pi * (angl_r[tps]**2 - angl_r[tps-1]**2) * bin_fdens[tps]
				tps = tps - 1

			cfi = cfi + np.pi * angl_r[0]**2 * bin_fdens[0]

			flux_arr[kk] = cfi + flux_arr[kk]

	return flux_arr

def over_dens_sb_func(data, weit_data, pix_size, cx, cy, z0, R_bins,):
	"""
	data: data used to measure (2D-array)
	Nbin: number of bins will devide
	R_bins : radius bin edges for SB measurement, in unit of pixels
	cx, cy: cluster central position in image frame (in inuit pixel)
	pix_size: pixel size
	z0 : the redshift of data
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

	rbin = R_bins + 0.  # have been divided bins, in unit of pixels
	angl_r = R_bins * pix_size

	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc

	N_bins = len( rbin )

	intens = np.zeros(N_bins, dtype = np.float)
	intens_r = np.zeros(N_bins, dtype = np.float)
	intens_err = np.zeros(N_bins, dtype = np.float)

	N_pix = np.zeros(N_bins, dtype = np.float)
	nsum_ratio = np.zeros(N_bins, dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)

	for k in range(N_bins - 1):
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
			#intens_r[k] = 0.5 * (r_iner + r_out) # in unit of kpc
			cen_r = np.nansum(dr[ ir ] * weit_arr) / np.nansum( weit_arr ) * pix_size
			intens_r[k] = cen_r * Da0 * 1e3 / rad2arcsec

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

			RMS = np.std(Tmpf)
			if len(Tmpf) > 1:
				intens_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_err[k] = RMS

	idzo = N_pix < 1

	Intns = intens.copy()
	Intns_err = intens_err.copy()
	Intns_r = intens_r.copy()

	Intns[ idzo ] = np.nan
	Intns_r[ idzo ] = np.nan
	Intns_err[ idzo ] = np.nan

	Intns, Intns_err = Intns / pix_size**2, Intns_err / pix_size**2
	Intns, Intns_err = Intns[:-1], Intns_err[:-1]
	Intns_r = Intns_r[:-1]

	# for rbin < 1
	cen_flux = data[cy, cx] / pix_size**2
	cen_err = 0
	cen_R = 0.5 * angl_r[0]
	fdens = np.r_[ cen_flux, Intns ]
	R_fdens = 0.5 * ( angl_r[1:] + angl_r[:-1] )
	R_fdens = np.r_[ cen_R, R_fdens]

	id_nul = np.isnan( fdens )
	cumuli_f = cumula_flux_func( angl_r[id_nul == False], fdens[id_nul == False] )

	bar_fdens = cumuli_f / ( np.pi * angl_r[id_nul == False]**2 )

	bar_fdens = bar_fdens[1:]
	R_aper = R_fdens[ id_nul == False][1:]
	phy_R_aper = ( R_aper * Da0 ) * 1e3 / rad2arcsec

	# over-fdens
	over_fdens = np.zeros( Intns.shape[0], dtype = np.float )

	N0 = len( Intns )

	for pp in range( N0 ): 

		identi = np.isnan( Intns_r[pp] )

		if identi == False:
			Ri = Intns_r[pp]
			dR = np.abs(phy_R_aper - Ri)

			idRx = dR == dR.min()
			over_fdens[pp] = bar_fdens[idRx] - Intns[pp]

		else:
			over_fdens[pp] = np.nan

			continue

	return Intns_r, Intns, Intns_err, over_fdens, N_pix, nsum_ratio


### covariance & correlarion matrix
def cov_MX_func(radius, pros, id_jack = True,):

	flux_array = np.array(pros)
	r_array = np.array(radius)
	Nt = len(flux_array)

	R_mean = np.nanmean(r_array, axis = 0)
	mean_lit = np.nanmean(flux_array, axis = 0)

	std_lit = np.nanstd(flux_array, axis = 0)
	nx, ny = flux_array.shape[1], flux_array.shape[0]

	cov_tt = np.zeros((nx, nx), dtype = np.float)
	cor_tt = np.zeros((nx, nx), dtype = np.float)

	for qq in range(nx):
		for tt in range(nx):
			cov_tt[qq, tt] = np.nansum( (flux_array[:,qq] - mean_lit[qq]) * (flux_array[:,tt] - mean_lit[tt]) ) / ny

	for qq in range(nx):
		for tt in range(nx):
			cor_tt[qq, tt] = cov_tt[qq, tt] / (std_lit[qq] * std_lit[tt])
	if id_jack == True:
		cov_MX = cov_tt * (ny - 1.) ## jackknife factor
	else:
		cov_MX = cov_tt * 1.
	cor_MX = cor_tt * 1.

	return R_mean, cov_MX, cor_MX


### dimming effect correction and flux scaling
def flux_recal(data, z0, zref):
	"""
	this function is used to rescale the pixel flux of cluster images to reference redshift
	"""
	f_obs = data
	z0 = z0
	z1 = zref
	Da0 = Test_model.angular_diameter_distance( z0 ).value
	Da1 = Test_model.angular_diameter_distance( z1 ).value
	f_ref = f_obs * (1 + z0)**4 * Da0**2 / ( (1 + z1)**4 * Da1**2 )
	return f_ref

def flux_scale(data, z0, zref, pix_z0):
	obs = data / pix_z0**2
	scaled_sb = obs *( (1 + z0)**4 / (1 + zref)**4 )

	Da0 = Test_model.angular_diameter_distance(z0).value
	Da1 = Test_model.angular_diameter_distance(zref).value
	s0 = pix_z0**2
	s1 = pix_z0**2 * ( Da0**2 / Da1**2 )

	pix_zf = np.sqrt(s1)
	sb_ref = scaled_sb * s1
	return sb_ref, pix_zf


### jackknife SB 
def jack_SB_func(SB_array, R_array, band_str, N_sample,):
	"""
	stacking profile based on surface brightness,
	SB_array : list of surface brightness profile, in unit of " nanomaggies / arcsec^2 "
	SB_array, R_array : list type
	band_str : 'str' type ['r', 'g', 'i', 'u', 'z'], for SB correction 
				band_id : 0, 1, 2, 3, 4 --> r, g, i, u, z 
	N_sample : number of sub-samples
	"""
	band_id = band.index( band_str )

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


### surface brightness profile measurement (weight version)
###		[set weit_data as ones-array for no weight case]
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

	#..center pixel point
	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*xn + 1) / 2)**2 + ((2*pix_id[1] + 1) / 2 - (2*yn + 1) / 2)**2)
	idu = (dr >= R_low) & (dr <= R_up)

	theta = np.arctan2((pix_id[1,:] - yn), (pix_id[0,:] - xn))
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

	#Angl_r = (0.5 * (R_low + R_up) ) * pix_size
	Angl_r = np.nansum( dr[idu] * weit_arr ) / np.nansum( weit_arr ) * pix_size

	Intns, Intns_err = Intns / pix_size**2, Intns_err / pix_size**2

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

	#..center pixel point
	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	theta = np.arctan2((pix_id[1,:] - yn), (pix_id[0,:] - xn))
	chi = theta * 180 / np.pi
	# radius in unit of pixel number
	rbin = R_bins + 0.

	N_bins = len( rbin )

	intens = np.zeros(N_bins, dtype = np.float)
	intens_err = np.zeros(N_bins, dtype = np.float)
	Angl_r = np.zeros(N_bins, dtype = np.float)
	N_pix = np.zeros(N_bins, dtype = np.float)
	nsum_ratio = np.zeros(N_bins, dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*xn + 1) / 2)**2 + ((2*pix_id[1] + 1) / 2 - (2*yn + 1) / 2)**2)

	for k in range(N_bins - 1):
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
			#Angl_r[k] = 0.5 * (r_iner + r_out) * pix_size
			Angl_r[k] = np.nansum( dr[ir] * weit_arr ) / np.nansum( weit_arr ) * pix_size

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

	Intns, Intns_err = Intns / pix_size**2, Intns_err / pix_size**2

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

	#..center pixel point
	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*xn + 1) / 2)**2 + ((2*pix_id[1] + 1) / 2 - (2*yn + 1) / 2)**2)
	idu = (dr >= R_pix_low) & (dr <= R_pix_up)

	theta = np.arctan2((pix_id[1,:] - yn), (pix_id[0,:] - xn))
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

	#Intns_r = (0.5 * (R_low + R_up) )
	cen_r = np.nansum(dr[idu] * weit_arr) / np.nansum( weit_arr ) * pix_size
	Intns_r = cen_r * Da0 * 1e3 / rad2arcsec

	Intns, Intns_err = Intns / pix_size**2, Intns_err / pix_size**2

	return Intns, Intns_r, Intns_err, N_pix, nsum_ratio

def light_measure_weit(data, weit_data, pix_size, cx, cy, z0, R_bins):
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
	Da0 = Test_model.angular_diameter_distance( z0 ).value ## in unit 'Mpc'
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	#..center pixel point
	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	theta = np.arctan2((pix_id[1,:] - yn), (pix_id[0,:] - xn))
	chi = theta * 180 / np.pi

	rbin = R_bins + 0.  # have been divided bins, in unit of pixels
	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc

	N_bins = len( rbin )

	intens = np.zeros(N_bins, dtype = np.float)
	intens_r = np.zeros(N_bins, dtype = np.float)
	intens_err = np.zeros(N_bins, dtype = np.float)

	N_pix = np.zeros(N_bins, dtype = np.float)
	nsum_ratio = np.zeros(N_bins, dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*xn + 1) / 2)**2 + ((2*pix_id[1] + 1) / 2 - (2*yn + 1) / 2)**2)

	for k in range(N_bins - 1):
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
			#intens_r[k] = 0.5 * (r_iner + r_out) # in unit of kpc
			cen_r = np.nansum(dr[ ir ] * weit_arr) / np.nansum( weit_arr ) * pix_size
			intens_r[k] = cen_r * Da0 * 1e3 / rad2arcsec

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
	Intns, Intns_err = Intns / pix_size**2, Intns_err / pix_size**2

	return Intns, Intns_r, Intns_err, N_pix, nsum_ratio


### surface brightness profile along given circles
def light_measure_circle( data, weit_data, pix_size, cx, cy, R_bins, id_phy = False, z0 = None):
	"""
	use for measuring surface brightness(SB) profile in angle coordinate,
		directly measure SB profile from observation img.
	data : the image use to measure SB profile
	pix_size : pixel size, in unit of "arcsec"
	cx, cy : the central position of objs in the image frame
	weit_data : the weight array for surface brightness profile measurement, it's must be 
				the same size as the 'data' array
	R_bins : radius bin edges for SB measurement, in unit of pixel
	
	---------
	if the observed redshift is given, z0 = z_obs, and set id_phy = True
	"""

	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	#..center pixel point
	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	theta = np.arctan2((pix_id[1,:] - yn), (pix_id[0,:] - xn))
	chi = theta * 180 / np.pi

	# radius in unit of pixel number
	rbin = R_bins.astype( int )

	N_bins = len( rbin )

	intens = np.zeros(N_bins, dtype = np.float)
	Angl_r = np.zeros(N_bins, dtype = np.float)
	N_pix = np.zeros(N_bins, dtype = np.float)
	nsum_ratio = np.zeros(N_bins, dtype = np.float)

	dr = np.sqrt(( pix_id[0] - xn )**2 + ( pix_id[1] - yn)**2)

	for k in range( N_bins ):

		ir = dr == rbin[ k ]

		bool_sum = np.sum(ir)

		if bool_sum == 0:
			Angl_r[k] = rbin[ k ] * pix_size		

		else:
			weit_arr = weit_data[ir]
			samp_flux = data[ir]
			samp_chi = chi[ir]

			tot_flux = np.nansum(samp_flux * weit_arr) / np.nansum(weit_arr)
			idnn = np.isnan( samp_flux )
			N_pix[k] = np.sum( idnn == False )
			nsum_ratio[k] = np.nansum(weit_arr) / np.sum( idnn == False )

			intens[k] = tot_flux + 0.
			Angl_r[k] = np.nansum( dr[ir] * weit_arr ) / np.nansum( weit_arr ) * pix_size

	idzo = N_pix < 1

	Intns = intens.copy()
	Intns[idzo] = 0.
	nsum_ratio[idzo] = 0.

	Intns = Intns / pix_size**2

	if id_phy:

		Da0 = Test_model.angular_diameter_distance( z0 ).value ## in unit 'Mpc'
		phy_r = Angl_r * Da0 * 1e3 / rad2arcsec # in unit of kpc
		return Intns, phy_r, N_pix, nsum_ratio

	else:
		return Intns, Angl_r, N_pix, nsum_ratio

def PA_SB_Zx_func(data, weit_data, pix_size, cx, cy, z0, R_bins):
	"""
	measure liht profile in physical coordinate (radius in units of kpc)
	---------------------
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

	#..center pixel point
	dev_05_x = cx - np.int( cx )
	dev_05_y = cy - np.int( cy )

	if dev_05_x > 0.5:
		xn = np.int( cx ) + 1
	else:
		xn = np.int( cx )

	if dev_05_y > 0.5:
		yn = np.int( cy ) + 1
	else:
		yn = np.int( cy )

	theta = np.arctan2((pix_id[1,:] - yn), (pix_id[0,:] - xn))
	chi = theta * 180 / np.pi

	rbin = R_bins # have been divided bins, in unit of pixels
	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc


	#. surface brightness along the row direction
	intens_r_h = np.zeros(len(rbin), dtype = np.float)
	intens_h = np.zeros(len(rbin), dtype = np.float)
	intens_h_err = np.zeros(len(rbin), dtype = np.float)

	N_pix_h = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio_h = np.zeros(len(rbin), dtype = np.float)


	#. surface brightness along the column direction
	intens_r_v = np.zeros(len(rbin), dtype = np.float)
	intens_v = np.zeros(len(rbin), dtype = np.float)
	intens_v_err = np.zeros(len(rbin), dtype = np.float)

	N_pix_v = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio_v = np.zeros(len(rbin), dtype = np.float)


	#. surface brightness along the diagonal direction
	intens_r_d = np.zeros(len(rbin), dtype = np.float)
	intens_d = np.zeros(len(rbin), dtype = np.float)
	intens_d_err = np.zeros(len(rbin), dtype = np.float)	

	N_pix_d = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio_d = np.zeros(len(rbin), dtype = np.float)


	dr = np.sqrt( ( (2*pix_id[0] + 1) / 2 - (2*xn + 1) / 2)**2 + ( (2*pix_id[1] + 1) / 2 - (2*yn + 1) / 2)**2)
	diff_x = (2 * pix_id[0] + 1) / 2 - (2 * xn + 1) / 2
	diff_y = (2 * pix_id[1] + 1) / 2 - (2 * yn + 1) / 2


	for k in range(len(rbin) - 1):

		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		N_phi = np.int(360 / d_phi) + 1
		phi = np.linspace(0, 360, N_phi)
		phi = phi - 180

		r_iner = set_r[k] ## useing radius in unit of kpc
		r_out = set_r[k + 1]

		ir = (dr >= rbin[k]) & (dr < rbin[k + 1])
		bool_sum = np.sum(ir)

		if bool_sum == 0:
			intens_r_h[k] = 0.5 * (r_iner + r_out) # in unit of kpc
			intens_r_v[k] = 0.5 * (r_iner + r_out) # in unit of kpc
			intens_r_d[k] = 0.5 * (r_iner + r_out) # in unit of kpc

			continue

		else:
			#. points located in radius bin
			id_vx = (dr >= rbin[k]) & (dr < rbin[k + 1])


			##. points along the row direction
			id_ux_0 = np.abs( diff_x ) < rbin[ k ]
			id_uy_0 = ( np.abs( diff_y ) >= rbin[ k ] ) & ( np.abs( diff_y ) < rbin[ k+1 ] )
			id_lim_0 = ( id_ux_0 & id_uy_0 ) & id_vx

			weit_arr_0 = weit_data[ id_lim_0 ]

			samp_flux = data[ id_lim_0 ]
			samp_chi = chi[ id_lim_0 ]
			tot_flux = np.nansum( samp_flux * weit_arr_0 ) / np.nansum( weit_arr_0 )

			idnn = np.isnan( samp_flux )
			N_pix_v[ k ] = np.sum( idnn == False )
			nsum_ratio_v[ k ] = np.nansum( weit_arr_0 ) / np.sum( idnn == False )			

			intens_v[ k ] = tot_flux
			cen_r = np.nansum( dr[ id_lim_0 ] * weit_arr_0 ) / np.nansum( weit_arr_0 ) * pix_size
			intens_r_v[ k ] = cen_r * Da0 * 1e3 / rad2arcsec

			tmpf = []
			for tt in range(len(phi) - 1):

				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				if np.sum( iv ) == 0:
					continue

				else:
					set_samp = samp_flux[ iv ]
					set_weit = weit_arr_0[ iv ]

					ttf = np.nansum( set_samp * set_weit ) / np.nansum( set_weit )
					tmpf.append( ttf )

			# rms of flux
			tmpf = np.array( tmpf )
			id_inf = np.isnan( tmpf )
			tmpf[ id_inf ] = np.nan
			id_zero = tmpf == 0
			tmpf[id_zero] = np.nan

			id_nan = np.isnan( tmpf )
			id_fals = id_nan == False
			Tmpf = tmpf[ id_fals ]

			RMS = np.std(Tmpf)

			if len(Tmpf) > 1:
				intens_v_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_v_err[k] = RMS


			##. points along the columns direction
			id_ux_1 = np.abs( diff_y ) < rbin[ k ]
			id_uy_1 = ( np.abs( diff_x ) >= rbin[ k ] ) & ( np.abs( diff_x ) < rbin[ k+1 ] )
			id_lim_1 = ( id_ux_1 & id_uy_1 ) & id_vx

			weit_arr_1 = weit_data[ id_lim_1 ]

			samp_flux = data[ id_lim_1 ]
			samp_chi = chi[ id_lim_1 ]
			tot_flux = np.nansum( samp_flux * weit_arr_1 ) / np.nansum( weit_arr_1 )

			idnn = np.isnan( samp_flux )
			N_pix_h[ k ] = np.sum( idnn == False )
			nsum_ratio_h[ k ] = np.nansum( weit_arr_1 ) / np.sum( idnn == False )			

			intens_h[ k ] = tot_flux
			cen_r = np.nansum( dr[ id_lim_1 ] * weit_arr_1 ) / np.nansum( weit_arr_1 ) * pix_size
			intens_r_h[ k ] = cen_r * Da0 * 1e3 / rad2arcsec

			tmpf = []
			for tt in range(len(phi) - 1):

				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				if np.sum( iv ) == 0:
					continue

				else:
					set_samp = samp_flux[ iv ]
					set_weit = weit_arr_1[ iv ]

					ttf = np.nansum( set_samp * set_weit ) / np.nansum( set_weit )
					tmpf.append( ttf )

			# rms of flux
			tmpf = np.array( tmpf )
			id_inf = np.isnan( tmpf )
			tmpf[ id_inf ] = np.nan
			id_zero = tmpf == 0
			tmpf[id_zero] = np.nan

			id_nan = np.isnan( tmpf )
			id_fals = id_nan == False
			Tmpf = tmpf[ id_fals ]

			RMS = np.std(Tmpf)

			if len(Tmpf) > 1:
				intens_h_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_h_err[k] = RMS


			##. points along the diagonal direction
			id_qx = ( np.abs( diff_x ) <= rbin[ k ] ) & ( np.abs( diff_y ) <= rbin[ k ] )
			id_lim_2 = id_qx & id_vx

			weit_arr_2 = weit_data[ id_lim_2 ]

			samp_flux = data[ id_lim_2 ]
			samp_chi = chi[ id_lim_2 ]
			tot_flux = np.nansum( samp_flux * weit_arr_2 ) / np.nansum( weit_arr_2 )

			idnn = np.isnan( samp_flux )
			N_pix_d[ k ] = np.sum( idnn == False )
			nsum_ratio_d[ k ] = np.nansum( weit_arr_2 ) / np.sum( idnn == False )			

			intens_d[ k ] = tot_flux
			cen_r = np.nansum( dr[ id_lim_2 ] * weit_arr_2 ) / np.nansum( weit_arr_2 ) * pix_size
			intens_r_d[ k ] = cen_r * Da0 * 1e3 / rad2arcsec

			tmpf = []
			for tt in range(len(phi) - 1):

				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				if np.sum( iv ) == 0:
					continue

				else:
					set_samp = samp_flux[ iv ]
					set_weit = weit_arr_2[ iv ]

					ttf = np.nansum( set_samp * set_weit ) / np.nansum( set_weit )
					tmpf.append( ttf )

			# rms of flux
			tmpf = np.array( tmpf )
			id_inf = np.isnan( tmpf )
			tmpf[ id_inf ] = np.nan
			id_zero = tmpf == 0
			tmpf[id_zero] = np.nan

			id_nan = np.isnan( tmpf )
			id_fals = id_nan == False
			Tmpf = tmpf[ id_fals ]

			RMS = np.std(Tmpf)

			if len(Tmpf) > 1:
				intens_d_err[k] = RMS / np.sqrt(len(Tmpf) - 1)
			else:
				intens_d_err[k] = RMS

	#..
	idzo = N_pix_h < 1

	SB_h = intens_h.copy()
	SB_h[idzo] = 0.
	SB_h_err = intens_h_err.copy()
	SB_h_err[idzo] = 0.

	SB_h_R = intens_r_h.copy()
	nsum_ratio_h[idzo] = 0.
	SB_h, SB_h_err = SB_h / pix_size**2, SB_h_err / pix_size**2


	idzo = N_pix_v < 1

	SB_v = intens_v.copy()
	SB_v[idzo] = 0.
	SB_v_err = intens_v_err.copy()
	SB_v_err[idzo] = 0.

	SB_v_R = intens_r_v.copy()
	nsum_ratio_v[idzo] = 0.
	SB_v, SB_v_err = SB_v / pix_size**2, SB_v_err / pix_size**2


	idzo = N_pix_d < 1

	SB_d = intens_d.copy()
	SB_d[idzo] = 0.
	SB_d_err = intens_d_err.copy()
	SB_d_err[idzo] = 0.

	SB_d_R = intens_r_d.copy()
	nsum_ratio_d[idzo] = 0.
	SB_d, SB_d_err = SB_d / pix_size**2, SB_d_err / pix_size**2


	h_array = [ SB_h_R, SB_h, SB_h_err, N_pix_h, nsum_ratio_h ]
	v_array = [ SB_v_R, SB_v, SB_v_err, N_pix_v, nsum_ratio_v ]
	d_array = [ SB_d_R, SB_d, SB_d_err, N_pix_d, nsum_ratio_d ]

	return h_array, v_array, d_array


### SB profile measure with modification in large scale
### 	[with weight array applied]
def lim_SB_pros_func(J_sub_img, J_sub_pix_cont, alter_sub_sb, alter_jk_sb, n_rbins, N_bin, SN_lim, 
	band_str, edg_bins = None, ):

	### stacking in angle coordinate

	lim_r = 0

	for nn in range( N_bin ):
		with h5py.File(J_sub_img % nn, 'r') as f:
			sub_jk_img = np.array(f['a'])

		xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
		id_nn = np.isnan(sub_jk_img)
		eff_y, eff_x = np.where(id_nn == False)
		dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
		dR_max = np.int( dR.max() ) + 1
		lim_r = np.max([lim_r, dR_max])

	r_bins = np.logspace(0, np.log10(lim_r), n_rbins)
	r_angl = r_bins * pixel

	for nn in range( N_bin ):

		with h5py.File(J_sub_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		with h5py.File(J_sub_pix_cont % nn, 'r') as f:
			tmp_cont = np.array(f['a'])

		xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

		Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(tmp_img, tmp_cont, pixel, xn, yn, r_bins)
		sb_arr, sb_err = Intns, Intns_err

		r_arr = Intns_r.copy()

		id_sn = nratio >= np.nanmax(nratio) / SN_lim ## limitation on S/N
		id_npix = npix >= 1.

		r_arr[id_npix == False] = np.nan
		sb_arr[id_npix == False] = np.nan
		sb_err[id_npix == False] = np.nan
		try:
			id_R = r_arr > 200 ## arcsec
			cri_R = r_arr[ id_R & (id_sn == False) ]

			id_bin = r_angl < cri_R[0]
			id_dex = np.sum(id_bin) - 1
		except IndexError:
			cri_R = np.array([600]) # arcsec
			id_bin = r_angl < cri_R[0]
			id_dex = np.sum(id_bin) - 1

		edg_R_low = r_bins[id_dex]
		edg_R_up = r_bins[ -1 ]

		if edg_bins is not None:
			## linear bins
			edg_R_bin = np.linspace(edg_R_low, edg_R_up, edg_bins,)
			Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(tmp_img, tmp_cont, pixel, xn, yn, edg_R_bin)
		else:
			## out-region as one bin
			Intns, Intns_r, Intns_err, npix, nratio = light_measure_rn_Z0_weit(tmp_img, tmp_cont, pixel, xn, yn, edg_R_low, edg_R_up)

		edg_sb, edg_sb_err = Intns, Intns_err
		edg_R = Intns_r.copy()

		id_edg = r_arr >= cri_R[0]
		r_arr[id_edg] = np.nan
		sb_arr[id_edg] = np.nan
		sb_err[id_edg] = np.nan

		r_arr = np.r_[r_arr, edg_R ]
		sb_arr = np.r_[sb_arr, edg_sb ]
		sb_err = np.r_[sb_err, edg_sb_err ]

		with h5py.File(alter_sub_sb % nn, 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err)

	tmp_sb = []
	tmp_r = []
	for nn in range( N_bin ):

		with h5py.File(alter_sub_sb % nn, 'r') as f:
			r_arr = np.array(f['r'])
			sb_arr = np.array(f['sb'])
			sb_err = np.array(f['sb_err'])

			tmp_sb.append(sb_arr)
			tmp_r.append(r_arr)

	dt_arr = np.array( tmp_r )
	medi_R = np.nanmedian( dt_arr, axis = 0 )
	cc_tmp_r = []
	cc_tmp_sb = []

	for nn in range( N_bin ):

		xx_R = tmp_r[ nn ] + 0
		xx_sb = tmp_sb[ nn ] + 0

		deviR = np.abs( xx_R - medi_R )

		idmx = deviR > 0
		xx_sb[ idmx ] = np.nan

		cc_tmp_sb.append( xx_sb )
		cc_tmp_r.append( medi_R )

	## only save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(cc_tmp_sb, cc_tmp_r, band_str, N_bin,)[4:]
	#tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_bin,)[4:]

	with h5py.File(alter_jk_sb, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)

	return

def zref_lim_SB_adjust_func(J_sub_img, J_sub_pix_cont, alter_sub_sb, alter_jk_sb, n_rbins, N_bin, SN_lim, z_ref,
	band_str, edg_bins = None,):

	### stacking in angle coordinate

	lim_r = 0

	for nn in range( N_bin ):
		with h5py.File(J_sub_img % nn, 'r') as f:
			sub_jk_img = np.array(f['a'])

		xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
		id_nn = np.isnan(sub_jk_img)
		eff_y, eff_x = np.where(id_nn == False)
		dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
		dR_max = np.int( dR.max() ) + 1
		lim_r = np.max([lim_r, dR_max])

	r_bins = np.logspace(0, np.log10(lim_r), n_rbins)
	r_angl = r_bins * pixel

	Da_ref = Test_model.angular_diameter_distance(z_ref).value
	phy_r = Da_ref * 1e3 * r_angl / rad2asec

	for nn in range( N_bin ):

		with h5py.File(J_sub_img % nn, 'r') as f:
			tmp_img = np.array(f['a'])
		with h5py.File(J_sub_pix_cont % nn, 'r') as f:
			tmp_cont = np.array(f['a'])

		xn, yn = np.int(tmp_img.shape[1] / 2), np.int(tmp_img.shape[0] / 2)

		Intns, Intns_r, Intns_err, npix, nratio = light_measure_weit(tmp_img, tmp_cont, pixel, xn, yn, z_ref, r_bins)
		sb_arr, sb_err = Intns, Intns_err
		r_arr = Intns_r.copy()

		id_npix = npix >= 1.
		r_arr[id_npix == False] = np.nan
		sb_arr[id_npix == False] = np.nan
		sb_err[id_npix == False] = np.nan

		id_sn = nratio >= np.nanmax(nratio) / SN_lim ## limitation on S/N

		try:
			id_R = r_arr > 500 # kpc
			cri_R = r_arr[ id_R & (id_sn == False) ]

			id_bin = phy_r < cri_R[0]
			id_dex = np.sum(id_bin) - 1

		except IndexError:
			cri_R = np.array([2000]) # kpc
			id_bin = phy_r < cri_R[0]
			id_dex = np.sum(id_bin) - 1

		edg_R_low = r_bins[id_dex]
		edg_R_up = r_bins[ -1 ]

		phy_edg_R_low = phy_r[id_dex]
		phy_edg_R_up = phy_r[ -1 ]

		if edg_bins is not None:
			edg_R_bin = np.linspace(edg_R_low, edg_R_up, edg_bins,)
			Intns, Intns_r, Intns_err, npix, nratio = light_measure_weit(tmp_img, tmp_cont, pixel, xn, yn, z_ref, edg_R_bin)
		else:
			Intns, Intns_r, Intns_err, npix, nratio = light_measure_rn_weit(
				tmp_img, tmp_cont, pixel, xn, yn, z_ref, phy_edg_R_low, phy_edg_R_up)			

		edg_sb, edg_sb_err = Intns, Intns_err
		edg_R = Intns_r.copy()

		id_edg = r_arr >= cri_R[0]
		r_arr[id_edg] = np.nan
		sb_arr[id_edg] = np.nan
		sb_err[id_edg] = np.nan

		r_arr = np.r_[r_arr, edg_R ]
		sb_arr = np.r_[sb_arr, edg_sb ]
		sb_err = np.r_[sb_err, edg_sb_err ]

		with h5py.File(alter_sub_sb % nn, 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err)

	tmp_sb = []
	tmp_r = []
	for nn in range( N_bin ):

		with h5py.File(alter_sub_sb % nn, 'r') as f:
			r_arr = np.array(f['r'])
			sb_arr = np.array(f['sb'])
			sb_err = np.array(f['sb_err'])

			tmp_sb.append(sb_arr)
			tmp_r.append(r_arr)

	dt_arr = np.array( tmp_r )
	medi_R = np.nanmedian( dt_arr, axis = 0 )
	cc_tmp_r = []
	cc_tmp_sb = []

	for nn in range( N_bin ):

		xx_R = tmp_r[ nn ] + 0
		xx_sb = tmp_sb[ nn ] + 0

		deviR = np.abs( xx_R - medi_R )

		idmx = deviR > 0
		xx_sb[ idmx ] = np.nan

		cc_tmp_sb.append( xx_sb )
		cc_tmp_r.append( medi_R )

	## only save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(cc_tmp_sb, cc_tmp_r, band_str, N_bin,)[4:]
	#tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band_str, N_bin,)[4:]

	with h5py.File(alter_jk_sb, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)

	return

