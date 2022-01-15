import h5py
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from numba import vectorize
import pandas as pds

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

def flux_recal(data, z0, zref):
	obs = data
	z0 = z0
	z1 = zref
	Da0 = Test_model.angular_diameter_distance(z0).value
	Da1 = Test_model.angular_diameter_distance(z1).value
	flux = obs * (1 + z0)**4 * Da0**2 / ((1 + z1)**4 * Da1**2)
	return flux

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

def angu_area(s0, z0, zref):
	s0 = s0
	z0 = z0
	z1 = zref
	Da0 = Test_model.angular_diameter_distance(z0).value
	Da1 = Test_model.angular_diameter_distance(z1).value
	angu_S = s0*Da0**2/Da1**2
	return angu_S

######### error test (for bins number and angle section number)
def light_measure_pn(data, Nbin, R_small, R_max, cx, cy, psize, z0, pn):

	Da0 = Test_model.angular_diameter_distance(z0).value
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)

	pix_id = np.array(np.meshgrid(x0,y0))
	theta = np.arctan2((pix_id[1,:]-cy), (pix_id[0,:]-cx))
	chi = theta * 180 / np.pi

	divi_r = np.logspace(np.log10(R_small), np.log10(R_max), Nbin)
	r = (divi_r * 1e-3 * rad2arcsec / Da0) / psize
	ia = r <= 1. # smaller than 1 pixel
	ib = r[ia]
	ic = len(ib)
	rbin = r[ic:]
	set_r = divi_r[ic:]

	intens = np.zeros(len(r) - ic, dtype = np.float)
	intens_r = np.zeros(len(r) - ic, dtype = np.float)
	intens_err = np.zeros(len(r) - ic, dtype = np.float)
	N_pix = np.zeros(len(r) - ic, dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)

	for k in range(len(rbin) - 1):
		cdr = rbin[k + 1] - rbin[k]
		d_phi = (pn * cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		N_phi = np.int(360 / d_phi) + 1
		phi = np.linspace(0, 360, N_phi)
		phi = phi - 180

		ir = (dr >= rbin[k]) & (dr < rbin[k + 1])
		bool_sum = np.sum(ir)

		r_iner = set_r[k] ## useing radius in unit of kpc
		r_out = set_r[k + 1]

		if bool_sum == 0:
			intens[k] = np.nan
			intens_err[k] = np.nan
			N_pix[k] = np.nan
			intens_r[k] = 0.5 * (r_iner + r_out)
		else:
			samp_f = data[ir]
			samp_chi = chi[ir]

			tot_flux = np.nanmean(samp_f)

			intens[k] = tot_flux
			N_pix[k] = len(samp_f)
			intens_r[k] = 0.5 * (r_iner + r_out)

			tmpf = []
			for tt in range(len(phi) - 1):
				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])
				set_samp = samp_f[iv]
				ttf = np.nanmean(set_samp)
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
			intens_err[k] = RMS / np.sqrt(len(Tmpf) - 1)

	intens[intens == 0] = np.nan
	Intns = intens * 1

	intens_r[intens_r == 0] = np.nan
	Intns_r = intens_r * 1
	
	intens_err[intens_err == 0] = np.nan
	Intns_err = intens_err * 1

	N_pix[ N_pix == 0] = np.nan
	Npix = N_pix * 1

	return Intns, Intns_r, Intns_err, Npix

######### SB profile measure with modification in large scale
def lim_SB_pros_func(J_sub_img, J_sub_pix_cont, alter_sub_sb, alter_jk_sb, n_rbins, N_bin, SN_lim, 
	id_band, edg_bins = None, ):

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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(cc_tmp_sb, cc_tmp_r, band[ id_band ], N_bin,)[4:]

	#tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band[ id_band ], N_bin,)[4:]

	with h5py.File(alter_jk_sb, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)

	return

def zref_lim_SB_adjust_func(J_sub_img, J_sub_pix_cont, alter_sub_sb, alter_jk_sb, n_rbins, N_bin, SN_lim, z_ref,
	id_band, edg_bins = None,):

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
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(cc_tmp_sb, cc_tmp_r, band[ id_band ], N_bin,)[4:]

	#tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, band[ id_band ], N_bin,)[4:]

	with h5py.File(alter_jk_sb, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)

	return

######### SB pros with mean, median, mode and flux hist of radius bins
def SB_measure_Z0_weit_func(data, weit_data, pix_size, cx, cy, R_bins, bin_flux_file,):
	"""
	use for measuring surface brightness(SB) profile in angle coordinate,
		directly measure SB profile from observation img.
	data : the image use to measure SB profile
	pix_size : pixel size, in unit of "arcsec"
	cx, cy : the central position of objs in the image frame
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	R_bins : radius bin edges for SB measurement, in unit of pixel
	bin_flux_file : output the pixel flux and pixel-contribution of effective radius bins,('.h5' files)
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

	mean_intens = np.zeros(len(rbin), dtype = np.float)
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

			mean_intens[k] = tot_flux

			#cen_r = 0.5 * (r_iner + r_out) * pix_size
			cen_r = np.nansum( dr[ir] * weit_arr ) / np.nansum( weit_arr ) * pix_size

			Angl_r[k] = cen_r

			## output the flux_arr and count_arr
			with h5py.File( bin_flux_file % cen_r, 'w') as f:
				f['flux'] = np.array( samp_flux.astype( np.float32 ) )
				f['pix_count'] = np.array( weit_arr.astype( np.float32 ) )

			tmpf = []
			for tt in range(len(phi) - 1):
				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])

				set_samp = samp_flux[iv]
				set_weit = weit_arr[iv]

				ttf = np.nansum(set_samp * set_weit) / np.nansum(set_weit)
				tmpf.append(ttf)

			# rms of flux
			tmpf = np.array( tmpf )
			id_inf = np.isnan( tmpf )
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

	mean_intens[idzo] = np.nan
	intens_err[idzo] = np.nan
	nsum_ratio[idzo] = np.nan

	mean_intens, intens_err = mean_intens / pix_size**2, intens_err / pix_size**2

	return mean_intens, Angl_r, intens_err, N_pix, nsum_ratio

def light_measure_weit(data, weit_data, pix_size, cx, cy, z0, R_bins, bin_flux_file,):
	"""
	data: data used to measure (2D-array)
	Nbin: number of bins will devide
	R_bins : radius bin edges for SB measurement, in unit of pixels
	cx, cy: cluster central position in image frame (in inuit pixel)
	pix_size: pixel size
	z : the redshift of data
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	bin_flux_file : output the pixel flux and pixel-contribution of effective radius bins,('.h5' files)
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
			#intens_r[k] = 0.5 * (r_iner + r_out) # in unit of kpc
			cen_r = np.nansum(dr[ ir ] * weit_arr) / np.nansum( weit_arr ) * pix_size
			intens_r[k] = cen_r * Da0 * 1e3 / rad2arcsec

			## output the flux_arr and count_arr
			tt_R = cen_r * Da0 * 1e3 / rad2arcsec

			with h5py.File( bin_flux_file % tt_R, 'w') as f:
				f['flux'] = np.array( samp_flux.astype( np.float32 ) )
				f['pix_count'] = np.array( weit_arr.astype( np.float32 ) )

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

######### average surface brightness for given radius annuli ??
def annu_aveg_SB_func(data, weit_data, pix_size, cx, cy, z0, R_bins):

	"""
	data: data used to measure (2D-array)
	Nbin: number of bins will devide
	R_bins : radius bin edges for SB measurement, in unit of pixels,
			the innest radius must be 1 pixel
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

	intens = np.zeros( len(rbin), dtype = np.float )
	intens_r = np.zeros( len(rbin), dtype = np.float )
	intens_err = np.zeros( len(rbin), dtype = np.float )

	N_pix = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio = np.zeros(len(rbin), dtype = np.float)

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

	dr = np.sqrt( ( (2 * pix_id[0] + 1) / 2 - (2 * xn + 1) / 2 )**2 + ( (2 * pix_id[1] + 1) / 2 - (2 * yn + 1) / 2)**2 )

	for k in range( 1,len(rbin) - 1):

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

	## center pixel
	intens[0] = data[yn, xn] / pix_size**2
	intens_err[0] = 0.
	intens_r[0] = np.sqrt( pix_size**2 / np.pi )

	N_pix[0] = 1.
	nsum_ratio[0] = 1.

	idzo = N_pix < 1

	Intns = intens.copy()
	Intns[idzo] = 0.
	Intns_err = intens_err.copy()
	Intns_err[idzo] = 0.

	Intns_r = intens_r.copy()
	nsum_ratio[idzo] = 0.
	Intns, Intns_err = Intns / pix_size**2, Intns_err / pix_size**2

	bins_edgs = np.r_[ np.sqrt( pix_size**2 / np.pi ), rbin[1:] * pix_size ]

	return Intns_r, bins_edgs, Intns, Intns_err, N_pix, nsum_ratio

def light_measure_weit_cm(data, weit_data, pix_size, cx, cy, z0, R_bins,):
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

	theta = np.arctan2( (pix_id[1,:] - yn), (pix_id[0,:] - xn) )
	chi = theta * 180 / np.pi

	rbin = R_bins # have been divided bins, in unit of pixels
	set_r = rbin * pix_size * Da0 * 1e3 / rad2arcsec # in unit of kpc

	intens = np.zeros(len(rbin), dtype = np.float)
	intens_r = np.zeros(len(rbin), dtype = np.float)
	intens_err = np.zeros(len(rbin), dtype = np.float)

	N_pix = np.zeros(len(rbin), dtype = np.float)
	nsum_ratio = np.zeros(len(rbin), dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*xn + 1) / 2)**2 + ((2*pix_id[1] + 1) / 2 - (2*yn + 1) / 2)**2)

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
