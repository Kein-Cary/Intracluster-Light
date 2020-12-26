import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from numba import vectorize
import pandas as pds

from light_measure import jack_SB_func

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


### ???
# SB pros with mean, median, mode and flux hist of radius bins
def SB_measure_Z0_weit_func(data, weit_data, pix_size, cx, cy, R_bins,
	bin_flux_file,):
	"""
	use for measuring surface brightness(SB) profile in angle coordinate,
		directly measure SB profile from observation img.
	data : the image use to measure SB profile
	pix_size : pixel size, in unit of "arcsec"
	cx, cy : the central position of objs in the image frame
	weit_data : the weight array for surface brightness profile measurement, it's must be 
	the same size as the 'data' array
	R_bins : radius bin edges for SB measurement, in unit of pixel
	bin_flux_file : output the pixel flux and pixel-contribution of effective radius bins,('.csv' files)
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
	medi_intens = np.zeros(len(rbin), dtype = np.float)
	mode_intens = np.zeros(len(rbin), dtype = np.float)

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
			#Angl_r[k] = 0.5 * (r_iner + r_out) * pix_size
			Angl_r[k] = np.nansum( dr[ir] * weit_arr ) / np.nansum( weit_arr ) * pix_size

			medi_intens[k] = np.nanmedian(samp_flux)
			mode_intens[k] = 3 * np.nanmedian(samp_flux) - 2 * tot_flux

			## output the flux_arr and count_arr
			#cen_r = 0.5 * (r_iner + r_out) * pix_size )
			cen_r = np.nansum( dr[ir] * weit_arr ) / np.nansum( weit_arr ) * pix_size

			keys = ['flux', 'pix_count']
			values = [ samp_flux, weit_arr ]
			fill = dict(zip( keys, values) )
			out_data = pds.DataFrame(fill)
			out_data.to_csv( bin_flux_file % cen_r,)

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

	mean_intens[idzo] = np.nan
	medi_intens[idzo] = np.nan
	mode_intens[idzo] = np.nan
	intens_err[idzo] = np.nan
	nsum_ratio[idzo] = np.nan

	return mean_intens, medi_intens, mode_intens, Angl_r, intens_err, N_pix, nsum_ratio

