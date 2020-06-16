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

def light_measure(data, Nbin, R_small, R_max, cx, cy, pix_size, z0):
	"""
	data: data used to measure (2D-array)
	Nbin: number of bins will devide
	Rp: radius in unit pixel number
	cx, cy: cluster central position in image frame (in inuit pixel)
	pix_size: pixel size
	z : the redshift of data
	"""
	Da0 = Test_model.angular_diameter_distance(z0).value
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	theta = np.arctan2((pix_id[1,:]-cy), (pix_id[0,:]-cx))
	chi = theta * 180 / np.pi

	divi_r = np.logspace(np.log10(R_small), np.log10(R_max), Nbin)
	r = (divi_r * 1e-3 * rad2arcsec / Da0) / pix_size
	ia = r <= 1. # smaller than 1 pixel
	ib = r[ia]
	ic = len(ib)
	rbin = r[ic:]
	set_r = divi_r[ic:]

	intens = np.zeros(len(r) - ic, dtype = np.float)
	intens_r = np.zeros(len(r) - ic, dtype = np.float)
	intens_err = np.zeros(len(r) - ic, dtype = np.float)

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
			intens[k] = np.nan
			intens_err[k] = np.nan
			intens_r[k] = 0.5 * (r_iner + r_out) # in unit of kpc

		else:
			samp_flux = data[ir]
			samp_chi = chi[ir]

			tot_flux = np.nanmean(samp_flux)
			tot_area = pix_size**2
			intens[k] = tot_flux
			intens_r[k] = 0.5 * (r_iner + r_out) # in unit of kpc

			tmpf = []
			for tt in range(len(phi) - 1):
				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])
				set_samp = samp_flux[iv]
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
			intens_err[k] = np.nanstd(tmpf) / np.sqrt(len(Tmpf) - 1)

	intens[intens == 0] = np.nan
	Intns = intens * 1

	intens_r[intens_r == 0] = np.nan
	Intns_r = intens_r * 1
	
	intens_err[intens_err == 0] = np.nan
	Intns_err = intens_err * 1

	return Intns, Intns_r, Intns_err

def light_measure_rn(data, R_low, R_up, cx, cy, pix_size, z0):
	"""
	use to get the surface brightness for given radius
	data : data used to measure brightness (2D-array)
	R_low, R_up : the low_limit and up_limit of the given radius (in unit of "kpc")
	cx, cy : the center location / the reference point of the radius
	pix_size : the pixel size in unit of arcsec
	z0 : the redshift of the data 
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
	Intns = np.nanmean( data[idu] )

	theta = np.arctan2((pix_id[1,:] - cy), (pix_id[0,:] - cx))
	chi = theta * 180 / np.pi
	samp_chi = chi[idu]
	samp_flux = data[idu]

	cdr = R_up - R_low
	d_phi = ( cdr / (0.5 * (R_low + R_up) ) ) * 180 / np.pi
	N_phi = np.int(360 / d_phi) + 1
	phi = np.linspace(0, 360, N_phi)
	phi = phi - 180.

	tmpf = []
	for tt in range(len(phi) - 1):
		idv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt + 1])
		set_samp = samp_flux[idv]

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
	Intns_err = np.nanstd(tmpf) / np.sqrt(len(Tmpf) - 1)

	Intns_r = (0.5 * (R_low + R_up) )

	return Intns_r, Intns, Intns_err

def light_measure_Z0(data, pix_size, r_lim, R_pix, cx, cy, bins):
	"""
	This part use for measuring surface brightness(SB) of objs those redshift 
		is too small or is zero. ie. the sky brightness.
	data : the image use to measure SB profile
	pix_size : pixel size, in unit of "arcsec"
	r_lim : the smallest radius of SB profile
	R_pix : the radius of objs in unit of pixel number
	cx, cy : the central position of objs in the image frame
	"""
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	theta = np.arctan2((pix_id[1,:] - cy), (pix_id[0,:] - cx))
	chi = theta * 180 / np.pi
	# radius in unit of pixel number
	rbin = np.logspace(np.log10(r_lim), np.log10(R_pix), bins)

	intens = np.zeros(len(rbin), dtype = np.float)
	intens_err = np.zeros(len(rbin), dtype = np.float)
	Angl_r = np.zeros(len(rbin), dtype = np.float)

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

		r_iner = rbin[k]
		r_out = rbin[k + 1]

		if bool_sum == 0:
			intens[k] = np.nan
			intens_err[k] = np.nan
			Angl_r[k] = 0.5 * (r_iner + r_out) * pix_size

		else:
			samp_flux = data[ir]
			samp_chi = chi[ir]
			tot_flux = np.nanmean(samp_flux)
			tot_area = pix_size**2
			intens[k] = tot_flux
			Angl_r[k] = 0.5 * (r_iner + r_out) * pix_size

			tmpf = []
			for tt in range(len(phi) - 1):
				iv = (samp_chi >= phi[tt]) & (samp_chi <= phi[tt+1])
				set_samp = samp_flux[iv]
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
			intens_err[k] = np.nanstd(tmpf) / np.sqrt(len(Tmpf) - 1)

	intens[intens == 0] = np.nan
	Intns = intens * 1

	intens_err[intens_err == 0] = np.nan
	Intns_err = intens_err * 1

	Angl_r[Angl_r == 0] = np.nan

	return Intns, Angl_r, Intns_err

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
