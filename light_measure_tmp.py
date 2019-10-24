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

def light_measure(data, Nbin, small, Rp, cx, cy, psize, z):
	"""
	data: data used to measure
	Nbin: number of bins will devide
	Rp: radius in unit pixel number
	cx, cy: cluster central position in image frame (in inuit pixel)
	psize: pixel size
	z : the redshift of data
	"""
	cx = cx
	cy = cy
	Nbins = Nbin
	f_data = data
	pixel = psize * 1
	z0 = z
	Da0 = Test_model.angular_diameter_distance(z0).value
	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0))

	theta = np.arctan2((pix_id[1,:]-cy), (pix_id[0,:]-cx))
	where_are_nan = np.isnan(theta)
	theta[where_are_nan] = 0
	chi = theta * 180 / np.pi

	divi_r = np.logspace(np.log10(small), 3.04, Nbins)
	r = (divi_r * 1e-3 * rad2arcsec / Da0) / pixel
	ia = r <= small
	ib = np.array(np.where(ia == True))
	ic = ib.shape[1]
	rbin = r[ic-1:]
	rbin[0] = np.mean(r[ia])
	set_r = divi_r[ic-1:]

	intens = np.zeros(len(r) - ic + 1, dtype = np.float)
	intens_r = np.zeros(len(r) - ic + 1, dtype = np.float)
	intens_err = np.zeros(len(r) - ic + 1, dtype = np.float)
	Angur = np.zeros(len(r) - ic + 1, dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)

	for k in range(len(rbin) - 1):
		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / ( 0.5 * (rbin[k] + rbin[k + 1]) ) ) * 180 / np.pi
		phi = np.arange(0, 360, d_phi)
		phi = phi - 180

		ir = (dr >= rbin[k]) & (dr < rbin[k + 1])
		io = np.where(ir == True)
		num = len(io[0])

		r_iner = set_r[k] ## useing radius in unit of kpc
		r_out = set_r[k + 1]

		if num == 0:
			intens[k] = np.nan
			intens_err[k] = np.nan

			## in unit of kpc
			Angur[k] = 0.5 * (r_iner + r_out) * 1e-3 * rad2arcsec / Da0
			intens_r[k] = 0.5 * (r_iner + r_out)

			#Angur[k] = np.sqrt(r_iner * r_out) * 1e-3 * rad2arcsec / Da0
			#intens_r[k] = np.sqrt(r_iner * r_out)

		else:
			iy = io[0]
			ix = io[1]
			sampf = f_data[iy, ix]

			tot_flux = np.nanmean(sampf)
			tot_area = pixel**2
			intens[k] = tot_flux

			## in unit of kpc
			Angur[k] = 0.5 * (r_iner + r_out) * 1e-3 * rad2arcsec / Da0
			intens_r[k] = 0.5 * (r_iner + r_out)

			#Angur[k] = np.sqrt(r_iner * r_out) * 1e-3 * rad2arcsec / Da0
			#intens_r[k] = np.sqrt(r_iner * r_out)

			tmpf = []
			for tt in range(len(phi) - 1):
				iv = (chi >= phi[tt]) & (chi <= phi[tt+1])
				iu = iv & ir
				set_samp = f_data[iu]

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

	Angur[Angur == 0] = np.nan
	Ar = Angur * 1

	intens[intens == 0] = np.nan
	Intns = intens * 1

	intens_r[intens_r == 0] = np.nan
	Intns_r = intens_r * 1
	
	intens_err[intens_err == 0] = np.nan
	Intns_err = intens_err * 1

	return Intns, Intns_r, Ar, Intns_err
