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

def flux_scale(data, z0, zref):
	obs = data
	z0 = z0
	z1 = zref
	flux = obs * (1 + z0)**4 / (1 + z1)**4
	return flux

def flux_recal(data, z0, zref):
	obs = data
	z0 = z0
	z1 = zref
	Da0 = Test_model.angular_diameter_distance(z0).value
	Da1 = Test_model.angular_diameter_distance(z1).value
	flux = obs * (1 + z0)**4 * Da0**2 / ((1 + z1)**4 * Da1**2)
	return flux

def angu_area(s0, z0, zref):
	s0 = s0
	z0 = z0
	z1 = zref
	Da0 = Test_model.angular_diameter_distance(z0).value
	Da1 = Test_model.angular_diameter_distance(z1).value
	angu_S = s0*Da0**2/Da1**2
	return angu_S

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
	cen_close = small
	pixel = psize
	R_pixel = Rp
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
	chi = theta * 180/np.pi

	r = np.logspace(0, np.log10(Rp), Nbins) # in unit "pixel"
	ia = r <= cen_close
	ib = np.array(np.where(ia == True))
	ic = ib.shape[1]
	rbin = r[ic-1:]
	rbin[0] = np.mean(r[ia])

	intens = np.zeros(len(r) - ic + 1, dtype = np.float)
	intens_r = np.zeros(len(r) - ic + 1, dtype = np.float)
	intens_err = np.zeros(len(r) - ic + 1, dtype = np.float)

	light = np.zeros(len(r) - ic + 1, dtype = np.float)
	R = np.zeros(len(r) - ic + 1, dtype = np.float)
	Angur = np.zeros(len(r) - ic + 1, dtype = np.float)
	SB_error = np.zeros(len(r) - ic + 1, dtype = np.float)
	PN_rbin = np.zeros(len(r) - ic + 1, dtype = np.float)

	dr = np.sqrt(((2*pix_id[0] + 1) / 2 - (2*cx + 1) / 2)**2 + 
		((2*pix_id[1] + 1) / 2 - (2*cy + 1) / 2)**2)

	for k in range(len(rbin) - 1):
		cdr = rbin[k + 1] - rbin[k]
		d_phi = (cdr / rbin[k]) * 180/np.pi
		phi = np.arange(0, 360, d_phi)
		phi = phi - 180

		ir = (dr > (2 * rbin[k] +1) / 2) & (dr <= (2 * rbin[k + 1] +1) / 2)
		io = np.where(ir == True)
		num = len(io[0])

		r_iner = (2 * rbin[k]+1) / 2
		r_out = (2 * rbin[k + 1] +1) / 2

		if num == 0:
			light[k] = np.nan
			SB_error[k] = np.nan
			R[k] = 0.5 * (r_iner + r_out) * pixel * Da0*10**3 / rad2arcsec
			Angur[k] = 0.5 * (r_iner + r_out) * pixel

			intens_r[k] = 0.5 * (r_iner + r_out) * pixel
			intens[k] = np.nan
			intens_err[k] = np.nan

			PN_rbin[k] = np.nan
		else:
			iy = io[0]
			ix = io[1]
			sampf = f_data[iy, ix]

			tot_flux = np.nanmean(sampf)
			tot_area = pixel**2
			light[k] = 22.5 - 2.5*np.log10(tot_flux) + 2.5*np.log10(tot_area)
			R[k] = 0.5 * (r_iner + r_out) * pixel * Da0*10**3/rad2arcsec
			Angur[k] = 0.5 * (r_iner + r_out) * pixel

			intens[k] = tot_flux
			intens_r[k] = 0.5 * (r_iner + r_out) * pixel

			terr = []
			tmpf = []
			for tt in range(len(phi) - 1):
				iv = (chi >= phi[tt]) & (chi <= phi[tt+1])
				iu = iv & ir
				set_samp = f_data[iu]

				ttf = np.nanmean(set_samp)
				SB_in = 22.5 - 2.5*np.log10(ttf) + 2.5*np.log10(tot_area)
				terr.append(SB_in)
				tmpf.append(ttf)
			# rms of SB
			terr = np.array(terr)
			where_are_inf = np.isinf(terr)
			terr[where_are_inf] = np.nan
			id_zero = terr == 0
			terr[id_zero] = np.nan
			Trms = np.nanstd(terr)
			id_nan = np.isnan(terr)
			id_fals = id_nan == False
			Terr = terr[id_fals]
			SB_error[k] = Trms / np.sqrt(len(Terr) - 1)

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

			PN_rbin[k] = num * 1

	light[light == 0] = np.nan
	ll = light * 1

	R[R == 0] = np.nan
	RR = R * 1

	Angur[Angur == 0] = np.nan
	AA = Angur * 1

	SB_error[SB_error == 0] = np.nan
	EE = SB_error * 1

	intens[intens == 0] = np.nan
	Intns = intens * 1

	intens_r[intens_r == 0] = np.nan
	Intns_r = intens_r * 1
	
	intens_err[intens_err == 0] = np.nan
	Intns_err = intens_err * 1

	PN_rbin[PN_rbin == 0] = np.nan

	return ll, RR, AA, EE, Intns, Intns_r, Intns_err, PN_rbin

def weit_l_measure(data, weit_M, Nbin, small, Rp, cx, cy, psize, z):
	"""
	## this function used for test
	data: data used to measure
	Nbin: number of bins will devide
	Rp: radius in unit pixel number
	cx, cy: cluster central position in image frame (in inuit pixel)
	psize: pixel size
	z : the redshift of data
	weit_M: weight metrix of flux
	"""
	cx = cx
	cy = cy
	Nbins = Nbin
	f_data = data
	cen_close = small
	pixel = psize
	R_pixel = Rp
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
	chi = theta * 180/np.pi

	r = np.logspace(-1, np.log10(Rp), Nbins) # in unit "pixel"
	ia = r<= cen_close
	ib = np.array(np.where(ia == True))
	ic = ib.shape[1]
	rbin = r[ic-1:]
	rbin[0] = np.mean(r[ia])
	light = np.zeros(len(r)-ic+1, dtype = np.float)
	R = np.zeros(len(r)-ic+1, dtype = np.float)
	Angur = np.zeros(len(r)-ic+1, dtype = np.float)
	SB_error = np.zeros(len(r)-ic+1, dtype = np.float)
	dr = np.sqrt(((2*pix_id[0]+1)/2-(2*cx+1)/2)**2+
			((2*pix_id[1]+1)/2-(2*cy+1)/2)**2)

	for k in range(len(rbin)):
		cdr = rbin[k] - rbin[k-1]
		d_phi = (cdr / rbin[k]) * 180/np.pi
		phi = np.arange(0, 360, d_phi)
		phi = phi - 180

		if rbin[k] <= cen_close:
			ig = rbin <= cen_close
			subr = rbin[ig]
			ih = rbin[ig]
			im = len(ih)

			ir = dr <= rbin[im-1]
			io = np.where(ir == True)
			num = len(io[0])

			if num == 0:
				light[k] = 0
				SB_error[k] = 0
				R[k] = np.mean(subr)*pixel*Da0*10**3/rad2arcsec
				Angur[k] = np.mean(subr)*pixel
			else:
				iy = io[0]
				ix = io[1]
				sampf = f_data[iy, ix][f_data[iy,ix] != 0]
				wm = weit_M[iy, ix][f_data[iy, ix] != 0]
				tot_flux = np.sum(sampf * wm) / np.sum(wm)
				tot_area = pixel**2

				light[k] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
				R[k] = np.mean(subr)*pixel*Da0*10**3/rad2arcsec
				Angur[k] = np.mean(subr)*pixel

				terr = []
				for tt in range(len(phi) - 1):
					iv = (chi >= phi[tt]) & (chi <= phi[tt+1])
					iu = iv & ir
					set_samp = f_data[iu][f_data[iu] != 0 ]
					err_wm = weit_M[iu][f_data[iu] != 0]

					ttf = np.sum(set_samp * err_wm) / np.sum(1 * err_wm)
					SB_in = 22.5-2.5*np.log10(ttf)+2.5*np.log10(tot_area)
					terr.append(SB_in)

				terr = np.array(terr)
				where_are_inf = np.isinf(terr)
				terr[where_are_inf] = 0
				where_are_nan = np.isnan(terr)
				terr[where_are_nan] = 0

				Terr = terr[terr != 0]
				Trms = np.std(Terr)
				SB_error[k] = Trms/np.sqrt(len(Terr) - 1)
			k = im+1

		else:
			ir = (dr > rbin[k-1]) & (dr <= rbin[k])
			io = np.where(ir == True)
			num = len(io[0])

			if num == 0:
				light[k] = 0
				SB_error[k] = 0
				R[k-im] = 0.5*(rbin[k-1]+rbin[k])*pixel*Da0*10**3/rad2arcsec
				Angur[k-im] = 0.5*(rbin[k-1]+rbin[k])*pixel
			else:
				iy = io[0]
				ix = io[1]
				sampf = f_data[iy, ix][f_data[iy,ix] != 0]				
				wm = weit_M[iy, ix][f_data[iy, ix] != 0]
				tot_flux = np.sum(sampf * wm) / np.sum(wm)
				tot_area = pixel**2

				light[k-im] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
				R[k-im] = 0.5*(rbin[k-1]+rbin[k])*pixel*Da0*10**3/rad2arcsec
				Angur[k-im] = 0.5*(rbin[k-1]+rbin[k])*pixel

				terr = []
				for tt in range(len(phi) - 1):
					iv = (chi >= phi[tt]) & (chi <= phi[tt+1])
					iu = iv & ir
					set_samp = f_data[iu][f_data[iu] != 0 ]
					err_wm = weit_M[iu][f_data[iu] != 0]

					ttf = np.mean(set_samp * err_wm) / np.sum(err_wm)
					SB_in = 22.5-2.5*np.log10(ttf)+2.5*np.log10(tot_area)
					terr.append(SB_in)

				terr = np.array(terr)
				where_are_inf = np.isinf(terr)
				terr[where_are_inf] = 0
				where_are_nan = np.isnan(terr)
				terr[where_are_nan] = 0

				Terr = terr[terr != 0]
				Trms = np.std(Terr)
				SB_error[k] = Trms/np.sqrt(len(Terr) - 1)

	# tick out the bad value
	where_are_nan1 = np.isnan(light)
	light[where_are_nan1] = 0
	where_are_inf1 = np.isinf(light)
	light[where_are_inf1] = 0

	where_are_nan2 = np.isnan(SB_error)
	SB_error[where_are_nan2] = 0
	where_are_inf2 = np.isinf(SB_error)
	SB_error[where_are_inf2] = 0

	ii = light != 0
	jj = SB_error != 0
	kk = ii & jj

	ll = light[kk]
	RR = R[kk]
	AA = Angur[kk]
	EE = SB_error[kk]
	return ll, RR, AA, EE

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
	rho_0 = M/((np.log(1+c)-c/(1+c))*4*np.pi*rs**3)
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
	Qc = kpc2m/Msun2kg # recrect parameter for rho_c
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