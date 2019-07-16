# use for light measure
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

def flux_scale(data, s0, z0, zref):
	obs = data
	s0 = s0
	z0 = z0
	z_stak = zref
	ref_data = (obs/s0)*(1+z0)**4/(1+z_stak)**4 
	return ref_data

def flux_recal(data, z0, zref):
	obs = data
	z0 = z0
	z1 = zref
	Da0 = Test_model.angular_diameter_distance(z0).value
	Da1 = Test_model.angular_diameter_distance(z1).value
	flux = obs*((1+z0)**2*Da0)**2/((1+z1)**2*Da1)**2
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
				#sampf = f_data[iy, ix][f_data[iy,ix] != 0]

				sub_img = np.isnan(f_data[iy, ix])
				ntt = np.where(sub_img == False)
				sampf = f_data[iy, ix][ntt]	

				tot_flux = np.mean(sampf)
				tot_area = pixel**2
				light[k] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
				R[k] = np.mean(subr)*pixel*Da0*10**3/rad2arcsec
				Angur[k] = np.mean(subr)*pixel

				terr = []
				for tt in range(len(phi) - 1):
					iv = (chi >= phi[tt]) & (chi <= phi[tt+1])
					iu = iv & ir
					#set_samp = f_data[iu][f_data[iu] != 0]

					sub_img = np.isnan(f_data[iu])
					ntt = np.where(sub_img == False)
					set_samp = f_data[iu][ntt]

					ttf = np.mean(set_samp)
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
				R[k-im] = 0.5*(rbin[k-1] + rbin[k])*pixel*Da0*10**3/rad2arcsec
				Angur[k-im] = 0.5*(rbin[k-1] + rbin[k])*pixel
			else:
				iy = io[0]
				ix = io[1]
				#sampf = f_data[iy, ix][f_data[iy,ix] != 0]

				sub_img = np.isnan(f_data[iy, ix])
				ntt = np.where(sub_img == False)
				sampf = f_data[iy, ix][ntt]	

				tot_flux = np.mean(sampf)
				tot_area = pixel**2
				light[k-im] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
				R[k-im] = 0.5*(rbin[k-1]+rbin[k])*pixel*Da0*10**3/rad2arcsec
				Angur[k-im] = 0.5*(rbin[k-1]+rbin[k])*pixel

				terr = []
				for tt in range(len(phi) - 1):
					iv = (chi >= phi[tt]) & (chi <= phi[tt+1])
					iu = iv & ir
					#set_samp = f_data[iu][f_data[iu] != 0 ]

					sub_img = np.isnan(f_data[iu])
					ntt = np.where(sub_img == False)
					set_samp = f_data[iu][ntt]

					ttf = np.mean(set_samp)
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

def weit_l_measure(data, weit_M, Nbin, small, Rp, cx, cy, psize, z):
	"""
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
	#r = np.linspace(0, np.ceil(Rp), Nbins)
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
	rho_0 = (kpc2m/Msun2kg)*(3*H0**2)/(8*np.pi*G)
	r200_c = (3*M/(4*np.pi*rho_0*200))**(1/3) 
	rs = r200_c/c
	# next similar variables are for comoving coordinate, with simble "_c"
	rho0_c = M/((np.log(1+c)-c/(1+c))*4*np.pi*rs**3)
	r200_c = (3*M/(4*np.pi*rho_0*200))**(1/3)
	f0_c = 2*rho0_c*rs # use for test
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
	return  sigma_c

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
	#sigmamc(100, 15, 5)
	rho2d = sigmam(100, 15, 0, 5)

if __name__ == '__main__':
	main()