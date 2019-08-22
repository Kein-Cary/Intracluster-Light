import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
from scipy import interpolate as interp

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy

from resamp import gen
from ICL_surface_mass_density import sigma_m_c
from light_measure import light_measure, flux_recal

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
Lsun2erg = U.L_sun.to(U.erg/U.s)
rad2arsec = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Angu_ref = rad2arsec / Da_ref
Rpp = Angu_ref / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*Jy # zero point in unit (erg/s)/cm^-2

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

Lstar = 2e10 # in unit Lsun
def lumi_fun(M_arr, Mag_c, alpha_c, phi_c):
	M_c = Mag_c 
	alpha = alpha_c
	phi = phi_c

	M_arr = M_arr
	X = 10**(-0.4 * (M_arr - M_c))
	lumi_f = phi * X**(alpha + 1) * np.exp(-1 * X)
	'''
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(M_arr, np.log10(lumi_f), 'r-')
	ax.set_title('galaxy luminosity function')
	ax.set_xlabel('$M_r - 5 log h$')
	ax.set_ylabel('$log(\phi) \, h^2 Mpc^{-2}$')
	ax.text(-17, 0, s = '$Mobasher \, et \, al.2003$' + '\n' + 'for Coma cluster')
	ax.invert_xaxis()
	plt.savefig('/home/xkchen/mywork/ICL/code/mock_lumifun.png', dpi = 300)
	plt.close()
	'''
	dM = M_arr[1:] - M_arr[:-1]
	mid_M = 0.5 * (M_arr[1:] + M_arr[:-1])
	Acum_l = np.zeros(len(dM), dtype = np.float)
	for kk in range(1, len(dM) + 1):
		sub_x = mid_M[:kk] * dM[:kk]
		Acum_l[kk - 1] = np.sum(sub_x)
	Acum_l = Acum_l / Acum_l[-1]
	'''
	plt.figure()
	ax = plt.subplot(111)
	ax.plot(mid_M, np.log10(Acum_l), 'r-')
	ax.set_title('integration of luminosity function')
	ax.set_xlabel('$M_r - 5 log h$')
	ax.set_ylabel('$log(\phi \, dM) \, h^2 Mpc^{-2}$')
	ax.invert_xaxis()
	plt.savefig('/home/xkchen/mywork/ICL/code/acum_lumifun.png', dpi = 300)
	plt.close()
	'''
	L_set = interp.interp1d(Acum_l, mid_M)
	new_P = np.linspace(np.min(Acum_l), np.max(Acum_l), len(M_arr))
	new_M = L_set(new_P)
	gal_L = 10**((M_c - new_M) / 2.5) * Lstar

	return gal_L

def cluster(Mh, Nbin, Nx1, Nx2, z):
	Mhc = Mh
	N = Nbin
	M2L = 250
	R, rho_2d, rbin = sigma_m_c(Mhc, N)
	Lc = rho_2d/M2L
	Lz = (Lc / (z+1)**4) / (4 * np.pi * rad2arsec**2)
	Lob = Lz*Lsun / kpc2cm**2

	R = np.max(rbin)
	r_sc = rbin / R
	flux_func = interp.interp1d(r_sc, Lob, kind = 'cubic')

	cx = np.int(Nx1 / 2)
	cy = np.int(Nx2 / 2)
	Da0 = Test_model.angular_diameter_distance(z).value
	Angu_r = (10**(-3) * R / Da0) * rad2arsec
	R_pixel = Angu_r / pixel
	r_in = (rbin * 10**(-3) / Da0) * rad2arsec

	y0 = np.linspace(0, Nx2 - 1, Nx2)
	x0 = np.linspace(0, Nx1 - 1, Nx1)
	frame = np.zeros((len(y0), len(x0)), dtype = np.float)
	pxl = np.meshgrid(x0, y0)

	dr = np.sqrt(((2 * pxl[0] + 1) / 2 - (2 * cx + 1) / 2)**2 + ((2 * pxl[1] + 1) / 2 - (2 * cy + 1) / 2)**2)
	dr_sc = dr / R_pixel
	ix = np.abs(x0 - cx)
	iy = np.abs(y0 - cy)
	ix0 = np.where(ix == np.min(ix))[0][0]
	iy0 = np.where(iy == np.min(iy))[0][0]

	test_dr = dr_sc[iy0, ix0 + 1: ix0 + 1 + np.int(R_pixel)]
	test = flux_func(test_dr)
	iat = r_sc <= test_dr[0]
	ibt = r_in[iat]
	ict = Lob[iat]
	for k in range(len(test_dr)):
		if k == 0:
			continue
		else:
			ia = (dr_sc >= test_dr[k-1]) & (dr_sc < test_dr[k])
			ib = np.where(ia == True)
			frame[ib[0], ib[1]] = flux_func(test_dr[k-1]) * pixel**2 / (f0*10**(-9))
	back_lel = np.ones((Nx2, Nx1), dtype = np.float) * 1.89 * test[-1] * pixel**2 / (f0*10**(-9))
	mock_ccd = back_lel + frame
	mock_ccd = mock_ccd - np.mean(mock_ccd)
	Gnose = np.random.random()

	plt.figure()
	plt.title('mock cluster with NFW model + Gaussian noise')
	plt.imshow(mock_ccd, cmap = 'rainbow', origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(label = 'flux[nMgy]', fraction = 0.035, pad = 0.003)
	#plt.savefig('/home/xkchen/mywork/ICL/code/mock_cluster_with_galaxy.png', dpi = 300)
	plt.show()
	
	raise
	return mock_ccd

def galaxy():
	ng = 1000
	m_set = np.linspace(-23, -16, ng)
	Mag_c = -21.79
	alpha_c = -1.18
	phi_c = 9.5
	ly = lumi_fun(m_set, Mag_c, alpha_c, phi_c)
	# sersic profile formula


	return

def mock_img():
	'''
	Mag_c = -21.79
	alpha_c = -1.18
	phi_c = 9.5 # in unit "h^2 Mpc^-2"
	M_arr = np.linspace(-23, -16, 100)
	ly = lumi_fun(M_arr, Mag_c, alpha_c, phi_c)
	'''
	Mh = 15-np.log10(8)
	Nbin = 131 
	Nx1 = 2048
	Nx2 = 1489
	z = 0.2
	mock_img = cluster(Mh, Nbin, Nx1, Nx2, z)


	return

def fig_out():
	#frame = mock_img()


	return

def main():
	mock_img()
	fig_out()

if __name__ == "__main__":
	main()
