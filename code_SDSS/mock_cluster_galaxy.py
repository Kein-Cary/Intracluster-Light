import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import random
import skimage
import numpy as np
import pandas as pds
from scipy import interpolate as interp

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy.optimize import curve_fit, minimize
#from ICL_surface_mass_density import sigma_m_c
from light_measure_tmp import light_measure, flux_recal
from resample_modelu import down_samp, sum_samp

## constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
Lsun2erg = U.L_sun.to(U.erg/U.s)
rad2asec = U.rad.to(U.arcsec)
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
Angu_ref = rad2asec / Da_ref
Rpp = Angu_ref / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*Jy # zero point in unit (erg/s)/cm^-2
Lstar = 2e10 # in unit L_sun ("copy from Galaxies in the Universe", use for Luminosity function calculation)

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
	gal_L = 10**((M_c - new_M) / 2.5) * Lstar #mock galaxy luminosity, in unit of L_sun

	return gal_L

def galaxy():
	ng = 500
	m_set = np.linspace(-23, -16, ng)
	Mag_c = -21.79
	alpha_c = -1.18
	phi_c = 9.5 # in unit "h^2 Mpc^-2"
	ly = lumi_fun(m_set, Mag_c, alpha_c, phi_c) # in unit "L_sun"
	raise
	return

def main():
	galaxy()

if __name__ == "__main__":
	main()