import h5py
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy

G = C.G.value
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
Ms = C.M_sun.value # solar mass

band = ['r', 'g', 'i', 'u', 'z']

def rich2R_Melchior(z, lamda):
	"""
	Based on Melchior et al. 2017, the result r200 is the r200m (virial radius)
	"""
	M0, lamd0, z0 = 14.37, 30, 0.5
	F_lamda, G_z = 1.12, 0.18
	V_num = 200

	M_lam_z = M0 + F_lamda * np.log10(lamda / lamd0) + G_z * np.log10( (1 + z) / (1 + z0) )
	Qc = kpc2m / Msun2kg # correction fractor of rho_c
	M = 10**M_lam_z
	Ez = np.sqrt(Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
	Hz = H0*Ez
	rho_c = Qc * (3 * Hz**2) / (8 * np.pi * G) # in unit Msun/kpc^3
	omega_z = Test_model.Om( z )
	rho_m = V_num * rho_c * omega_z
	r200 = ( 3 * M / (4 * np.pi * rho_m) )**(1/3) # in unit kpc
	M200 = M

	return M200, r200

def rich2R_Simet(z, lamda, N_dist = 501):
	"""
	Based on Simet et al. 2017, the reuslt is r200m (virial radius)
	"""
	M = np.logspace(13.5, 15.5, N_dist)
	lamda0, M0, sigma_lnM = 40, 10**14.344, 0.25
	alpha = 1.33
	V_num = 200

	R200 = np.zeros(len(lamda), dtype = np.float)
	M200 = np.zeros(len(lamda), dtype = np.float)

	for tt in range(len(lamda)):

		dM = M[1:] - M[:-1]
		M_bar = 0.5 * (M[1:] + M[:-1])

		err_lnM = (alpha**2 / lamda[tt] + sigma_lnM**2)**(1 / 2)

		A = 1. / ( np.sqrt(2 * np.pi) * err_lnM )
		B = (-1. / 2) * ( np.log(M_bar) - (np.log(M0) + alpha * np.log(lamda[tt] / lamda0)) )**2 / err_lnM**2
		P_lnM_lamd = A * np.exp(B)
		P_M_lamd = P_lnM_lamd / M_bar
		est_M = np.sum(P_M_lamd * M_bar * dM) ## in unit of M_sun / h
		M200[tt] = (est_M / h) * 0.98 ## in unit of M_sun

		Qc = kpc2m / Msun2kg # correction fractor of rho_c
		Ez = np.sqrt(Omega_m * (1 + z[tt])**3 + Omega_k * (1 + z[tt])**2 + Omega_lambda)
		Hz = H0 * Ez
		rho_c = Qc * (3 * Hz**2) / (8 * np.pi * G) # in unit Msun/kpc^3
		omega_z = Test_model.Om( z[tt] ) # density parameter
		rho_m = V_num * rho_c * omega_z
		r200 = ( 3 * M200[tt] / (4 * np.pi * rho_m) )**(1/3) # in unit kpc
		R200[tt] = r200 * 1.

	return M200, R200

def rich2R_critical(z, lamda, N_dist = 501):
	"""
	Based on Simet et al. 2017, the reuslt is r200, but the 200 mean 
	a radius in which the mean density is 200 times of the critical density
	for given redshift (r200c).
	"""
	M = np.logspace(13.5, 15.5, N_dist)
	lamda0, M0, sigma_lnM = 40, 10**14.344, 0.25
	alpha = 1.33
	V_num = 200

	R200 = np.zeros(len(lamda), dtype = np.float)
	M200 = np.zeros(len(lamda), dtype = np.float)
	for tt in range(len(lamda)):

		dM = M[1:] - M[:-1]
		M_bar = 0.5 * (M[1:] + M[:-1])

		err_lnM = (alpha**2 / lamda[tt] + sigma_lnM**2)**(1 / 2)

		A = 1. / ( np.sqrt(2 * np.pi) * err_lnM )
		B = (-1. / 2) * ( np.log(M_bar) - (np.log(M0) + alpha * np.log(lamda[tt] / lamda0)) )**2 / err_lnM**2
		P_lnM_lamd = A * np.exp(B)
		P_M_lamd = P_lnM_lamd / M_bar
		est_M = np.sum(P_M_lamd * M_bar * dM) ## in unit of M_sun / h
		M200[tt] = (est_M / h) * 0.98 ## in unit of M_sun

		Qc = kpc2m / Msun2kg # correction fractor of rho_c
		Ez = np.sqrt(Omega_m * (1 + z[tt])**3 + Omega_k * (1 + z[tt])**2 + Omega_lambda)
		Hz = H0 * Ez
		rho_c = Qc * (3 * Hz**2) / (8 * np.pi * G) # in unit Msun/kpc^3
		rho_m = V_num * rho_c
		r200 = ( 3 * M200[tt] / (4 * np.pi * rho_m) )**(1/3) # in unit kpc
		R200[tt] = r200 * 1.

	return M200, R200

if __name__ == "__main__":
	main()

