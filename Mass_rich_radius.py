import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

def rich2R(z, lamda,  M0, lamd0, z0, F_lamda, G_z, V_num):
	M_lam_z = M0 + F_lamda * np.log10(lamda / lamd0) + G_z * np.log10( (1 + z) / (1 + z0) )

	Qc = kpc2m / Msun2kg # correction fractor of rho_c
	M = 10**M_lam_z
	Ez = np.sqrt(Omega_m * (1 + z)**3 + Omega_k * (1 + z)**2 + Omega_lambda)
	Hz = H0*Ez
	rho_c = Qc * (3 * Hz**2) / (8 * np.pi * G) # in unit Msun/kpc^3
	rho_m = V_num * rho_c
	r200 = ( 3 * M / (4 * np.pi * rho_m) )**(1/3) # in unit kpc
	return r200

def main():
	## relation from Melchior et al.2017, mass_rich relation
	M0, lamd0, z0 = 14.37, 30, 0.5
	F_lamda, G_z = 1.12, 0.18
	V_num = 200

	## case 1:
	z = np.linspace(0.2, 0.3, 6)
	lamda = 32
	R = rich2R(z, lamda, M0, lamd0, z0, F_lamda, G_z, V_num)

	plt.figure()
	ax = plt.subplot(111)
	ax.plot(z, R)
	ax.set_xlabel('z')
	ax.set_ylabel('$ R_{200}[kpc] $')
	plt.savefig('R200_z.png', dpi = 300)
	plt.close()

	z = 0.3
	lamda = np.linspace(20, 50, 6)
	R = rich2R(z, lamda, M0, lamd0, z0, F_lamda, G_z, V_num)

	plt.figure()
	ax = plt.subplot(111)
	ax.plot(lamda, R)
	ax.set_xlabel('$ \\lambda $')
	ax.set_ylabel('$ R_{200}[kpc] $')
	plt.savefig('R200_lambda.png', dpi = 300)
	plt.close()	

if __name__ == "__main__":
	main()
