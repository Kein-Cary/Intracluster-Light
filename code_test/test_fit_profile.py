import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from light_measure import sigmamc
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

M_dot = 4.83 # the absolute magnitude of SUN
pixel = 0.396 # in unit of arcsec
z_ref = 0.25
def SB_fit(r, m0, mc, c, m2l):
	bl = m0
	f_bl = 10**(0.4 * ( 22.5 - bl + 2.5 * np.log10(pixel**2) ) )

	surf_mass = sigmamc(r, mc, c)
	surf_lit = surf_mass / m2l
	SB_mod = M_dot - 2.5 * np.log10(surf_lit * 1e-6) + 10 * np.log10(1 + z_ref) + 21.572
	f_mod = 10**(0.4 * ( 22.5 - SB_mod + 2.5 * np.log10(pixel**2) ) )

	f_ref = f_mod + f_bl

	return f_ref

def crit_r(Mc, c):
	c = c
	M = 10**Mc
	rho_c = (kpc2m / Msun2kg)*(3*H0**2) / (8*np.pi*G)
	r200_c = (3*M / (4*np.pi*rho_c*200))**(1/3) 
	rs = r200_c / c
	return rs, r200_c

def sers_pro(r, mu_e, r_e, n):
	belta_n = 2*n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def main():
	bins = 65
	r = np.logspace(0, 3, bins)
	######## sersic
	mu_e = np.linspace(20, 25, 11)
	r_e = 20.
	n_e = 4.
	M_sb = np.zeros((len(mu_e), bins), dtype = np.float)
	for kk in range(len(mu_e)):
		SB = sers_pro(r, mu_e[kk], r_e, n_e)
		M_sb[kk, :] += SB
	plt.figure()
	ax = plt.subplot(111)
	for kk in range(len(mu_e)):
		ax.plot(r, M_sb[kk], linestyle = '-', color = mpl.cm.plasma(kk / len(mu_e)), label = '$\mu_{e} = %.1f$' % mu_e[kk], alpha = 0.5)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylim(20, 33)
	ax.set_ylabel('SB[mag/arcsec^2]')
	ax.legend(loc = 3)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_title('$ Sersic \; profile \; as \; function \; of \; \mu_{e} $')
	plt.savefig('sersic_SB_mu_e.png', dpi = 300)
	plt.close()

	mu_e = 23.5
	r_e = np.linspace(19, 29, 11)
	n_e = 4.
	M_sb = np.zeros((len(r_e), bins), dtype = np.float)
	for kk in range(len(r_e)):
		SB = sers_pro(r, mu_e, r_e[kk], n_e)
		M_sb[kk, :] += SB
	plt.figure()
	ax = plt.subplot(111)
	for kk in range(len(r_e)):
		ax.plot(r, M_sb[kk], linestyle = '-', color = mpl.cm.plasma(kk / len(r_e)), label = '$ R_{e} = %.1f$' % r_e[kk], alpha = 0.5)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylim(20, 33)
	ax.set_ylabel('SB[mag/arcsec^2]')
	ax.legend(loc = 3)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_title('$ Sersic \; profile \; as \; function \; of \; R_{e} $')
	plt.savefig('sersic_SB_R_e.png', dpi = 300)
	plt.close()

	mu_e = 23.5
	r_e = 20
	n_e = np.linspace(1.5, 4.5, 11)
	M_sb = np.zeros((len(n_e), bins), dtype = np.float)
	for kk in range(len(n_e)):
		SB = sers_pro(r, mu_e, r_e, n_e[kk])
		M_sb[kk, :] += SB
	plt.figure()
	ax = plt.subplot(111)
	for kk in range(len(n_e)):
		ax.plot(r, M_sb[kk], linestyle = '-', color = mpl.cm.plasma(kk / len(n_e)), label = '$ n = %.1f$' % n_e[kk], alpha = 0.5)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylim(20, 33)
	ax.set_ylabel('SB[mag/arcsec^2]')
	ax.legend(loc = 3)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_title('$ Sersic \; profile \; as \; function \; of \; index(n) $')
	plt.savefig('sersic_SB_n_e.png', dpi = 300)
	plt.close()
	raise

	######## NFW + C
	## test for Mc
	Mc = np.linspace(13.5, 15.5, 11) # M_sun
	M0 = 27.5 #mag/arcsec^2
	Cc = 5
	M2L = 50
	M_sb = np.zeros((len(Mc), bins), dtype = np.float)
	for kk in range(len(Mc)):
		flux = SB_fit(r, M0, Mc[kk], Cc, M2L)
		SB = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
		M_sb[kk, :] += SB

	plt.figure()
	ax = plt.subplot(111)
	for kk in range(len(Mc)):
		rs, r200 = crit_r(Mc[kk], Cc)
		ax.axvline(x = rs, linestyle = '--', color = mpl.cm.plasma(kk / len(Mc)), alpha = 0.5)
		ax.plot(r, M_sb[kk], linestyle = '-', color = mpl.cm.plasma(kk / len(Mc)), label = '$Mc = %.1f M_{\odot}$' % Mc[kk], alpha = 0.5)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylabel('SB[mag/arcsec^2]')
	ax.legend(loc = 3)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_title('SB profile as function of halo Mass')
	plt.savefig('SB_test_for_Mc.png', dpi = 300)
	plt.close()

	## test for M0
	Mc = 14
	M0 = np.linspace(26.5, 28.5, 11)
	Cc = 5
	M2L = 50
	M_sb = np.zeros((len(M0), bins), dtype = np.float)
	for kk in range(len(M0)):
		flux = SB_fit(r, M0[kk], Mc, Cc, M2L)
		SB = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
		M_sb[kk, :] += SB

	plt.figure()
	ax = plt.subplot(111)
	for kk in range(len(M0)):
		ax.plot(r, M_sb[kk], linestyle = '-', color = mpl.cm.plasma(kk / len(M0)), label = '$M0 = %.1f$' % M0[kk], alpha = 0.5)
	rs, r200 = crit_r(Mc, Cc)
	ax.axvline(x = rs, linestyle = '--', linewidth = 1, color = 'k', alpha = 0.5)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylabel('SB[mag/arcsec^2]')
	ax.legend(loc = 3)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_title('SB profile as function of residual sky')
	plt.savefig('SB_test_for_bl.png', dpi = 300)
	plt.close()

	## test for Cc
	Mc = 14
	M0 = 27.5
	Cc = np.linspace(2.5, 7.5, 11)
	M_sb = np.zeros((len(Cc), bins), dtype = np.float)
	for kk in range(len(Cc)):
		flux = SB_fit(r, M0, Mc, Cc[kk], M2L)
		SB = 22.5 - 2.5 * np.log10(flux) + 2.5 * np.log10(pixel**2)
		M_sb[kk, :] += SB

	plt.figure()
	ax = plt.subplot(111)
	for kk in range(len(Cc)):
		rs, r200 = crit_r(Mc, Cc[kk])
		ax.axvline(x = rs, linestyle = '--', linewidth = 1, color = mpl.cm.plasma(kk / len(Cc)), alpha = 0.5)		
		ax.plot(r, M_sb[kk], linestyle = '-', color = mpl.cm.plasma(kk / len(Cc)), label = 'C = %.1f' % Cc[kk], alpha = 0.5)
	ax.set_xscale('log')
	ax.set_xlabel('R[kpc]')
	ax.set_ylabel('SB[mag/arcsec^2]')
	ax.legend(loc = 3)
	ax.invert_yaxis()
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax.set_title('SB profile as function of concentration')
	plt.savefig('SB_test_for_Concen.png', dpi = 300)
	plt.close()

	return

if __name__ == "__main__":
	main()