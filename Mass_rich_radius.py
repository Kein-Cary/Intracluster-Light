import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

	## view the R200 of samples
	load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
	band = ['r', 'g', 'i', 'u', 'z']
	rich_a0, rich_a1, rich_a2 = 20, 30, 50

	for kk in range(3):
		## R200 calculate
		with h5py.File(load + 'sky_select_img/%s_band_sky_0.80Mpc_select.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		set_z, set_rich = set_array[2,:], set_array[4,:]
		R_vir = rich2R(set_z, set_rich, M0, lamd0, z0, F_lamda, G_z, V_num)

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('%s band R200 PDF' % band[kk] )

		for lk in range(3):
			if lk == 0:
				idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
				sub_r200 = R_vir[idx]
				ax.hist(sub_r200, bins = 20, histtype = 'step', color = 'r', density = True, 
					label = '$ 20 \\leqslant \\lambda \\leqslant 30 $')
				ax.axvline(x = np.nanmean(sub_r200), linestyle = '--', color = 'r', label = 'mean')
				ax.axvline(x = np.nanmedian(sub_r200), linestyle = ':', color = 'r', label = 'median')
			elif lk == 1:
				idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
				sub_r200 = R_vir[idx]
				ax.hist(sub_r200, bins = 20, histtype = 'step', color = 'g', density = True, 
					label = '$ 30 \\leqslant \\lambda \\leqslant 50 $')
				ax.axvline(x = np.nanmean(sub_r200), linestyle = '--', color = 'g', label = 'mean')
				ax.axvline(x = np.nanmedian(sub_r200), linestyle = ':', color = 'g', label = 'median')
			else:
				idx = (set_rich >= rich_a2)
				sub_r200 = R_vir[idx]
				ax.hist(sub_r200, bins = 20, histtype = 'step', color = 'b', density = True, 
					label = '$ 50 \\leqslant \\lambda $')
				ax.axvline(x = np.nanmean(sub_r200), linestyle = '--', color = 'b', label = 'mean')
				ax.axvline(x = np.nanmedian(sub_r200), linestyle = ':', color = 'b', label = 'median')
		ax.axvline(x = 1e3, linestyle = '-', color = 'k', label = '1Mpc')
		ax.axvline(x = 1.1e3, linestyle = '--', color = 'k', label = '1.1Mpc')
		ax.legend(loc = 1)
		ax.set_ylabel('$ pdf $')
		ax.set_xlabel('$ R_{200}[kpc] $')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		plt.savefig(load + '%s_band_rich_R200.png' % band[kk], dpi = 300)
		plt.close()

if __name__ == "__main__":
	main()
