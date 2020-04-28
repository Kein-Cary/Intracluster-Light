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

## view the R200 of samples
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
band = ['r', 'g', 'i', 'u', 'z']

def rich2R_2017(z, lamda):
	"""
	Based on Melchior et al. 2017
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
	rho_m = V_num * rho_c
	r200 = ( 3 * M / (4 * np.pi * rho_m) )**(1/3) # in unit kpc

	return r200

def rich2R_2019(z, lamda, N_dist):
	"""
	Based on Simet et al. 2017
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

def main():
	N_dist = 301 ## use for rich2R_2019
	rich_a0, rich_a1, rich_a2 = 20, 30, 50

	for kk in range(3):
		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]

		M_vir, R_vir = rich2R_2019(set_z, set_rich, N_dist)

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('%s band R200 PDF' % band[kk] )

		for lk in range(3):
			if lk == 0:
				idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
				sub_rich = set_rich[idx]
				sub_r200 = R_vir[idx]
				sub_m200 = M_vir[idx]
				dmp_array = np.array([sub_r200, sub_m200])
				with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200.h5' % (band[kk], lk), 'w') as f:
					f['a'] = np.array(dmp_array)
				with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200.h5' % (band[kk], lk) ) as f:
					for mm in range(len(dmp_array)):
						f['a'][mm,:] = dmp_array[mm,:]

				ax.hist(sub_r200, bins = 20, histtype = 'step', color = 'r', density = True, 
					label = '$ 20 \\leq \\lambda \\leq 30 $')
				ax.axvline(x = np.nanmean(sub_r200), linestyle = '--', color = 'r')
				ax.axvline(x = np.nanmedian(sub_r200), linestyle = ':', color = 'r')
			elif lk == 1:
				idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
				sub_rich = set_rich[idx]
				sub_r200 = R_vir[idx]
				sub_m200 = M_vir[idx]
				dmp_array = np.array([sub_r200, sub_m200])
				with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200.h5' % (band[kk], lk), 'w') as f:
					f['a'] = np.array(dmp_array)
				with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200.h5' % (band[kk], lk) ) as f:
					for mm in range(len(dmp_array)):
						f['a'][mm,:] = dmp_array[mm,:]

				ax.hist(sub_r200, bins = 20, histtype = 'step', color = 'g', density = True, 
					label = '$ 30 \\leq \\lambda \\leq 50 $')
				ax.axvline(x = np.nanmean(sub_r200), linestyle = '--', color = 'g', label = 'mean')
				ax.axvline(x = np.nanmedian(sub_r200), linestyle = ':', color = 'g', label = 'median')
			else:
				idx = (set_rich >= rich_a2)
				sub_rich = set_rich[idx]
				sub_r200 = R_vir[idx]
				sub_m200 = M_vir[idx]
				dmp_array = np.array([sub_r200, sub_m200])
				with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200.h5' % (band[kk], lk), 'w') as f:
					f['a'] = np.array(dmp_array)
				with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200.h5' % (band[kk], lk) ) as f:
					for mm in range(len(dmp_array)):
						f['a'][mm,:] = dmp_array[mm,:]

				ax.hist(sub_r200, bins = 20, histtype = 'step', color = 'b', density = True, 
					label = '$ \\lambda \\geq 50 $')
				ax.axvline(x = np.nanmean(sub_r200), linestyle = '--', color = 'b')
				ax.axvline(x = np.nanmedian(sub_r200), linestyle = ':', color = 'b')

		ax.axvline(x = 1e3, linestyle = '-', color = 'k', label = '1Mpc')
		ax.axvline(x = 1.1e3, linestyle = '--', color = 'k', label = '1.1Mpc')
		ax.legend(loc = 1)
		ax.set_ylabel('$ pdf $')
		ax.set_xlabel('$ R_{200}[kpc] $')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.tight_layout()
		plt.savefig( home + '%s_band_rich_R200.png' % band[kk], dpi = 300)
		plt.close()

if __name__ == "__main__":
	main()
