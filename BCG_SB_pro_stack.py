import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C

import h5py
import time
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits
import statistics as sts

from scipy import ndimage
from astropy import cosmology as apcy
from light_measure import light_measure
from Mass_rich_radius import rich2R

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
Lsun = C.L_sun.value*10**7
G = C.G.value

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel, z_ref = 0.396, 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel
M_dot = 4.83 # the absolute magnitude of SUN

home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

## profile catalogue [in unit of 'arcsec']
cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
					4.63,  7.43,  11.42,  18.20,  28.20, 
					44.21, 69.00, 107.81, 168.20, 263.00])
## the band info. of SDSS BCG pro. : 0, 1, 2, 3, 4 --> u, g, r, i, z

def BCG_pro_stack(band_id, rich_id, set_z, set_ra, set_dec, sample_id):

	if band_id == 0:
		pro_id = 2
	if band_id == 1:
		pro_id = 1
	if band_id == 2:
		pro_id = 3

	zn = len(set_z)
	d_load = load + 'rich_sample/jackknife/'
	# calculate the cat_Rii at z = 0.25 in physical unit (kpc)
	ref_Rii = Da_ref * cat_Rii * 10**3 / rad2asec # in unit kpc
	Nr = len(ref_Rii)
	SB_i = np.zeros((zn, Nr), dtype = np.float) + np.nan

	for tt in range(zn):
		z_g, ra_g, dec_g = set_z[tt], set_ra[tt], set_dec[tt]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		cat_pro = pds.read_csv(load + 'BCG_pros/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)

		dat_band = np.array(cat_pro.band)
		dat_bins = np.array(cat_pro.bin)
		dat_pro = np.array(cat_pro.profMean) # in unit of nmaggy / arcsec^2
		dat_pro_err = np.array(cat_pro.profErr)

		idx = dat_band == pro_id
		tt_pro = dat_pro[idx] * ((1 + z_g) / (1 + z_ref))**4 ## change the BCG pro to ref_z
		tt_proErr = dat_pro_err[idx] * ((1 + z_g) / (1 + z_ref))**4
		tt_bin = dat_bins[idx]
		tt_r = (cat_Rii[tt_bin] * Da_g / rad2asec) * 1e3 # arcsec --> kpc

		for mm in range( len(tt_r) ):
			dr = np.abs( ref_Rii - tt_r[mm] )
			idy = dr == np.min(dr)
			SB_i[tt,:][idy] = tt_pro[mm]

	sub_nr = np.zeros(Nr, dtype = np.float)
	for q in range(Nr):
		id_nan = np.isnan(SB_i[:, q])
		sub_nr[q] = len( SB_i[:,q][id_nan == False] )

	SB_mean = np.nanmean(SB_i, axis = 0)
	SB_std = np.nanstd(SB_i, axis = 0)
	SB_std = SB_std / np.sqrt(sub_nr)
	SB_pro = 22.5 - 2.5 * np.log10(SB_mean) + mag_add[band_id]

	SB0 = 22.5 - 2.5 * np.log10(SB_mean + SB_std) + mag_add[band_id]
	SB1 = 22.5 - 2.5 * np.log10(SB_mean - SB_std) + mag_add[band_id]
	err0 = SB_pro - SB0
	err1 = SB1 - SB_pro
	# save the result
	tmp_array = np.array([ref_Rii, SB_pro, err0, err1, SB_mean, SB_std])
	with h5py.File(d_load + '%s_band_%d_rich_%d_sub-samp_SB_pro.h5' % (band[band_id], rich_id, sample_id), 'w') as f:
		f['a'] = np.array(tmp_array)
	with h5py.File(d_load + '%s_band_%d_rich_%d_sub-samp_SB_pro.h5' % (band[band_id], rich_id, sample_id), ) as f:
		for ll in range( len(tmp_array) ):
			f['a'][ll, :] = tmp_array[ll, :]

	return

def main():
	rich_a0, rich_a1, rich_a2 = 20, 30, 50 # for lamda_k = 0, 1, 2
	N_bin = 30

	## stack BCG pros.
	for kk in range(rank, rank + 1):
		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		set_ra, set_dec, set_z, set_rich = set_array[0,:], set_array[1,:], set_array[2,:], set_array[3,:]

		for lamda_k in range(3):
			if lamda_k == 0:
				idx = (set_rich >= rich_a0) & (set_rich <= rich_a1)
			elif lamda_k == 1:
				idx = (set_rich >= rich_a1) & (set_rich <= rich_a2)
			else:
				idx = (set_rich >= rich_a2)

			lis_z = set_z[idx]
			lis_ra = set_ra[idx]
			lis_dec = set_dec[idx]

			zN = len(lis_z)
			n_step = zN // N_bin
			id_arr = np.linspace(0, zN - 1, zN)
			id_arr = id_arr.astype(int)

			for nn in range(N_bin):
				if nn == N_bin - 1:
					dot = id_arr[nn * n_step:]
				else:
					dot = id_arr[nn * n_step: (nn + 1) * n_step]
				id_use = list( set( id_arr ).difference( set( dot ) ) )
				id_use = np.array(id_use)

				z_use = lis_z[id_use]
				ra_use = lis_ra[id_use]
				dec_use = lis_dec[id_use]
				BCG_pro_stack(kk, lamda_k, z_use, ra_use, dec_use, nn)
	raise

if __name__ == "__main__":
	main()
