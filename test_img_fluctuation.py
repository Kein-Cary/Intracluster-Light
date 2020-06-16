import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
from matplotlib.patches import Circle
from astropy import cosmology as apcy
from scipy.ndimage import map_coordinates as mapcd
from astropy.io import fits as fits
from scipy import interpolate as interp
from light_measure import light_measure, light_measure_rn

import time
from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Rpp = (rad2asec / Da_ref) / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'

def jack_SB(SB_array, R_array, band_id, N_bins):
	## stacking profile based on flux
	dx_r = np.array(R_array)
	dy_sb = np.array(SB_array)
	Stack_R = np.nanmean(dx_r, axis = 0)
	Stack_SB = np.nanmean(dy_sb, axis = 0)
	std_Stack_SB = np.nanstd(dy_sb, axis = 0)
	jk_Stack_err = np.sqrt(N_bins - 1) * std_Stack_SB

	## change flux to magnitude
	jk_Stack_SB = 22.5 - 2.5 * np.log10(Stack_SB) + mag_add[band_id]
	dSB0 = 22.5 - 2.5 * np.log10(Stack_SB + jk_Stack_err) + mag_add[band_id]
	dSB1 = 22.5 - 2.5 * np.log10(Stack_SB - jk_Stack_err) + mag_add[band_id]
	err0 = jk_Stack_SB - dSB0
	err1 = dSB1 - jk_Stack_SB
	id_nan = np.isnan(jk_Stack_SB)
	jk_Stack_SB, jk_Stack_R = jk_Stack_SB[id_nan == False], Stack_R[id_nan == False]
	jk_Stack_err0, jk_Stack_err1 = err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	jk_Stack_err1[idx_nan] = 100.
	return jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err

N_bin = 30
d_load = load + 'rich_sample/scale_fig/R200_m/' ## jackknife samples
sub_load = load + 'rich_sample/jack_sub-sample/'  ## individual sub-samples

line_id = 1 # 0: background subtracted, 1: before background subtraction
bl_set = 0.75

for kk in range(rank, rank + 1):

	plt.figure(figsize = (12, 6))
	ax0 = plt.subplot(121)
	ax1 = plt.subplot(122)
	if line_id == 0:
		ax0.set_title( '%s img ICL + sky ICL SB profile' % band[kk] )
	if line_id == 1:
		ax0.set_title( '%s img ICL + sky ICL + background SB profile' % band[kk] )

	for lamda_k in range(2,3):
		## read the virial radius cat.
		with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200m.h5' % (band[kk], lamda_k), 'r') as f:
			r200_array = np.array(f['a'])
		R200 = np.median(r200_array[0])
		r_a0, r_a1 = bl_set * R200, bl_set * R200 + 100.

		## record the process data and calculate the Jackknife varinace
		SB_flux, R_arr = [], []
		stack_sb, stack_r = [], []
		add_sb, add_r = [], []

		check_sb, check_r = [], []

		for nn in range(N_bin):
			## jackknife samples
			with h5py.File(d_load + '%s_band_%d_rich_%d_sub-stack_clust_SB_%.1f-%.1f_Mpc.h5' % (band[kk], lamda_k, nn, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
				dmp_array = np.array(f['a'])
			stack_sb.append(dmp_array[1])
			stack_r.append(dmp_array[0])

			with h5py.File(d_load + '%s_band_%d_rich_%d_sub-stack_add_SB_%.1f-%.1f_Mpc.h5' % (band[kk], lamda_k, nn, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
				dmp_array = np.array(f['a'])
			add_sb.append(dmp_array[1])
			add_r.append(dmp_array[0])

			add_flux = dmp_array[1]
			add_R = dmp_array[0]
			add_SB = 22.5 - 2.5 * np.log10(add_flux) + mag_add[kk]
			id_nan = np.isnan(add_SB)
			add_SB = add_SB[id_nan == False]
			add_R = add_R[id_nan == False]
			add_flux = add_flux[id_nan == False]

			with h5py.File(d_load + '%s_band_%d_rich_%d_sub-stack_cli_SB_%.1f-%.1f_Mpc.h5' % (band[kk], lamda_k, nn, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
				dmp_array = np.array(f['a'])
			SB_flux.append(dmp_array[1])
			R_arr.append(dmp_array[0])

			cli_flux = dmp_array[1]
			cli_R = dmp_array[0]
			cli_SB = 22.5 - 2.5 * np.log10(cli_flux) + mag_add[kk]

			## individual sub-samples
			with h5py.File(sub_load + '%s_band_%d_rich_%d_sub-sample_R200m.h5' % (band[kk], lamda_k, nn), 'r') as f:
				dmp_array_m = np.array(f['a'])
			indivi_r200 = dmp_array_m[0]
			indivi_r_a0, indivi_r_a1 = bl_set * indivi_r200, bl_set * indivi_r200 + 100.

			with h5py.File(sub_load + '%s_band_%d_rich_%d_sub-sample_clust_stack_SB.h5' % (band[kk], lamda_k, nn), 'r') as f:
				dmp_array = np.array(f['a'])
			sub_clust_R, sub_clust_flux = dmp_array[0], dmp_array[1]

			with h5py.File(sub_load + '%s_band_%d_rich_%d_sub-sample_add-img_SB.h5' % (band[kk], lamda_k, nn), 'r') as f:
				dmp_array = np.array(f['a'])
			sub_add_R, sub_add_flux = dmp_array[0], dmp_array[1]
			check_sb.append(sub_add_flux)
			check_r.append(sub_add_R)

			sub_add_SB = 22.5 - 2.5 * np.log10(sub_add_flux) + mag_add[kk]
			id_nan = np.isnan(sub_add_SB)
			sub_add_SB = sub_add_SB[id_nan == False]
			sub_add_R = sub_add_R[id_nan == False]
			sub_add_flux = sub_add_flux[id_nan == False]

			with h5py.File(sub_load + '%s_band_%d_rich_%d_sub-sample_ICL_SB.h5' % (band[kk], lamda_k, nn), 'r') as f:
				dmp_array = np.array(f['a'])
			sub_cli_R, sub_cli_flux = dmp_array[0], dmp_array[1]
			sub_cli_SB = 22.5 - 2.5 * np.log10(sub_cli_flux) + mag_add[kk]
			id_nan = np.isnan(sub_cli_SB)
			sub_cli_SB = sub_cli_SB[id_nan == False]
			sub_cli_R = sub_cli_R[id_nan == False]
			sub_cli_flux = sub_cli_flux[id_nan == False]

			if line_id == 0:
				ax0.plot(cli_R, cli_flux, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)
				ax1.plot(cli_R, cli_SB, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)

				#ax0.plot(sub_cli_R, sub_cli_flux, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)
				#ax1.plot(sub_cli_R, sub_cli_SB, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)
			if line_id == 1:
				ax0.plot(add_R, add_flux, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)
				ax1.plot(add_R, add_SB, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)

				#ax0.plot(sub_add_R, sub_add_flux, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)
				#ax1.plot(sub_add_R, sub_add_SB, ls = '--', color = mpl.cm.rainbow(nn / N_bin), alpha = 0.5)

		########## jackknife resampling result
		## mean of SB profile
		jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err = jack_SB(stack_sb, stack_r, kk, N_bin)
		id_nan = np.isnan(Stack_SB)
		Stack_R, Stack_SB, jk_Stack_err = Stack_R[id_nan == False], Stack_SB[id_nan == False], jk_Stack_err[id_nan == False]

		jk_add_SB, jk_add_R, jk_add_err0, jk_add_err1, Add_R, Add_SB, jk_Add_err = jack_SB(add_sb, add_r, kk, N_bin)
		id_nan = np.isnan(Add_SB)
		Add_R, Add_SB, jk_Add_err = Add_R[id_nan == False], Add_SB[id_nan == False], jk_Add_err[id_nan == False]

		JK_SB, JK_R, JK_err0, JK_err1, jk_cli_R, jk_cli_SB, jk_cli_err = jack_SB(SB_flux, R_arr, kk, N_bin)
		id_nan = np.isnan(jk_cli_SB)
		jk_cli_R, jk_cli_SB, jk_cli_err = jk_cli_R[id_nan == False], jk_cli_SB[id_nan == False], jk_cli_err[id_nan == False]

		if line_id == 0:
			ax0.errorbar(jk_cli_R, jk_cli_SB, yerr = jk_cli_err, xerr = None, color = 'k', marker = 'None', 
				ls = '-', linewidth = 1, ecolor = 'k', elinewidth = 1, alpha = 0.5, label = 'Jackknife SB')
		if line_id == 1:
			ax0.errorbar(Add_R, Add_SB, yerr = jk_Add_err, xerr = None, color = 'k', marker = 'None', 
				ls = '-', linewidth = 1, ecolor = 'k', elinewidth = 1, alpha = 0.5, label = 'Jackknife SB')

		ax0.axvline(x = bl_set * R200, color = 'k', alpha = 0.5, linestyle = '--', label = 'Median of 0.75 R200c %.1f Mpc' % (bl_set * R200) )

		ax0.set_xlabel('$ R[kpc] $')
		ax0.set_ylabel('$ SB[nmaggy / arcsec^2] $')
		ax0.set_xscale('log')
		ax0.set_yscale('log')

		if line_id == 0:
			ax0.set_xlim(1, 2e3)
			ax0.set_ylim(1e-5, 1e1)
		if line_id == 1:
			ax0.set_xlim(2e2, 2e3)
			ax0.set_ylim(2e-3, 2e-2)
			#ax0.set_ylim(2e-4, 3e-2)

		ax0.legend(loc = 1, frameon = False)
		#ax0.grid(which = 'minor', axis = 'both')
		ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

		if line_id == 0:
			ax1.errorbar(JK_R, JK_SB, yerr = [JK_err0, JK_err1], xerr = None, color = 'k', marker = 'None', 
				ls = '-', linewidth = 1, ecolor = 'k', elinewidth = 1, alpha = 0.5, label = 'Jackknife SB')
		if line_id == 1:
			ax1.errorbar(jk_add_R, jk_add_SB, yerr = [jk_add_err0, jk_add_err1], xerr = None, color = 'k', marker = 'None', 
				ls = '-', linewidth = 1, ecolor = 'k', elinewidth = 1, alpha = 0.5, label = 'Jackknife SB')

		ax1.axvline(x = bl_set * R200, color = 'k', alpha = 0.5, linestyle = '--', )
		ax1.set_xlabel('$R[kpc]$')
		ax1.set_ylabel('$SB[mag / arcsec^2]$')
		ax1.set_xscale('log')

		if line_id == 0:
			ax1.set_xlim(1, 2e3)
			ax1.set_ylim(19, 34)
		if line_id == 1:
			ax1.set_xlim(2e2, 2e3)
			ax1.set_ylim(27, 31)

		ax1.legend(loc = 1, frameon = False)
		ax1.invert_yaxis()
		#ax1.grid(which = 'minor', axis = 'both')
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.tight_layout()

		if line_id == 0:
			plt.savefig(home + '%s_ICL_%d_rich_SB.png' % (band[kk], lamda_k), dpi = 300)
			#plt.savefig(home + '%s_ICL_%d_rich_sub-sample_SB.png' % (band[kk], lamda_k), dpi = 300)
		if line_id == 1:
			plt.savefig(home + '%s_ICL_%d_rich_SB_before_sub-BL.png' % (band[kk], lamda_k), dpi = 300)
			#plt.savefig(home + '%s_ICL_%d_rich_sub-sample_SB_before_sub-BL.png' % (band[kk], lamda_k), dpi = 300)
		plt.close()

