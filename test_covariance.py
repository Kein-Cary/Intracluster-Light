import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C

import h5py
import numpy as np
import pandas as pds
import statistics as sts
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.fits as fits

from scipy import interpolate as interp
from matplotlib.patches import Circle
from astropy import cosmology as apcy
from light_measure import light_measure, light_measure_rn

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
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def betwn_SB(data, R_low, R_up, cx, cy, pix_size, z0, band_id):

	betwn_r, betwn_Intns, betwn_err = light_measure_rn(data, R_low, R_up, cx, cy, pix_size, z0)
	betwn_lit = 22.5 - 2.5 * np.log10(betwn_Intns) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
	flux0 = betwn_Intns + betwn_err
	flux1 = betwn_Intns - betwn_err
	dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
	dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[band_id]
	btn_err0 = betwn_lit - dSB0
	btn_err1 = dSB1 - betwn_lit
	id_nan = np.isnan(dSB1)
	if id_nan == True:
		btn_err1 = 100.

	return betwn_r, betwn_lit, btn_err0, btn_err1, betwn_Intns, betwn_err

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

def SB_pro(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg, band_id):
	kk = band_id
	Intns, Intns_r, Intns_err = light_measure(img, R_bins, R_min, R_max, Cx, Cy, pix_size, zg)
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	err0 = SB - dSB0
	err1 = dSB1 - SB
	id_nan = np.isnan(SB)
	SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	out_err1[idx_nan] = 100.

	return R_out, SB_out, out_err0, out_err1, Intns, Intns_r, Intns_err

def cov_MX(radius, pros):
	flux_array = np.array(pros)
	r_array = np.array(radius)
	Nt = len(flux_array)
	SB_value = []
	R_value = []
	for ll in range(Nt):
		id_nan = np.isnan(flux_array[ll])
		setx = flux_array[ll][id_nan == False]
		setr = r_array[ll][id_nan == False]
		SB_value.append(setx)
		R_value.append(setr)
	SB_value = np.array(SB_value)
	R_value = np.array(R_value)
	R_mean_img = np.nanmean(R_value, axis = 0)

	mean_lit = np.nanmean(SB_value, axis = 0)
	std_lit = np.nanstd(SB_value, axis = 0)
	nx, ny = SB_value.shape[1], SB_value.shape[0]

	cov_tt = np.zeros((nx, nx), dtype = np.float)
	cor_tt = np.zeros((nx, nx), dtype = np.float)

	for qq in range(nx):
		for tt in range(nx):
			cov_tt[qq, tt] = np.sum( (SB_value[:,qq] - mean_lit[qq]) * (SB_value[:,tt] - mean_lit[tt]) ) / ny

	for qq in range(nx):
		for tt in range(nx):
			cor_tt[qq, tt] = cov_tt[qq, tt] / (std_lit[qq] * std_lit[tt])
	cov_MX_img = cov_tt * (ny - 1.) ## jackknife factor
	cor_MX_img = cor_tt * 1.
	return R_mean_img, cov_MX_img, cor_MX_img

N_bin = 30
#d_load = load + 'rich_sample/jackknife/'
#r_a0, r_a1 = 1.6e3, 1.8e3 ## also for correlation matrix comparison
d_load = load + 'rich_sample/scale_fig/R200_m/'
bl_set = 0.75 # bl_set must be 1.0 for 'R200_c/' case

for kk in range(rank, rank + 1):

	plt.figure()
	#plt.suptitle('BL %.1f-%.1f Mpc' % (r_a0 / 1e3, r_a1 / 1e3) )
	gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])
	ax0.set_title( '%s img ICL + sky ICL SB profile' % band[kk] )

	sub_SB = []
	sub_R = []
	sub_err0 = []
	sub_err1 = []
	scale_r = []

	for lamda_k in range(3):
		## read the virial radius cat.
		with h5py.File( load + 'rich_sample/jackknife/%s_band_%d_rich_R200m.h5' % (band[kk], lamda_k), 'r') as f:
			r200_array = np.array(f['a'])
		R200 = np.median(r200_array[0]) * bl_set
		#R200 = np.mean(r200_array[0])
		scale_r.append(R200)
		r_a0, r_a1 = R200, R200 + 100.

		## record the process data and calculate the Jackknife varinace
		SB_flux, R_arr = [], []
		stack_sb, stack_r = [], []
		add_sb, add_r = [], []
		sky_lit, sky_r = [], []

		for nn in range(N_bin):

			with h5py.File(d_load + '%s_band_%d_rich_%d_sub-stack_clust_SB_%.1f-%.1f_Mpc.h5' % 
				(band[kk], lamda_k, nn, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
				dmp_array = np.array(f['a'])
			stack_sb.append(dmp_array[1])
			stack_r.append(dmp_array[0])

			with h5py.File(d_load + '%s_band_%d_rich_%d_sub-stack_sky_ICL.h5' % (band[kk], lamda_k, nn), 'r') as f:
				dmp_array = np.array(f['a'])
			sky_lit.append(dmp_array[1])
			sky_r.append(dmp_array[0])

			with h5py.File(d_load + '%s_band_%d_rich_%d_sub-stack_add_SB_%.1f-%.1f_Mpc.h5' % 
				(band[kk], lamda_k, nn, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
				dmp_array = np.array(f['a'])
			add_sb.append(dmp_array[1])
			add_r.append(dmp_array[0])

			with h5py.File(d_load + '%s_band_%d_rich_%d_sub-stack_cli_SB_%.1f-%.1f_Mpc.h5' % 
				(band[kk], lamda_k, nn, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
				dmp_array = np.array(f['a'])
			SB_flux.append(dmp_array[1])
			R_arr.append(dmp_array[0])

		########## jackknife resampling result
		## mean of SB profile
		jk_Stack_SB, jk_Stack_R, jk_Stack_err0, jk_Stack_err1, Stack_R, Stack_SB, jk_Stack_err = jack_SB(stack_sb, stack_r, kk, N_bin)
		id_nan = np.isnan(Stack_SB)
		Stack_R, Stack_SB, jk_Stack_err = Stack_R[id_nan == False], Stack_SB[id_nan == False], jk_Stack_err[id_nan == False]

		jk_add_SB, jk_add_R, jk_add_err0, jk_add_err1, Add_R, Add_SB, jk_Add_err = jack_SB(add_sb, add_r, kk, N_bin)
		id_nan = np.isnan(Add_SB)
		Add_R, Add_SB, jk_Add_err = Add_R[id_nan == False], Add_SB[id_nan == False], jk_Add_err[id_nan == False]

		jk_sky_SB, jk_sky_R, jk_sky_err0, jk_sky_err1, sky_R, m_sky_SB, jk_sky_err = jack_SB(sky_lit, sky_r, kk, N_bin)
		id_nan = np.isnan(m_sky_SB)
		sky_R, m_sky_SB, jk_sky_err = sky_R[id_nan == False], m_sky_SB[id_nan == False], jk_sky_err[id_nan == False]

		JK_SB, JK_R, JK_err0, JK_err1, jk_cli_R, jk_cli_SB, jk_cli_err = jack_SB(SB_flux, R_arr, kk, N_bin)
		id_nan = np.isnan(jk_cli_SB)
		jk_cli_R, jk_cli_SB, jk_cli_err = jk_cli_R[id_nan == False], jk_cli_SB[id_nan == False], jk_cli_err[id_nan == False]

		## re-calculate SB profile
		sub_pros = 22.5 - 2.5 * np.log10(jk_cli_SB) + mag_add[kk]
		dSB0 = 22.5 - 2.5 * np.log10(jk_cli_SB + jk_cli_err) + mag_add[kk]
		dSB1 = 22.5 - 2.5 * np.log10(jk_cli_SB - jk_cli_err) + mag_add[kk]
		err0 = sub_pros - dSB0
		err1 = dSB1 - sub_pros
		idx_nan = np.isnan(dSB1)
		err1[idx_nan] = 100.
		sub_SB.append(sub_pros)
		sub_R.append(jk_cli_R)
		sub_err0.append(err0)
		sub_err1.append(err1)

		if lamda_k == 0:
			#ax0.errorbar(JK_R, JK_SB, yerr = [JK_err0, JK_err1], xerr = None, color = 'b', marker = 'None', ls = '-', linewidth = 1, 
			#	ecolor = 'b', elinewidth = 1, alpha = 0.5, label = '$ 20 \\leq \\lambda \\leq 30 $')
			ax0.plot(JK_R, JK_SB, color = 'b', alpha = 0.5, ls = '-',)
			ax0.fill_between(JK_R, y1 = JK_SB - JK_err0, y2 = JK_SB + JK_err1, color = 'b', alpha = 0.30, label = '$ 20 \\leq \\lambda \\leq 30 $')

		elif lamda_k == 1:
			#ax0.errorbar(JK_R, JK_SB, yerr = [JK_err0, JK_err1], xerr = None, color = 'g', marker = 'None', ls = '-', linewidth = 1, 
			#	ecolor = 'g', elinewidth = 1, alpha = 0.5, label = '$ 30 \\leq \\lambda \\leq 50 $')
			ax0.plot(JK_R, JK_SB, color = 'g', alpha = 0.5, ls = '-',)
			ax0.fill_between(JK_R, y1 = JK_SB - JK_err0, y2 = JK_SB + JK_err1, color = 'g', alpha = 0.30, label = '$ 30 \\leq \\lambda \\leq 50 $')

		else:
			#ax0.errorbar(JK_R, JK_SB, yerr = [JK_err0, JK_err1], xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1, 
			#	ecolor = 'r', elinewidth = 1, alpha = 0.5, label = '$ \\lambda \\geq 50 $')
			ax0.plot(JK_R, JK_SB, color = 'r', alpha = 0.5, ls = '-',)
			ax0.fill_between(JK_R, y1 = JK_SB - JK_err0, y2 = JK_SB + JK_err1, color = 'r', alpha = 0.30, label = '$ \\lambda \\geq 50 $')

	ax0.set_xlabel('$R[kpc]$')
	ax0.set_ylabel('$SB[mag / arcsec^2]$')
	ax0.set_xscale('log')
	ax0.set_ylim(19, 34)
	ax0.set_xlim(1, 2e3)
	ax0.legend(loc = 1, frameon = False)
	ax0.invert_yaxis()
	ax0.grid(which = 'both', axis = 'both')
	ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

	## deviation comparison
	id_nan = np.isnan(sub_SB[1])
	id_inf = np.isinf(sub_SB[1])
	idu = id_nan | id_inf
	inter_r = sub_R[1][idu == False]
	inter_sb = sub_SB[1][idu == False]
	f_SB = interp.interp1d(inter_r, inter_sb, kind = 'cubic')

	id_nan = np.isnan(sub_SB[0])
	id_inf = np.isinf(sub_SB[0])
	idu = id_nan | id_inf
	id_R0 = sub_R[0][idu == False]
	id_SB0 = sub_SB[0][idu == False]
	id_err0, id_err1 = sub_err0[0][idu == False], sub_err1[0][idu == False]
	idx = (id_R0 > np.min(inter_r)) & (id_R0 < np.max(inter_r))
	dev_R0 = id_R0[idx]
	dev_SB0 = id_SB0[idx] - f_SB(dev_R0)
	dev_err0_0, dev_err0_1 = id_err0[idx], id_err1[idx]

	id_nan = np.isnan(sub_SB[2])
	id_inf = np.isinf(sub_SB[2])
	idu = id_nan | id_inf
	id_R2 = sub_R[2][idu == False]
	id_SB2 = sub_SB[2][idu == False]
	id_err0, id_err1 = sub_err0[2][idu == False], sub_err1[2][idu == False]
	idx = (id_R2 > np.min(inter_r)) & (id_R2 < np.max(inter_r))
	dev_R2 = id_R2[idx]
	dev_SB2 = id_SB2[idx] - f_SB(dev_R2)
	dev_err2_0, dev_err2_1 = id_err0[idx], id_err1[idx]
	"""
	ax1.errorbar(dev_R2, dev_SB2, yerr = [dev_err2_0, dev_err2_1], xerr = None, color = 'r', marker = 'None', 
		ls = '-', linewidth = 1, ecolor = 'r', elinewidth = 1, alpha = 0.5,)
	ax1.errorbar(sub_R[1], sub_SB[1] - sub_SB[1], yerr = [ sub_err0[1], sub_err1[1] ], xerr = None, color = 'g', marker = 'None', 
		ls = '-', linewidth = 1, ecolor = 'g', elinewidth = 1, alpha = 0.5,)
	ax1.errorbar(dev_R0, dev_SB0, yerr = [dev_err0_0, dev_err0_1], xerr = None, color = 'b', marker = 'None', 
		ls = '-', linewidth = 1, ecolor = 'b', elinewidth = 1, alpha = 0.5,)
	"""
	ax1.plot(dev_R2, dev_SB2, color = 'r', alpha = 0.5, ls = '-')
	ax1.fill_between(dev_R2, y1 = dev_SB2 - dev_err2_0, y2 = dev_SB2 + dev_err2_1, color = 'r', alpha = 0.30,)
	ax1.plot(sub_R[1], sub_SB[1] - sub_SB[1], color = 'g', alpha = 0.5, ls = '-')
	ax1.fill_between(sub_R[1], y1 = 0 - sub_err0[1], y2 = 0 + sub_err1[1], color = 'g', alpha = 0.30,)
	ax1.plot(dev_R0, dev_SB0, color = 'b', alpha = 0.5, ls = '-')
	ax1.fill_between(dev_R0, y1 = dev_SB0 - dev_err0_0, y2 = dev_SB0 + dev_err0_1, color = 'b', alpha = 0.30,)

	ax1.set_xlim(ax0.get_xlim())
	ax1.set_ylim(-0.5, 0.5)
	ax1.set_xscale('log')
	ax1.set_xlabel('$ R[kpc] $')
	ax1.set_ylabel('$ SB - SB_{30 \\leq \\lambda \\leq 50} $')
	ax1.grid(which = 'both', axis = 'both')
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax0.set_xticklabels([])

	plt.subplots_adjust(hspace = 0.05)
	#plt.savefig( home + '%s_band_SB_rich_binned_%.1f-%.1f_Mpc.png' % (band[kk], r_a0 / 1e3, r_a1 / 1e3), dpi = 300)
	plt.savefig( home + '%s_band_SB_rich_binned.png' % band[kk], dpi = 300)
	plt.close()

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('$ %s \; band \; R_{200} \; scaled \; SB $' % band[kk])
	"""
	ax.errorbar(sub_R[2] / scale_r[2], sub_SB[2], yerr = [ sub_err0[2], sub_err1[2] ], xerr = None, color = 'r', marker = 'None', 
		ls = '-', linewidth = 1, ecolor = 'r', elinewidth = 1, alpha = 0.5, label = '$ \\lambda \\geq 50 $')
	ax.errorbar(sub_R[1] / scale_r[1], sub_SB[1], yerr = [ sub_err0[1], sub_err1[1] ], xerr = None, color = 'g', marker = 'None', 
		ls = '-', linewidth = 1, ecolor = 'g', elinewidth = 1, alpha = 0.5, label = '$ 30 \\leq \\lambda \\leq 50 $')
	ax.errorbar(sub_R[0] / scale_r[0], sub_SB[0], yerr = [ sub_err0[0], sub_err1[0] ], xerr = None, color = 'b', marker = 'None', 
		ls = '-', linewidth = 1, ecolor = 'b', elinewidth = 1, alpha = 0.5, label = '$ 20 \\leq \\lambda \\leq 30 $')
	"""
	ax.plot(sub_R[2] / scale_r[2], sub_SB[2], color = 'r', alpha = 0.5, ls = '-', label = '$ \\lambda \\geq 50 $')
	ax.fill_between(sub_R[2] / scale_r[2], y1 = sub_SB[2] - sub_err0[2], y2 = sub_SB[2] + sub_err1[2], color = 'r', alpha = 0.30,)
	ax.plot(sub_R[1] / scale_r[1], sub_SB[1], color = 'g', alpha = 0.5, ls = '-', label = '$ 30 \\leq \\lambda \\leq 50 $')
	ax.fill_between(sub_R[1] / scale_r[1], y1 = sub_SB[1] - sub_err0[1], y2 = sub_SB[1] + sub_err1[1], color = 'g', alpha = 0.30,)
	ax.plot(sub_R[0] / scale_r[0], sub_SB[0], color = 'b', alpha = 0.5, ls = '-', label = '$ 20 \\leq \\lambda \\leq 30 $')
	ax.fill_between(sub_R[0] / scale_r[0], y1 = sub_SB[0] - sub_err0[0], y2 = sub_SB[0] + sub_err1[0], color = 'b', alpha = 0.30,)	

	ax.set_xlabel('$ R / R_{200}$')
	ax.set_ylabel('$SB[mag / arcsec^2]$')
	ax.set_xscale('log')
	ax.set_ylim(19, 34)
	ax.set_xlim(1e-3, 2e0)
	ax.legend(loc = 1, frameon = False)
	ax.invert_yaxis()
	ax.grid(which = 'both', axis = 'both')
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.savefig( home + '%s_band_scaled_SB.png' % band[kk], dpi = 300)
	plt.close()

"""
		########## covariance matrix calculation
		R_mean_img, cov_Mx_img, cor_Mx_img = cov_MX(stack_r, stack_sb)
		R_mean_add, cov_Mx_add, cor_Mx_add = cov_MX(add_r, add_sb)
		R_mean_ICL, cov_Mx_ICL, cor_Mx_ICL = cov_MX(R_arr, SB_flux)

		cov_Mx_img, cov_Mx_add, cov_Mx_ICL = cov_Mx_img * (N_bin - 1.), cov_Mx_add * (N_bin - 1), cov_Mx_ICL * (N_bin - 1)

		with h5py.File( d_load + '%s_band_%d_rich_Jack_cov_Mx_%.1f-%.1f_Mpc.h5' % (band[kk], lamda_k, r_a0 / 1e3, r_a1 / 1e3), 'w') as f:
			f['a'] = np.array(cov_Mx_img)
		with h5py.File( d_load + '%s_band_%d_rich_Jack_cor_Mx_%.1f-%.1f_Mpc.h5' % (band[kk], lamda_k, r_a0 / 1e3, r_a1 / 1e3), 'w') as f:
			f['a'] = np.array(cor_Mx_img)

		## sub-sample case result
		with h5py.File( load + 'rich_sample/correlation_check/%s_band_%d_rich_cov_Mx_%.1f-%.1f_Mpc.h5' % 
			(band[kk], lamda_k, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
			cov_Mx_img_param = np.array(f['a'])
		with h5py.File( load + 'rich_sample/correlation_check/%s_band_%d_rich_cor_Mx_%.1f-%.1f_Mpc.h5' % 
			(band[kk], lamda_k, r_a0 / 1e3, r_a1 / 1e3), 'r') as f:
			cor_Mx_img_param = np.array(f['a'])

		differ_cov = cov_Mx_img - cov_Mx_img_param
		differ_cor = cor_Mx_img - cor_Mx_img_param

		#############
		plt.figure()
		if lamda_k == 0:
			plt.suptitle('$ %s \; band \; SB \; Covariance \; matrix [20 \\leqslant \\lambda \\leqslant 30] $' % band[kk], va = 'center')
		elif lamda_k == 1:
			plt.suptitle('$ %s \; band \; SB \; Covariance \; matrix [30 \\leqslant \\lambda \\leqslant 50] $' % band[kk], va = 'center')
		else:
			plt.suptitle('$ %s \; band \; SB \; Covariance \; matrix [50 \\leqslant \\lambda ] $' % band[kk], va = 'center')
		ax = plt.subplot(111)
		ax.set_title('Diagonal element of SB Covariance matrix')
		ax.plot(np.diag(cov_Mx_img), np.diag(cov_Mx_img_param), 'g*', alpha = 0.5, label = 'Jackknife vs sub-sample')
		ax.plot(np.diag(cov_Mx_img), np.diag(cov_Mx_img), 'r-', alpha = 0.5, label = 'y=x')

		ax.set_xlabel('Jackknife resampling $ [nanomaggies / arcsec^2]^2 $')
		ax.set_ylabel('sub-sample result $ [nanomaggies / arcsec^2]^2 $')
		if kk == 0:
			ax.set_xlim(1e-7, 2e0)
			ax.set_ylim(1e-7, 2e0)
		if kk == 1:
			ax.set_xlim(1e-7, 1e-1)
			ax.set_ylim(1e-7, 1e-1)
		if kk == 2:
			ax.set_xlim(1e-6, 5e0)
			ax.set_ylim(1e-6, 5e0)

		ax.set_xscale('log')
		ax.set_yscale('log')
		ax.legend(loc = 2, frameon = False)
		ax.grid(which = 'major', axis = 'both')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		plt.savefig(home + '%s_band_%d_rich_cov_matrix_%.1f-%.1f_Mpc.png' % (band[kk], lamda_k, r_a0 / 1e3, r_a1 / 1e3), dpi = 300)
		plt.close()

		plt.figure( figsize = (18, 6) )
		if lamda_k == 0:
			plt.suptitle('$ %s \; band \; SB \; correlation \; matrix [20 \\leqslant \\lambda \\leqslant 30] $' % band[kk], va = 'center')
		elif lamda_k == 1:
			plt.suptitle('$ %s \; band \; SB \; correlation \; matrix [30 \\leqslant \\lambda \\leqslant 50] $' % band[kk], va = 'center')
		else:
			plt.suptitle('$ %s \; band \; SB \; correlation \; matrix [50 \\leqslant \\lambda ] $' % band[kk], va = 'center')

		ax0 = plt.subplot(131)
		ax1 = plt.subplot(132)
		ax2 = plt.subplot(133)
		ax0.set_title('Jackknife resampling')
		ax1.set_title('sub-sample')
		ax2.set_title('Jackknife - sub-sample')

		tf = ax0.imshow(cor_Mx_img, cmap = 'seismic', origin = 'lower', vmin = -1, vmax = 1)
		plt.colorbar(tf, ax = ax0, fraction = 0.045, pad = 0.01)
		ax0.set_xlim(0, len(R_mean_img) - 1)
		ax0.set_ylim(0, len(R_mean_img) - 1)
		xtick = ax0.get_xticks(minor = False)
		idx = xtick < len(R_mean_img)
		ax0.set_xticks(xtick[idx])
		ax0.set_xticklabels('%.1f' % ll for ll in R_mean_img[ xtick[idx].astype(int) ])
		ax0.set_xlabel('R [kpc]')
		ytick = ax0.get_yticks(minor = False)
		idx = ytick < len(R_mean_img)
		ax0.set_yticks(ytick[idx])
		ax0.set_yticklabels('%.1f' % ll for ll in R_mean_img[ ytick[idx].astype(int) ])
		ax0.set_ylabel('R [kpc]')

		tf = ax1.imshow(cor_Mx_img_param, cmap = 'seismic', origin = 'lower', vmin = -1, vmax = 1)
		plt.colorbar(tf, ax = ax1, fraction = 0.045, pad = 0.01)
		ax1.set_xlim(0, len(R_mean_img) - 1)
		ax1.set_ylim(0, len(R_mean_img) - 1)
		xtick = ax1.get_xticks(minor = False)
		idx = xtick < len(R_mean_img)
		ax1.set_xticks(xtick[idx])
		ax1.set_xticklabels('%.1f' % ll for ll in R_mean_img[ xtick[idx].astype(int) ])
		ax1.set_xlabel('R [kpc]')
		ytick = ax1.get_yticks(minor = False)
		idx = ytick < len(R_mean_img)
		ax1.set_yticks(ytick[idx])
		ax1.set_yticklabels('%.1f' % ll for ll in R_mean_img[ ytick[idx].astype(int) ])
		ax1.set_ylabel('R [kpc]')

		tf = ax2.imshow(differ_cor, cmap = 'seismic', origin = 'lower', vmin = -1, vmax = 1)
		plt.colorbar(tf, ax = ax2, fraction = 0.045, pad = 0.01)
		ax2.set_xlim(0, len(R_mean_img) - 1)
		ax2.set_ylim(0, len(R_mean_img) - 1)
		xtick = ax2.get_xticks(minor = False)
		idx = xtick < len(R_mean_img)
		ax2.set_xticks(xtick[idx])
		ax2.set_xticklabels('%.1f' % ll for ll in R_mean_img[ xtick[idx].astype(int) ])
		ax2.set_xlabel('R [kpc]')
		ytick = ax2.get_yticks(minor = False)
		idx = ytick < len(R_mean_img)
		ax2.set_yticks(ytick[idx])
		ax2.set_yticklabels('%.1f' % ll for ll in R_mean_img[ ytick[idx].astype(int) ])
		ax2.set_ylabel('R [kpc]')

		plt.tight_layout()
		plt.savefig(home + '%s_band_%d_rich_cor_matrix_%.1f-%.1f_Mpc.png' % (band[kk], lamda_k, r_a0 / 1e3, r_a1 / 1e3), dpi = 300)
		plt.close()
"""
