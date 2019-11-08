import matplotlib as mpl
mpl.use('Agg')
import handy.scatter as hsc
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
import astropy.io.ascii as asc
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d as interp
from light_measure import light_measure, flux_recal, sigmamc

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
f0 = 3631*10**(-23) # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel
M_dot = 4.83 # the absolute magnitude of SUN

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5', 'r') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

goal_data = fits.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)

z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]

red_z = z_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_ra = ra_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_dec = dec_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_rich = rich_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def SB_fit(r, m0, mc, c, m2l):
	bl = m0
	f_bl = 10**(0.4 * ( 22.5 - bl + 2.5 * np.log10(pixel**2) ) )

	surf_mass = sigmamc(r, mc, c)
	surf_lit = surf_mass / m2l
	SB_mod = M_dot - 2.5 * np.log10(surf_lit * 1e-6) + 10 * np.log10(1 + z_ref) + 21.572
	f_mod = 10**(0.4 * ( 22.5 - SB_mod + 2.5 * np.log10(pixel**2) ) )

	f_ref = f_mod + f_bl

	return f_ref

def chi2(X, *args):
	m0 = X[0]
	mc = X[1]
	c = X[2]
	m2l = X[3]
	r, data, yerr = args
	m0 = m0
	mc = mc
	m2l = m2l
	c = c
	mock_L = SB_fit(r, m0, mc, c, m2l)
	chi = np.sum( ( (mock_L - data) / yerr)**2 )
	return chi

def crit_r(Mc, c):
	c = c
	M = 10**Mc
	rho_c = (kpc2m / Msun2kg)*(3*H0**2) / (8*np.pi*G)
	r200_c = (3*M / (4*np.pi*rho_c*200))**(1/3)
	rs = r200_c / c
	return rs, r200_c

def stack_process(band_number, subz, subra, subdec):

	stack_N = len(subz)
	ii = np.int(band_number)
	sub_z = subz
	sub_ra = subra
	sub_dec = subdec

	x0 = 2427
	y0 = 1765
	bins = 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	sum_array_B = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_B = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_B = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		except_cat = pds.read_csv(load + 'Except_%s_sample.csv' % band[ii])
		except_ra = ['%.3f' % ll for ll in except_cat['ra'] ]
		except_dec = ['%.3f' % ll for ll in except_cat['dec'] ]
		except_z = ['%.3f' % ll for ll in except_cat['z'] ]
		## rule out bad image (once a cluster image is bad in a band, all the five band image will be ruled out!)
		identi = ('%.3f'%ra_g in except_ra) & ('%.3f'%dec_g in except_dec) & ('%.3f'%z_g in except_z)

		if  identi == True: 
			continue
		else:
			## stack A mask

			# 1.5sigma
			data_A = fits.getdata(load + 
				'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			'''
			# Zibetti
			data_A = fits.getdata(load + 
				'resample/Zibetti/A_mask/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			'''
			img_A = data_A[0]
			xn = data_A[1]['CENTER_X']
			yn = data_A[1]['CENTER_Y']

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + img_A.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + img_A.shape[1])

			idx = np.isnan(img_A)
			idv = np.where(idx == False)

			sum_array_A[la0:la1, lb0:lb1][idv] = sum_array_A[la0:la1, lb0:lb1][idv] + img_A[idv]
			count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
			id_nan = np.isnan(count_array_A)
			id_fals = np.where(id_nan == False)
			p_count_A[id_fals] += 1.
			p_count_A[0, 0] += 1.
			count_array_A[la0: la1, lb0: lb1][idv] = np.nan

			## stack B mask

			# 1.5sigma
			data_B = fits.getdata(load + 
				'resample/resam_B/frameB-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			'''
			# Zibetti
			data_B = fits.getdata(load + 
				'resample/Zibetti/B_mask/frameB-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			'''
			img_B = data_B[0]
			xn = data_B[1]['CENTER_X']
			yn = data_B[1]['CENTER_Y']

			la0 = np.int(y0 - yn)
			la1 = np.int(y0 - yn + img_B.shape[0])
			lb0 = np.int(x0 - xn)
			lb1 = np.int(x0 - xn + img_B.shape[1])

			idx = np.isnan(img_B)
			idv = np.where(idx == False)

			sum_array_B[la0:la1, lb0:lb1][idv] = sum_array_B[la0:la1, lb0:lb1][idv] + img_B[idv]
			count_array_B[la0: la1, lb0: lb1][idv] = img_B[idv]
			id_nan = np.isnan(count_array_B)
			id_fals = np.where(id_nan == False)
			p_count_B[id_fals] += 1.
			p_count_B[0, 0] += 1.
			count_array_B[la0: la1, lb0: lb1][idv] = np.nan

	with h5py.File(load + 'test_h5/stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)

	with h5py.File(load + 'test_h5/stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)

	with h5py.File(load + 'test_h5/stack_Bmask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_B)

	with h5py.File(load + 'test_h5/stack_Bmask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_B)

	return

def main():
	t0 = time.time()

	bins, ta0, ta1 = 65, 2051, 2166
	RichN, zN, raN, decN = red_rich[ta0 : ta1], red_z[ta0 : ta1], red_ra[ta0 : ta1], red_dec[ta0 : ta1]
	stackn = len(zN)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	#commd.Barrier()
	#for tt in range(len(band)):
	for tt in range(3):
		m, n = divmod(stackn, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		stack_process(tt, zN[N_sub0 :N_sub1], raN[N_sub0 :N_sub1], decN[N_sub0 :N_sub1])
	commd.Barrier()

	Num = np.array([3388, 3382, 3368])
	### stack all of the sub-stack image
	# stack sub-stack img of A mask
	mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	if rank == 0:
		for qq in range(3):
			# read the SB of BCG + ICL + residual sky
			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_2.csv' % band[qq], skiprows = 1 )
			R_obs, SB_obs = SB_tt['Mpc'] * 1e3, SB_tt['mag/arcsec^2']

			tot_N = 0
			#tot_N = Num[qq]

			for pp in range(cpus):

				with h5py.File(load + 'test_h5/stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[qq]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(load + 'test_h5/stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[qq]), 'r') as f:
					sum_img = np.array(f['a'])

				sub_Num = np.nanmax(p_count)
				tot_N += sub_Num
				id_zero = p_count == 0
				ivx = id_zero == False
				mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
				p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

				# check sub-results
				p_count[id_zero] = np.nan
				sum_img[id_zero] = np.nan
				sub_mean = sum_img / p_count
				
				plt.figure()
				plt.title('[A]stack_%d_%s_band_%d_cpus' % (np.int(sub_Num), band[qq], pp) )
				ax = plt.imshow(sub_mean, cmap = 'Greys', vmin = 1e-7, vmax = 5e1, origin = 'lower', norm = mpl.colors.LogNorm())
				plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
				hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
				hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)
				plt.xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
				plt.ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)
				plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/A_stack_%d_%s_band_%d_cpus.png' % (np.int(sub_Num), band[qq], pp), dpi = 300)
				plt.close()
			
			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)
			stack_img = mean_img / p_add_count

			## save the stack image
			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (tot_N, band[qq]), 'w') as f:
				f['a'] = np.array(stack_img)

			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (tot_N, band[qq]), 'r') as f:
				stack_img = np.array(f['a'])

			R_cut = 900
			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, 10, Rpp, R_cut, R_cut, pixel, z_ref)
			flux0 = Intns + Intns_err
			flux1 = Intns - Intns_err
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[qq]
			SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[qq]
			SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[qq]
			err0 = SB - SB0
			err1 = SB1 - SB
			id_nan = np.isnan(SB)
			SBt, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
			Rt, err0, err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
			id_nan = np.isnan(SB1)
			err1[id_nan] = 100.
			'''
			## fit with NFW + C[residual sky]
			ix = Intns_r >= 100
			iy = Intns_r <= 900
			iz = ix & iy
			r_fit, f_obs, err_fit = Intns_r[iz], Intns[iz], Intns_err[iz]

			m0 = np.arange(26.5, 28.5, 0.25) # residual sky
			mc = np.arange(13.5, 15, 0.25)   # the total gravity mass of cluster
			cc = np.arange(1., 7.5, 0.25)       # concentration
			m2l = np.arange(35, 65, 1)     # mass to light ratio (Mo . book)
			popt = minimize(chi2, x0 = np.array([m0[0], mc[0], cc[0], m2l[0]]), args = (r_fit, f_obs, err_fit), 
				method = 'L-BFGS-B', bounds = ( (26, 29), (13.5, 15.5), (1, 7.5), (30, 70) ) )
			M0, Mc, Cc, M2L = popt.x[0], popt.x[1], popt.x[2], popt.x[3]

			R_sky = 10**(0.4 * (22.5 - M0 + 2.5 * np.log10(pixel**2) ) )
			SB_rem = 22.5 - 2.5 * np.log10(Intns - R_sky) + 2.5 * np.log10(pixel**2)
			SB_m0 = 22.5 - 2.5 * np.log10(Intns - R_sky + Intns_err) + 2.5 * np.log10(pixel**2)
			SB_m1 = 22.5 - 2.5 * np.log10(Intns - R_sky - Intns_err) + 2.5 * np.log10(pixel**2)
			err_m0 = SB_rem - SB_m0
			err_m1 = SB_m1 - SB_rem
			id_nan = np.isnan(SB_m1)
			err_m1[id_nan] = 100.
			'''
			plt.figure()
			plt.title('[A]stack_%d_%s_band' % (tot_N, band[qq]) )
			ax = plt.imshow(stack_img, cmap = 'Greys', vmin = 1e-7, vmax = 5e1, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
			hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
			hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)	
			plt.xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
			plt.ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)
			plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/A_stack_%d_%s_band.png' % (tot_N, band[qq]), dpi = 300)
			plt.close()

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('[A]stack_%d_%s_band_SB' % (tot_N, band[qq]) )
			ax.errorbar(Rt, SBt, yerr = [err0, err1], xerr = None, color = 'r', marker = '.', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'r', elinewidth = 1, label = 'stack image', alpha = 0.5)
			ax.scatter(R_obs, SB_obs, s = 6.5, marker = '>', color = 'b', label = 'Z05', alpha = 0.5)
			'''
			ax.errorbar(Intns_r, SB_rem, yerr = [err_m0, err_m1], xerr = None, color = 'g', marker = '.', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'g', elinewidth = 1, label = 'stack image - residual BL', alpha = 0.5)
			ax.text(300, 26, s = '$ \; M_{c} = %.3f M_{\odot} $' % Mc + '\n' + '$ \; C = %.3f $' % Cc + '\n' + 
				'$ \; R_{sky} = %.3f $' % M0 + '\n' + '$ \; M2L = %.3f $' % M2L)
			'''
			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(21, 34)
			ax.legend(loc = 1)
			ax.invert_yaxis()
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/A_mask/A_stack_%d_%s_band_SB.png' % (tot_N, band[qq]), dpi = 300)
			plt.close()

	commd.Barrier()

	# stack sub-stack img of B mask
	mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	if rank == 1:
		for qq in range(3):
			# read the SB of total cluster emission
			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_3.csv' % band[qq], skiprows = 1 )
			R_obs, SB_obs = SB_tt['Mpc'] * 1e3, SB_tt['mag/arcsec^2']
			tot_N = 0
			for pp in range(cpus):
				
				with h5py.File(load + 'test_h5/stack_Bmask_pcount_%d_in_%s_band.h5' % (pp, band[qq]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(load + 'test_h5/stack_Bmask_sum_%d_in_%s_band.h5' % (pp, band[qq]), 'r') as f:
					sum_img = np.array(f['a'])

				sub_Num = np.nanmax(p_count)
				tot_N += sub_Num
				id_zero = p_count == 0
				ivx = id_zero == False
				mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
				p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

				# check sub-results
				p_count[id_zero] = np.nan
				sum_img[id_zero] = np.nan
				sub_mean = sum_img / p_count
				
				plt.figure()
				plt.title('[B]stack_%d_%s_band_%d_cpus' % (np.int(sub_Num), band[qq], pp) )
				ax = plt.imshow(sub_mean, cmap = 'Greys', vmin = 1e-7, vmax = 5e1, origin = 'lower', norm = mpl.colors.LogNorm())
				plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
				hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
				hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)
				plt.xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
				plt.ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)
				plt.savefig(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/B_mask/B_stack_%d_%s_band_%d_cpus.png' % (np.int(sub_Num), band[qq], pp), dpi = 300)
				plt.close()

			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)

			t1 = time.time()
			stack_img = mean_img / p_add_count
			## save the stack image
			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Bmask_%d_in_%s_band.h5' % (tot_N, band[qq]), 'w') as f:
				f['a'] = np.array(stack_img)

			R_cut = 900
			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, 10, Rpp, R_cut, R_cut, pixel, z_ref)
			flux0 = Intns + Intns_err
			flux1 = Intns - Intns_err
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[qq]
			SB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[qq]
			SB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[qq]
			err0 = SB - SB0
			err1 = SB1 - SB
			id_nan = np.isnan(SB)
			SBt, SB0, SB1 = SB[id_nan == False], SB0[id_nan == False], SB1[id_nan == False] 
			Rt, err0, err1 = Intns_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
			id_nan = np.isnan(SB1)
			err1[id_nan] = 100.
			t2 = time.time() - t1
			t3 = time.time() - t0

			plt.figure()
			plt.title('[B]stack_%d_%s_band' % (tot_N, band[qq]) )
			ax = plt.imshow(stack_img, cmap = 'Greys', vmin = 1e-7, vmax = 5e1, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
			hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
			hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)	
			plt.xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
			plt.ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)
			plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/B_mask/B_stack_%d_%s_band.png' % (tot_N, band[qq]), dpi = 300)
			plt.close()

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('[B]stack_%d_%s_band_SB' % (tot_N, band[qq]) )
			ax.errorbar(Rt, SBt, yerr = [err0, err1], xerr = None, color = 'r', marker = '.', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'r', label = 'stack image', elinewidth = 1, alpha = 0.5)
			ax.scatter(R_obs, SB_obs, s = 6.5, marker = '>', color = 'b', label = 'Z05', alpha = 0.5)
			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.legend(loc = 1)
			ax.invert_yaxis()
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/B_mask/B_stack_%d_%s_band_SB.png' % (tot_N, band[qq]), dpi = 300)
			plt.close()

	commd.Barrier()
	print( time.time() - t0 )

if __name__ == "__main__":
	main()
