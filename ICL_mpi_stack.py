import matplotlib as mpl
mpl.use('Agg')
import handy.scatter as hsc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import h5py
import time
import numpy as np
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits
from light_measure import light_measure
from scipy.interpolate import interp1d as interp

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

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

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
	sum_grid = np.array(np.meshgrid(Nx, Ny))

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

		Da_g = Test_model.angular_diameter_distance(z_g).value

		data_A = fits.getdata(load + 
		    'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
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
		p_count_A[id_fals] = p_count_A[id_fals] + 1
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

		data_B = fits.getdata(load + 
		    'resample/resam_B/frameB-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
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
		p_count_B[id_fals] = p_count_B[id_fals] + 1
		count_array_B[la0: la1, lb0: lb1][idv] = np.nan

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Bmask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_B)

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Bmask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_B)

	return

def main():
	t0 = time.time()

	bins = 65
	'''
	ix = red_rich >= 39
	RichN = red_rich[ix]
	zN = red_z[ix]
	raN = red_ra[ix]
	decN = red_dec[ix]
	stackn = np.int(690)
	'''
	RichN = red_rich * 1
	zN = red_z * 1
	raN = red_ra * 1
	decN = red_dec * 1
	stackn = len(zN)

	GR_minus = []
	Mag_minus = []
	R_record = []
	x0 = 2427
	y0 = 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	commd.Barrier()
	for tt in range(2):
		m, n = divmod(stackn, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		if rank == 14:
			print('*' * 20)
			print(rank, '-----', '\n', 
				zN[N_sub0 :N_sub1], '\n',
				raN[N_sub0 :N_sub1], '\n', 
				decN[N_sub0 :N_sub1])

		stack_process(tt, zN[N_sub0 :N_sub1], raN[N_sub0 :N_sub1], decN[N_sub0 :N_sub1])
		commd.Barrier()

	mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	if rank == 0:
		for pp in range(cpus):
			with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Amask_pcount_%d_in_g_band.h5' % pp, 'r') as f:
				p_count = np.array(f['a'])
			with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Amask_sum_%d_in_g_band.h5' % pp, 'r') as f:
				sum_img = np.array(f['a'])

			id_zero = p_count == 0
			ivx = id_zero == False
			mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
			p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

			# check sub-results
			p_count[id_zero] = np.nan
			sum_img[id_zero] = np.nan
			sub_mean = sum_img / p_count

			plt.figure()
			plt.title('stack_%d_r_band.png' % pp)
			ax = plt.imshow(sub_mean, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
			plt.xlim(x0 - 1.2*Rpp, x0 + 1.2*Rpp)
			plt.ylim(y0 - 1.2*Rpp, y0 + 1.2*Rpp)
			plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_g_band.png' % pp, dpi = 300)
			plt.close()

		id_zero = p_add_count == 0
		mean_img[id_zero] = np.nan
		p_add_count[id_zero] = np.nan

		t1 = time.time()
		stack_img = mean_img / p_add_count
		SBt, Rt, Art, errt = light_measure(stack_img, bins, 1, Rpp, x0, y0, pixel, z_ref)[:4]
		SBt = SBt + mag_add[0]
		t2 = time.time() - t1
		t3 = time.time() - t0

		print('*' * 20)
		print('t = ', t2)
		print('t_tot = ', t3)

		plt.figure()
		plt.title('stack_%d_r_band.png' % stackn)
		ax = plt.imshow(stack_img, cmap = 'Greys', vmin = 1e-3, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
		plt.xlim(x0 - 1.2*Rpp, x0 + 1.2*Rpp)
		plt.ylim(y0 - 1.2*Rpp, y0 + 1.2*Rpp)
		plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_g_band.png' % stackn, dpi = 300)
		plt.close()

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('stack_%d_r_band_SB.png' % stackn)
		ax.errorbar(Rt, SBt, yerr = errt, xerr = None, color = 'r', marker = 'o', ls = '', linewidth = 1, markersize = 5, 
			ecolor = 'r', elinewidth = 1, alpha = 0.5)
		ax.set_xlabel('$R[kpc]$')
		ax.set_ylabel('$SB[mag / arcsec^2]$')
		ax.set_xscale('log')
		ax.invert_yaxis()
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_g_band_SB.png' % stackn, dpi = 300)
		plt.close()


if __name__ == "__main__":
	main()
