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

# bad data list
except_ra_r = ['128.611', '133.229', '177.809', '183.580', '221.403', '242.344', '130.537',
				'32.906', '38.382', '168.057', '202.923', '212.342', '237.591', '180.927']
except_dec_r = ['36.515', '18.196', '33.577', '32.835', '8.243', '36.393', '38.226', '0.387',
				'1.911', '15.760', '13.325', '39.952', '2.756', '1.032']
except_z_r = ['0.289', '0.257', '0.212', '0.226', '0.268', '0.281', '0.244', '0.295', '0.243',
				'0.275', '0.242', '0.267', '0.240', '0.255']

except_ra_g = ['168.215', '8.320', '128.611', '133.229', '177.809', '242.344', '351.680',
				'35.888', '180.927', '186.666', '130.537', '34.675', '121.566', '124.701',
				'172.033', '188.950', '215.244', '227.367']
except_dec_g = ['56.464', '-0.633', '36.515', '18.196', '33.577', '36.393', '1.134', '-7.228',
				'1.032', '3.383', '38.226', '0.114', '19.247', '4.898', '41.322', '15.556', 
				'0.229', '0.899']
except_z_g = ['0.227', '0.261', '0.289', '0.257', '0.212', '0.281', '0.277', '0.279', '0.255',
				'0.226', '0.244', '0.272', '0.285', '0.252', '0.288', '0.285', '0.277', '0.263']

except_ra_i = ['5.200', '8.320', '10.708', '21.224', '24.320', '37.108', '37.662', '117.410',
				'117.585', '126.251', '129.462', '141.919', '143.950', '148.541', '162.793',
				'174.080', '179.024', '180.927', '183.580', '186.372', '130.537', '190.946',
				'196.951', '214.913', '218.217', '242.344', '330.691']
except_dec_i = ['1.585', '-0.633', '0.215', '2.508', '7.882', '1.503', '-4.991', '27.287',
				'26.293', '4.430', '6.777', '4.352', '12.151', '36.054', '9.066', '50.425',
				'42.024', '1.032', '32.835', '0.726', '38.226', '51.831', '43.153', '43.195',
				'30.489', '36.393', '-8.544']
except_z_i = ['0.208', '0.261', '0.269', '0.217', '0.258', '0.264', '0.292', '0.243', '0.205',
				'0.224', '0.236', '0.275', '0.255', '0.291', '0.221', '0.286', '0.245', '0.255',
				'0.226', '0.238', '0.244', '0.268', '0.210', '0.219', '0.266', '0.281', '0.210']

except_ra_z = ['24.320', '126.251', '130.537', '140.078', '148.541', '153.281', '186.372', '190.640',
				'192.743', '196.951', '222.069', '228.094', '242.344', '334.357', '346.984']
except_dec_z = ['7.882', '4.430', '38.226', '37.105', '36.054', '17.932', '0.726', '35.537', '9.056',
				'43.153', '13.622', '59.217', '36.393', '27.836', '15.373']
except_z_z = ['0.258', '0.224', '0.244', '0.235', '0.291', '0.261', '0.238', '0.269', '0.298', '0.210',
				'0.226', '0.295', '0.281', '0.298', '0.249']

except_ra_u = ['35.888', '130.537', '140.296', '147.783', '155.783', '189.004', '204.525', '208.380',
				'219.787', '242.344', '244.422', '327.180', '334.357', '346.734']
except_dec_u = ['-7.228', '38.226', '34.860', '43.745', '7.108', '53.394', '5.434', '53.575', '29.039',
				'36.393', '42.539', '7.024', '27.836', '15.375']
except_z_u = ['0.279', '0.244', '0.238', '0.287', '0.290', '0.275', '0.237', '0.255', '0.251', '0.281',
				'0.294', '0.282', '0.298', '0.220']
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

		identi = (( ('%.3f'%ra_g in except_ra_r) & ('%.3f'%dec_g in except_dec_r) & ('%.3f'%z_g in except_z_r) ) | 
					( ('%.3f'%ra_g in except_ra_g) & ('%.3f'%dec_g in except_dec_g) & ('%.3f'%z_g in except_z_g) ) | 
					( ('%.3f'%ra_g in except_ra_i) & ('%.3f'%dec_g in except_dec_i) & ('%.3f'%z_g in except_z_i) ) | 
					( ('%.3f'%ra_g in except_ra_u) & ('%.3f'%dec_g in except_dec_u) & ('%.3f'%z_g in except_z_u) ) | 
					( ('%.3f'%ra_g in except_ra_z) & ('%.3f'%dec_g in except_dec_z) & ('%.3f'%z_g in except_z_z) ))
		if  identi == True: 
			continue
		else:
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
	for tt in range(len(band)):
		m, n = divmod(stackn, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		stack_process(tt, zN[N_sub0 :N_sub1], raN[N_sub0 :N_sub1], decN[N_sub0 :N_sub1])
		commd.Barrier()

	mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	if rank == 0:
		for qq in range(len(band)):

			tot_N = 0
			for pp in range(cpus):
				
				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[qq]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[qq]), 'r') as f:
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
				plt.title('stack_%d_%s_band_%d_cpus.png' % (np.int(sub_Num), band[qq], pp) )
				ax = plt.imshow(sub_mean, cmap = 'Greys', vmin = 1e-7, origin = 'lower', norm = mpl.colors.LogNorm())
				plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
				hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
				hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)
				plt.xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
				plt.ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)
				plt.savefig(
					'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_%s_band_%d_cpus.png' % (np.int(sub_Num), band[qq], pp), dpi = 300)
				plt.close()

			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)

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
			plt.title('stack_%d_%s_band.png' % (tot_N, band[qq]) )
			ax = plt.imshow(stack_img, cmap = 'Greys', vmin = 1e-7, origin = 'lower', norm = mpl.colors.LogNorm())
			plt.colorbar(ax, fraction = 0.035, pad =  0.01, label = '$flux[nmaggy]$')
			hsc.circles(x0, y0, s = Rpp, fc = '', ec = 'r', linestyle = '-', alpha = 0.5)
			hsc.circles(x0, y0, s = 0.2 * Rpp, fc = '', ec = 'g', linestyle = '--', alpha = 0.5)	
			plt.xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
			plt.ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)
			plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_%s_band.png' % (tot_N, band[qq]), dpi = 300)
			plt.close()

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('stack_%d_%s_band_SB.png' % (tot_N, band[qq]) )
			ax.errorbar(Rt, SBt, yerr = errt, xerr = None, color = 'r', marker = 'o', ls = '', linewidth = 1, markersize = 5, 
				ecolor = 'r', elinewidth = 1, alpha = 0.5)
			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.invert_yaxis()
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_%s_band_SB.png' % (tot_N, band[qq]), dpi = 300)
			plt.close()

			print('Now %s band finished!' % band[qq])
if __name__ == "__main__":
	main()
