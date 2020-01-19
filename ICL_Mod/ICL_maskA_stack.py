import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import astropy.io.fits as fits

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/' ## save the catalogue data
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'  ## save the process data
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

def img_stack(band_id, sub_z, sub_ra, sub_dec):

	stack_N = len(sub_z)
	ii = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	## initial image which is about twice of the biggest size of images, use to stack images
	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	f2_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)  ## also save flux^2 img for Variance calculation
	id_nm = 0.

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		data_A = fits.getdata(tmp + 'test/resam-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g) header = True)

		img_A = data_A[0]
		xn = data_A[1]['CENTER_X']
		yn = data_A[1]['CENTER_Y']

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
		f2_sum[la0: la1, lb0: lb1][idv] = f2_sum[la0: la1, lb0: lb1][idv] + img_A[idv]**2
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan
		id_nm += 1.

	p_count_A[0, 0] = id_nm
	with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(p_count_A)
	with h5py.File(tmp + 'stack_Amask_Var_%d_in_%s_band.h5' % (rank, band[ii]), 'w') as f:
		f['a'] = np.array(f2_sum)

	return

def main():
	### sub-stack mask A
	#for kk in range(len(band)):
	for kk in range( 3 ):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk], 'r') as f:
			cat = np.array(f['a'])
		ra, dec, z = cat[0,:], cat[1,:], cat[2,:]
		zN = len(z)

		Ns = 100
		np.random.seed(1)
		tt0 = np.random.choice(zN, size = Ns, replace = False)
		set_z, set_ra, set_dec = z[tt0], ra[tt0], dec[tt0]

		m, n = divmod(Ns, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		img_stack(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

	### combine all of the sub-stack A mask image
	if rank == 0:
		#for kk in range(len(band)):
		for kk in range( 3 ):

			tot_N = 0
			mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			sqare_f = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for pp in range(cpus):

				with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])
				with h5py.File(tmp + 'stack_Amask_Var_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					f2_sum = np.array(f['a'])

				tot_N += p_count[0, 0]
				id_zero = p_count == 0
				ivx = id_zero == False
				mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
				p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]
				sqare_f[ivx] = sqare_f[ivx] + f2_sum[ivx]

			## save the stack image
			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)
			stack_img = mean_img / p_add_count
			where_are_inf = np.isinf(stack_img)
			stack_img[where_are_inf] = np.nan

			with h5py.File(tmp + 'test/stack_maskA_%d_in_%s_band.h5' % (tot_N, band[kk]), 'w') as f:
				f['a'] = np.array(stack_img)

			## save the variance image
			sqare_f[id_zero] = np.nan
			E_f2 = sqare_f / p_add_count
			id_inf = np.isinf(E_f2)
			E_f2[id_inf] = np.nan
			Var_f = E_f2 - stack_img**2  ## Variance img
			with h5py.File(tmp + 'test/stack_maskA_pix_Var_%d_in_%s_band.h5' % (tot_N, band[kk]), 'w') as f:
				f['a'] = np.array(Var_f)

	commd.Barrier()

if __name__ == "__main__":
	main()
