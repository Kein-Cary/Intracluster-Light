import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import h5py
import numpy as np
import astropy.io.fits as fits
import astropy.units as U
from astropy import cosmology as apcy
from matplotlib.patches import Circle, Ellipse

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
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

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/photo_data/' ## save the catalogue data
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
		try:
			data_A = fits.getdata(load + 
				'photo_z/resample/pho_z-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
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
		except FileNotFoundError:
			continue

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
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	'''
	## deteck "bad" images
	with h5py.File(load + 'mpi_h5/photo_z_difference_sample.h5', 'r') as f:
		dat = np.array(f['a'])
	ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
	zN = len(z)
	da0, da1 = 764, 817 # zN - 1
	Ntt = np.int(da1 - da0)
	'''
	for kk in range( 3 ):
		'''
		set_z, set_ra, set_dec = z[da0: da1], ra[da0: da1], dec[da0: da1]
		m, n = divmod(Ntt, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		img_stack(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		'''
		with h5py.File(load + 'mpi_h5/phot_z_%s_band_stack_cat.h5' % band[kk], 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
		zN = len(z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		img_stack(kk, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		commd.Barrier()

	### combine all of the sub-stack A mask image
	if rank == 0:
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

				## record process img
				sub_img = sum_img / p_count
				id_inf = np.isinf(sub_img)
				sub_img[id_inf] = np.nan

				plt.figure()
				ax = plt.subplot(111)
				ax.set_title('stack %s band %d imgs %d cpus' % (band[kk], p_count[0, 0], pp) )
				clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5)
				tf = ax.imshow(sub_img, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
				plt.colorbar(tf, ax = ax, fraction = 0.040, pad = 0.01, label = 'flux[nmaggy]')
				ax.add_patch(clust)
				ax.set_xlim(x0 - 2 * np.int(Rpp), x0 + 2 * np.int(Rpp))
				ax.set_ylim(y0 - 2 * np.int(Rpp), y0 + 2 * np.int(Rpp))
				plt.savefig(load + 'photo_z/stack/stack_%s_band_%d_imgs_%d_cpus.png' % (band[kk], p_count[0, 0], pp), dpi = 300)
				plt.close()

			## save the stack image
			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)
			stack_img = mean_img / p_add_count
			where_are_inf = np.isinf(stack_img)
			stack_img[where_are_inf] = np.nan

			with h5py.File(load + 'photo_z/stack/stack_maskA_%d_in_%s_band.h5' % (tot_N, band[kk]), 'w') as f:
				f['a'] = np.array(stack_img)

			## save the variance image
			sqare_f[id_zero] = np.nan
			E_f2 = sqare_f / p_add_count
			id_inf = np.isinf(E_f2)
			E_f2[id_inf] = np.nan
			Var_f = E_f2 - stack_img**2  ## Variance img
			with h5py.File(load + 'photo_z/stack/stack_maskA_pix_Var_%d_in_%s_band.h5' % (tot_N, band[kk]), 'w') as f:
				f['a'] = np.array(Var_f)

			## record the stack img
			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('stack %s band %d imgs' % (band[kk], tot_N) )
			clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5)
			tf = ax.imshow(stack_img, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = ax, fraction = 0.040, pad = 0.01, label = 'flux[nmaggy]')
			ax.add_patch(clust)
			ax.set_xlim(x0 - 2 * np.int(Rpp), x0 + 2 * np.int(Rpp))
			ax.set_ylim(y0 - 2 * np.int(Rpp), y0 + 2 * np.int(Rpp))
			plt.savefig(load + 'photo_z/stack/stack_%s_band_%d_imgs.png' % (band[kk], tot_N), dpi = 300)
			plt.close()

	commd.Barrier()

if __name__ == "__main__":
	main()
