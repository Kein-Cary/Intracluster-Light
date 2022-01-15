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

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'  ## save the process data
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

"""
sky image stacking include: 
1) skg_img - np.nanmean(sky_img) [cor_id == 0] or sky_img - np.nanmedian(sky_img) [cor_id == 1]
2) stacking the img : center on BCG (--img_1) and random center (img_2)
3) calculating the difference img: img_1 - img_2 (this is the signal need to add back to 
   the result of cluster image stacking)
"""

def sky_stack_BCG(band_id, sub_z, sub_ra, sub_dec, cor_id):
	stack_N = len(sub_z)
	kk = np.int(band_id)
	open_id = cor_id

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	f2_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	id_nm = 0
	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		'''
		## scaled
		data = fits.open( load + 
			'photo_z/sky/sky_resam_img/resampl_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		## rule out the edge pixels
		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan
		'''
		## sky-select sample
		data = fits.open( load + 
			'photo_z/sky/sky_cut_img/Cut_edge_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']

		la0 = np.int(y0 - cy)
		la1 = np.int(y0 - cy + img.shape[0])
		lb0 = np.int(x0 - cx)
		lb1 = np.int(x0 - cx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		if open_id == 0 :
			img_add = img - np.nanmean(img)
		if open_id == 1 :
			img_add = img - np.nanmedian(img)

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img_add[idv]
		f2_sum[la0: la1, lb0: lb1][idv] = f2_sum[la0: la1, lb0: lb1][idv] + img_add[idv]**2
		count_array[la0: la1, lb0: lb1][idv] = img_add[idv]
		id_nan = np.isnan(count_array)
		id_fals = np.where(id_nan == False)
		p_count[id_fals] = p_count[id_fals] + 1
		count_array[la0: la1, lb0: lb1][idv] = np.nan
		id_nm += 1.

	p_count[0, 0] = id_nm
	with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(sum_array)
	with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(p_count)

	with h5py.File(tmp + 'sky_f2_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(f2_sum)

	return

def sky_stack_rndm(band_id, sub_z, sub_ra, sub_dec, cor_id):

	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	## random center as comparison
	rndm_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	rndm_count = np.zeros((len(Ny), len(Nx)), dtype = np.float) * np.nan
	rndm_pcont = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	f2_sum = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	pos_x = np.zeros(stack_N, dtype = np.float)
	pos_y = np.zeros(stack_N, dtype = np.float)
	n_add = 0.
	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]
		'''
		## scaled
		data = fits.open( load + 
			'photo_z/sky/sky_resam_img/resampl_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']
		## rule out the edge pixels
		img[0,:] = np.nan
		img[-1,:] = np.nan
		img[:,0] = np.nan
		img[:,-1] = np.nan
		'''
		## sky-select sample
		data = fits.open( load + 
			'photo_z/sky/sky_cut_img/Cut_edge_sky-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )
		img = data[0].data
		cx, cy = data[0].header['CENTER_X'], data[0].header['CENTER_Y']

		## the random center test
		rnx, rny = np.random.choice(img.shape[1], 1, replace = False), np.random.choice(img.shape[0], 1, replace = False) ## random center
		pos_x[jj], pos_y[jj] = rnx, rny

		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		if cor_id == 0:
			img_add = img - np.nanmean(img)
		if cor_id == 1:
			img_add = img - np.nanmedian(img)

		rndm_sum[la0:la1, lb0:lb1][idv] = rndm_sum[la0:la1, lb0:lb1][idv] + img_add[idv]
		f2_sum[la0:la1, lb0:lb1][idv] = f2_sum[la0:la1, lb0:lb1][idv] + img_add[idv]**2
		rndm_count[la0:la1, lb0:lb1][idv] = img_add[idv]
		id_nan = np.isnan(rndm_count)
		id_fals = np.where(id_nan == False)
		rndm_pcont[id_fals] = rndm_pcont[id_fals] + 1
		rndm_count[la0:la1, lb0:lb1][idv] = np.nan

		n_add += 1.

	rndm_pcont[0, 0] = n_add
	## save the random center data
	with h5py.File(tmp + 'rndm_sum_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(rndm_sum)
	with h5py.File(tmp + 'rndm_sum_pcount_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(rndm_pcont)
	## save the random poisition
	rdn_pos = np.array([pos_x, pos_y])
	with h5py.File(tmp + 'rdnm_pos_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(rdn_pos)
	with h5py.File(tmp + 'rdnm_pos_%d_in_%s_band.h5' % (rank, band[kk]) ) as f:
		for ll in range(len(rdn_pos)):
			f['a'][ll,:] = rdn_pos[ll,:]

	with h5py.File(tmp + 'rndm_Var_%d_in_%s_band.h5' % (rank, band[kk]), 'w') as f:
		f['a'] = np.array(f2_sum)

	return

def main():
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	cor_id = 1 # /0 (see description at the beginning)

	### sub-stack
	for kk in range( 3 ):

		#with h5py.File(load + 'mpi_h5/phot_z_%s_band_stack_cat.h5' % band[kk], 'r') as f:
		with h5py.File(load + 'photo_z/%s_band_img-center_cat.h5' % band[kk], 'r') as f:
			dat = np.array(f['a'])
		set_ra, set_dec, set_z, set_rich = dat[0,:], dat[1,:], dat[2,:], dat[3,:]
		zN = len(set_z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		sky_stack_BCG(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], cor_id)
		commd.Barrier()

		## combine all of the sub-stack imgs
		if rank == 0:
			tt_N = 0
			bcg_stack = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			bcg_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for pp in range(cpus):
				with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])

				tt_N += p_count[0, 0]
				id_zero = p_count == 0
				ivx = id_zero == False
				bcg_stack[ivx] = bcg_stack[ivx] + sum_img[ivx]
				bcg_count[ivx] = bcg_count[ivx] + p_count[ivx]
			## sample sky SB
			tt_N = np.int(tt_N)
			## centered on BCG
			id_zero = bcg_count == 0
			bcg_stack[id_zero] = np.nan
			bcg_count[id_zero] = np.nan
			stack_img = bcg_stack / bcg_count
			id_inf = np.isinf(stack_img)
			stack_img[id_inf] = np.nan
			if cor_id == 0:
				with h5py.File(load + 'photo_z/stack/stack_sky_mean_%d_imgs_%s_band.h5' % (tt_N, band[kk]), 'w') as f:
					f['a'] = np.array(stack_img)
			if cor_id == 1:
				with h5py.File(load + 'photo_z/stack/stack_sky_median_%d_imgs_%s_band.h5' % (tt_N, band[kk]), 'w') as f:
					f['a'] = np.array(stack_img)

		commd.Barrier()

	##### for random case
	for d_record in range(1, 6):
		for kk in range( 3 ):

			#with h5py.File(load + 'mpi_h5/phot_z_%s_band_stack_cat.h5' % band[kk], 'r') as f:
			with h5py.File(load + 'photo_z/%s_band_img-center_cat.h5' % band[kk], 'r') as f:
				dat = np.array(f['a'])
			set_ra, set_dec, set_z, set_rich = dat[0,:], dat[1,:], dat[2,:], dat[3,:]
			zN = len(set_z)
			m, n = divmod(zN, cpus)
			N_sub0, N_sub1 = m * rank, (rank + 1) * m
			if rank == cpus - 1:
				N_sub1 += n
			sky_stack_rndm(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], cor_id)
			commd.Barrier()

			## sum sub-stack imgs
			if rank == 0:
				tt_N = 0
				rand_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				rand_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)

				for pp in range(cpus):
					## random center case
					with h5py.File(tmp + 'rndm_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
						rndm_sum = np.array(f['a'])
					with h5py.File(tmp + 'rndm_sum_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
						rndm_pcont = np.array(f['a'])
					tt_N += rndm_pcont[0, 0]
					id_zero = rndm_pcont == 0
					ivx = id_zero == False
					rand_img[ivx] = rand_img[ivx] + rndm_sum[ivx]
					rand_cnt[ivx] = rand_cnt[ivx] + rndm_pcont[ivx]

				tt_N = np.int(tt_N)
				id_zero = rand_cnt == 0
				rand_img[id_zero] = np.nan
				rand_cnt[id_zero] = np.nan
				random_stack = rand_img / rand_cnt
				id_inf = np.isinf(random_stack)
				random_stack[id_inf] = np.nan

				if cor_id == 0:
					with h5py.File(load + 'photo_z/stack/%d_sky_rndm_mean_%d_imgs_%s_band.h5' % (d_record, tt_N, band[kk]), 'w') as f:
						f['a'] = np.array(random_stack)

				if cor_id == 1:
					with h5py.File(load + 'photo_z/stack/%d_sky_rndm_median_%d_imgs_%s_band.h5' % (d_record, tt_N, band[kk]), 'w') as f:
						f['a'] = np.array(random_stack)

			commd.Barrier()

	#N_sum = np.array([1497, 1503, 1492])
	N_sum = np.array([1176, 1176, 1169])

	## calculate the image average for random case
	if rank == 0:
		for kk in range(3):

			m_rndm_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			rndm_cnt = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for id_rec in range(1, 6):

				if cor_id == 0:
					with h5py.File(load + 'photo_z/stack/%d_sky_rndm_mean_%d_imgs_%s_band.h5' % (id_rec, N_sum[kk], band[kk]), 'r') as f:
						sub_img = np.array(f['a'])

				if cor_id == 1:
					with h5py.File(load + 'photo_z/stack/%d_sky_rndm_median_%d_imgs_%s_band.h5' % (id_rec, N_sum[kk], band[kk]), 'r') as f:
						sub_img = np.array(f['a'])

				id_nan = np.isnan(sub_img)
				id_fals = id_nan == False
				m_rndm_img[id_fals] = m_rndm_img[id_fals] + sub_img[id_fals]
				rndm_cnt[id_fals] = rndm_cnt[id_fals] + 1.

			m_rndm_img = m_rndm_img / rndm_cnt

			if cor_id == 0:
				with h5py.File(load + 'photo_z/stack/M_sky_rndm_mean_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'w') as f:
					f['a'] = np.array(m_rndm_img)

			if cor_id == 1:
				with h5py.File(load + 'photo_z/stack/M_sky_rndm_median_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'w') as f:
					f['a'] = np.array(m_rndm_img)

	commd.Barrier()

if __name__ == "__main__":
	main()
