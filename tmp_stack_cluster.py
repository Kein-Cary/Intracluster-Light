import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import handy.scatter as hsc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits

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
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

band = ['r', 'g', 'i', 'u', 'z']
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

def mask_A(band_id, z_set, ra_set, dec_set):

	kk = np.int(band_id)
	Nz = len(z_set)
	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'
	out_load_A = '/mnt/ddnfs/data_users/cxkttwl/PC/A_mask_%d_cpus.cat' % rank
	## size test
	r_res = 2.8 # 2.8, 5.6, 7.4
	for q in range(Nz):
		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		pro_f = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], ra_g, dec_g, z_g)

		data_f = fits.open(pro_f)
		img = data_f[0].data
		head_inf = data_f[0].header
		wcs = awc.WCS(head_inf)
		cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
		R_ph = rad2asec / (Test_model.angular_diameter_distance(z_g).value)
		R_p = R_ph / pixel

		hdu = fits.PrimaryHDU()
		hdu.data = img
		hdu.header = head_inf
		hdu.writeto('/mnt/ddnfs/data_users/cxkttwl/PC/source_data_%d.fits' % rank, overwrite = True)

		file_source = '/mnt/ddnfs/data_users/cxkttwl/PC/source_data_%d.fits' % rank
		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_A, out_cat)
		a = subpro.Popen(cmd, shell = True)
		a.wait()

		source = asc.read(out_load_A)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE']) - 1
		cy = np.array(source['Y_IMAGE']) - 1
		p_type = np.array(source['CLASS_STAR'])

		Kron = 6 * r_res # iso_radius set as 3 times rms (2.8 from size test)
		a = Kron * A
		b = Kron * B

		mask = load + 'bright_star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		tau = 10 * r_res # the mask size set as 10 * FWHM from dr12

		set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
		set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
		set_chi = np.zeros(set_A.shape[1], dtype = np.float)

		lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
		lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
		sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])
		# bright stars
		x, y = wcs.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)
		ia = (x >= 0) & (x <= img.shape[1])
		ib = (y >= 0) & (y <= img.shape[0])
		ie = (set_mag <= 20)
		iq = lln >= 2
		ig = OBJ == 6
		ic = (ia & ib & ie & ig & iq)
		sub_x0 = x[ic]
		sub_y0 = y[ic]
		sub_A0 = lr_iso[ic]
		sub_B0 = sr_iso[ic]
		sub_chi0 = set_chi[ic]

		# saturated source(may not stars)
		xa = ['SATURATED' in qq for qq in xt]
		xv = np.array(xa)
		idx = xv == True
		ipx = (idx & ia & ib)

		sub_x2 = x[ipx]
		sub_y2 = y[ipx]
		sub_A2 = 3 * lr_iso[ipx]
		sub_B2 = 3 * sr_iso[ipx]
		sub_chi2 = set_chi[ipx]

		comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
		comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
		Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
		Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
		phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]

		cx = np.r_[cx, comx]
		cy = np.r_[cy, comy]
		a = np.r_[a, Lr]
		b = np.r_[b, Sr]
		theta = np.r_[theta, phi]
		Numb = Numb + len(comx)

		mask_A = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
		oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
		basic_coord = np.array(np.meshgrid(ox, oy))
		major = a / 2
		minor = b / 2 # set the star mask based on the major and minor radius
		senior = np.sqrt(major**2 - minor**2)

		tdr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)
		dr00 = np.where(tdr == np.min(tdr))[0]

		for k in range(Numb):
			xc = cx[k]
			yc = cy[k]

			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = theta[k]*np.pi/180

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r +1), img.shape[1] - 1] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r +1), img.shape[0] - 1] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
			iv[iu] = np.nan
			mask_A[lb0: lb1, la0: la1] = mask_A[lb0: lb1, la0: la1] * iv

		mirro_A = mask_A * img

		hdu = fits.PrimaryHDU()
		hdu.data = mirro_A
		hdu.header = head_inf
		hdu.writeto(home + 'tmp_stack/real_cluster/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g), overwrite = True)

	return

def clust_stack(band_id, sub_z, sub_ra, sub_dec):

	stack_N = len(sub_z)
	ii = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	id_nm = 0.

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		data_A = fits.open(home + 'tmp_stack/real_cluster/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[ii], ra_g, dec_g, z_g))
		img_A = data_A[0].data
		head = data_A[0].header
		wcs_lis = awc.WCS(head)
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])
		'''
		## stacking centered on image center (rule out BCG region)
		rnx, rny = np.int(img_A.shape[1] / 2), np.int(img_A.shape[0] / 2) ## image frame center
		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img_A.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img_A.shape[1])
		'''
		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
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

	return

def sky_stack(band_id, sub_z, sub_ra, sub_dec):
	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	id_nm = 0
	for jj in range(stack_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		data = fits.open(load + 'sky/sky_arr/sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]))
		img = data[0].data
		head = data[0].header
		wcs_lis = awc.WCS(head)
		cx, cy = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		'''
		## catalog (ra, dec)
		la0 = np.int(y0 - cy)
		la1 = np.int(y0 - cy + img.shape[0])
		lb0 = np.int(x0 - cx)
		lb1 = np.int(x0 - cx + img.shape[1])
		'''
		## image frame center / random center
		#rnx, rny = np.random.choice(img.shape[1], 1, replace = False), np.random.choice(img.shape[0], 1, replace = False)
		rnx, rny = np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)
		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)

		img = img - np.nanmedian(img)

		sum_array[la0: la1, lb0: lb1][idv] = sum_array[la0:la1, lb0:lb1][idv] + img[idv]
		count_array[la0: la1, lb0: lb1][idv] = img[idv]
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
	return

def main():
	"""
	## use r band only
	for kk in range( 1 ):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]

		DN = len(z)
		set_ra, set_dec, set_z = ra[:DN], dec[:DN], z[:DN]

		DN = len(set_z)
		m, n = divmod(DN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		mask_A(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])

	commd.Barrier()
	"""
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	for kk in range( 1 ):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f:
			set_array = np.array(f['a'])
		ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]

		DN = len(z)
		set_ra, set_dec, set_z = ra[:DN], dec[:DN], z[:DN]

		DN = len(set_z)
		m, n = divmod(DN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		clust_stack(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

		if rank == 0:

			tot_N = 0
			mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for pp in range(cpus):

				with h5py.File(tmp + 'stack_Amask_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'stack_Amask_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])

				tot_N += p_count[0, 0]
				id_zero = p_count == 0
				ivx = id_zero == False
				mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
				p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

				## save sub-sample sky
				sub_mean = sum_img / p_count
				id_zero = sub_mean == 0.
				id_inf = np.isinf(sub_mean)
				sub_mean[id_zero] = np.nan
				sub_mean[id_inf] = np.nan

				#with h5py.File(home + 'tmp_stack/%s_band_center-stack_cluster-img_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:
				#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_cluster-img_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:
				#	f['a'] = np.array(sub_mean)

				## sub-sample for jackknife
				with h5py.File(home + 'tmp_stack/jack_random/apply_bcgs/%s_band_bcgs-stack_clust-img_%d-sub-smp.h5' % (band[kk], pp), 'w') as f:
					f['a'] = np.array(sub_mean)

			## save the stack image
			id_zero = p_add_count == 0
			mean_img[id_zero] = np.nan
			p_add_count[id_zero] = np.nan
			tot_N = np.int(tot_N)
			stack_img = mean_img / p_add_count
			where_are_inf = np.isinf(stack_img)
			stack_img[where_are_inf] = np.nan

			#with h5py.File(home + 'tmp_stack/%s_band_center-stack_cluster-img.h5' % band[kk], 'w') as f:
			#with h5py.File(home + 'tmp_stack/%s_band_cat-stack_cluster-img.h5' % band[kk], 'w') as f:
			#	f['a'] = np.array(stack_img)

		commd.Barrier()
	raise
	for kk in range( 1 ):

		with h5py.File(load + 'mpi_h5/%s_band_sky_catalog.h5' % band[kk], 'r') as f: ## now using sample
			set_array = np.array(f['a'])
		ra, dec, z = set_array[0,:], set_array[1,:], set_array[2,:]

		DN = 1000
		set_ra, set_dec, set_z = ra[:DN], dec[:DN], z[:DN]

		m, n = divmod(DN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		sky_stack(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

		## combine all of the sub-stack imgs
		if rank == 0:
			tot_N = 0.
			bcg_stack = np.zeros((len(Ny), len(Nx)), dtype = np.float)
			bcg_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)

			for pp in range(cpus):

				with h5py.File(tmp + 'sky_sum_pcount_%d_in_%s_band.h5' % (pp, band[kk]), 'r')as f:
					p_count = np.array(f['a'])
				with h5py.File(tmp + 'sky_sum_%d_in_%s_band.h5' % (pp, band[kk]), 'r') as f:
					sum_img = np.array(f['a'])
				tot_N += p_count[0, 0]
				id_zero = p_count == 0
				ivx = id_zero == False
				bcg_stack[ivx] = bcg_stack[ivx] + sum_img[ivx]
				bcg_count[ivx] = bcg_count[ivx] + p_count[ivx]

				## save sub-sample sky
				sub_mean = sum_img / p_count
				id_zero = sub_mean == 0.
				id_inf = np.isinf(sub_mean)
				sub_mean[id_zero] = np.nan
				sub_mean[id_inf] = np.nan

				#with h5py.File(home + 'tmp_stack/%s_band_center-stack_cluster-sky_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:
				with h5py.File(home + 'tmp_stack/%s_band_center-stack_clust_minu-media_sky_%d-sub-sample.h5' % (band[kk], pp), 'w') as f:
					f['a'] = np.array(sub_mean)

			## centered on BCG
			id_zero = bcg_count == 0
			bcg_stack[id_zero] = np.nan
			bcg_count[id_zero] = np.nan
			stack_img = bcg_stack / bcg_count
			id_inf = np.isinf(stack_img)
			stack_img[id_inf] = np.nan

			#with h5py.File(home + 'tmp_stack/%s_band_center-stack_cluster-sky-img.h5' % band[kk], 'w') as f:
			with h5py.File(home + 'tmp_stack/%s_band_center-stack_clust_minu-media_sky-img.h5' % band[kk], 'w') as f:
				f['a'] = np.array(stack_img)

		commd.Barrier()

if __name__ == "__main__":
	main()
