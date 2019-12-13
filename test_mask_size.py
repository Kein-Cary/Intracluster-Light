import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.gridspec as gridspec

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

from dustmaps.sfd import SFDQuery
from extinction_redden import A_wave
from astropy.coordinates import SkyCoord
from resample_modelu import sum_samp, down_samp
from light_measure import light_measure, flux_recal

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
import time

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

pixel = 0.396
z_ref = 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

Rv = 3.1
sfd = SFDQuery()

def sers_pro(r, mu_e, r_e, n):
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def A_mask(band_id, z_set, ra_set, dec_set, r_star, r_galx):

	kk = np.int(band_id)
	Nz = len(z_set)
	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'
	out_load_A = '/mnt/ddnfs/data_users/cxkttwl/PC/A_mask_%d_cpus.cat' % rank

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

		x0 = np.linspace(0, img.shape[1] - 1, img.shape[1])
		y0 = np.linspace(0, img.shape[0] - 1, img.shape[0])
		img_grid = np.array(np.meshgrid(x0, y0))
		ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
		pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
		BEV = sfd(pos)
		Av = Rv * BEV * 0.86
		Al = A_wave(l_wave[kk], Rv) * Av
		img = img * 10 ** (Al / 2.5)

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

		Kron = 6 * r_galx # iso_radius set as 3 times rms
		a = Kron*A
		b = Kron*B

		mask = load + 'bright_star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		tau = r_star * 10 # the mask size set as 10 * FWHM from dr12

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

			if k == dr00[0] :
				continue
			else:
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
		hdu.writeto(load + 
			'mask_data/A_plane/1.5sigma/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[kk], ra_g, dec_g, z_g),overwrite = True)
	return	

def A_resamp(band_id, sub_z, sub_ra, sub_dec):
	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value

		data = fits.getdata(load + 
			'mask_data/A_plane/1.5sigma/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), header = True)
		img = data[0]
		head_mean = data[1]
		cx0 = data[1]['CRPIX1']
		cy0 = data[1]['CRPIX2']
		RA0 = data[1]['CRVAL1']
		DEC0 = data[1]['CRVAL2']

		wcs = awc.WCS(data[1])
		cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		Angur = (R0*rad2asec/Da_g)
		Rp = Angur/pixel
		L_ref = Da_ref*pixel/rad2asec
		L_z0 = Da_g*pixel/rad2asec
		b = L_ref/L_z0
		Rref = (R0*rad2asec/Da_ref)/pixel

		f_goal = flux_recal(img, z_g, z_ref)
		ix0 = np.int(cx0/b)
		iy0 = np.int(cy0/b)

		if b > 1:
			resam, xn, yn = sum_samp(b, b, f_goal, cx, cy)
		else:
			resam, xn, yn = down_samp(b, b, f_goal, cx, cy)

		xn = np.int(xn)
		yn = np.int(yn)
		x0 = resam.shape[1]
		y0 = resam.shape[0]

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x0, y0, ix0, iy0, xn, yn, RA0, DEC0, ra_g, dec_g, z_g, pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(load + 
			'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), resam, header = fil, overwrite=True)
	return

def A_stack(band_number, subz, subra, subdec, r_star, r_galx):

	stack_N = len(subz)
	ii = np.int(band_number)
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = subra[jj]
		dec_g = subdec[jj]
		z_g = subz[jj]
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
		'''
		## select the 1 Mpc ~ 1.1 Mpc region as background
		grd_x = np.linspace(0, img_A.shape[1] - 1, img_A.shape[1])
		grd_y = np.linspace(0, img_A.shape[0] - 1, img_A.shape[0])
		grd = np.array( np.meshgrid(grd_x, grd_y) )
		ddr = np.sqrt( (grd[0,:] - xn)**2 + (grd[1,:] - yn)**2 )
		idu = (ddr > Rpp) & (ddr < 1.1 * Rpp) # using SB in 1 Mpc ~ 1.1 Mpc region as residual sky
		'''
		#BL = np.nanmean(img_A[idu])
		BL = 0.
		sub_BL_img = img_A - BL

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + sub_BL_img[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = sub_BL_img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		p_count_A[0, 0] = p_count_A[0, 0] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

	with h5py.File(load + 
		'test_h5/stack_Amask_sum_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % (rank, band[ii], r_star, r_galx), 'w') as f:
		f['a'] = np.array(sum_array_A)
	with h5py.File(load + 
		'test_h5/stack_Amask_pcount_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % (rank, band[ii], r_star, r_galx), 'w') as f:
		f['a'] = np.array(p_count_A)

	return

def main():
	## sersic pro of Zibetti 05
	mu_e = np.array([23.87, 25.22, 23.4])
	r_e = np.array([19.29, 19.40, 20])

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	R_cut, bins = 1280, 80
	R_smal, R_max = 1, 1.7e3 # kpc

	### set the mask parameter
	ext_r_star = np.array([2., 2.2, 2.4, 2.6, 2.8])
	ext_r_galx = np.array([2., 2.2, 2.4, 2.6, 2.8])
	N_tt = 500
	'''
	ext_r_star = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
	ext_r_galx = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
	N_tt = 200
	'''
	N_set = len(ext_r_star)
	'''
	for tt in range(3):
		with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/%s_band_%d_sample.h5' % (band[tt], N_tt), 'r') as f:
			set_info = np.array(f['a'])
		set_z, set_ra, set_dec = set_info[0,:], set_info[1,:], set_info[2,:]
		zN = len(set_z)

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		for kk in range(N_set):
			A_mask(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], ext_r_star[kk], ext_r_galx[kk])
			A_resamp(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
			A_stack(tt, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1], ext_r_star[kk], ext_r_galx[kk])
		commd.Barrier()

	if rank == 1:
		for tt in range(3):
			for kk in range(N_set):
				tot_N = 0
				mean_img = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				p_add_count = np.zeros((len(Ny), len(Nx)), dtype = np.float)
				for pp in range(cpus):

					with h5py.File(load + 
						'test_h5/stack_Amask_pcount_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % (pp, band[tt], ext_r_star[kk], ext_r_galx[kk]), 'r')as f:
						p_count = np.array(f['a'])
					with h5py.File(load + 
						'test_h5/stack_Amask_sum_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % (pp, band[tt], ext_r_star[kk], ext_r_galx[kk]), 'r') as f:
						sum_img = np.array(f['a'])

					sub_Num = np.nanmax(p_count)
					tot_N += sub_Num
					id_zero = p_count == 0
					ivx = id_zero == False
					mean_img[ivx] = mean_img[ivx] + sum_img[ivx]
					p_add_count[ivx] = p_add_count[ivx] + p_count[ivx]

				id_zero = p_add_count == 0
				mean_img[id_zero] = np.nan
				p_add_count[id_zero] = np.nan
				tot_N = np.int(tot_N)
				stack_img = mean_img / p_add_count
				where_are_inf = np.isinf(stack_img)
				stack_img[where_are_inf] = np.nan
				with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_Amask_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % 
					(tot_N, band[tt], ext_r_star[kk], ext_r_galx[kk]), 'w') as f:
					f['a'] = np.array(stack_img)
	commd.Barrier()
	'''
	if rank == 1:
		for tt in range(3):
			for kk in range(N_set):
				with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_Amask_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % 
					(N_tt, band[tt], ext_r_star[kk], ext_r_galx[kk]), 'r') as f:
					stack_img = np.array(f['a'])

				plt.figure()
				clust = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5, label = 'cluster region[1Mpc]')
				ax = plt.subplot(111)
				ax.set_title('2D stacking %d image in %s band %.2fr_star %.2fr_galx' % (N_tt, band[tt], ext_r_star[kk], ext_r_galx[kk]) )
				tf = ax.imshow(stack_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
				plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux[nmaggy]')
				ax.add_patch(clust)
				ax.set_xlim(x0 - 2 * Rpp, x0 + 2 * Rpp)
				ax.set_ylim(y0 - 2 * Rpp, y0 + 2 * Rpp)
				ax.legend(loc = 1)
				plt.subplots_adjust(left = 0.125, right = 0.9, bottom = 0.1, top = 0.9)
				plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_%d_img_%s_band_%.2frstar_%.2frgalx.png' % 
					(N_tt, band[tt], ext_r_star[kk], ext_r_galx[kk]), dpi = 300)
				plt.close()

	commd.Barrier()

	### test for RBL region
	cen_pos = 1280
	R_bl = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
	'''
	if rank == 1:
		for kk in range(3):

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt, band[kk]), 'r') as f:
				stack_img = np.array(f['a'])

			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
			SB_r0 = SB + mag_add[kk]
			R_r0 = Intns_r * 1

			BL_img = stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos]
			grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
			grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
			grd = np.array( np.meshgrid(grd_x, grd_y) )

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('%s band SB as function of RBL region [%d img] ' % (band[kk], N_tt) )
			ax.plot(R_r0, SB_r0, 'k-', label = '$ stack \; img $', alpha = 0.5)
			ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic Z05', alpha = 0.5)

			for qq in range(len(R_bl) - 1):
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > R_bl[qq] * Rpp) & (ddr < R_bl[qq + 1] * Rpp)
				Resi_bl = np.nanmean( BL_img[idu] )
				dd_SB = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
				ax.plot(R_r0, dd_SB, linestyle = '--', color = mpl.cm.rainbow(qq / (len(R_bl) - 1) ), 
					label = '$ subtracted \, RBL \, in \, %.2f \sim %.2fMpc $' % (R_bl[qq], R_bl[qq + 1]), alpha = 0.5)

			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 33)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 3, fontsize = 7.5)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')			
			plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/SB_%d_img_%s_band_BL_sub.png' % (N_tt, band[kk]), dpi = 300)
			plt.close()
	commd.Barrier()	
	'''
	ext_r_star = np.array([1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8])
	ext_r_galx = np.array([1.6, 1.8, 2., 2.2, 2.4, 2.6, 2.8])
	N_set = len(ext_r_star)

	### test for RBL extimate
	if rank == 1:
		for kk in range(3):

			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt, band[kk]), 'r') as f:
				stack_img = np.array(f['a'])
			BL_img = stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos]
			grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
			grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
			grd = np.array( np.meshgrid(grd_x, grd_y) )
			ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )

			ref_RBL = []
			for qq in range(len(R_bl) - 1):
				idu = (ddr > R_bl[qq] * Rpp) & (ddr < R_bl[qq + 1] * Rpp)
				Resi_bl = np.nanmean( BL_img[idu] )
				ref_RBL.append(Resi_bl)
			ref_RBL = np.array(ref_RBL) / pixel**2
			BL_r = 0.5 * (R_bl[:-1] + R_bl[1:])

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('RBL estimate as function of mask size and radius')
			ax.plot(BL_r, ref_RBL, 'k--', label = '$ r_{mask} / r_{mask}^{0} = 1.$', alpha = 0.5)

			for tt in range(N_set):
				with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_Amask_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % 
					(N_tt, band[kk], ext_r_star[tt], ext_r_galx[tt]), 'r') as f:
					tt_stack_img = np.array(f['a'])
				BL_img = tt_stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos]

				sub_RBL = []
				for qq in range(len(R_bl) - 1):
					idu = (ddr > R_bl[qq] * Rpp) & (ddr < R_bl[qq + 1] * Rpp)
					Resi_bl = np.nanmean( BL_img[idu] )
					sub_RBL.append(Resi_bl)
				sub_RBL = np.array(sub_RBL) / pixel**2

				ax.plot(BL_r, sub_RBL, linestyle = '-', color = mpl.cm.rainbow(tt / N_set), 
					label = '$ r_{mask} / r_{mask}^{0} = %.2f $' % ext_r_star[tt], alpha = 0.5)
			ax.set_xlabel('$ \overline{R} [Mpc] $')
			ax.set_ylabel('$ RBL(r) [nmaggy / pixel^2]$')
			ax.legend(loc = 1)
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/%s_band_RBL_estimate_%d_img.png' % (band[kk], N_tt), dpi = 300)
			plt.close()
	commd.Barrier()

	### test for differ mask size
	if rank == 1:
		for kk in range(3):

			with h5py.File(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_Amask_%d_in_%s_band.h5' % (N_tt, band[kk]), 'r') as f:
				stack_img = np.array(f['a'])

			BL_img = stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos] # 1280 pixel, for z = 0.25, larger than 2Mpc
			grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
			grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
			grd = np.array( np.meshgrid(grd_x, grd_y) )
			ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
			idu = (ddr > 1.0 * Rpp) & (ddr < 1.1 * Rpp)
			Resi_bl = np.nanmean( BL_img[idu] )

			ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
			Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
			SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2)
			sub_SB_r0 = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
			SB_r0 = SB + mag_add[kk]
			R_r0 = Intns_r * 1

			SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
			R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
			R_obs = R_obs**4
			## sersic part
			Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
			SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('$ %s \, band \, SB [%d imgs \,] \, [RBL %.2f \sim %.2fMpc]$' % (band[kk], N_tt, 1.0, 1.1) )

			for tt in range(N_set):
				with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_Amask_%d_in_%s_band_%.2frstar_%.2frgalx.h5' % 
					(N_tt, band[kk], ext_r_star[tt], ext_r_galx[tt]), 'r') as f:
					sub_stack_img = np.array(f['a'])
				ss_img = sub_stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
				Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)			
				SB0 = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]

				BL_img = sub_stack_img[y0 - cen_pos: y0 + cen_pos, x0 - cen_pos: x0 + cen_pos]
				ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
				idu = (ddr > 1.0 * Rpp) & (ddr < 1.1 * Rpp)
				Resi_bl = np.nanmean( BL_img[idu] )
				SB1 = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]

				ax.plot(Intns_r, SB0, linestyle = '-', color = mpl.cm.rainbow(tt / N_set), 
					label = '$ stack \; image [r_{mask} / r_{mask}^{0} = %.2f] $' % ext_r_star[tt], alpha = 0.5)
				ax.plot(Intns_r, SB1, linestyle = '--', color = mpl.cm.rainbow(tt / N_set), alpha = 0.5)

			ax.plot(R_obs, SB_obs, 'm-', label = 'Z05', alpha = 0.5)
			ax.plot(R_obs, SB_Z05, 'm:', label = 'Sersic Z05', alpha = 0.5)
			ax.plot(R_r0, SB_r0, 'k-', label = '$ stack \; img [r_{mask} / r_{mask}^{0} = 1.] $', alpha = 0.5)
			ax.plot(R_r0, sub_SB_r0, 'k--', label = '$ RBL \; subtracted $', alpha = 0.5)

			ax.set_xlabel('$R[kpc]$')
			ax.set_ylabel('$SB[mag / arcsec^2]$')
			ax.set_xscale('log')
			ax.set_ylim(20, 33)
			ax.set_xlim(1, 1.5e3)
			ax.legend(loc = 3, fontsize = 7.5)
			ax.invert_yaxis()
			ax.grid(which = 'both', axis = 'both')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/SB_pro_%s_band_%d_img.png' % (band[kk], N_tt), dpi = 300)
			plt.close()
	commd.Barrier()

if __name__=="__main__":
	main()
