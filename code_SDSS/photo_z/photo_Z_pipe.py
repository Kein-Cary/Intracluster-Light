import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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

from light_measure import flux_recal
from resample_modelu import sum_samp, down_samp
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

R0 = 1 # in unit Mpc
pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/photo_data/'
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
Rv = 3.1
sfd = SFDQuery()

def pho_mask(band_id, z_set, ra_set, dec_set):

	kk = np.int(band_id)
	Nz = len(z_set)
	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'
	out_load_A = tmp + 'A_mask_%d_cpus.cat' % rank
	## size test
	r_res = 2.8 # 2.8 for larger R setting
	for q in range(Nz):
		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]
		try:
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
			hdu.writeto(tmp + 'source_data_%d.fits' % rank, overwrite = True)

			file_source = tmp + 'source_data_%d.fits' % rank
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
			a = Kron*A
			b = Kron*B

			mask = load + 'photo_z/star_cat/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
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
			hdu.writeto(load + 'photo_z/mask/A_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[kk], ra_g, dec_g, z_g),overwrite = True)
		except FileNotFoundError:
			continue
	return

def pho_resample(band_id, sub_z, sub_ra, sub_dec):
	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		try:
			data = fits.getdata(load + 'photo_z/mask/A_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
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
			fits.writeto(load + 'photo_z/resample/pho_z-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), 
				resam, header = fil, overwrite=True)
		except FileNotFoundError:
			continue
	return

def pho_edg_cut(band_id, sub_z, sub_ra, sub_dec):

	ii = np.int(band_id)
	zn = len(sub_z)
	for k in range(zn):
		ra_g = sub_ra[k]
		dec_g = sub_dec[k]
		z_g = sub_z[k]
		try:
			data = fits.getdata(load + 
				'photo_z/resample/pho_z-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
			img = data[0]
			BCGx, BCGy = data[1]['CENTER_X'], data[1]['CENTER_Y']
			RA0, DEC0 = data[1]['CRVAL1'], data[1]['CRVAL2']

			xc, yc = np.int(img.shape[1] / 2), np.int(img.shape[0] / 2)
			## keep the image size but set np.nan for egde pixels
			re_img = np.zeros( (img.shape[0], img.shape[1]), dtype = np.float) + np.nan
			( re_img[yc - np.int(Rpp): yc + np.int(Rpp), xc - np.int(1.3 * Rpp): xc + np.int(1.3 * Rpp)] ) = ( 
				img[yc - np.int(Rpp): yc + np.int(Rpp), xc - np.int(1.3 * Rpp): xc + np.int(1.3 * Rpp)] )

			New_bcgx = BCGx + 0
			New_bcgy = BCGy + 0

			Lx = re_img.shape[1]
			Ly = re_img.shape[0]
			Crx = xc + 0
			Cry = yc + 0

			keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
					'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
			value = ['T', 32, 2, Lx, Ly, Crx, Cry, New_bcgx, New_bcgy, RA0, DEC0, ra_g, dec_g, z_g, pixel]
			ff = dict(zip(keys,value))
			fil = fits.Header(ff)
			fits.writeto(load + 
			'photo_z/resamp_cut/cut_edge-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % 
			(band[ii], ra_g, dec_g, z_g), re_img, header = fil, overwrite=True)
		except FileNotFoundError:
			continue

	return

def main():

	## source masking
	for tt in range(3):

		with h5py.File(load + 'mpi_h5/photo_z_difference_sample.h5', 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
		zN = len(z)

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		pho_mask(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

	## pixel resampling
	for tt in range(3):

		with h5py.File(load + 'mpi_h5/photo_z_difference_sample.h5', 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
		zN = len(z)

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		pho_resample(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

	## rule out the edge pixels
	for tt in range(3):

		with h5py.File(load + 'mpi_h5/photo_z_difference_sample.h5', 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
		zN = len(z)

		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		pho_edg_cut(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

if __name__ == "__main__":
	main()
