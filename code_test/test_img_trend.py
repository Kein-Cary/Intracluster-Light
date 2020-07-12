import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import astropy.units as U
import astropy.constants as C

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import subprocess as subpro
import astropy.io.fits as fits

from scipy import ndimage
from astropy import cosmology as apcy

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

home = '/media/xkchen/My Passport/data/SDSS/'
load = '/media/xkchen/My Passport/data/SDSS/'
dfile = '/media/xkchen/My Passport/data/SDSS/redMap_random/'

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

def medi_mask(img_data, source_file):

	r_res = 2.8

	source = asc.read(source_file)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	xn = np.array(source['X_IMAGE']) - 1
	yn = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])

	Kron = 6 * r_res
	a = Kron * A
	b = Kron * B

	mask_path = np.ones((img_data.shape[0], img_data.shape[1]), dtype = np.float)
	ox = np.linspace(0, img_data.shape[1] - 1, img_data.shape[1])
	oy = np.linspace(0, img_data.shape[0] - 1, img_data.shape[0])
	basic_coord = np.array(np.meshgrid(ox, oy))
	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	for k in range(Numb):
		xk = xn[k]
		yk = yn[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi/180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xk - set_r), 0])
		la1 = np.min( [np.int(xk + set_r +1), img_data.shape[1] ] )
		lb0 = np.max( [np.int(yk - set_r), 0] ) 
		lb1 = np.min( [np.int(yk + set_r +1), img_data.shape[0] ] )

		df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xk)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yk)* np.sin(chi)
		df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yk)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xk)* np.sin(chi)
		fr = df1**2 / lr**2 + df2**2 / sr**2
		jx = fr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
		iv[iu] = np.nan
		mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img_data

	return mask_img

def mask_A(band_id, z_set, ra_set, dec_set):

	kk = np.int(band_id)
	Nz = len(z_set)

	param_A = 'default_mask_A.sex'
	#param_A = 'default_adjust.sex'  ## adjust someparameter
	out_cat = 'default_mask_A.param'
	out_load_0 = '/home/xkchen/mywork/ICL/code/SEX/result/rand_adjust_0.cat'
	out_load_1 = '/home/xkchen/mywork/ICL/code/SEX/result/rand_adjust_1.cat'

	r_res = 2.8

	for q in range(Nz):
		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		pro_f = dfile + 'rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], ra_g, dec_g, z_g)
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
		hdu.writeto('source_data.fits', overwrite = True)

		file_source = 'source_data.fits'
		cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_0, out_cat)
		a = subpro.Popen(cmd, shell = True)
		a.wait()

		### first masking
		mask_img_0 = medi_mask(img, out_load_0)

		hdu = fits.PrimaryHDU()
		hdu.data = mask_img_0
		hdu.header = head_inf
		hdu.writeto('first_mask.fits', overwrite = True)

		file = 'first_mask.fits'
		cmd = 'sex '+ file + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_1, out_cat)
		a = subpro.Popen(cmd, shell = True)
		a.wait()

		source = asc.read(out_load_1)
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

		mask = load + 'random_cat/star_cat/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		tau = 10 * 2.8

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

		### second masking
		mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
		ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
		oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
		basic_coord = np.array(np.meshgrid(ox, oy))
		major = a / 2
		minor = b / 2
		senior = np.sqrt(major**2 - minor**2)

		for k in range(Numb):
			xc = cx[k]
			yc = cy[k]

			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = theta[k] * np.pi / 180

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(xc - set_r), 0])
			la1 = np.min( [np.int(xc + set_r +1), img.shape[1] ] )
			lb0 = np.max( [np.int(yc - set_r), 0] ) 
			lb1 = np.min( [np.int(yc + set_r +1), img.shape[0] ] )

			df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
			df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
			fr = df1**2 / lr**2 + df2**2 / sr**2
			jx = fr <= 1

			iu = np.where(jx == True)
			iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
			iv[iu] = np.nan
			mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

		mask_img = mask_path * mask_img_0

		hdu = fits.PrimaryHDU()
		hdu.data = mask_img
		hdu.header = head_inf
		hdu.writeto(load + 'random_cat/mask_adjust/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g), overwrite = True)

	return

def rand_pont(band_id, sub_z, sub_ra, sub_dec):

	stack_N = len(sub_z)
	kk = np.int(band_id)

	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		data_A = fits.open(load + 'random_cat/mask_adjust/random_mask_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g) )

		img_A = data_A[0].data
		head = data_A[0].header
		'''
		wcs_lis = awc.WCS(head)
		xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

		# centered on cat.(ra, dec)
		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img_A.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img_A.shape[1])
		'''
		rnx, rny = np.int(img_A.shape[1] / 2), np.int(img_A.shape[0] / 2) ## image center
		la0 = np.int(y0 - rny)
		la1 = np.int(y0 - rny + img_A.shape[0])
		lb0 = np.int(x0 - rnx)
		lb1 = np.int(x0 - rnx + img_A.shape[1])

		idx = np.isnan(img_A)
		idv = np.where(idx == False)

		sum_array_A[la0: la1, lb0: lb1][idv] = sum_array_A[la0: la1, lb0: lb1][idv] + img_A[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img_A[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1.
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

	id_zero = p_count_A == 0
	p_count_A[id_zero] = np.nan
	sum_array_A[id_zero] = np.nan

	stack_img = sum_array_A / p_count_A
	where_are_inf = np.isinf(stack_img)
	stack_img[where_are_inf] = np.nan

	with h5py.File(load + 'random_cat/angle_stack/%s_band_stack_mask-only_adjust-param.h5' % band[kk], 'w') as f:
		f['a'] = np.array(stack_img)

	return

def main():

	for tt in range( 1 ):

		with h5py.File(load + 'random_cat/cat_select/rand_%s_band_catalog.h5' % (band[tt]), 'r') as f:
			tmp_array = np.array(f['a'])
		ra, dec, z, rich = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2]), np.array(tmp_array[3])
		zN = len(z)

		DN = 1000
		set_ra, set_dec, set_z = ra[:DN], dec[:DN], z[:DN]

		mask_A(tt, set_z, set_ra, set_dec)

		rand_pont(tt, set_z, set_ra, set_dec)

if __name__ == "__main__":
	main()
