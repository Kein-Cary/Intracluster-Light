import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits
import subprocess as subpro

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/' ## save the catalogue data
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'  ## save the process data
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

def mask_A(band_id, sub_z, sub_ra, sub_dec):
	tot_N = len(sub_z)
	ii = np.int(band_id)
	## set the masking parameter, and list the parameter need to save, save the source catalogue
	mask_para_set = '/home/cxkttwl/ICL/code/SEX/default_mask_A.sex'
	out_para_lis = '/home/cxkttwl/ICL/code/SEX/default_mask_A.param'
	out_cat = tmp + 'A_mask_%d_cpus.cat' % rank  ## the file save the catalogue

	r_res = 2.8 ## for larger R mask set based on test
	for jj in range(tot_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		file = tmp + 'test/Extinction_correct_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[ii], ra_g, dec_g, z_g)
		data = fits.open(file)
		img = data[0].data
		head_inf = data[0].header
		wcs = awc.WCS(head_inf)
		cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

		## run SExTractor for source finding
		cmd = 'sex '+ file + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(mask_para_set, out_cat, out_para_lis) # 1.5sigma
		tpp = subpro.Popen(cmd, shell = True)
		tpp.wait()

		source = asc.read(out_cat)
		Numb = np.array(source['NUMBER'][-1])
		A = np.array(source['A_IMAGE'])
		B = np.array(source['B_IMAGE'])
		theta = np.array(source['THETA_IMAGE'])
		cx = np.array(source['X_IMAGE']) - 1
		cy = np.array(source['Y_IMAGE']) - 1
		p_type = np.array(source['CLASS_STAR'])

		Kron = 6 * r_res
		# iso_radius set as 3 times isophoto (rms) times (2.8 from size test), 3 is according to handlebook, 2.8 based on our test
		a = Kron * A
		b = Kron * B

		## introduce the star catalogue based on SDSS, Dr8 photometric
		mask = load + 'bright_star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		tau = 10 * r_res # the mask size set as 10 * FWHM from dr12 times the test result

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

		## creat the mask image, rule out pixels are filled with np.nan
		mask_arr = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
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
				mask_arr[lb0: lb1, la0: la1] = mask_arr[lb0: lb1, la0: la1] * iv

		mirro_A = mask_arr * img

		hdu = fits.PrimaryHDU()
		hdu.data = mirro_A
		hdu.header = head_inf
		hdu.writeto(tmp + 'test/A_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), overwrite = True)

	return

def main():

	#for kk in range(len(band)):
	for kk in range( 3 ):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk], 'r') as f:
			cat = np.array(f['a'])
		ra, dec, z = cat[0,:], cat[1,:], cat[2,:]

		zN = len(z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		mask_A(kk, set_z[N_sub0 :N_sub1], set_ra[N_sub0 :N_sub1], set_dec[N_sub0 :N_sub1])
		commd.Barrier()

if __name__ == "__main__":
	main()
