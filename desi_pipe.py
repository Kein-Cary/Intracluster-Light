import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import astropy.io.ascii as asc
from astropy import cosmology as apcy

import subprocess as subpro

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

R0 = 2 # Mpc / h
pixel = 0.45
band = ['g', 'r', 'z']
home = '/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/'
d_load = '/mnt/ddnfs/data_users/cxkttwl/ICL/BASS/bass_img/'

def medi_mask(img_data, source_file, bcg_x, bcg_y):

	source = asc.read(source_file)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	xn = np.array(source['X_IMAGE']) - 1
	yn = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])

	Kron = 6.
	a = Kron*A
	b = Kron*B

	mask_path = np.ones((img_data.shape[0], img_data.shape[1]), dtype = np.float)
	ox = np.linspace(0, img_data.shape[1] - 1, img_data.shape[1])
	oy = np.linspace(0, img_data.shape[0] - 1, img_data.shape[0])
	basic_coord = np.array(np.meshgrid(ox, oy))
	major = a / 2
	minor = b / 2 # set the star mask based on the major and minor radius
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
		la1 = np.min( [np.int(xk + set_r +1), img_data.shape[1] - 1] )
		lb0 = np.max( [np.int(yk - set_r), 0] ) 
		lb1 = np.min( [np.int(yk + set_r +1), img_data.shape[0] - 1] )

		df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xk)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yk)* np.sin(chi)
		df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yk)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xk)* np.sin(chi)
		fr = df1**2 / lr**2 + df2**2 / sr**2
		jx = fr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
		iv[iu] = np.nan
		mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img_data
	## add back the BCG region
	tdr = np.sqrt((xn - bcg_x)**2 + (yn - bcg_y)**2)
	idx = tdr == np.min(tdr)

	lr = major[idx]
	sr = minor[idx]
	cr = senior[idx]

	set_r = np.int(np.ceil(1.0 * lr))
	la0 = np.max( [np.int(xn[idx] - set_r), 0])
	la1 = np.min( [np.int(xn[idx] + set_r +1), img_data.shape[1] - 1] )
	lb0 = np.max( [np.int(yn[idx] - set_r), 0] )
	lb1 = np.min( [np.int(yn[idx] + set_r +1), img_data.shape[0] - 1] )
	mask_img[lb0: lb1, la0: la1] = img_data[lb0: lb1, la0: la1]

	return mask_img

def mask(img_data, source_x, source_y, source_a, source_b, source_chi, bcg_x, bcg_y):

	xn = np.r_[source_x[0], source_x[1]]
	yn = np.r_[source_y[0], source_y[1]]
	a = np.r_[source_a[0], source_a[1]]
	b = np.r_[source_b[0], source_b[1]]
	theta = np.r_[source_chi[0], source_chi[1]]

	Numb = len(xn)
	mask_path = np.ones((img_data.shape[0], img_data.shape[1]), dtype = np.float)
	ox = np.linspace(0, img_data.shape[1] - 1, img_data.shape[1])
	oy = np.linspace(0, img_data.shape[0] - 1, img_data.shape[0])
	basic_coord = np.array(np.meshgrid(ox, oy))
	major = a / 2
	minor = b / 2 # set the star mask based on the major and minor radius
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
		la1 = np.min( [np.int(xk + set_r +1), img_data.shape[1] - 1] )
		lb0 = np.max( [np.int(yk - set_r), 0] ) 
		lb1 = np.min( [np.int(yk + set_r +1), img_data.shape[0] - 1] )

		df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xk)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yk)* np.sin(chi)
		df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yk)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xk)* np.sin(chi)
		fr = df1**2 / lr**2 + df2**2 / sr**2
		jx = fr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
		iv[iu] = np.nan
		mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img_data
	## add back the BCG region
	tdr = np.sqrt((source_x[0] - bcg_x)**2 + (source_y[0] - bcg_y)**2)
	idx = tdr == np.min(tdr)

	lr = source_a[0][idx] / 2
	sr = source_b[0][idx] / 2
	cr = np.sqrt(lr**2 - sr**2)

	set_r = np.int(np.ceil(1.0 * lr))
	la0 = np.max( [np.int(source_x[0][idx] - set_r), 0])
	la1 = np.min( [np.int(source_x[0][idx] + set_r +1), img_data.shape[1] - 1] )
	lb0 = np.max( [np.int(source_y[0][idx] - set_r), 0] )
	lb1 = np.min( [np.int(source_y[0][idx] + set_r +1), img_data.shape[0] - 1] )
	mask_img[lb0: lb1, la0: la1] = img_data[lb0: lb1, la0: la1]

	return mask_img

def lod_img(band_id, set_z, set_ra, set_dec):

	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'
	out_load_0 = '/mnt/ddnfs/data_users/cxkttwl/PC/%d_cpus_mask_test_0.cat' % rank
	out_load_1 = '/mnt/ddnfs/data_users/cxkttwl/PC/%d_cpus_mask_test_1.cat' % rank

	kk = band_id
	zN = len(set_z)

	doc = open(home + 'BASS_obj/err_lis/%s_band_%d_err_stop_mask.txt' % (band[kk], rank), 'w')
	for ll in range(zN):
		ra_g, dec_g, z_g = set_ra[ll], set_dec[ll], set_z[ll]
		Da_g = Test_model.angular_diameter_distance(z_g).value
		R_pix = ( (rad2asec * 2 / h ) / Da_g ) / pixel

		try:
			data = fits.open(d_load + 'bass_img_ra%.3f_dec%.3f_z%.3f_%s_band.fits' % (ra_g, dec_g, z_g, band[kk]) )
			img = data[0].data
			head = data[0].header
			w = awc.WCS(head)
			cx, cy = w.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

			file = d_load + 'bass_img_ra%.3f_dec%.3f_z%.3f_%s_band.fits' % (ra_g, dec_g, z_g, band[kk])
			cmd = 'sex '+ file + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_load_0, out_cat)
			a = subpro.Popen(cmd, shell = True)
			a.wait()

			mask_img = medi_mask(img, out_load_0, cx, cy) ## first masking

			hdu = fits.PrimaryHDU()
			hdu.data = mask_img
			hdu.header = head
			hdu.writeto('/mnt/ddnfs/data_users/cxkttwl/PC/first_mask_%d_cpus.fits' % rank, overwrite = True)
			file = '/mnt/ddnfs/data_users/cxkttwl/PC/first_mask_%d_cpus.fits' % rank
			cmd = 'sex '+ file + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_load_1, out_cat)
			a = subpro.Popen(cmd, shell = True)
			a.wait()

			## load sectractor result
			source_0 = asc.read(out_load_0)
			Numb = np.array(source_0['NUMBER'][-1])
			A = np.array(source_0['A_IMAGE'])
			B = np.array(source_0['B_IMAGE'])
			theta = np.array(source_0['THETA_IMAGE'])
			xn = np.array(source_0['X_IMAGE']) - 1
			yn = np.array(source_0['Y_IMAGE']) - 1
			p_type = np.array(source_0['CLASS_STAR'])

			source_1 = asc.read(out_load_1)

			Numb = Numb + np.array(source_1['NUMBER'][-1])
			A = np.r_[A, np.array(source_1['A_IMAGE'])]
			B = np.r_[B, np.array(source_1['B_IMAGE'])]
			theta = np.r_[theta, np.array(source_1['THETA_IMAGE'])]
			xn = np.r_[xn, np.array(source_1['X_IMAGE']) - 1]
			yn = np.r_[yn, np.array(source_1['Y_IMAGE']) - 1]
			p_type = np.r_[p_type, np.array(source_1['CLASS_STAR'])]

			Kron = 10.
			a = Kron * A
			b = Kron * B

			## including the PSF sources in Tractor_cat
			Tract = fits.open(home + 'Tractor_cat/desi_tractor-cat_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g) )
			ra_sor = Tract[1].data['ra']
			dec_sor = Tract[1].data['dec']
			source_type = Tract[1].data['type']
			pox, poy = w.all_world2pix(ra_sor * U.deg, dec_sor * U.deg, 1)

			bnd_str = 'psfsize_%s' % band[kk]
			FWHM = (Tract[1].data[bnd_str] / pixel) ## FWHM in unit of pix-number

			## use flux / SB to select stars / psf sources
			apf_flux = Tract[1].data['apflux_r'][:, 0]
			apf_mag = 22.5 - 2.5 * np.log10(apf_flux) + 2.5 * np.log10(np.pi * 0.5**2)

			## divide sources into: point source, galaxy
			idx_pont = source_type == 'PSF'
			pont_x, pont_y = pox[idx_pont], poy[idx_pont]
			star_mag = apf_mag[[idx_pont]]
			sub_FWHM = FWHM[[idx_pont]]

			idx0 = star_mag <= 19
			pont_x0, pont_y0 = pont_x[idx0], pont_y[idx0]
			pont_r0 = 15 * sub_FWHM[idx0]

			idx1 = (star_mag > 19) & (star_mag < 24)
			pont_x1, pont_y1 = pont_x[idx1], pont_y[idx1]
			pont_r1 = 5 * sub_FWHM[idx1]

			idx2 = (star_mag > 24)
			pont_x2, pont_y2 = pont_x[idx2], pont_y[idx2]
			pont_r2 = 2 * sub_FWHM[idx2]

			x_star = np.r_[pont_x0, pont_x1, pont_x2]
			y_star = np.r_[pont_y0, pont_y1, pont_y2]
			r_star = np.r_[pont_r0, pont_r1, pont_r2]
			chi = np.zeros(len(x_star), dtype = np.float)

			sor_x = [xn, x_star]
			sor_y = [yn, y_star]
			sor_a = [a, r_star]
			sor_b = [b, r_star]
			sor_chi = [theta, chi]

			mask_img = mask(img, sor_x, sor_y, sor_a, sor_b, sor_chi, cx, cy)
			hdu = fits.PrimaryHDU()
			hdu.data = mask_img
			hdu.header = head
			hdu.writeto(home + 'mask_record/mask_bass_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g), overwrite = True)

			### img record
			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('source masking [%s band ra%.3f dec%.3f z%.3f]' % (band[kk], ra_g, dec_g, z_g) )
			tf = ax.imshow(mask_img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e-1, norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = ax, fraction = 0.050, pad =  0.01, label = 'flux[nmaggy]')
			circl0 = Circle(xy = (cx, cy), radius = R_pix, fill = False, ec = 'b', ls = '-', linewidth = 0.5, alpha = 0.5, label = '2 Mpc/h')
			circl1 = Circle(xy = (cx, cy), radius = 0.5 * R_pix, fill = False, ec = 'b', ls = '--', linewidth = 0.5, alpha = 0.5, label = '1 Mpc/h')
			ax.add_patch(circl0)
			ax.add_patch(circl1)
			ax.set_xlim(0, img.shape[1])
			ax.set_ylim(0, img.shape[0])
			plt.savefig(home + 'fig_view/mask/%s_band_source_mask_ra%.3f_dec%.3f_z%.3f.png' % (band[kk], ra_g, dec_g, z_g), dpi = 300)
			plt.close()

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('cluster [%s band ra%.3f dec%.3f z%.3f]' % (band[kk], ra_g, dec_g, z_g) )
			tf = ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e-1, norm = mpl.colors.LogNorm())
			plt.colorbar(tf, ax = ax, fraction = 0.050, pad =  0.01, label = 'flux[nmaggy]')
			circl0 = Circle(xy = (cx, cy), radius = R_pix, fill = False, ec = 'b', ls = '-', linewidth = 0.5, alpha = 0.5, label = '2 Mpc/h')
			circl1 = Circle(xy = (cx, cy), radius = 0.5 * R_pix, fill = False, ec = 'b', ls = '--', linewidth = 0.5, alpha = 0.5, label = '1 Mpc/h')
			ax.add_patch(circl0)
			ax.add_patch(circl1)
			ax.set_xlim(0, img.shape[1])
			ax.set_ylim(0, img.shape[0])
			plt.savefig(home + 'fig_view/data_view/cluster_%s_band_ra%.3f_dec%.3f_z%.3f.png' % (band[kk], ra_g, dec_g, z_g), dpi = 300)
			plt.close()

		except:
			s = '%s, %d, %.3f, %.3f, %.3f' % (band[kk], ll, ra_g, dec_g, z_g)
			print(s, file = doc, )
			continue
	doc.close()

	return

def main():

	with h5py.File(home + 'BASS_obj/mpi_h5/sdss_z-lambda_cat.h5', 'r') as f:
		tmp_array = np.array(f['a'])
	ra, dec, z = np.array(tmp_array[0]), np.array(tmp_array[1]), np.array(tmp_array[2])
	zN = len(z)

	for kk in range(len(band)):
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		lod_img(kk, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		#commd.Barrier()

if __name__ == "__main__":
	main()
