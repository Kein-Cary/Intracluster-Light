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
import time
##
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
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

# sample catalog (total 3434 images in each band)
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/mpi_h5/sample_catalog.h5', 'r') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
#d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/sky_sub_img/' # add sky information

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])
zopt = np.array([22.5, 22.5, 22.5, 22.46, 22.52])
sb_lim = np.array([24.5, 25, 24, 24.35, 22.9])
Rv = 3.1
sfd = SFDQuery()

def mask_A(band_id, z_set, ra_set, dec_set):
	kk = np.int(band_id)
	Nz = len(z_set)
	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'
	out_load_A = '/mnt/ddnfs/data_users/cxkttwl/PC/A_mask_%d_cpus.cat' % rank

	for q in range(Nz):
		z_g = z_set[q]
		ra_g = ra_set[q]
		dec_g = dec_set[q]

		pro_f = d_file + 'Revis-ra%.3f-dec%.3f-z%.3f-%s-band.fits' % (ra_g, dec_g, z_g, band[kk]) # add sky information
		#pro_f = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[kk], ra_g, dec_g, z_g)

		data_f = fits.open(pro_f)
		img = data_f[0].data
		head_inf = data_f[0].header
		wcs = awc.WCS(head_inf)
		cx_BCG, cy_BCG = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
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
		dete_thresh = sb_lim[kk] + 10*np.log10((1 + z_g) / (1 + z_ref))
		dete_thresh = '%.3f' % dete_thresh + ',%.2f' % zopt[kk]
		dete_min = '10'
		ana_thresh = dete_thresh *1
		cmd = (
			'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s'
			%(param_A, out_load_A, out_cat, dete_min, dete_thresh, ana_thresh))
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
		#Kron = source['KRON_RADIUS']
		Kron = 6 * 2.8 ## mask with the test result
		a = Kron * A
		b = Kron * B

		mask = load + 'bright_star_dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
		cat = pds.read_csv(mask, skiprows = 1)
		set_ra = np.array(cat['ra'])
		set_dec = np.array(cat['dec'])
		set_mag = np.array(cat['r'])
		OBJ = np.array(cat['type'])
		xt = cat['Column1']
		tau = 10 * 2.8 # the mask size set as 10 * FWHM from dr12

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
		basic_coord = np.array(np.meshgrid(ox,oy))
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
			chi = theta[k] * np.pi / 180

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
		hdu.writeto(load + 'mask_data/A_plane/Zibetti/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits' % (band[kk], ra_g, dec_g, z_g),overwrite = True)

		plt.figure()
		plt.imshow(mirro_A, cmap = 'Greys', origin = 'lower', vmin = 1e-5, norm = mpl.colors.LogNorm())
		hsc.circles(cx_BCG, cy_BCG, s = R_p, fc = '', ec = 'b', )
		hsc.circles(cx_BCG, cy_BCG, s = 1.1 * R_p, fc = '', ec = 'b', ls = '--')
		plt.plot(cx_BCG, cy_BCG, 'bo', alpha = 0.5)
		plt.title('A mask %s ra%.3f dec%.3f z%.3f' % (band[kk], ra_g, dec_g, z_g) )
		plt.xlim(0, img.shape[1])
		plt.ylim(0, img.shape[0])
		plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_class/Amask_Zibetti/A_mask_%s_ra%.3f_dec%.3f_z%.3f.png'%(band[kk], ra_g, dec_g, z_g), dpi = 300)
		plt.close()

	return

def main():

	Ntot = len(z)
	#commd.Barrier()
	for tt in range(3):
		m, n = divmod(Ntot, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n
		mask_A(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()

if __name__ == "__main__":
	main()
