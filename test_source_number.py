import matplotlib as mpl
mpl.use('Agg')
import handy.scatter as hsc
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pd
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import subprocess as subpro
from astropy.coordinates import SkyCoord

import time
# constant
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
Jy = 10**(-23) # in unit of (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

# total catalogue with redshift
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

# dust correct
mag_add = np.array([-0.04, 0, 0, 0, 0.02])
sb_lim = np.array([25, 25, 24.5, 24, 22.9]) # SB limit at z_ref
zopt = np.array([22.46, 22.5, 22.5, 22.5, 22.52]) # zero point

def bins_viw():
	bins = 30
	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))
	load = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
	file = 'frame-r-ra150.040-dec17.588-redshift0.291.fits.bz2'
	rag = 150.040
	decg = 17.588
	zg = 0.291

	data_f = fits.open(load+file)
	img = data_f[0].data
	head_inf = data_f[0].header
	wcs = awc.WCS(head_inf)
	cx, cy = wcs.all_world2pix(rag*U.deg, decg*U.deg, 1)
	cr = np.sqrt((img_grid[0,:] - cx)**2 + (img_grid[1,:] - cy)**2)

	Da = Test_model.angular_diameter_distance(zg).value
	Rp = (rad2asec/Da)/pixel
	Rk = np.logspace(1e-2, np.log10(Rp), bins)

	theta = np.arctan2(img_grid[1,:]-cy, img_grid[0,:]-cx)
	phi = theta * 180/np.pi
	eta = np.linspace(0, 360, 25)
	eta = eta - 180
	# mine
	plt.figure()
	plt.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
	hsc.circles(cx, cy, s = Rk, fc = '', ec = 'b')
	hsc.circles(cx, cy, s = 1.1*Rp, fc = '', ec = 'b', ls = '--')
	for k in range(len(Rk) - 1):
		dr = Rk[k+1] - Rk[k]
		ir = (cr >= Rk[k]) & (cr <= Rk[k+1])

		for q in range(len(eta)-1):
			dci = (phi >= eta[q]) & (phi <= eta[q+1])
			iv = dci & ir
			iu = np.where(iv == True)
			setx = x0[iu[1]]
			sety = y0[iu[0]]

			plt.plot(setx, sety, color = mpl.cm.rainbow(q/len(eta)), alpha = 0.5)
	plt.title('$bins \; for \; SB \; uncertainty \; estimate \; (me)$')
	plt.savefig('error_estimate_mine.png', dpi = 600)
	plt.close()
	# Zibetti
	plt.figure()
	plt.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-3, norm = mpl.colors.LogNorm())
	hsc.circles(cx, cy, s = Rk, fc = '', ec = 'b')
	hsc.circles(cx, cy, s = 1.1*Rp, fc = '', ec = 'b', ls = '--')
	for k in range(len(Rk) - 1):
		dr = Rk[k+1] - Rk[k]
		ir = (cr >= Rk[k]) & (cr <= Rk[k+1])
		d_chi = (dr / Rk[k]) * 180/np.pi
		chi = np.arange(0, 360, d_chi)
		chi = chi - 180

		c_num = 0	
		for q in range(len(chi)-1):
			dci = (phi >= chi[q]) & (phi <= chi[q+1])
			iv = dci & ir
			iu = np.where(iv == True)
			if len(iu[0]) != 0:
				c_num = c_num + 1
				setx = x0[iu[1]]
				sety = y0[iu[0]]
				plt.plot(setx, sety, color = mpl.cm.rainbow(q/len(chi)), alpha = 0.5)
			else:
				c_num = c_num
	plt.title('$bins \; for \; SB \; uncertainty \; estimate \; in \; Zibetti_{2005}$')
	plt.savefig('error_estimate_Ze.png', dpi = 600)
	plt.close()

	return

def sex_number():
	stack_N = np.int(100)
	bins = 60
	x0 = np.linspace(0, 2047, 2048)
	y0 = np.linspace(0, 1488, 1489)
	img_grid = np.array(np.meshgrid(x0, y0))

	param_A = 'default_mask_A.sex'
	param_A_Tal = 'default_mask_A_Tal.sex'
	out_cat = 'default_mask_A.param'
	out_load_No = './result/mask_A_number.cat'

	band_add = ['r', 'i', 'z']
	band_fil = ['u', 'g', 'r', 'i', 'z']
	d_file = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'
	load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'

	N_source_15 = np.zeros((len(band_fil), bins), dtype = np.float)
	N_source_Tal = np.zeros((len(band_fil), bins), dtype = np.float)
	N_source_ZIT = np.zeros((len(band_fil), bins), dtype = np.float)

	for l in range(len(band_fil)):

		sub_N_15 = np.zeros((stack_N, bins), dtype = np.float)
		sub_count_15 = np.zeros((stack_N, bins), dtype = np.float)

		sub_N_Tal = np.zeros((stack_N, bins), dtype = np.float)
		sub_count_Tal = np.zeros((stack_N, bins), dtype = np.float)

		sub_N_ZIT = np.zeros((stack_N, bins), dtype = np.float)
		sub_count_ZIT = np.zeros((stack_N, bins), dtype = np.float)

		for q in range(stack_N):
			pro_f = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band_fil[l], ra[q], dec[q], z[q])
			z_g = z[q]
			ra_g = ra[q]
			dec_g = dec[q]

			data_f = fits.open(pro_f)
			img = data_f[0].data
			head_inf = data_f[0].header
			wcs = awc.WCS(head_inf)
			cx_BCG, cy_BCG = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)
			R_ph = rad2asec/(Test_model.angular_diameter_distance(z_g).value)
			R_p = R_ph/pixel
			Rk = np.linspace(0, R_p, bins+1)

			hdu = fits.PrimaryHDU()
			hdu.data = data_f[0].data
			hdu.header = head_inf
			hdu.writeto('source_data_num.fits', overwrite = True)
			file_source = './source_data_num.fits'
			# 1.5sigma case
			cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_No, out_cat)
			print(cmd)
			a = subpro.Popen(cmd, shell = True)
			a.wait()
			source = asc.read(out_load_No)
			Numb = np.array(source['NUMBER'][-1])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1
			dr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)

			for k in range(len(Rk) - 1):
				ix = (dr >= Rk[k]) & (dr <= Rk[k+1])
				iy = np.where(ix == True)
				sub_N_15[q, k] = sub_N_15[q, k] + len(iy[0])
				if len(iy[0]) == 0 :
					sub_count_15[q, k] = sub_count_15[q, k]
				else:
					sub_count_15[q, k] = sub_count_15[q, k] + 1

			# Tal case
			combine = np.zeros((1489, 2048), dtype = np.float)
			for t in range(len(band_add)):
				file_p = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band_add[t], ra[q], dec[q], z[q])
				data_p = fits.open(file_p)
				img_p = data_p[0].data
				combine = combine + img_p
			hdu = fits.PrimaryHDU()
			hdu.data = combine
			hdu.header = head_inf
			hdu.writeto('combine_data_No.fits', overwrite = True)
			file_source = './combine_data_No.fits'
			cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A_Tal, out_load_No, out_cat)
			print(cmd)
			a = subpro.Popen(cmd, shell = True)
			a.wait()						
			source = asc.read(out_load_No)
			Numb = np.array(source['NUMBER'][-1])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1
			dr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)

			for k in range(len(Rk) - 1):
				ix = (dr >= Rk[k]) & (dr <= Rk[k+1])
				iy = np.where(ix == True)
				sub_N_Tal[q, k] = sub_N_Tal[q, k] + len(iy[0])
				if len(iy[0]) == 0 :
					sub_count_Tal[q, k] = sub_count_Tal[q, k]
				else:
					sub_count_Tal[q, k] = sub_count_Tal[q, k] + 1

			# ZIT case
			if (l == 4) | (l == 0):
				combine = np.zeros((1489, 2048), dtype = np.float)
				for t in range(len(band_add)):
					file_p = d_file + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(band_add[t], ra[q], dec[q], z[q])
					data_p = fits.open(file_p)
					img_p = data_p[0].data
					combine = combine + img_p
				hdu = fits.PrimaryHDU()
				hdu.data = combine
				hdu.header = head_inf
				hdu.writeto('combine_data_No.fits', overwrite = True)
				file_source = './combine_data_No.fits'
				cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A_Tal, out_load_No, out_cat)
			else:
				hdu = fits.PrimaryHDU()
				hdu.data = data_f[0].data
				hdu.header = head_inf
				hdu.writeto('source_data_zit.fits', overwrite = True)
				file_source = './source_data_zit.fits'	
				dete_thresh = sb_lim[l] + 10*np.log10((1 + z_g)/(1 + z_ref))
				dete_thresh = '%.3f' % dete_thresh + ',%.2f' % zopt[l]
				dete_min = '10'
				ana_thresh = dete_thresh *1
				cmd = (
					'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s -DETECT_MINAREA %s -DETECT_THRESH %s -ANALYSIS_THRESH %s'
					%(param_A, out_load_No, out_cat, dete_min, dete_thresh, ana_thresh))

			print(cmd)
			a = subpro.Popen(cmd, shell = True)
			a.wait()
			source = asc.read(out_load_No)
			Numb = np.array(source['NUMBER'][-1])
			cx = np.array(source['X_IMAGE']) - 1
			cy = np.array(source['Y_IMAGE']) - 1
			dr = np.sqrt((cx - cx_BCG)**2 + (cy - cy_BCG)**2)

			for k in range(len(Rk) - 1):
				ix = (dr >= Rk[k]) & (dr <= Rk[k+1])
				iy = np.where(ix == True)
				sub_N_ZIT[q, k] = sub_N_ZIT[q, k] + len(iy[0])
				if len(iy[0]) == 0 :
					sub_count_ZIT[q, k] = sub_count_ZIT[q, k]
				else:
					sub_count_ZIT[q, k] = sub_count_ZIT[q, k] + 1

		N_source_15[l,:] = np.sum(sub_N_15, axis = 0) / np.sum(sub_count_15, axis = 0)
		N_source_Tal[l,:] = np.sum(sub_N_Tal, axis = 0) / np.sum(sub_count_Tal, axis = 0)
		N_source_ZIT[l,:] = np.sum(sub_N_ZIT, axis = 0) / np.sum(sub_count_ZIT, axis = 0)

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_15sigma.h5', 'w') as f:
		f['a'] = N_source_15
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_15sigma.h5') as f:
		for tt in range(len(N_source_15)):
			f['a'][tt,:] = N_source_15[tt,:]

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_Tal.h5', 'w') as f:
		f['a'] = N_source_Tal
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_Tal.h5') as f:
		for tt in range(len(N_source_Tal)):
			f['a'][tt,:] = N_source_Tal[tt,:]	

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_ZIT.h5', 'w') as f:
		f['a'] = N_source_ZIT
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_ZIT.h5') as f:
		for tt in range(len(N_source_ZIT)):
			f['a'][tt,:] = N_source_ZIT[tt,:]

	return

def fig_out():

	Rp = (rad2asec/Da_ref)/pixel
	Ar = rad2asec/Da_ref
	bins = 30
	Rk = np.linspace(0, Rp, bins+1)
	Rs = np.zeros(np.int(bins), dtype = np.float)
	for k in range(len(Rk)-1):
		Rs[k] = 0.5*(Rk[k] + Rk[k+1])
	Rs = Rs / Rp

	band = ['u', 'g', 'r', 'i', 'z']

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_15sigma.h5') as f:
		N_source_15 = np.array(f['a'])
	Ns_15_u = N_source_15[0,:]
	where_are_nan = np.isnan(Ns_15_u)
	Ns_15_u[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_15_u)
	Ns_15_u[where_are_inf] = 0

	Ns_15_g = N_source_15[1,:]
	where_are_nan = np.isnan(Ns_15_g)
	Ns_15_g[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_15_g)
	Ns_15_g[where_are_inf] = 0

	Ns_15_r = N_source_15[2,:]
	where_are_nan = np.isnan(Ns_15_r)
	Ns_15_r[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_15_r)
	Ns_15_r[where_are_inf] = 0

	Ns_15_i = N_source_15[3,:]
	where_are_nan = np.isnan(Ns_15_i)
	Ns_15_i[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_15_i)
	Ns_15_i[where_are_inf] = 0

	Ns_15_z = N_source_15[4,:]
	where_are_nan = np.isnan(Ns_15_z)
	Ns_15_z[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_15_z)
	Ns_15_z[where_are_inf] = 0

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_Tal.h5') as f:
		N_source_Tal = np.array(f['a'])
	Ns_Tal_u = N_source_Tal[0,:]
	where_are_nan = np.isnan(Ns_Tal_u)
	Ns_Tal_u[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_Tal_u)
	Ns_Tal_u[where_are_inf] = 0

	Ns_Tal_g = N_source_Tal[1,:]
	where_are_nan = np.isnan(Ns_Tal_g)
	Ns_Tal_g[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_Tal_g)
	Ns_Tal_g[where_are_inf] = 0

	Ns_Tal_r = N_source_Tal[2,:]
	where_are_nan = np.isnan(Ns_Tal_r)
	Ns_Tal_r[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_Tal_r)
	Ns_Tal_r[where_are_inf] = 0

	Ns_Tal_i = N_source_Tal[3,:]
	where_are_nan = np.isnan(Ns_Tal_i)
	Ns_Tal_i[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_Tal_i)
	Ns_Tal_i[where_are_inf] = 0

	Ns_Tal_z = N_source_Tal[4,:]
	where_are_nan = np.isnan(Ns_Tal_z)
	Ns_Tal_z[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_Tal_z)
	Ns_Tal_z[where_are_inf] = 0

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/source_Number_ZIT.h5') as f:
		N_source_ZIT = np.array(f['a'])
	Ns_ZIT_u = N_source_ZIT[0,:]
	where_are_nan = np.isnan(Ns_ZIT_u)
	Ns_ZIT_u[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_ZIT_u)
	Ns_ZIT_u[where_are_inf] = 0

	Ns_ZIT_g = N_source_ZIT[1,:]
	where_are_nan = np.isnan(Ns_ZIT_g)
	Ns_ZIT_g[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_ZIT_g)
	Ns_ZIT_g[where_are_inf] = 0

	Ns_ZIT_r = N_source_ZIT[2,:]
	where_are_nan = np.isnan(Ns_ZIT_r)
	Ns_ZIT_r[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_ZIT_r)
	Ns_ZIT_r[where_are_inf] = 0

	Ns_ZIT_i = N_source_ZIT[3,:]
	where_are_nan = np.isnan(Ns_ZIT_i)
	Ns_ZIT_i[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_ZIT_i)
	Ns_ZIT_i[where_are_inf] = 0

	Ns_ZIT_z = N_source_ZIT[4,:]
	where_are_nan = np.isnan(Ns_ZIT_z)
	Ns_ZIT_z[where_are_nan] = 0
	where_are_inf = np.isinf(Ns_ZIT_z)
	Ns_ZIT_z[where_are_inf] = 0

	plt.figure()
	plt.plot(Rs[Ns_15_u != 0], Ns_15_u[Ns_15_u != 0], 'ro', label = '$1.5 \sigma$', alpha = 0.5)
	plt.plot(Rs[Ns_Tal_u != 0], Ns_Tal_u[Ns_Tal_u != 0], 'bs', label = '$Tal_{2011}$', alpha = 0.5)
	plt.plot(Rs[Ns_ZIT_u != 0], Ns_ZIT_u[Ns_ZIT_u != 0], 'gx', label = '$Zibetti_{2005}$', alpha = 0.5)
	plt.xlabel(r'$ \frac {R}{1Mpc}$')
	plt.ylabel('$Number \; of \; source$')
	plt.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.legend(loc = 2)
	plt.title('$u \; band \; source \; number \; comparation$')
	plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/source_number_u_band.png', dpi = 600)
	plt.close()

	plt.figure()
	plt.plot(Rs[Ns_15_g != 0], Ns_15_g[Ns_15_g != 0], 'ro', label = '$1.5 \sigma$', alpha = 0.5)
	plt.plot(Rs[Ns_Tal_g != 0], Ns_Tal_g[Ns_Tal_g != 0], 'bs', label = '$Tal_{2011}$', alpha = 0.5)
	plt.plot(Rs[Ns_ZIT_g != 0], Ns_ZIT_g[Ns_ZIT_g != 0], 'gx', label = '$Zibetti_{2005}$', alpha = 0.5)
	plt.xlabel(r'$ \frac {R}{1Mpc}$')
	plt.ylabel('$Number \; of \; source$')
	plt.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.legend(loc = 2)
	plt.title('$g \; band \; source \; number \; comparation$')
	plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/source_number_g_band.png', dpi = 600)
	plt.close()

	plt.figure()
	plt.plot(Rs[Ns_15_r != 0], Ns_15_r[Ns_15_r != 0], 'ro', label = '$1.5 \sigma$', alpha = 0.5)
	plt.plot(Rs[Ns_Tal_r != 0], Ns_Tal_r[Ns_Tal_r != 0], 'bs', label = '$Tal_{2011}$', alpha = 0.5)
	plt.plot(Rs[Ns_ZIT_r != 0], Ns_ZIT_r[Ns_ZIT_r != 0], 'gx', label = '$Zibetti_{2005}$', alpha = 0.5)
	plt.xlabel(r'$ \frac {R}{1Mpc}$')
	plt.ylabel('$Number \; of \; source$')
	plt.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.legend(loc = 2)
	plt.title('$r \; band \; source \; number \; comparation$')
	plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/source_number_r_band.png', dpi = 600)
	plt.close()

	plt.figure()
	plt.plot(Rs[Ns_15_i != 0], Ns_15_i[Ns_15_i != 0], 'ro', label = '$1.5 \sigma$', alpha = 0.5)
	plt.plot(Rs[Ns_Tal_i != 0], Ns_Tal_i[Ns_Tal_i != 0], 'bs', label = '$Tal_{2011}$', alpha = 0.5)
	plt.plot(Rs[Ns_ZIT_i != 0], Ns_ZIT_i[Ns_ZIT_i != 0], 'gx', label = '$Zibetti_{2005}$', alpha = 0.5)
	plt.xlabel(r'$ \frac {R}{1Mpc}$')
	plt.ylabel('$Number \; of \; source$')
	plt.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.legend(loc = 2)
	plt.title('$i \; band \; source \; number \; comparation$')
	plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/source_number_i_band.png', dpi = 600)
	plt.close()

	plt.figure()
	plt.plot(Rs[Ns_15_z != 0], Ns_15_z[Ns_15_z != 0], 'ro', label = '$1.5 \sigma$', alpha = 0.5)
	plt.plot(Rs[Ns_Tal_z != 0], Ns_Tal_z[Ns_Tal_z != 0], 'bs', label = '$Tal_{2011}$', alpha = 0.5)
	plt.plot(Rs[Ns_ZIT_z != 0], Ns_ZIT_z[Ns_ZIT_z != 0], 'gx', label = '$Zibetti_{2005}$', alpha = 0.5)
	plt.xlabel(r'$ \frac {R}{1Mpc}$')
	plt.ylabel('$Number \; of \; source$')
	plt.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.legend(loc = 2)
	plt.title('$z \; band \; source \; number \; comparation$')
	plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/source_number_z_band.png', dpi = 600)
	plt.close()

	return

def main():
	bins_viw()
	#sex_number()
	#fig_out()

if __name__ == "__main__":
	main()
