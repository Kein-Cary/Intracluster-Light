import h5py
import numpy as np
from scipy import interpolate as interp
from numba import vectorize

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy

from resamp import gen
from resample_modelu import down_samp, sum_samp
from ICL_surface_mass_density import sigma_m_c
from light_measure import light_measure, flux_recal

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import time
# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
Lsun2erg = U.L_sun.to(U.erg/U.s)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7

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

with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
set_z0 = z[ z > 0.25]
set_z1 = z[ z < 0.25]
ra_z0 = ra[ z > 0.25]
ra_z1 = ra[ z < 0.25]
dec_z0 = dec[ z > 0.25]
dec_z1 = dec[ z < 0.25]
load = '/home/xkchen/mywork/ICL/data/mock_frame/'

def SB_lightpro():
	"""
	sigma_m_c: calculate the 2D density profile of cluster, rho_2d
	R : r200, rbin: a radius array
	"""
	Mh = 14.3 # make the radius close to 1 Mpc
	N = 151
	Concen = 5.
	R, rho_2d, rbin = sigma_m_c(Mh, N, Concen)

	Lc = rho_2d / 100 # in unit L_sun/kpc^2, Lc = Lc(r)
	# creat series cluster intrinsic SB profile
	set_z = np.r_[ set_z0[:10], set_z1[:10] ]

	'''
	a = 1 / (1 + set_z)
	Lz = np.tile(a**4, (N,1)).T * np.tile(Lc, (len(set_z), 1)) / (4 * np.pi * rad2asec**2) # L at z in unit: (Lsun/kpc^2)/arcsec^2
	Lob = Lz * Lsun / kpc2cm**2
	Iner_SB = 22.5 - 2.5*np.log10(Lob / (10**(-9) * f0))
	'''
	Da = Test_model.angular_diameter_distance(set_z).value
	Dl = Da * (1 + set_z)**2
	AS = (rad2asec * 10**(-3) / Da)**2
	F = np.zeros((len(set_z), N), dtype = np.float)
	Iner_SB = np.zeros((len(set_z), N), dtype = np.float)
	for kk in range(len(set_z)):
		ttf = (Lc * Lsun / (4 * np.pi * Dl[kk]**2)) * Mpc2cm**(-2)
		F[kk, :] = ttf / (10**(-9) * f0)
		Iner_SB[kk, :] = 22.5 - 2.5 * np.log10(ttf / (10**(-9) * f0) ) + 2.5 * np.log10(AS[kk])

	plt.figure(figsize = (16, 8))
	ax0 = plt.subplot(121)
	ax1 = plt.subplot(122)
	ax0.set_title('Intrinsic  SB profile')
	ax0.plot(rbin, Lc, 'r-')
	ax0.set_xscale('log')
	ax0.set_xlabel('$R[kpc]$')
	ax0.set_ylabel('$SB[L_{\odot} / kpc^2]$')
	ax0.set_yscale('log')
	ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

	lines = []
	ax1.set_title('Observation SB [20]')
	for ll in range(10):
		l1, = ax1.plot(rbin, Iner_SB[ll, :], '--', color = mpl.cm.rainbow(ll/10), label = '$z%.3f$' % set_z0[ll])
		l2, = ax1.plot(rbin, Iner_SB[ll + 10, :], '-', color = mpl.cm.rainbow(ll/10), label = '$z%.3f$' % set_z1[ll])
		handles, labels = plt.gca().get_legend_handles_labels()
		lines.append([l1, l2])
	legend = plt.legend(lines[0], ['z > 0.5', 'z < 0.5'], loc = 3)
	ax1.set_xscale('log')
	ax1.set_xlabel('$R[kpc]$')
	ax1.set_ylabel('$SB[mag / arcsec^2]$')
	ax1.invert_yaxis()
	ax1.add_artist(legend)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.savefig('mock_SB_profile.png', dpi = 300)
	plt.show()
	raise

	with h5py.File(load + 'mock_flux_data.h5', 'w') as f:
		f['a'] = Lob

	with h5py.File(load + 'mock_mag_data.h5', 'w') as f:
		f['a'] = Iner_SB

	c_s = np.array([Lc, rbin])
	with h5py.File(load + 'mock_intric_SB.h5', 'w') as f:
		f['a'] = c_s
	with h5py.File(load + 'mock_intric_SB.h5') as f:
		for t in range(len(c_s)):
			f['a'][t,:] = c_s[t,:]

	return

def mock_ccd():
	with h5py.File(load +'mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File(load + 'mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File(load + 'mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])
	R = np.max(rbin)
	r_sc = rbin / np.max(rbin)
	set_z = np.r_[ set_z0[:10], set_z1[:10] ]
	set_ra = np.r_[ ra_z0[:10], ra_z1[:10] ]
	set_dec = np.r_[ dec_z0[:10], dec_z1[:10] ]

	def centered_loc(xc, yc, zc, kd):
		cc_Lob = Lob[kd]

		y0 = np.linspace(0, 1488, 1489)
		x0 = np.linspace(0, 2047, 2048)
		frame = np.zeros((len(y0), len(x0)), dtype = np.float)
		pxl = np.meshgrid(x0, y0)

		flux_func = interp.interp1d(r_sc, cc_Lob, kind = 'cubic')
		Da0 = Test_model.angular_diameter_distance(zc).value
		Angu_r = (10**(-3) * R / Da0) * rad2asec
		R_pixel = Angu_r / pixel

		dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )
		dr_sc = dr / R_pixel

		ix = np.abs(x0 - xc)
		iy = np.abs(y0 - yc)
		ix0 = np.where(ix == np.min(ix))[0][0]
		iy0 = np.where(iy == np.min(iy))[0][0]

		## set ccd flux pixel by pixel
		for kk in range(dr_sc.shape[0]):
			for jj in range(dr_sc.shape[1]):
				if (dr_sc[kk, jj] >= np.max(r_sc) ) | (dr_sc[kk, jj] <= np.min(r_sc) ):
					continue
				else:
					frame[kk, jj] = flux_func( dr_sc[kk, jj] ) * pixel**2 /(f0 * 10**(-9) )
		x = frame.shape[1]
		y = frame.shape[0]

		# add noise
		xfree = np.random.normal(0, 0.15, (1489, 2048))
		#Noise = (xfree / np.max( np.abs(xfree) )) * np.min(cc_Lob * pixel**2 /(f0 * 10**(-9) )) / 100.
		#frame1 = frame + Noise
		Dev = 10.
		frame1 = frame * (1 + xfree / Dev)

		#from light_measure_tmp import light_measure
		SBt, Rt, Art, errt = light_measure(frame1, 65, 1, R_pixel, xc, yc, pixel, zc)[:4]
		cc_SB = Iner_SB[kd, :]
		f_SB = interp.interp1d(rbin, cc_SB, kind = 'cubic')
		id_nan = np.isnan(SBt)
		ivx = id_nan == False
		ss_R = Rt[ivx]
		ss_SB = SBt[ivx]
		ddsb = ss_SB[ (ss_R > np.min(rbin)) & (ss_R < np.max(rbin)) ] - f_SB( ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))] )
		ddsr = ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))]

		plt.figure()
		gs0 = gridspec.GridSpec(5, 1)
		ax0 = plt.subplot(gs0[:4])
		ax1 = plt.subplot(gs0[-1])
		##
		ax0.set_title('$ \Delta noise %.2f $' % (1 / Dev) )
		ax0.plot(rbin, cc_SB, 'r-', label = '$ Intrinsic $', alpha = 0.5)
		ax0.plot(ss_R, ss_SB, 'g--', label = '$ Measurement $', alpha = 0.5)
		ax0.set_xscale('log')
		ax0.set_ylabel('$SB[mag/arcsec^2]$')
		ax0.invert_yaxis()
		ax0.legend(loc = 1)
		ax0.set_xlim(1e1, 1e3)
		ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

		bx1 = ax0.twiny()
		xtik = ax0.get_xticks()
		xtik = np.array(xtik)
		xR = xtik * 10**(-3) * rad2asec / Da0
		bx1.set_xscale('log')
		bx1.set_xticks(xtik)
		bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx1.set_xlim(ax0.get_xlim())
		ax0.set_xticks([])
		##
		ax1.plot(ddsr, ddsb, 'g-', alpha = 0.5)
		ax1.axhline(y = 0, linestyle = '--', color = 'r', alpha = 0.5, label = '$ \Delta{SB} = 0 $')
		ax1.set_xscale('log')
		ax1.set_xlim( ax0.get_xlim() )
		ax1.set_xlabel('$R[kpc]$')
		ax1.set_ylabel('$ SB_{M} - SB_{I} $')
		ax1.set_ylim(-1e-2, 1e-2)
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.tight_layout()
		plt.savefig('test_Dev%.2f.png' % (1 / Dev), dpi = 300)
		plt.close()
		raise

		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
				'CENTER_X','CENTER_Y','CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		values = ['T', 32, 2, x, y, np.int(ix0), np.int(iy0), ix0, iy0, ix0, iy0, set_ra[kd], set_dec[kd], zc, pixel]
		ff = dict(zip(keys, values))
		fil = fits.Header(ff)

		fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/mock/mock_z%.3f_ra%.3f_dec%.3f.fits' % 
			(zc, set_ra[kd], set_dec[kd]), frame, header = fil, overwrite=True)
		fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % 
			(zc, set_ra[kd], set_dec[kd]), frame1, header = fil, overwrite=True)

		# add noise + mask
		N_s = 450
		xa0 = np.max( [xc - np.int(R_pixel), 0] )
		xa1 = np.min( [xc + np.int(R_pixel), 2047] )
		ya0 = np.max( [yc - np.int(R_pixel), 0] )
		ya1 = np.min( [yc + np.int(R_pixel), 1488] )
		lox = np.array([random.randint(xa0, xa1) for ll in range(N_s)])
		loy = np.array([random.randint(ya0, ya1) for ll in range(N_s)])
		Lr = np.abs(np.random.normal(0, 1.2, N_s) * 18)
		Sr = Lr * np.random.random(N_s)
		Phi = np.random.random(N_s) * 180

		mask = np.ones((1489, 2048), dtype = np.float)
		ox = np.linspace(0, frame.shape[1] - 1, frame.shape[1])
		oy = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
		basic_coord = np.array(np.meshgrid(ox,oy))
		major = Lr / 2
		minor = Sr / 2
		senior = np.sqrt(major**2 - minor**2)
		tdr = np.sqrt((lox - xc)**2 + (loy - yc)**2)
		dr00 = np.where(tdr == np.min(tdr))[0]
		for k in range(N_s):
			posx = lox[k]
			posy = loy[k]
			lr = major[k]
			sr = minor[k]
			cr = senior[k]
			chi = Phi[k] * np.pi / 180

			set_r = np.int(np.ceil(1.2 * lr))
			la0 = np.max( [np.int(posx - set_r), 0])
			la1 = np.min( [np.int(posx + set_r +1), frame1.shape[1] - 1] )
			lb0 = np.max( [np.int(posy - set_r), 0] ) 
			lb1 = np.min( [np.int(posy + set_r +1), frame1.shape[0] - 1] )

			if k == dr00[0] :
				continue
			else:
				df1 = lr**2 - cr**2*np.cos(chi)**2
				df2 = lr**2 - cr**2*np.sin(chi)**2
				fr = ((basic_coord[0,:][lb0: lb1, la0: la1] - posx)**2*df1 + (basic_coord[1,:][lb0: lb1, la0: la1] - posy)**2*df2
					- cr**2*np.sin(2*chi)*(basic_coord[0,:][lb0: lb1, la0: la1] - posx)*(basic_coord[1,:][lb0: lb1, la0: la1] - posy))
				idr = fr / (lr**2*sr**2)
				jx = idr <= 1

				iu = np.where(jx == True)
				iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
				iv[iu] = np.nan
				mask[lb0: lb1, la0: la1] = mask[lb0: lb1, la0: la1] * iv
		frame2 = mask * frame1
		fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % 
			(zc, set_ra[kd], set_dec[kd]), frame2, header = fil, overwrite=True)

		return

	for kk in range(len(set_z)):
		xc = random.randint(900, 1100)
		yc = random.randint(800, 1000)
		centered_loc(xc, yc, set_z[kk], kk)

	return

def mock_image():
	# ccd image
	with h5py.File(load +'mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File(load + 'mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File(load + 'mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])
	R = np.max(rbin)
	r_sc = rbin / np.max(rbin)
	set_z = np.r_[ set_z0[:10], set_z1[:10] ]
	set_ra = np.r_[ ra_z0[:10], ra_z1[:10] ]
	set_dec = np.r_[ dec_z0[:10], dec_z1[:10] ]

	fig = plt.figure(figsize = (18,9))
	gs = gridspec.GridSpec(5,4)
	for k in range(20):
		data = fits.getdata(load + 'noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		img = data[0]
		ax = plt.subplot(gs[ k // 4, k % 4])
		ax.set_title('sample %d [z%.3f]' %(k, set_z[k]) )
		tf = ax.imshow(img, origin = 'lower', cmap = 'rainbow', vmin = 1e9, vmax = 1e12, norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])
	plt.tight_layout()
	plt.savefig('noise_sample_view.png', dpi = 300)
	plt.close()

	fig = plt.figure(figsize = (18,9))
	gs = gridspec.GridSpec(5,4)
	for k in range(20):
		data = fits.getdata(load + 'noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		img = data[0]
		ax = plt.subplot(gs[ k // 4, k % 4])
		ax.set_title('sample %d [z%.3f]' %(k, set_z[k]) )
		tf = ax.imshow(img, origin = 'lower', cmap = 'rainbow', vmin = 1e9, vmax = 1e12, norm = mpl.colors.LogNorm())
		plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = '$flux[nmaggy]$')
		ax.set_xlim(0, img.shape[1])
		ax.set_ylim(0, img.shape[0])
	plt.tight_layout()
	plt.savefig('add_mask_sample_view.png', dpi = 300)
	plt.close()

	raise
	return

def light_test():
	bins = 65
	with h5py.File(load + 'mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File(load + 'mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File(load + 'mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])
	R = np.max(rbin)
	r_sc = rbin / np.max(rbin)
	set_z = np.r_[ set_z0[:10], set_z1[:10] ]
	set_ra = np.r_[ ra_z0[:10], ra_z1[:10] ]
	set_dec = np.r_[ dec_z0[:10], dec_z1[:10] ]

	# SB at z = 0.25
	a_ref = 1 / (1 + z_ref)
	Lref = Lc * a_ref**4 / (4 * np.pi * rad2asec**2) # L at z in unit: (Lsun/kpc^2)/arcsec^2
	Lob_ref = Lref * Lsun / kpc2cm**2
	SB_ref = 22.5 - 2.5*np.log10(Lob_ref / (10**(-9) * f0))
	# measure SB
	Nz = len(set_z)
	R_t = np.zeros((Nz, bins), dtype = np.float)
	SB_t = np.zeros((Nz, bins), dtype = np.float)
	err_t = np.zeros((Nz, bins), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'mock/mock_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		#data = fits.getdata(load + 'noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		#data = fits.getdata(load + 'noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)

		img = data[0]
		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		Rp = (rad2asec / Dag) / pixel
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']
		SBt, Rt, Art, errt = light_measure(img, bins, 1, Rp, cenx, ceny, pixel, set_z[k])[:4]
		SB_t[k, :] = SBt
		R_t[k,:] = Rt
		err_t[k,:] = errt

	plt.figure(figsize = (20, 24))
	gs = gridspec.GridSpec(5, 4)
	for k in range(Nz):
		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		cc_SB = Iner_SB[k,:]
		f_SB = interp.interp1d(rbin, cc_SB, kind = 'cubic')
		id_nan = np.isnan(SB_t[k,:])
		ivx = id_nan == False
		ss_R = R_t[k, ivx]
		ss_SB = SB_t[k, ivx]
		ddsb = ss_SB[ (ss_R > np.min(rbin)) & (ss_R < np.max(rbin)) ] - f_SB( ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))] )
		ddsr = ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))]
		gs0 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec = gs[ k // 4, k % 4])
		ax0 = plt.subplot(gs0[:4])
		ax1 = plt.subplot(gs0[-1])
		##
		ax0.plot(rbin, cc_SB, 'r-', label = '$ Intrinsic $', alpha = 0.5)
		ax0.plot(ss_R, ss_SB, 'g--', label = '$ Smooth $', alpha = 0.5)
		#ax0.plot(ss_R, ss_SB, 'g--', label = '$ Noise $', alpha = 0.5)
		#ax0.plot(ss_R, ss_SB, 'g--', label = '$ Add \; mask $', alpha = 0.5)

		ax0.set_xscale('log')
		ax0.set_ylabel('$SB[mag/arcsec^2]$')
		ax0.invert_yaxis()
		ax0.legend(loc = 1)
		ax0.set_xlim(1e1, 1e3)
		ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

		bx1 = ax0.twiny()
		xtik = ax0.get_xticks()
		xtik = np.array(xtik)
		xR = xtik * 10**(-3) * rad2asec / Dag
		bx1.set_xscale('log')
		bx1.set_xticks(xtik)
		bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx1.set_xlim(ax0.get_xlim())
		ax0.set_xticks([])
		##
		ax1.plot(ddsr, ddsb, 'g-', alpha = 0.5)
		ax1.axhline(y = 0, linestyle = '--', color = 'r', alpha = 0.5, label = '$ \Delta{SB} = 0 $')
		ax1.set_xscale('log')
		ax1.set_xlim(1e1, 1e3)
		ax1.set_xlabel('$R[kpc]$')
		ax1.set_ylabel('$ SB_{M} - SB_{I} $')
		ax1.set_ylim(-1e-2, 1e-2)
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('mock_light_measure_test.pdf', dpi = 300)
	#plt.savefig('noise_light_measure_test.pdf', dpi = 300)
	#plt.savefig('add_mask_light_measure_test.pdf', dpi = 300)

	plt.close()
	raise
	return

def resample_test():
	bins = 65
	with h5py.File(load + 'mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File(load + 'mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File(load + 'mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])

	R0 = np.max(rbin)
	Rpp = (rad2asec * 10**(-3) * R0 / Da_ref) / pixel
	r_sc = rbin / np.max(rbin)
	set_z = np.r_[ set_z0[:10], set_z1[:10] ]
	set_ra = np.r_[ ra_z0[:10], ra_z1[:10] ]
	set_dec = np.r_[ dec_z0[:10], dec_z1[:10] ]
	a_ref = 1 / (1 + z_ref)
	Lref = Lc * a_ref**4 / (4 * np.pi * rad2asec**2) # L at z in unit: (Lsun/kpc^2)/arcsec^2
	Lob_ref = Lref * Lsun / kpc2cm**2
	SB_ref = 22.5 - 2.5*np.log10(Lob_ref / (10**(-9) * f0))
	f_SB = interp.interp1d(rbin, SB_ref)

	Nz = len(set_z)
	R_t = np.zeros((Nz, bins), dtype = np.float)
	SB_t = np.zeros((Nz, bins), dtype = np.float)
	err_t = np.zeros((Nz, bins), dtype = np.float)
	R_01 = np.zeros((Nz, bins), dtype = np.float)
	SB_01 = np.zeros((Nz, bins), dtype = np.float)
	err_01 = np.zeros((Nz, bins), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'mock/mock_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		#data = fits.getdata(load + 'noise/noise_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)
		#data = fits.getdata(load + 'noise_mask/add_mask_frame_z%.3f_ra%.3f_dec%.3f.fits' % (set_z[k], set_ra[k], set_dec[k]), header = True)

		img = data[0]
		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		Rp = (rad2asec * 10**(-3) * R0 / Dag) / pixel
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']
		Len_ref = Da_ref * pixel / rad2asec
		Len_z0 = Dag * pixel / rad2asec
		eta = Len_ref / Len_z0
		mu = 1 / eta
		scale_img = flux_recal(img, set_z[k], z_ref) # scale the flux to the reference redshift
		if eta > 1:
			resamt, xn, yn = sum_samp(eta, eta, scale_img, cenx, ceny)
		else:
			resamt, xn, yn = down_samp(eta, eta, scale_img, cenx, ceny)

		xn = np.int(xn)
		yn = np.int(yn)
		Nx = resamt.shape[1]
		Ny = resamt.shape[0]

		keys = ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'CENTER_X', 'CENTER_Y', 'ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, Nx, Ny, xn, yn, set_z[k], pixel]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)
		fits.writeto(load + 'resamp-mock-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), resamt, header = fil, overwrite=True)
		#fits.writeto(load + 'resamp-noise-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), resamt, header = fil, overwrite=True)
		#fits.writeto(load + 'resamp-mask-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), resamt, header = fil, overwrite=True)

		SBt, Rt, Art, errt = light_measure(resamt, bins, 1, Rpp, xn, yn, pixel, z_ref)[:4]
		SB_t[k, :] = SBt
		R_t[k, :] = Rt
		err_t[k, :] = errt

		SB, R, Anr, err = light_measure(scale_img, bins, 1, Rp, cenx, ceny, pixel * mu, z_ref)[:4]
		SB_01[k, :] = SB
		R_01[k, :] = R
		err_01[k, :] = err

	plt.figure(figsize = (20, 24))
	gs = gridspec.GridSpec(5, 4)
	for k in range(Nz):
		Dag = Test_model.angular_diameter_distance(set_z[k]).value
		id_nan = np.isnan(SB_t[k,:])
		ivx = id_nan == False
		ss_R = R_t[k, ivx]
		ss_SB = SB_t[k, ivx]
		ddsb = ss_SB[ (ss_R > np.min(rbin)) & (ss_R < np.max(rbin)) ] - f_SB( ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))] )
		ddsr = ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))]

		id_nan = np.isnan(SB_01[k,:])
		ivx = id_nan == False
		st_R = R_01[k, ivx]
		st_SB = SB_01[k, ivx]
		ddtb = st_SB[ (st_R > np.min(rbin)) & (st_R < np.max(rbin)) ] - f_SB( st_R[(st_R > np.min(rbin)) & (st_R < np.max(rbin))] )
		ddtr = st_R[ (st_R > np.min(rbin)) & (st_R < np.max(rbin))]

		gs0 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec = gs[ k // 4, k % 4])
		ax0 = plt.subplot(gs0[:4])
		ax1 = plt.subplot(gs0[-1])

		ax0.plot(rbin, SB_ref, 'r-', label = '$ Intrinsic $', alpha = 0.5)
		ax0.plot(ss_R, ss_SB, 'g--', label = '$ Smooth $', alpha = 0.5)
		ax0.plot(st_R, st_SB, 'b-.', label = '$ Smooth + resampling $', alpha = 0.5)

		#ax0.plot(ss_R, ss_SB, 'g--', label = '$ Noise $', alpha = 0.5)
		#ax0.plot(st_R, st_SB, 'b-.', label = '$ Noise + resampling $', alpha = 0.5)

		#ax0.plot(ss_R, ss_SB, 'g--', label = '$ Add \; mask $', alpha = 0.5)
		#ax0.plot(st_R, st_SB, 'b-.', label = '$ Mask + resampling $', alpha = 0.5)		

		ax0.set_xscale('log')
		ax0.set_xlim(1e1, 1e3)
		ax0.set_ylabel('$SB[mag/arcsec^2]$')
		ax0.invert_yaxis()
		ax0.legend(loc = 1)
		ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

		bx1 = ax0.twiny()
		xtik = ax0.get_xticks()
		xtik = np.array(xtik)
		xR = xtik * 10**(-3) * rad2asec / Dag
		bx1.set_xscale('log')
		bx1.set_xticks(xtik)
		bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
		bx1.set_xlim(ax0.get_xlim())
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax0.set_xticks([])

		ax1.plot(ddsr, ddsb, 'g-', alpha = 0.5)
		ax1.plot(ddtr, ddtb, 'b-.', alpha = 0.5)
		ax1.axhline(y = 0, linestyle = '--', color = 'k', alpha = 0.5, label = '$ \Delta{SB} = 0 $')
		ax1.set_xscale('log')
		ax1.set_xlim(1e1, 1e3)
		ax1.set_xlabel('$R[kpc]$')
		ax1.set_ylabel('$ SB_{M} - SB_{I} $')
		ax1.set_ylim(-1e-2, 1e-2)
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.tight_layout()
	plt.savefig('mock_resample_SB.pdf', dpi = 300)
	#plt.savefig('noise_resample_SB.pdf', dpi = 300)
	#plt.savefig('mask_resample_SB.pdf', dpi = 300)
	plt.close()
	raise
	return

def random_test():
	bins = 65
	with h5py.File(load + 'mock_flux_data.h5')as f:
		Lob = np.array(f['a'])
	with h5py.File(load + 'mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File(load + 'mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])

	R0 = np.max(rbin)
	r_sc = rbin / np.max(rbin)
	Rpp = (rad2asec * 10**(-3) * R0 / Da_ref) / pixel
	set_z = np.r_[ set_z0[:10], set_z1[:10] ]
	set_ra = np.r_[ ra_z0[:10], ra_z1[:10] ]
	set_dec = np.r_[ dec_z0[:10], dec_z1[:10] ]
	a_ref = 1 / (1 + z_ref)
	Lref = Lc * a_ref**4 / (4 * np.pi * rad2asec**2) # L at z in unit: (Lsun/kpc^2)/arcsec^2
	Lob_ref = Lref * Lsun / kpc2cm**2
	SB_ref = 22.5 - 2.5*np.log10(Lob_ref / (10**(-9) * f0))
	f_SB = interp.interp1d(rbin, SB_ref)

	Nz = len(set_z)
	x0, y0 = 2427, 1765
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	sum_array_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_D = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'resamp-noise-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), header = True)
		img = data[0]
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']

		la0 = np.int(y0 - ceny)
		la1 = np.int(y0 - ceny + img.shape[0])
		lb0 = np.int(x0 - cenx)
		lb1 = np.int(x0 - cenx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)
		sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + img[idv]
		count_array_D[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array_D)
		id_fals = np.where(id_nan == False)
		p_count_D[id_fals] = p_count_D[id_fals] + 1
		count_array_D[la0: la1, lb0: lb1][idv] = np.nan

	mean_array_D = sum_array_D / p_count_D
	where_are_inf = np.isinf(mean_array_D)
	mean_array_D[where_are_inf] = np.nan
	id_zeros = np.where(p_count_D == 0)
	mean_array_D[id_zeros] = np.nan
	SBt0, Rt0, Art0, errt0 = light_measure(mean_array_D, bins, 1, Rpp, x0, y0, pixel, z_ref)[:4]

	id_nan = np.isnan(SBt0)
	ivx = id_nan == False
	ss_R = Rt0[ivx]
	ss_SB = SBt0[ivx]
	ddsb0 = ss_SB[ (ss_R > np.min(rbin)) & (ss_R < np.max(rbin)) ] - f_SB( ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))] )
	ddsr0 = ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))]

	sum_array_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_D = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'resamp-mask-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), header = True)
		img = data[0]
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']

		la0 = np.int(y0 - ceny)
		la1 = np.int(y0 - ceny + img.shape[0])
		lb0 = np.int(x0 - cenx)
		lb1 = np.int(x0 - cenx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)
		sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + img[idv]
		count_array_D[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array_D)
		id_fals = np.where(id_nan == False)
		p_count_D[id_fals] = p_count_D[id_fals] + 1
		count_array_D[la0: la1, lb0: lb1][idv] = np.nan

	mean_array_D = sum_array_D / p_count_D
	where_are_inf = np.isinf(mean_array_D)
	mean_array_D[where_are_inf] = np.nan
	id_zeros = np.where(p_count_D == 0)
	mean_array_D[id_zeros] = np.nan
	SBt1, Rt1, Art1, errt1 = light_measure(mean_array_D, bins, 1, Rpp, x0, y0, pixel, z_ref)[:4]

	id_nan = np.isnan(SBt1)
	ivx = id_nan == False
	ss_R = Rt1[ivx]
	ss_SB = SBt1[ivx]
	ddsb1 = ss_SB[ (ss_R > np.min(rbin)) & (ss_R < np.max(rbin)) ] - f_SB( ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))] )
	ddsr1 = ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))]

	sum_array_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_D = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_D = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	for k in range(Nz):
		data = fits.getdata(load + 'resamp-mock-ra%.3f-dec%.3f-redshift%.3f.fits' % (set_ra[k], set_dec[k], set_z[k]), header = True)
		img = data[0]
		cenx = data[1]['CENTER_X']
		ceny = data[1]['CENTER_Y']

		la0 = np.int(y0 - ceny)
		la1 = np.int(y0 - ceny + img.shape[0])
		lb0 = np.int(x0 - cenx)
		lb1 = np.int(x0 - cenx + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)
		sum_array_D[la0:la1, lb0:lb1][idv] = sum_array_D[la0:la1, lb0:lb1][idv] + img[idv]
		count_array_D[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array_D)
		id_fals = np.where(id_nan == False)
		p_count_D[id_fals] = p_count_D[id_fals] + 1
		count_array_D[la0: la1, lb0: lb1][idv] = np.nan

	mean_array_D = sum_array_D / p_count_D
	where_are_inf = np.isinf(mean_array_D)
	mean_array_D[where_are_inf] = np.nan
	id_zeros = np.where(p_count_D == 0)
	mean_array_D[id_zeros] = np.nan
	SBt2, Rt2, Art2, errt2 = light_measure(mean_array_D, bins, 1, Rpp, x0, y0, pixel, z_ref)[:4]

	id_nan = np.isnan(SBt2)
	ivx = id_nan == False
	ss_R = Rt2[ivx]
	ss_SB = SBt2[ivx]
	ddsb2 = ss_SB[ (ss_R > np.min(rbin)) & (ss_R < np.max(rbin)) ] - f_SB( ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))] )
	ddsr2 = ss_R[(ss_R > np.min(rbin)) & (ss_R < np.max(rbin))]

	plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])

	ax0.set_title('stack image')
	ax0.plot(rbin, SB_ref, 'r-', label = '$ reference $', alpha = 0.5)
	ax0.plot(Rt2, SBt2, 'g--', label = '$ No \; noise + No \; mask $', alpha = 0.5)
	ax0.plot(Rt1, SBt1, 'b-.', label = '$ noise + mask $', alpha = 0.5)
	ax0.plot(Rt0, SBt0, 'm:', label = '$ noise $', alpha = 0.5)

	ax0.set_xscale('log')
	ax0.set_xlabel('$R[kpc]$')
	ax0.set_xlim(1e1, 1e3)
	ax0.set_ylabel('$SB[mag/arcsec^2]$')
	ax0.invert_yaxis()
	ax0.legend(loc = 1)
	ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

	bx1 = ax0.twiny()
	xtik = ax0.get_xticks()
	xtik = np.array(xtik)
	Dag = Test_model.angular_diameter_distance(z_ref).value
	xR = xtik * 10**(-3) * rad2asec / Dag
	bx1.set_xscale('log')
	bx1.set_xticks(xtik)
	bx1.set_xticklabels(['$%.2f^{ \prime \prime }$' % uu for uu in xR])
	bx1.set_xlim(ax0.get_xlim())
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')
	ax0.set_xticks([])

	ax1.plot(ddsr0, ddsb0, 'm--', alpha = 0.5)
	ax1.plot(ddsr1, ddsb1, 'b--', alpha = 0.5)
	ax1.plot(ddsr2, ddsb2, 'g--', alpha = 0.5)
	ax1.axhline(y = 0, linestyle = '--', color = 'k', alpha = 0.5, label = '$ \Delta{SB} = 0 $')
	ax1.set_xscale('log')
	ax1.set_xlim(1e1, 1e3)
	ax1.set_ylim(-1e-2, 1e-2)
	ax1.set_xlabel('$R[kpc]$')
	ax1.set_ylabel('$ SB_{stacking} - SB_{reference} $')
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(hspace = 0)
	plt.savefig('stack_test.png', dpi = 300)
	plt.close()

	raise
	return

def test():
	#SB_lightpro()
	mock_ccd()
	#mock_image()
	#light_test()
	resample_test()
	random_test()

def main():
	test()

if __name__ == '__main__' :
	main()
