"""
# this file use to creat a mock cluster with NFW model
and M/L = 50 M_sun/L_sun. 
accrding the light profile, we'll ctreat an image with 
real data(include redshift, pixel_size)
"""
import h5py
import numpy as np
from scipy import interpolate as interp
from numba import vectorize

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy

from resamp import gen
from ICL_surface_mass_density import sigma_m_c
from light_measure import light_measure, flux_recal, flux_scale

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
# constant
c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
c3 = U.L_sun.to(U.erg/U.s)
c4 = U.rad.to(U.arcsec)
c5 = U.pc.to(U.cm)
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

def SB_lightpro():
	"""
	sigma_m_c: calculate the 2D density profile of cluster, rho_2d
	R : r200, rbin: a radius array
	"""
	Mh = 15-np.log10(8) # make the radius close to 1 Mpc
	N = 131
	R, rho_2d, rbin = sigma_m_c(Mh, N)
	Lc = rho_2d/100 # in unit L_sun/kpc^2, Lc = Lc(r)
	# creat series cluster intrinsic SB profile
	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	a = 1/(1+z)
	SB = 21.572 + 4.75 - 2.5*np.log10(10**(-6)*np.tile(Lc, (len(z),1))) + 10*np.log10(np.tile(z+1,(N,1)).T)

	Lz = np.tile(a**4,(N,1)).T*np.tile(Lc, (len(z),1))/(4*np.pi*c4**2) # L at z in unit: (Lsun/kpc^2)/arcsec^2
	Lob = Lz*Lsun/c0**2
	Iner_SB = 22.5-2.5*np.log10(Lob/(10**(-9)*f0))
	raise
	zt = z[0]
	Dat = Test_model.angular_diameter_distance(zt).value
	st = (10**(-6)/Dat**2)*c4**2
	ft = Lc*Lsun/(4*np.pi*(1+zt)**4*Dat**2)
	SBt = 22.5 - 2.5*np.log10(ft/(c2**2*f0*10**(-9)))+2.5*np.log10(st)

	with h5py.File('mock_flux_data.h5', 'w') as f:
		f['a'] = Lob

	with h5py.File('mock_mag_data.h5', 'w') as f:
		f['a'] = Iner_SB

	c_s = np.array([Lc, rbin])
	with h5py.File('mock_intric_SB.h5', 'w') as f:
		f['a'] = c_s
	with h5py.File('mock_intric_SB.h5') as f:
		for t in range(len(c_s)):
			f['a'][t,:] = c_s[t,:]

	return Lob, Iner_SB, z, rbin, R

def mock_image():

	Npi = 1000

	with h5py.File('mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File('mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File('mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])
	R = np.max(rbin)
	r_sc = rbin/np.max(rbin)
	flux_func = interp.interp1d(r_sc, Lob[0], kind = 'cubic')

	x1 = np.linspace(-1.2*R, 1.2*R, 2*Npi+1)
	y1 = np.linspace(-1.2*R, 1.2*R, 2*Npi+1)
	plane = np.meshgrid(x1, y1)
	dr = np.sqrt((plane[0]-0)**2+(plane[1]-0)**2)
	dr_sc = dr/R
	test_dr = dr_sc[Npi+1, Npi+1:Npi+833]
	test = flux_func(test_dr)
	iat = r_sc <= test_dr[0]
	ibt = rbin[iat]
	ict = Lob[0][iat]

	mock_ana = np.zeros((2*Npi+1, 2*Npi+1), dtype = np.float)
	for k in range(len(test_dr)):
		if k == 0:
			mock_ana[Npi+1, Npi+1] = ict[-2]
		elif k == 1:
			mock_ana[Npi+1-1, Npi+1-1:Npi+1+2] = ict[-1]
			mock_ana[Npi+1+1, Npi+1-1:Npi+1+2] = ict[-1]
			mock_ana[Npi+1-1:Npi+1+2, Npi+1-1] = ict[-1]
			mock_ana[Npi+1-1:Npi+1+2, Npi+1+2] = ict[-1]
		else:
			ia = (dr_sc >= test_dr[k-1]) & (dr_sc < test_dr[k])
			ib = np.where(ia == True)
			mock_ana[ib[0], ib[1]] = flux_func(test_dr[k-1])

	plt.pcolormesh(plane[0], plane[1], mock_ana, cmap = 'plasma', norm = mpl.colors.LogNorm())
	plt.colorbar(label = '$flux[(erg/s)/cm^2]$')
	plt.xlabel('R[kpc]')
	plt.ylabel('R[kpc]')
	plt.savefig('mock_cluster_phy.png', dpi = 600)
	plt.show()
	plt.close()

	return

def mock_ccd(xc, yc):
	xc = xc
	yc = yc
	#kz = 0 # for z < 0.25
	kz = 3 # for z > 0.25
	with h5py.File('mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File('mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File('mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])
	R = np.max(rbin)
	r_sc = rbin/np.max(rbin)
	flux_func = interp.interp1d(r_sc, Lob[kz], kind = 'cubic')

	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]

	Da0 = Test_model.angular_diameter_distance(z[kz]).value
	Angu_r = (10**(-3)*R/Da0)*c4
	R_pixel = Angu_r/pixel
	r_in = (rbin*10**(-3)/Da0)*c4
	"""
	# scale as the size of observation
	(in case the cluster center is the frame center)
	"""
	p_scale = 1
	y0 = np.linspace(0, 1488, 1489*p_scale)
	x0 = np.linspace(0, 2047, 2048*p_scale)
	frame = np.zeros((len(y0), len(x0)), dtype = np.float)
	pxl = np.meshgrid(x0, y0)

	def centered_loc(xc, yc):
		xc = xc
		yc = yc
		dr = np.sqrt(((2*pxl[0]+1)/2-(2*xc+1)/2)**2+((2*pxl[1]+1)/2-(2*yc+1)/2)**2)
		dr_sc = dr/R_pixel
		
		ix = np.abs(x0 - xc)
		iy = np.abs(y0 - yc)
		ix0 = np.where(ix == np.min(ix))[0][0]
		iy0 = np.where(iy == np.min(iy))[0][0]

		test_dr = dr_sc[iy0, ix0+1: ix0+1+np.int(R_pixel*p_scale)]
		test = flux_func(test_dr)
		iat = r_sc <= test_dr[0]
		ibt = r_in[iat]
		ict = Lob[kz][iat]

		for k in range(len(test_dr)):
			if k == 0:
				continue
			else:
				ia = (dr_sc >= test_dr[k-1]) & (dr_sc < test_dr[k])
				ib = np.where(ia == True)
				frame[ib[0], ib[1]] = flux_func(test_dr[k-1])*((1/p_scale)*pixel)**2/(f0*10**(-9))
		'''
		plt.imshow(frame, cmap = 'rainbow', origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(label = 'flux[nMgy]', fraction = 0.035,pad = 0.003)
		plt.savefig('mock_frame.png', dpi =600)
		plt.show()
		'''
		x = frame.shape[1]
		y = frame.shape[0]
		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
		        'CENTER_X','CENTER_Y','CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE']
		value = ['T', 32, 2, x, y, np.int(ix0), np.int(iy0), ix0, iy0, ix0, iy0, ra[kz], dec[kz], z[kz], p_scale]
		ff = dict(zip(keys,value))
		fil = fits.Header(ff)

		#fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/mock_frame_z%.3f_ra%.3f_dec%.3f.fits'
		#				%(z[kz], ra[kz], dec[kz]), frame, header = fil, overwrite=True)

		#fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/corner_frame_z%.3f_ra%.3f_dec%.3f.fits'
		#				%(z[kz], ra[kz], dec[kz]), frame, header = fil, overwrite=True)

		#fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/short_edge_frame_z%.3f_ra%.3f_dec%.3f.fits'
		#				%(z[kz], ra[kz], dec[kz]), frame, header = fil, overwrite=True)

		#fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/long_edge_frame_z%.3f_ra%.3f_dec%.3f.fits'
		#		%(z[kz], ra[kz], dec[kz]), frame, header = fil, overwrite=True)

		## random test
		fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/random_frame_z%.3f_randx%.0f_randy%.0f.fits'
				%(z[kz], xc, yc), frame, header = fil, overwrite=True)
		return

	# xc = 1025, yc = 745 # center location
	# xc = 2, yc = 2, # corner location
	# xc = 2, yc = 745, # short edge location
	# xc = 1025, yc = 2, # long edge location
	centered_loc(xc = xc, yc = yc)

	return

def light_test():

	#kz = 0 # for z < 0.25
	kz = 3 # for z > 0.25
	with h5py.File('mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File('mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File('mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])

	r_sc = rbin/np.max(rbin)
	flux_func = interp.interp1d(r_sc, Iner_SB[kz], kind = 'cubic')

	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]
	Da0 = Test_model.angular_diameter_distance(z[kz]).value

	mock_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/mock_frame_z%.3f_ra%.3f_dec%.3f.fits'
				%(z[kz], ra[kz], dec[kz]), header = True)

	#mock_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/corner_frame_z%.3f_ra%.3f_dec%.3f.fits'
	#			%(z[kz], ra[kz], dec[kz]), header = True)

	#mock_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/short_edge_frame_z%.3f_ra%.3f_dec%.3f.fits'
	#			%(z[kz], ra[kz], dec[kz]), header = True)

	#mock_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/long_edge_frame_z%.3f_ra%.3f_dec%.3f.fits'
	#			%(z[kz], ra[kz], dec[kz]), header = True)

	f = mock_data[0]
	r_in = ((rbin/10**3)/Da0)*c4
	Angur = (np.max(rbin/1000)/Da0)*c4
	Rp = Angur/pixel
	cx = mock_data[1]['CENTER_X']
	cy = mock_data[1]['CENTER_Y']
	p_scale = mock_data[1]['P_SCALE']

	test_data = f[cy, cx:]
	test_x = np.linspace(cx, f.shape[1]-1, f.shape[1]-cx)
	test_lit = 22.5-2.5*np.log10(test_data)+2.5*np.log10((1/p_scale)**2*pixel**2)
	test_com = test_lit[test_lit != np.inf]
	test_r = (test_x[test_lit != np.inf]- cx)*(1/p_scale)*pixel
	test_R = test_r*Da0*10**3/c4

	SB_down = np.max([np.min(test_com), np.min(Iner_SB[kz])])
	SB_up = np.min([np.max(test_com), np.max(Iner_SB[kz])])

	compare_l0 = test_com[(test_com >= SB_down) & (test_com <= SB_up)]
	ib = test_R[(test_com >= SB_down) & (test_com <= SB_up)]

	nr = np.linspace(np.min(ib), np.max(ib), len(compare_l0))
	compare_l1 = flux_func(nr/np.max(rbin))
	'''
	plt.figure(figsize = (16,9))
	gs1 = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax1 = plt.subplot(gs1[0])
	ax2 = plt.subplot(gs1[1])

	ax1.plot(ib, compare_l0, 'r--', label = '$SB_{ccd \\ cubic}$', alpha = 0.5)
	ax1.plot(ib, compare_l1, 'b-', label = '$SB_{theory}$', alpha = 0.5)
	ax1.set_title('cubic interpolation')
	ax1.legend(loc = 3, fontsize = 15)
	ax1.set_xlabel('R[kpc]')
	ax1.set_ylabel('$SB[mag/arcsec^2]$')
	ax1.set_xscale('log')
	ax1.invert_yaxis()

	ax2.plot(ib, compare_l0 - compare_l1, label = '$SB_{ccd \\ cubic}-SB_{theory}$')
	ax2.legend(loc = 3, fontsize = 15)
	ax2.set_xlabel('R[kpc]')
	ax2.set_ylabel('$SB[mag/arcsec^2]$')
	ax2.set_xscale('log')
	plt.savefig('mock2ccd_test.png', dpi = 600)
	plt.show()
	'''
	bin_number = 80
	t0 = time.time()
	light, R, Ar1, err = light_measure(f, bin_number, 1, Rp, cx, cy, pixel, z[kz])
	t1 = time.time() - t0
	print(t1)
	raise
	light_com = flux_func(R/np.max(rbin))
	'''
	plt.figure(figsize = (16,9))
	gs1 = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax1 = plt.subplot(gs1[0])
	ax2 = plt.subplot(gs1[1])

	ax1.plot(Ar1, light, 'r*--', label = '$SB_{ccd \\ binned \\ measure}$', alpha = 0.5)
	ax1.plot(Ar1, light_com, 'b-', label = '$SB_{theory}$', alpha = 0.5)
	ax1.legend(loc = 1, fontsize = 15)
	ax1.set_title('binned measure with %.0f'%bin_number)
	ax1.set_xscale('log')
	ax1.set_xlabel('R[arcsec]')
	ax1.set_ylabel('$SB[mag/arcsec^2]$')
	ax1.invert_yaxis()

	ax2.plot(Ar1, light - light_com, label = '$SB_{ccd \\ binned \\ measure} - SB_{theory}$')
	ax2.legend(loc = 1, fontsize = 15)
	ax2.set_xlabel('R[arcsec]')
	ax2.set_xscale('log')
	ax2.set_ylabel('$SB[mag/arcsec^2]$')

	plt.savefig('ccd_light_measure.png', dpi = 600)
	plt.show()
	'''
	bins = 110
	L_ref = Da_ref*pixel/c4 
	L_z0 = Da0*pixel/c4
	b = L_ref/L_z0
	xn, yn, resam = gen(f, 1, b, cx, cy)
	test_resam, R_resam, r0_resam = light_measure(resam, bins, 1, Rp/b, xn, yn, pixel*b, z[kz])

	test_resam = test_resam[1:]
	R_resam = R_resam[1:]
	r0_resam = r0_resam[1:]
	test_light = flux_func(R_resam/np.max(rbin))

	plt.figure(figsize = (16,9))
	gs1 = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax1 = plt.subplot(gs1[0])
	ax2 = plt.subplot(gs1[1])

	ax1.plot(r0_resam, test_resam, 'r*--', label = '$SB_{resample}$', alpha = 0.5)
	ax1.plot(r0_resam, test_light, 'b-', label = '$SB_{theory}$', alpha = 0.5)
	ax1.set_title('resample measure with bin %.0f'%bins)
	ax1.set_xlabel('R[arcsec]')
	ax1.set_xscale('log')
	ax1.set_ylabel('$SB[mag/arcsec^2]$')
	ax1.legend(loc = 3, fontsize = 15)
	ax1.invert_yaxis()

	ax2.plot(r0_resam, test_resam - test_light, label = '$SB_{resample} - SB_{theory}$')
	ax2.set_xlabel('R[arcsec]')
	ax2.set_xscale('log')
	ax2.legend(loc = 1, fontsize = 15)
	ax2.set_ylabel('$SB[mag/arcsec^2]$')

	plt.savefig('mock_resam_test.png', dpi = 600)
	plt.show()
	raise
	return

def resample_test():

	#kz = 0 # for z < 0.25
	kz = 3 # for z > 0.25
	with h5py.File('mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File('mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File('mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])
	SB0 = Iner_SB[kz]
	R0 = np.max(rbin)*10**(-3)

	r_sc = rbin/np.max(rbin)
	flux_f1 = interp.interp1d(r_sc, SB0, kind = 'cubic')

	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]
	SB_0 = SB0 + 10*np.log10((1+z_ref)/(1+z[kz]))
	flux_f2 = interp.interp1d(r_sc, SB_0, kind = 'cubic')

	f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/mock_frame_z%.3f_ra%.3f_dec%.3f.fits'
				%(z[kz], ra[kz], dec[kz]), header = True)

	#f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/corner_frame_z%.3f_ra%.3f_dec%.3f.fits'
	#			%(z[kz], ra[kz], dec[kz]), header = True)

	#f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/short_edge_frame_z%.3f_ra%.3f_dec%.3f.fits'
	#			%(z[kz], ra[kz], dec[kz]), header = True)

	#f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/long_edge_frame_z%.3f_ra%.3f_dec%.3f.fits'
	#			%(z[kz], ra[kz], dec[kz]), header = True)

	Da0 = Test_model.angular_diameter_distance(z[kz]).value
	Angur = (R0/Da0)*c4
	Rp = Angur/pixel
	cx = f_data[1]['CENTER_X']
	cy = f_data[1]['CENTER_Y']
	L_ref = Da_ref*pixel/c4 
	L_z0 = Da0*pixel/c4
	b = L_ref/L_z0
	Rref = (R0*c4/Da_ref)/pixel

	bins = 110
	f = f_data[0]
	f_goal = flux_recal(f, z[kz], z_ref)

	xn, yn, resam = gen(f_goal, 1, b, cx, cy)
	
	# rule out the effect of resample code
	if b > 1:
		resam = resam[1:, 1:]
	elif b == 1:
		resam = resam[1:-1, 1:-1]
	else:
		resam = resam
	
	SB_1, R_1, r0_1, error_1 = light_measure(resam, bins, 1, Rref, xn, yn, pixel, z_ref)
	
	SB_measure = SB_1[1:]
	R_measure = R_1[1:]
	Ar_measure = r0_1[1:]
	SB_compare0 = flux_f1(R_measure/np.max(rbin))
	SB_compare1 = flux_f2(R_measure/np.max(rbin))

	fig = plt.figure(figsize = (16,9))
	fig.suptitle('ccd measure at z_ref with bin%.0f'%bins)
	gs1 = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax1 = plt.subplot(gs1[0])
	ax2 = plt.subplot(gs1[1])

	ax1.plot(R_measure, SB_measure, 'r-', lw = 2, label = '$SB_{ccd \\ at \\ z_{ref}}$', alpha = 0.5)
	ax1.plot(R_measure, SB_compare0, 'b--', lw = 2, label = '$SB_{theory \\ at \\ z_0}$', alpha = 0.5)
	ax1.plot(R_measure, SB_compare1, 'g--', lw = 2, label = '$SB_{theory \\ at \\ z_{ref}}$', alpha = 0.5)

	ax1.set_xlabel('R[kpc]')
	ax1.set_xscale('log')
	ax1.set_ylabel('$SB[mag/arcsec^2]$')
	ax3 = ax1.twiny()
	ax3.plot(Ar_measure, SB_measure, 'r-', lw = 2, alpha = 0.5)
	ax3.set_xlabel('R[arcsec]')
	ax3.set_xscale('log')
	ax3.tick_params(axis = 'x', which = 'both', direction = 'in')
	ax1.legend(loc = 3, fontsize = 15)
	ax1.invert_yaxis()

	ax2.plot(R_measure, SB_compare0 - SB_compare1, 'b-', label = '$SB_{z_0} - SB_{theory \\ z_{ref}}$', alpha = 0.5)
	ax2.plot(R_measure, SB_compare0 - SB_measure, 'r--', label = '$SB_{z_0} - SB_{ccd \\ z_{ref}}$', alpha = 0.5)
	ax2.set_xlabel('R[kpc]')
	ax2.set_xscale('log')
	ax2.set_ylabel('$SB[mag/arcsec^2]$')
	ax2.legend(loc = 4, fontsize = 15)

	plt.savefig('rescale_resample_test.png')
	plt.show()
	
	raise
	return

def random_test():

	#kz = 0 # for z < 0.25
	kz = 3 # for z > 0.25
	with h5py.File('mock_flux_data.h5') as f:
		Lob = np.array(f['a'])
	with h5py.File('mock_mag_data.h5') as f:
		Iner_SB = np.array(f['a'])
	with h5py.File('mock_intric_SB.h5') as f:
		Lc = np.array(f['a'][0])
		rbin = np.array(f['a'][1])
	SB0 = Iner_SB[kz]
	R0 = np.max(rbin)*10**(-3)

	r_sc = rbin/np.max(rbin)
	flux_f1 = interp.interp1d(r_sc, SB0, kind = 'cubic')

	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]
	SB_0 = SB0 + 10*np.log10((1+z_ref)/(1+z[kz]))
	flux_f2 = interp.interp1d(r_sc, SB_0, kind = 'cubic')
	## random set cluster center
	Nsample = 25
	'''
	rand_x0 = np.random.random_sample((1, Nsample))*10
	rand_x1 = np.random.random_sample((1, Nsample))*200
	rand_x = np.ceil(rand_x0*rand_x1)[0]

	rand_y0 = np.random.random_sample((1, Nsample))*10
	rand_y1 = np.random.random_sample((1, Nsample))*140
	rand_y = np.ceil(rand_y0*rand_y1)[0]
	
	pos = np.array([rand_x, rand_y])
	with h5py.File('/home/xkchen/mywork/ICL/data/mock_frame/random_catalog_%.0f.h5'%kz, 'w') as f:
		f['a'] = pos
	with h5py.File('/home/xkchen/mywork/ICL/data/mock_frame/random_catalog_%.0f.h5'%kz) as f:
		for q in range(len(pos)):
			f['a'][q,:] = pos[q,:]
	for k in range(Nsample):
		mock_ccd(xc = rand_x[k], yc = rand_y[k])
	'''
	with h5py.File('/home/xkchen/mywork/ICL/data/mock_frame/random_catalog_%.0f.h5'%kz ) as f:
		pos = np.array(f['a'])

	posx = pos[0]
	posy = pos[1]
	'''
	fig = plt.figure(figsize = (10,10))
	fig.suptitle('position random with z%.3f'%z[kz])
	gs = gridspec.GridSpec(5,5)
	for k in range(Nsample):
		f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/random_frame_z%.3f_randx%.0f_randy%.0f.fits'
			%(z[kz], posx[k], posy[k]), header = True)
		data = f_data[0]
		ax = plt.subplot(gs[np.int(k/5), k%5])
		im = ax.imshow(data, cmap = 'rainbow', origin = 'lower', norm = mpl.colors.LogNorm())
		ax.set_title('cx = %.0f cy = %.0f'%(posx[k], posy[k]))
		ax.set_xticks([])
		ax.set_yticks([])
		#plt.colorbar(im, label = 'flux[nMgy]', fraction = 0.035, pad = 0.001)
	plt.tight_layout()
	plt.savefig('position_random.pdf', dpi = 300)
	plt.close()
	'''
	f1 = plt.figure(figsize = (20, 28))
	#f1.suptitle('light measure for random position with z%.3f'%z[kz])
	gs1 = gridspec.GridSpec(5, 5)
	for k in range(Nsample):
		f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/random_frame_z%.3f_randx%.0f_randy%.0f.fits'
			%(z[kz], posx[k], posy[k]), header = True)
		data = f_data[0]
		Da0 = Test_model.angular_diameter_distance(z[kz]).value
		Angur = (R0/Da0)*c4
		Rp = Angur/pixel
		cx = f_data[1]['CENTER_X']
		cy = f_data[1]['CENTER_Y']
		L_ref = Da_ref*pixel/c4 
		L_z0 = Da0*pixel/c4
		b = L_ref/L_z0
		Rref = (R0*c4/Da_ref)/pixel

		bins = 70
		f = f_data[0]
		f_goal = flux_recal(f, z[kz], z_ref)

		xn, yn, resam = gen(f_goal, 1, b, cx, cy)
		
		# rule out the effect of resample code
		if b > 1:
			resam = resam[1:, 1:]
		elif b == 1:
			resam = resam[1:-1, 1:-1]
		else:
			resam = resam
		SB_1, R_1, r0_1, error_1 = light_measure(resam, bins, 1, Rref, xn, yn, pixel, z_ref)
		
		SB_measure = SB_1[1:]
		R_measure = R_1[1:]
		Ar_measure = r0_1[1:]
		SB_error = error_1[1:]
		SB_compare0 = flux_f1(R_measure/np.max(rbin))
		SB_compare1 = flux_f2(R_measure/np.max(rbin))

		ia = SB_error == 0
		SB_error[ia] = SB_compare1[ia] - SB_measure[ia]
		
		gs00 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec = gs1[np.int(k/5), k % 5])
		ax = plt.subplot(gs00[:4])
		ax.plot(R_measure, SB_measure, 'r-', label = '$SB_{ccd \\ at \\ z_{ref}}$', alpha = 0.5)
		ax.plot(R_measure, SB_compare0, 'b--', label = '$SB_{theory \\ at \\ z_0}$', alpha = 0.5)
		ax.plot(R_measure, SB_compare1, 'g--', label = '$SB_{theory \\ at \\ z_{ref}}$', alpha = 0.5)
		ax.text(1e2, -12, ' cx = %.0f \n cy = %.0f'%(posx[k], posy[k]))
		ax.set_xscale('log')
		ax.set_ylabel('$SB[mag/arcsec^2]$')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax1 = ax.twiny()
		ax1.plot(Ar_measure, SB_measure, 'r-', alpha = 0.5)
		ax1.set_xlabel('R[arcsec]')
		ax1.set_xscale('log')
		ax1.tick_params(axis = 'x', which = 'both', direction = 'in')
		ax.legend(loc = 3)
		ax.invert_yaxis()

		ax2 = plt.subplot(gs00[4])
		ax2.plot(R_measure, SB_compare1 - SB_measure, 'r-', label = '$SB_{z_{ref}} - SB_{ccd \\ z_{ref}}$', alpha = 0.5)
		ax2.set_xlabel('R[kpc]')
		ax2.set_xscale('log')
		ax2.set_ylabel('$\Delta SB[mag/arcsec^2]$')
		ax2.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax2.legend(loc = 3)

	plt.tight_layout()
	plt.savefig('light_random.pdf', dpi = 300)
	plt.close()
	raise
	return

def test():
	#SB_lightpro()
	#mock_image()
	#mock_ccd(xc = 1025, yc = 745)
	#light_test()
	resample_test()
	#random_test()

def main():
	test()

if __name__ == '__main__' :
	main()
