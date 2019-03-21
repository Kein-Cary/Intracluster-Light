"""
# this file use to creat a mock cluster with NFW model
and M/L = 50 M_sun/L_sun. 
accrding the light profile, we'll ctreat an image with 
real data(include redshift, pixel_size)
"""
import h5py
import numpy as np
from scipy import interpolate as interp
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy
from resamp import gen
from ICL_surface_mass_density import sigma_m_c
from light_measure import light_measure, flux_recal

import matplotlib as mpl
import matplotlib.pyplot as plt
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

	Mh = 15-np.log10(8) # make the radius close to 1 Mpc
	N = 131
	R, rho_2d, rbin = sigma_m_c(Mh, N)
	Lc = rho_2d/100 # in unit L_sun/kpc^2, Lc = Lc(r)
	'''
	#plt.plot(rbin, rho_2d, 'b-')
	plt.plot(rbin, Lc, 'r-')
	plt.xlabel('R[kpc]')
	plt.ylabel('$SB[L_\odot/kpc^2]$')
	plt.yscale('log')
	plt.xscale('log')
	plt.title('mock SB profile')
	plt.savefig('iner_SB.png', dpi = 600)
	plt.show()
	print(R)
	'''
	# creat series cluster intrinsic SB profile
	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	a = 1/(1+z)
	SB = 21.572 + 4.75 - 2.5*np.log10(10**(-6)*np.tile(Lc, (len(z),1))) + 10*np.log10(np.tile(z+1,(N,1)).T)

	Lz = np.tile(a**4,(N,1)).T*np.tile(Lc, (len(z),1))/(4*np.pi*c4**2) # L at z in unit: (Lsun/kpc^2)/arcsec^2
	Lob = Lz*Lsun/c0**2
	Iner_SB = 22.5-2.5*np.log10(Lob/(10**(-9)*f0))

	zt = z[0]
	Dat = Test_model.angular_diameter_distance(zt).value
	st = (10**(-6)/Dat**2)*c4**2
	ft = Lc*Lsun/(4*np.pi*(1+zt)**4*Dat**2)
	SBt = 22.5 - 2.5*np.log10(ft/(c2**2*f0*10**(-9)))+2.5*np.log10(st)
	'''
	plt.plot(rbin, SBt, 'r-', alpha = 0.5, label = 'direct')
	plt.plot(rbin, Iner_SB[0,:], 'b--', alpha = 0.5, label = 'indrect')
	#plt.plot(rbin, SB[0,:], 'g-')
	plt.xscale('log')
	plt.xlabel('R[kpc]')
	plt.legend(loc = 1)
	plt.title('SB compare')
	plt.ylabel('$SB[Mag/arcsec^2]$')
	plt.gca().invert_yaxis()
	plt.savefig('SB_measuer.png', dpi = 600)
	plt.show()
	'''
	'''
	plt.figure(figsize = (8, 8))
	ax1 = plt.subplot(121)
	for k in range(len(z)):
		if k % 500 == 0:
			ax1.plot(rbin, SB[k,:], color = mpl.cm.rainbow(k/len(z)))
	plt.gca().invert_yaxis()
	plt.ylabel('$SB [Mag/arcsec^2]$')
	plt.xlabel('R [kpc]')
	plt.xscale('log')

	ax2 = plt.subplot(122)
	for k in range(len(z)):
		if k % 500 == 0:
			plt.plot(rbin, Iner_SB[k,:], color = mpl.cm.rainbow(k/len(z)))
	plt.gca().invert_yaxis()
	plt.ylabel('$SB [Mag/arcsec^2]$')
	plt.xlabel('R [kpc]')
	plt.xscale('log')
	plt.tight_layout()
	plt.savefig('note_compare.png', dpi = 600)
	plt.show()
	'''
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
	Lob, Iner_SB, z, rbin, R = SB_lightpro()
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
	'''
	plt.plot(r_sc, Lob[0], 'r--', label = 'origin')
	plt.plot(test_dr, test, 'b-', label = 'interp', alpha = 0.5)
	plt.legend(loc = 1)
	plt.yscale('log')
	plt.show()
	'''
	mock_ana = np.zeros((2*Npi+1, 2*Npi+1), dtype = np.float)
	fc = 0
	for k in range(len(ibt)-1):
		fc = fc + 2*np.pi*(ibt[k+1]-ibt[k])*ict[k]*ibt[k]

	for k in range(len(test_dr)):
		if k == 0:
			mock_ana[Npi+1, Npi+1] = fc
		elif k == 1:
			mock_ana[Npi+1, Npi+1] = fc
		else:
			ia = (dr_sc >= test_dr[k-1]) & (dr_sc < test_dr[k])
			ib = np.where(ia == True)
			mock_ana[ib[0], ib[1]] = flux_func(test_dr[k-1])

	#plt.imshow(frame, cmap = 'plasma', origin = 'lower', norm = mpl.colors.LogNorm())
	plt.pcolormesh(plane[0], plane[1], mock_ana, cmap = 'plasma', norm = mpl.colors.LogNorm())
	plt.colorbar(label = '$flux[(erg/s)/cm^2]$')
	#plt.savefig('mock_cluster.png', dpi = 600)
	plt.xlabel('R[kpc]')
	plt.ylabel('R[kpc]')
	plt.savefig('mock_cluster_phy.png', dpi = 600)
	plt.show()
	plt.close()
	
	return

def mock_ccd():

	Lob, Iner_SB, z, rbin, R = SB_lightpro()
	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	ra = catalogue[1]
	dec = catalogue[2]
	
	r_sc = rbin/np.max(rbin)
	flux_func = interp.interp1d(r_sc, Lob[0], kind = 'cubic')

	Da0 = Test_model.angular_diameter_distance(z).value
	Angu_r = (10**(-3)*R/Da0)*c4
	R_pixel = Angu_r/pixel
	r_in = (rbin*10**(-3)/Da0[0])*c4
	"""
	# scale as the size of observation
	(in case the cluster center is the frame center)
	"""
	y0 = np.linspace(0, 1488, 1489)
	x0 = np.linspace(0, 2047, 2048)
	frame = np.zeros((len(y0), len(x0)), dtype = np.float)
	pxl = np.meshgrid(x0, y0)
	xc = 1025
	yc = 745
	dr = np.sqrt((pxl[0]-xc)**2+(pxl[1]-yc)**2)
	dr_sc = dr/R_pixel[0]
	
	test_dr = dr_sc[745, 1026:1732]
	test = flux_func(test_dr)
	iat = r_sc <= test_dr[0]
	ibt = r_in[iat]
	ict = Lob[0][iat]
	fc = 0
	for k in range(len(ibt)-1):
		fc = fc + 2*np.pi*(ibt[k+1]-ibt[k])*ict[k]*ibt[k]

	for k in range(len(test_dr)):
		if k == 0:
			frame[xc, yc] = pixel**2*fc/(f0*10**(-9))
		elif k == 1:
			frame[xc, yc] = pixel**2*fc/(f0*10**(-9))
		else:
			ia = (dr_sc >= test_dr[k-1]) & (dr_sc < test_dr[k])
			ib = np.where(ia == True)
			frame[ib[0], ib[1]] = flux_func(test_dr[k-1])*pixel**2/(f0*10**(-9))
	
	plt.imshow(frame, cmap = 'rainbow', origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(label = 'flux[nMgy]', fraction = 0.035,pad = 0.003)
	#plt.savefig('interp.png', dpi =600)
	plt.savefig('mock_frame.png', dpi =600)
	plt.show()
	
	raise
	x = frame.shape[1]
	y = frame.shape[0]
	keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
	        'CENTER_X','CENTER_Y','CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z',]
	value = ['T', 32, 2, x, y, np.int(xc), np.int(yc), xc, yc, xc, yc, ra[0], dec[0], z[0]]
	ff = dict(zip(keys,value))
	fil = fits.Header(ff)
	fits.writeto('/home/xkchen/mywork/ICL/data/mock_frame/mock_frame_z%.3f_ra%.3f_dec%.3f.fits'
					%(z[0], ra[0], dec[0]),frame, header = fil, overwrite=True)
	return

def light_test():
	Iner_SB, rbin = SB_lightpro()[1], SB_lightpro()[3]

	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]

	mock_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/mock_frame_z%.3f_ra%.3f_dec%.3f.fits'
				%(z[0], ra[0], dec[0]), header = True)
	f = mock_data[0]

	test_data = f[745, 1025:]
	test_x = np.linspace(1025, 2047, 1023)
	test_r = (test_x-1025)*0.396
	test_lit = 22.5-2.5*np.log10(test_data)+2.5*np.log10(pixel**2)

	x0 = np.linspace(0,2047,2048)
	y0 = np.linspace(0,1488,1489)
	pix_id = np.array(np.meshgrid(x0,y0))
	Da0 = Test_model.angular_diameter_distance(z[0]).value

	r_in = ((rbin/10**3)/Da0)*c4
	Angur = (np.max(rbin/1000)/Da0)*c4
	Rp = Angur/pixel
	cx = mock_data[1]['CENTER_X']
	cy = mock_data[1]['CENTER_Y']
	'''
	plt.figure(figsize = (16,9))
	plt.plot(test_r, test_lit, 'r--', label = 'test', alpha = 0.5)
	plt.plot(r_in, Iner_SB[0], 'b-', label = 'orin', alpha = 0.5)
	plt.legend(loc = 3)
	plt.xlabel('R[arcsec]')
	plt.ylabel('$SB[mag/arcsec^2]$')
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.savefig('mock2ccd_test.png', dpi = 600)
	plt.show()
	'''
	Nbins = 131
	r_in = ((rbin/10**3)/Da0)*c4
	dr = np.sqrt(((2*pix_id[0]+1)/2-(2*cx+1)/2)**2+
				((2*pix_id[1]+1)/2-(2*cy+1)/2)**2)
	
	r = np.logspace(-3, np.log10(Rp), Nbins)
	ia = r<= 1
	ib = np.array(np.where(ia == True))
	ic = ib.shape[1]
	R = (r/Rp)*10**3 # in unit kpc
	R = R[np.max(ib):]
	r0 = r[np.max(ib):]
	Ar1 = ((R/10**3)/Da0)*c4 # in unit arcsec
	light = np.zeros(len(r)-ic+1, dtype = np.float)
	for k in range(1,len(r)):
	        if r[k] <= 1:
	            ig = r <= 1
	            ih = np.array(np.where(ig == True))
	            im = np.max(ih)
	            ir = dr < r[im]
	            io = np.where(ir == True)
	            iy = io[0]
	            ix = io[1]
	            num = len(ix)
	            tot_flux = np.sum(f[iy,ix])/num
	            tot_area = pixel**2
	            light[0] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
	            k = im+1 
	        else:
	            ir = (dr >= r[k-1]) & (dr < r[k])
	            io = np.where(ir == True)
	            iy = io[0]
	            ix = io[1]
	            num = len(ix)
	            tot_flux = np.sum(f[iy,ix])/num
	            tot_area = pixel**2
	            light[k-im] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)

	plt.figure(figsize = (16,9))
	plt.plot(R, light, 'r--', label = 'log_bin')
	plt.plot(rbin, Iner_SB[0], 'b-', label = 'assum')
	plt.legend(loc = 1)
	plt.xscale('log')
	plt.gca().invert_yaxis()
	plt.xlabel('R[kpc]')
	plt.ylabel('$SB[mag/arcsec^2]$')
	'''
	#plt.plot(Ar1, light, 'r--', label = 'log_bin')
	#plt.plot(r_in, Iner_SB[0], 'b-', label = 'assum')
	#plt.legend(loc = 1)
	#plt.xscale('log')
	#plt.gca().invert_yaxis()
	#plt.xlabel('R[arcsec]')
	#plt.ylabel('$SB[mag/arcsec^2]$')
	'''
	plt.savefig('ccd_light_measure.png', dpi = 600)
	plt.show()
	plt.close()
	raise
	return

def resample_test():

	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]

	f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/mock_frame_z%.3f_ra%.3f_dec%.3f.fits'
				%(z[0], ra[0], dec[0]), header = True)
	Da0 = Test_model.angular_diameter_distance(z[0]).value

	Iner_SB, rbin = SB_lightpro()[1], SB_lightpro()[3]
	SB0 = Iner_SB[0]
	r0 = rbin
	r0_0 = (rbin/1000)*c4/Da0

	Angur = (1/Da0)*c4
	Rp = Angur/pixel
	cx = f_data[1]['CENTER_X']
	cy = f_data[1]['CENTER_Y']
	L_ref = Da_ref*pixel/c4 
	L_z0 = Da0*pixel/c4
	b = L_ref/L_z0
	Rref = (1*c4/Da_ref)/pixel

	f = f_data[0]
	f_goal = flux_recal(f, z[0], z_ref)
	SB_1, R_1, r0_1 = light_measure(f_goal, 30, 2, Rp, cx, cy, pixel)
	SB_2 = SB0 + 10*np.log10((1+z_ref)/(1+z[0]))

	xn, yn, resam = gen(f_goal, 1, b, cx, cy)
	SB_3, R_3, r0_3 = light_measure(resam, 30, 2, Rref, xn, yn, pixel)
	'''
	plt.plot(rbin, SB_2, 'r--', label = 'SB_note')
	plt.plot(R_1, SB_1, 'b-', label = 'SB_sclae')
	plt.plot(R_3, SB_3, 'g-', label = 'SB_resam')
	plt.xlabel('R[kpc]')
	plt.ylabel('$ SB[mag/arcsec^2] $')
	plt.xscale('log')
	plt.legend(loc = 3)
	plt.ylim(-13,-7.5)
	plt.gca().invert_yaxis()
	plt.savefig('c_cen_resam_line.png', dpi = 600)
	plt.show()
	'''
	return

def test():
	#SB_lightpro()
	#mock_image()
	#mock_ccd()
	light_test()
	#resample_test()

def main():
	test()

if __name__ == '__main__' :

	main()