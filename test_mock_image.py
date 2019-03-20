"""
# this file use to creat a mock cluster with NFW model
and M/L = 50 M_sun/L_sun. 
accrding the light profile, we'll ctreat an image with 
real data(include redshift, pixel_size)
"""
import h5py
import numpy as np
from scipy import interpolate
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy
from ICL_surface_mass_density import sigma_m_c_unlog

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
	R, rho_2d, rbin = sigma_m_c_unlog(Mh, N)
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

	Lob, Iner_SB, z, rbin, R = SB_lightpro()
	Npi = 1000
	x1 = np.linspace(-1.2*R, 1.2*R, 2*Npi)
	y1 = np.linspace(-1.2*R, 1.2*R, 2*Npi)
	plane = np.meshgrid(x1, y1)
	dr = np.sqrt((plane[0]-0)**2+(plane[1]-0)**2)
	mock_ana = np.zeros((2*Npi, 2*Npi), dtype = np.float)

	ia = rbin <= np.min(dr)
	ib = np.where(ia == True)
	pro = rbin[ia]
	imk = len(pro)
	for k in range(len(rbin)):
		if k <= imk-1:
			ia = dr <= rbin[imk-1]
			ib = np.where(ia == True)
			mock_ana[ib[1], ib[0]] = np.sum(Lob[0][:imk])/imk
			k = imk
		else:
			ia = (dr > rbin[k-1]) & (dr <= rbin[k])
			ib = np.where(ia == True)
			mock_ana[ib[1], ib[0]] = Lob[0][k]
	'''
	#plt.imshow(mock_ana, cmap = 'plasma', origin = 'lower', norm = mpl.colors.LogNorm())
	plt.pcolormesh(plane[0], plane[1], mock_ana, cmap = 'plasma', norm = mpl.colors.LogNorm())
	plt.colorbar(label = '$flux[(erg/s)/cm^2]$')
	plt.xlabel('R[kpc]')
	plt.ylabel('R[kpc]')
	#plt.savefig('mock_cluster.png', dpi = 600)
	plt.savefig('mock_cluster_phy.png', dpi = 600)
	plt.close()
	'''
	return

def mock_ccd():

	Lob, Iner_SB, z, rbin, R = SB_lightpro()
	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	ra = catalogue[1]
	dec = catalogue[2]
	
	r_sc = rbin/np.max(rbin)
	Da0 = Test_model.angular_diameter_distance(z).value
	Angu_r = (10**(-3)*R/Da0)*c4
	R_pixel = Angu_r/pixel
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
	for k in range(len(r_sc)):
		if k == 0:
			frame[0,0] = pixel**2*Lob[0][0]/(f0*10**(-9))
		else:
			ia = (dr_sc >= r_sc[k-1]) & (dr_sc < r_sc[k])
			ib = np.where(ia == True)
			frame[ib[0], ib[1]] = Lob[0][k]*pixel**2/(f0*10**(-9))
	'''
	plt.imshow(frame, cmap = 'rainbow', origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(label = 'flux [nMgy]', fraction = 0.035,pad = 0.003)
	plt.savefig('mock_frame.png', dpi = 600)
	plt.show()
	'''
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

def light_measure():
	Iner_SB, rbin = SB_lightpro()[1], SB_lightpro()[3]

	with h5py.File('/home/xkchen/mywork/ICL/code/sample_catalog.h5') as f:
		catalogue = np.array(f['a'])
	z = catalogue[0]
	ra = catalogue[1]
	dec = catalogue[2]

	mock_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/mock_frame_z%.3f_ra%.3f_dec%.3f.fits'
				%(z[0], ra[0], dec[0]), header = True)
	f = mock_data[0]
	x0 = np.linspace(0,2047,2048)
	y0 = np.linspace(0,1488,1489)
	pix_id = np.array(np.meshgrid(x0,y0))
	Nbins = 30
	Da0 = Test_model.angular_diameter_distance(z[0]).value
	Angur = (1/Da0)*c4
	Rp = Angur/pixel
	cx = mock_data[1]['CENTER_X']
	cy = mock_data[1]['CENTER_Y']

	r_in = ((rbin/10**3)/Da0)*c4
	dr = np.sqrt((pix_id[0]-cx)**2+(pix_id[1]-cy)**2)
	# case 1: log bins
	r = np.logspace(-2, np.log10(Rp), Nbins)
	ia = r<= 2
	ib = np.array(np.where(ia == True))
	ic = ib.shape[1]
	R = (r/Rp)*10**3 # in unit kpc
	R = R[np.max(ib):]
	r0 = r[np.max(ib):]
	Ar1 = ((R/10**3)/Da0)*c4 # in unit arcsec
	light = np.zeros(len(r)-ic+1, dtype = np.float)
	for k in range(1,len(r)):
	        if r[k] <= 2:
	            ig = r <= 2
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
	'''
	plt.figure(figsize = (8,8))
	ax1 = plt.subplot(121)
	ax1.plot(R, light, 'r--', label = 'log_bin')
	ax1.plot(rbin, Iner_SB[0], 'b-', label = 'assum')
	ax1.legend(loc = 1)
	ax1.set_xscale('log')
	plt.gca().invert_yaxis()
	ax1.set_xlabel('R[kpc]')
	ax1.set_ylabel('$SB[mag/arcsec^2]$')

	ax2 = plt.subplot(122)
	ax2.plot(Ar1, light, 'r--', label = 'log_bin')
	ax2.plot(r_in, Iner_SB[0], 'b-', label = 'assum')
	ax2.legend(loc = 1)
	ax2.set_xscale('log')
	plt.gca().invert_yaxis()
	ax2.set_xlabel('R[arcsec]')
	ax2.set_ylabel('$SB[mag/arcsec^2]$')
	plt.tight_layout()
	plt.savefig('log_bin.png', dpi = 600)
	plt.show()
	plt.close()
	'''
	# case 2: linear bins
	r_l = np.linspace(0, Rp, Nbins)
	ia_l = r_l<= 2
	ib_l = np.array(np.where(ia_l == True))
	ic_l = ib_l.shape[1]
	R_l = (r_l/Rp)*10**3 # in unit kpc
	R_l = R_l[np.max(ib_l):]
	r0_l = r_l[np.max(ib_l):]
	Ar1_l = ((R_l/10**3)/Da0)*c4 # in unit arcsec
	light_l = np.zeros(len(r_l)-ic_l+1, dtype = np.float)
	for k in range(1,len(r_l)):
			if k == 0:
				tot_flux_l = f[cx, cy]
				tot_area_l = pixel**2
				light_l[0] = 22.5-2.5*np.log10(tot_flux_l)+2.5*np.log10(tot_area_l)
			else:
			    ir_l = (dr >= r_l[k-1]) & (dr < r_l[k])
			    io_l = np.where(ir_l == True)
			    iy_l = io_l[0]
			    ix_l = io_l[1]
			    num_l = len(ix_l)
			    tot_flux_l = np.sum(f[iy_l,ix_l])/num_l
			    tot_area_l = pixel**2
			    light_l[k] = 22.5-2.5*np.log10(tot_flux_l)+2.5*np.log10(tot_area_l)
    '''
	plt.figure(figsize = (8,8))
	ax1 = plt.subplot(121)
	ax1.plot(R_l, light_l, 'r--', label = 'line_bin')
	ax1.plot(rbin, Iner_SB[0], 'b-', label = 'assum')
	ax1.legend(loc = 1)
	ax1.set_xscale('log')
	ax1.set_ylim(-13,-7.5)
	plt.gca().invert_yaxis()
	ax1.set_xlabel('R[kpc]')
	ax1.set_ylabel('$SB[mag/arcsec^2]$')

	ax2 = plt.subplot(122)
	ax2.plot(Ar1_l, light_l, 'r--', label = 'line_bin')
	ax2.plot(r_in, Iner_SB[0], 'b-', label = 'assum')
	ax2.legend(loc = 1)
	ax2.set_xscale('log')
	ax2.set_ylim(-13,-7.5)
	plt.gca().invert_yaxis()
	ax2.set_xlabel('R[kpc]')
	ax2.set_ylabel('$SB[mag/arcsec^2]$')
	plt.tight_layout()
	plt.savefig('line_bin.png', dpi = 600)
	plt.show()
	plt.close()
	'''
	return

def test():
	SB_lightpro()
	#mock_image()
	#mock_ccd()
	light_measure()

def main():
	test()

if __name__ == '__main__' :

	main()