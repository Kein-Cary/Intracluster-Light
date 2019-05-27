# this file use to test the combine of image data
import h5py
import numpy as np

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from resamp import gen # test model
from scipy import interpolate as interp
from light_measure import light_measure, flux_recal
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

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

def add_light():
	bins = 90
	kz = 0 # for z < 0.25
	#kz = 3 # for z > 0.25
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

	## sample load
	with h5py.File('/home/xkchen/mywork/ICL/data/mock_frame/random_catalog_%.0f.h5'%kz ) as f:
		pos = np.array(f['a'])
	posx = pos[0]
	posy = pos[1]

	Nsample = len(pos[0])
	x0 = 2049
	y0 = 1490
	Nx = np.linspace(0, 4097, 4098)
	Ny = np.linspace(0, 2979, 2980)
	sum_grid = np.meshgrid(Nx, Ny)

	get_array = np.zeros((2980, 4098), dtype = np.float) # sum the flux value for each time
	count_array = np.zeros((2980, 4098), dtype = np.float) # sum array but use for pixel count for each time
	p_count_1 = np.zeros((2980, 4098), dtype = np.float) # how many times of each pixel get value

	t0 = time.time()
	for k in range(Nsample):
		f_data = fits.getdata('/home/xkchen/mywork/ICL/data/mock_frame/random_frame_z%.3f_randx%.0f_randy%.0f.fits'
			%(z[kz], posx[k], posy[k]), header = True)
		
		data = f_data[0]
		cx = f_data[1]['CENTER_X']
		cy = f_data[1]['CENTER_Y']

		Da0 = Test_model.angular_diameter_distance(z[kz]).value
		Angur = (R0/Da0)*rad2asec
		Rp = Angur/pixel
		cx = f_data[1]['CENTER_X']
		cy = f_data[1]['CENTER_Y']
		L_ref = Da_ref*pixel/rad2asec
		L_z0 = Da0*pixel/rad2asec
		b = L_ref/L_z0
		Rref = (R0*rad2asec/Da_ref)/pixel

		f_goal = flux_recal(data, z[kz], z_ref)
		xn, yn, resam = gen(f_goal, 1, b, cx, cy)
		if b > 1:
			resam = resam[1:, 1:]
		elif b == 1:
			resam = resam[1:-1, 1:-1]
		else:
			resam = resam
		re_x = np.int(xn)
		re_y = np.int(yn)
		'''
		SB_t, R_t, Ar_t, error_t = light_measure(resam, bins, 1, Rref, xn, yn, pixel, z_ref)
		SB_tt = SB_t[1:]
		R_tt = R_t[1:]
		Ar_tt = Ar_t[1:]
		erro_tt = error_t[1:]
		plt.figure()
		ax1 = plt.subplot(111)
		ax1.plot(Ar_tt, SB_tt, 'b-', label = '$single$')
		ax1.set_xscale('log')
		ax1.set_xlabel('$R[arcsec]$')
		ax1.set_ylabel('$M_r[mag/arcsec^2]$')
		ax1.legend(loc = 1)
		ax1.set_title('measure in r band')
		ax1.tick_params(axis = 'both', which = 'both', direction = 'in')
		ax2 = ax1.twiny()
		ax2.plot(R_tt, SB_tt, 'b-')
		ax2.set_xscale('log')
		ax2.set_xlabel('$R[kpc]$')
		ax2.tick_params(axis = 'x', which = 'both', direction = 'in')
		ax1.invert_yaxis()
		plt.savefig('single_test_%.0f.png'%k, dpi = 600)
		plt.close()
		'''
		get_array[y0 -re_y: y0 -re_y +resam.shape[0], x0 -re_x: x0 -re_x +resam.shape[1]] = \
		get_array[y0 -re_y: y0 -re_y +resam.shape[0], x0 -re_x: x0 -re_x +resam.shape[1]] + resam
		count_array[y0 -re_y: y0 -re_y +resam.shape[0], x0 -re_x: x0 -re_x +resam.shape[1]] = resam
		ia = np.where(count_array != 0)
		p_count_1[ia[0], ia[1]] = p_count_1[ia[0], ia[1]] + 1
		count_array[y0 -re_y: y0 -re_y +resam.shape[0], x0 -re_x: x0 -re_x +resam.shape[1]] = 0		
	
	mean_array = get_array/p_count_1
	where_are_nan = np.isnan(mean_array)
	mean_array[where_are_nan] = 0
	t1 = time.time()-t0
	'''
	#plt.imshow(get_array, cmap = 'rainbow', origin = 'lower', norm = mpl.colors.LogNorm())
	plt.imshow(mean_array, cmap = 'rainbow', origin = 'lower', norm = mpl.colors.LogNorm())
	plt.xlim(1445, 2650)
	plt.ylim(870, 2090)
	plt.savefig('stacking_test.png', dpi = 600)
	plt.show()
	'''
	Angur = (R0/Da_ref)*rad2asec
	Rp = Angur/pixel
	SB_1, R_1, r0_1, error_1 = light_measure(mean_array, bins, 1, Rp, x0, y0, pixel, z_ref)
	SB_measure = SB_1[1:]
	R_measure = R_1[1:]
	Ar_measure = r0_1[1:]
	SB_error = error_1[1:]
	SB_compare0 = flux_f1(R_measure/np.max(rbin))
	SB_compare1 = flux_f2(R_measure/np.max(rbin))

	fig = plt.figure(figsize = (16,9))
	fig.suptitle('stacked measure with bin%.0f'%bins)
	gs1 = gridspec.GridSpec(2,1, height_ratios = [4,1])
	ax1 = plt.subplot(gs1[0])
	ax2 = plt.subplot(gs1[1])

	ax1.plot(R_measure, SB_measure, 'r-', lw = 2, label = '$SB_{ccd \\ at \\ z_{ref}}$', alpha = 0.5)
	#ax1.plot(R_measure, SB_compare0, 'b--', lw = 2, label = '$SB_{theory \\ at \\ z_0}$', alpha = 0.5)
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

	ax2.plot(R_measure, SB_compare1 - SB_measure, 'r--', label = '$SB_{theory \\ at \\ z_{ref}} - SB_{ccd \\ at \\ z_{ref}}$', alpha = 0.5)
	ax2.set_xlabel('R[kpc]')
	ax2.set_xscale('log')
	ax2.set_ylabel('$\Delta SB[mag/arcsec^2]$')
	ax2.legend(loc = 1, fontsize = 15)

	plt.savefig('rescale_resample_stack.png')
	plt.show()
	raise
	return
def test():
	add_light()

def main():
	test()

if __name__ == '__main__' :
	main()