import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import random
import skimage
import numpy as np
import pandas as pds
from scipy import interpolate as interp

import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits
from astropy import cosmology as apcy
from scipy.optimize import curve_fit, minimize
from resample_modelu import down_samp, sum_samp
from light_measure_tmp import light_measure, flux_recal
## constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
Lsun2erg = U.L_sun.to(U.erg/U.s)
rad2asec = U.rad.to(U.arcsec)
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
Angu_ref = rad2asec / Da_ref
Rpp = Angu_ref / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*Jy # zero point in unit (erg/s)/cm^-2
# in unit L_sun ("copy from Galaxies in the Universe", use for Luminosity function calculation)
Lstar = 2e10

load = '/home/xkchen/mywork/ICL/data/mock_frame/'
def pro_err():
	NMGY = 5e-3 # mean value of the data sample
	gain = 4.735  # for r band (mean value)
	V_dark =  1.2 # for r band (mean value)
	exp_time = 54 # exposure time, in unit second
	sky = 21. # mag/arcsec^2 (from SDSS dr14: the image quality)
	N_sky = 10**( (22.5 - sky + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY

	ins_SB = pds.read_csv(load + 'mock_intrinsic_SB.csv')
	r, SB_r = ins_SB['r'], ins_SB['0.250']
	r_sc = r / 10**3
	r_max = np.max(r_sc)
	r_min = np.min(r_sc)
	DN = 10**( (22.5 - SB_r + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY
	## mock a ccd frame
	f_DN = interp.interp1d(r_sc, DN, kind = 'cubic')
	y0 = np.linspace(0, 1488, 1489)
	x0 = np.linspace(0, 2047, 2048)
	frame = np.zeros((len(y0), len(x0)), dtype = np.float)
	Nois = np.zeros((len(y0), len(x0)), dtype = np.float)
	pxl = np.meshgrid(x0, y0)

	xc, yc = 1025, 745
	dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )
	dr_sc = dr / Rpp
	DN_min = np.min(DN)
	for kk in range(dr_sc.shape[0]):
		for jj in range(dr_sc.shape[1]):
			if (dr_sc[kk, jj] >= r_max ) | (dr_sc[kk, jj] <= r_min ):
				lam_x = DN_min * gain / 10
				N_e = lam_x + N_sky * gain
				rand_x = np.random.poisson( N_e )
				frame[kk, jj] += lam_x # electrons number
				Nois[kk, jj] += rand_x
			else:
				lam_x = f_DN( dr_sc[kk, jj] ) * gain
				N_e = lam_x + N_sky * gain
				rand_x = np.random.poisson( N_e )
				frame[kk, jj] += lam_x # electrons number
				Nois[kk, jj] += rand_x

	N_mock = frame / gain
	N_ele = (frame + N_sky * gain) / gain
	N_sub = Nois / gain - N_sky
	Noise = N_mock - N_sub

	## change N_sub to flux in unit 'nmaggy'
	N_mooth = N_mock * NMGY
	N_flux = N_sub * NMGY
	Intns, Intns_r, Intns_err, Npix = light_measure(N_flux, 65, 10, Rpp, xc, yc, pixel, z_ref, 1.)
	ref_err0 = Intns_err * 1.
	Intns, Intns_r, Intns_err, Npix = light_measure(N_mooth, 65, 10, Rpp, xc, yc, pixel, z_ref, 1.)
	ref_err1 = np.sqrt( (Intns / NMGY) / (Npix * gain) + N_sky / (gain * Npix) ) * NMGY
	ref_eta = ref_err0 / ref_err1

	## test the err measurement
	bins = np.arange(35, 75, 5)
	pn = 1.

	#bins = 65
	#pn = np.arange(1, 10, 1)
	#pn = np.arange(0.2, 1.2, 0.2)

	plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios = [2,1])
	ax = plt.subplot(gs[0])
	bx = plt.subplot(gs[1])
	ax.set_title('err ratio -- bins relation')
	#ax.set_title('err ratio -- pn relation')
	for aa in range( len(bins) ):

		Intns, Intns_r, Intns_err, Npix = light_measure(N_flux, bins[aa], 10, Rpp, xc, yc, pixel, z_ref, pn)
		#Intns, Intns_r, Intns_err, Npix = light_measure(N_flux, bins, 10, Rpp, xc, yc, pixel, z_ref, pn[aa])
		cc_err0 = Intns_err * 1.

		Intns, Intns_r, Intns_err, Npix = light_measure(N_mooth, bins[aa], 10, Rpp, xc, yc, pixel, z_ref, pn)
		#Intns, Intns_r, Intns_err, Npix = light_measure(N_mooth, bins, 10, Rpp, xc, yc, pixel, z_ref, pn[aa])

		## for smooth image, the err should be the Poisson Noise
		f_err = np.sqrt( (Intns / NMGY) / (Npix * gain) + N_sky / (gain * Npix) ) * NMGY # in single pix term
		#f_err = np.sqrt( (Intns * Npix / NMGY) / gain + N_sky * Npix / gain ) * NMGY / Npixc # calculate the total flux and then convert to err
		cc_err1 = f_err * 1.

		eta = cc_err0 / cc_err1

		ax.plot(Intns_r, eta, linestyle = '-', color = mpl.cm.rainbow(aa / len(bins) ), label = 'bins %d' % bins[aa], alpha = 0.5)
		#ax.plot(Intns_r, eta, linestyle = '-', color = mpl.cm.rainbow(aa / len(pn) ), label = 'pn = %.1f' % pn[aa], alpha = 0.5)
		#bx.plot(Intns_r, eta - ref_eta, linestyle = '--', color = mpl.cm.rainbow(aa / len(pn) ), alpha = 0.5)

	ax.set_xscale('log')
	#ax.set_xlabel('R[kpc]')
	ax.legend(loc = 1, fontsize = 5)
	ax.set_ylabel('$err_{Z05} / err_{Poisson}$')
	ax.set_xticks([])

	bx.axhline(y = 0, linestyle = ':', color = 'k', alpha = 0.75)
	bx.set_xscale('log')
	bx.set_xlim(ax.get_xlim())
	bx.set_xlabel('R[kpc]')
	bx.set_ylabel('err ratio deviation')

	plt.subplots_adjust(hspace = 0)
	plt.savefig('err_test_bins.png', dpi = 300)
	#plt.savefig('err_test_pn.png', dpi = 300)
	plt.close()

	raise

def main():
	pro_err()

if __name__ == "__main__":
	main()