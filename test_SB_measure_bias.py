import matplotlib as mpl
mpl.use('Agg')
import handy.scatter as hsc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from scipy.interpolate import interp1d as interp
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle
from light_measure import light_measure, flux_recal
# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)

# global various
Lsun = C.L_sun.value*10**7
G = C.G.value
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23)
f0 = 3631 * Jy
R0 = 1
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
	catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]

band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

def hist_plot():
	bins = 65
	load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'
	for kk in range(len(band)):
		for tt in range(len(z)):
			z_g = z[tt]
			ra_g = ra[tt]
			dec_g = dec[tt]
			Da_g = Test_model.angular_diameter_distance(z_g).value

			f0 = fits.getdata(load + 
				'mask_data/A_plane/1.5sigma/A_mask_data_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[kk], ra_g, dec_g, z_g), header = True)
			img0 = f0[0]
			cx0 = f0[1]['CRPIX1']
			cy0 = f0[1]['CRPIX2']
			RA0 = f0[1]['CRVAL1']
			DEC0 = f0[1]['CRVAL2']
			wcs = awc.WCS(f0[1])
			cx, cy = wcs.all_world2pix(ra_g*U.deg, dec_g*U.deg, 1)

			Angur = (R0*rad2asec/Da_g)
			Rp = Angur/pixel
			L_ref = Da_ref*pixel/rad2asec
			L_z0 = Da_g*pixel/rad2asec
			b = L_ref/L_z0
			mu = 1 / b

			SBt, Rt, Anrt, errt = light_measure(img0, bins, 1, Rp, cx, cy, pixel, z_g)[:4]
			id_nan = np.isnan(SBt)
			ivx = id_nan == False
			SB_ref = SBt[ivx] + 10*np.log10((1 + z_ref) / (1 + z_g)) + mag_add[kk]
			Ar_ref = Anrt[ivx] * mu
			f_SB = interp(Ar_ref, SB_ref, kind = 'cubic')

			f1 = fits.getdata(load + 
				'resample/1_5sigma/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[kk], ra_g, dec_g, z_g), header = True)
			img1 = f1[0]
			xn = f1[1]['CENTER_X']
			yn = f1[1]['CENTER_Y']
			SB, R, Anr, err = light_measure(img1, bins, 1, Rpp, xn, yn, pixel, z_ref)[:4]
			SB = SB + mag_add[kk]

			plt.figure(figsize = (16, 10))
			gs = gridspec.GridSpec(2, 1, height_ratios = [4, 1])
			ax = plt.subplot(gs[0])
			bx = plt.subplot(gs[1], sharex = ax)

			ax.set_title('Test of SB profile measurement [ra%.3f dec%.3f z%.3f %s band]' %(ra_g, dec_g, z_g, band[kk]) )
			ax.plot(Ar_ref, SB_ref, 'r--', label = '$ Reference $', alpha = 0.5)
			ax.plot(Anr, SB, 'g-', label = '$ Measuring \; from \; resampling \; image $', alpha = 0.5)

			ddsb = SB[(Anr > np.min(Ar_ref)) & (Anr < np.max(Ar_ref))] - f_SB(Anr[(Anr > np.min(Ar_ref)) & (Anr < np.max(Ar_ref))])

			bx.plot(Anr[(Anr > np.min(Ar_ref)) & (Anr< np.max(Ar_ref))], ddsb, 'g*', alpha = 0.5)
			bx.axhline(y = 0, ls = '--', color = 'b', alpha = 0.5)

			#ax.set_xscale('log')
			#ax.set_xlabel('$R[arcsec]$')
			ax.set_ylabel('$SB[mag/arcsec^2]$')
			ax.tick_params(axis = 'both', which = 'both', direction = 'in')
			ax.invert_yaxis()
			ax.legend(loc = 3, fontsize = 12)
			#ax.set_xlim(1, 2.5e2)

			bx.set_xlabel('$R[arcsec]$')
			bx.set_xscale('log')
			bx.set_ylabel('$ \Delta{SB}[mag/arcsec^2] $')
			bx.tick_params(axis = 'both', which = 'both', direction = 'in')
			bx.set_xlim(1, 2.5e2)

			plt.subplots_adjust(hspace = 0)
			plt.tight_layout()
			plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_summary/SB_bias/SB_bias_test_ra%.3f_dec%.3f_z%.3f_%s_band.png' % (ra_g, dec_g, z_g, band[kk]), dpi = 300)
			plt.close()

			id_nan = np.isnan(ddsb)
			ivx = id_nan == False
			plt.figure()
			plt.title('Measurement deviation distribution [ra%.3f dec%.3f z%.3f %s band]' % (ra_g, dec_g, z_g, band[kk]))
			plt.hist(ddsb[ivx], histtype = 'step', color = 'b', alpha = 0.5)
			plt.axvline(x = 0, ls = '--', color = 'r', alpha = 0.5)
			plt.savefig(
				'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_summary/SB_bias/SB_deviation_hist_ra%.3f_dec%.3f_z%.3f_%sband.png' % (ra_g, dec_g, z_g, band[kk]), dpi = 300)
			plt.close()

	return

def main():
	hist_plot()

if __name__ == "__main__":
	main()