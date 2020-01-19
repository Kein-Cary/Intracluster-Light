import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C

from astropy import cosmology as apcy
from light_measure import light_measure, flux_recal

rad2asec = U.rad.to(U.arcsec)

Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100

pixel, z_ref = 0.396, 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2
R0 = 1 # Mpc, physical size of cluster
Angu_ref = (R0 / Da_ref) * rad2asec
Rp_ref = Angu_ref / pixel

band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])  ## magnitude correction
l_wave = np.array([6166, 4686, 7480, 3551, 8932])

dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/' ## save the catalogue data
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'  ## save the process data
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

def sers_pro(r, mu_e, r_e, n):
	belta_n = 2 * n - 0.324
	fn = 1.086 * belta_n * ( (r/r_e)**(1/n) - 1)
	mu_r = mu_e + fn
	return mu_r

def main():
	x0, y0 = 2427, 1765  ## center of stack image
	R_cut, bins = 1280, 80
	R_smal, R_max = 1, 1.7e3 # kpc

	r_a0, r_a1 = 1.0, 1.1
	N_sum = 100
	## read the stack image
	#for kk in range(len(band)):
	for kk in range( 3 ):

		SB_tt = pds.read_csv( load + 'Zibetti_SB/%s_band_BCG_ICL.csv' % band[kk])
		R_obs, SB_obs = SB_tt['(1000R)^(1/4)'], SB_tt['mag/arcsec^2']
		R_obs = R_obs**4
		## sersic part
		Mu_e, R_e, n_e = mu_e[kk], r_e[kk], 4.
		SB_Z05 = sers_pro(R_obs, Mu_e, R_e, n_e)

		with h5py.File(tmp + 'test/stack_maskA_%d_in_%s_band.h5' % (N_sum, band[kk]), 'r') as f:
			stack_img = np.array(f['a'])

		ss_img = stack_img[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
		Intns, Intns_r, Intns_err, Npix = light_measure(ss_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
		SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		id_nan = np.isnan(SB)
		SBt, Rt = SB[id_nan == False], Intns_r[id_nan == False]

		with h5py.File(tmp + 'test/stack_sky_mean_%d_imgs_%s_band.h5' % (N_sum, band[kk]), 'r') as f:
		#with h5py.File(tmp + 'test/stack_sky_median_%d_imgs_%s_band.h5' % (N_sum, band[kk]), 'r') as f:
			BCG_add = np.array(f['a'])

		with h5py.File(tmp + 'test/M_sky_rndm_mean_%d_imgs_%s_band.h5' % (N_sum, band[kk]), 'r') as f:
		#with h5py.File(tmp + 'test/M_sky_rndm_median_%d_imgs_%s_band.h5' % (N_sum, band[kk]), 'r') as f:
			shlf_add = np.array(f['a'])

		resi_add = BCG_add - shlf_add

		### save the difference image (make sure the size is the same)
		with h5py.File(tmp + 'test/sky_difference_img_%d_imgs_%s_band.h5' % (N_sum, band[kk]), 'w') as f:
			f['a'] = np.array(resi_add)

		### save the corrected image
		correct_img = stack_img + resi_add - Resi_bl
		with h5py.File(tmp + 'test/Correct_stack_maskA_%d_imgs_%s_band.h5' % (N_sum[kk], band[kk]), 'w') as f:
			f['a'] = np.array(correct_img)

		add_img = ss_img + resi_add[y0 - R_cut: y0 + R_cut, x0 - R_cut: x0 + R_cut]
		Intns, Intns_r, Intns_err, Npix = light_measure(add_img, bins, R_smal, R_max, R_cut, R_cut, pixel, z_ref)
		SB_add = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		R_add = Intns_r * 1

		cen_pos = R_cut * 1 # 1280 pixel, for z = 0.25, larger than 2Mpc
		BL_img = add_img * 1
		grd_x = np.linspace(0, BL_img.shape[1] - 1, BL_img.shape[1])
		grd_y = np.linspace(0, BL_img.shape[0] - 1, BL_img.shape[0])
		grd = np.array( np.meshgrid(grd_x, grd_y) )
		ddr = np.sqrt( (grd[0,:] - cen_pos)**2 + (grd[1,:] - cen_pos)**2 )
		idu = (ddr > r_a0 * Rpp) & (ddr < r_a1 * Rpp)
		Resi_bl = np.nanmean( BL_img[idu] )

		# minus the RBL
		sub_SB = 22.5 - 2.5 * np.log10(Intns - Resi_bl) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		flux0 = Intns + Intns_err - Resi_bl
		flux1 = Intns - Intns_err - Resi_bl
		dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
		err0 = sub_SB - dSB0
		err1 = dSB1 - sub_SB

		id_nan = np.isnan(sub_SB)
		cli_SB, cli_R, cli_err0, cli_err1 = sub_SB[id_nan == False], R_add[id_nan == False], err0[id_nan == False], err1[id_nan == False]
		dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
		idx_nan = np.isnan(dSB1)
		cli_err1[idx_nan] = 100.

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('$ %s \, band \, SB \, %d imgs\,[subtract \, SB \, in \, %.2f \sim %.2f Mpc] $' % (band[kk], N_sum, r_a0, r_a1) )
		ax.errorbar(cli_R, cli_SB, yerr = [cli_err0, cli_err1], xerr = None, color = 'b', marker = '.', ls = '', linewidth = 1, markersize = 5, 
			ecolor = 'b', elinewidth = 1, label = 'Pipe.', alpha = 0.5)

		ax.plot(R_obs, SB_obs, 'k-.', label = 'Z05', alpha = 0.5)
		ax.plot(R_obs, SB_Z05, 'k:', label = 'Sersic', alpha = 0.5)
		ax.set_xlabel('$R[kpc]$')
		ax.set_ylabel('$SB[mag / arcsec^2]$')
		ax.set_xscale('log')
		ax.set_ylim(20, 33)
		ax.set_xlim(1, 1.5e3)
		ax.legend(loc = 3, fontsize = 7.5)
		ax.invert_yaxis()
		ax.grid(which = 'both', axis = 'both')
		ax.tick_params(axis = 'both', which = 'both', direction = 'in')
		plt.savefig(tmp + 'test/stack_maskA_%d_imgs_SB_pro_%s_band.png' %(N_sum, band[kk]), dpi = 300)
		plt.close()

if __name__ == "__main__":
	main()
