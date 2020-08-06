import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits
import astropy.wcs as awc
import astropy.io.ascii as asc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from scipy.stats import binned_statistic as binned
from light_measure import light_measure_Z0
from light_measure_tmp import light_measure_Z0_weit

rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

##############
def SB_pro_z0(img, pix_size, r_lim, R_pix, cx, cy, R_bins, band_id):

	kk = band_id
	Intns, Angl_r, Intns_err = light_measure_Z0(img, pix_size, r_lim, R_pix, cx, cy, R_bins)
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	err0 = SB - dSB0
	err1 = dSB1 - SB
	id_nan = np.isnan(SB)

	SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Angl_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	out_err1[idx_nan] = 100.

	return R_out, SB_out, out_err0, out_err1, Intns, Angl_r, Intns_err

def SB_pro_z0_weit(img, weit_img, pix_size, r_lim, R_pix, cx, cy, R_bins, band_id):

	kk = band_id
	Intns, Angl_r, Intns_err = light_measure_Z0_weit(img, weit_img, pix_size, r_lim, R_pix, cx, cy, R_bins)[:3]
	SB = 22.5 - 2.5 * np.log10(Intns) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	flux0 = Intns + Intns_err
	flux1 = Intns - Intns_err
	dSB0 = 22.5 - 2.5 * np.log10(flux0) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	dSB1 = 22.5 - 2.5 * np.log10(flux1) + 2.5 * np.log10(pixel**2) + mag_add[kk]
	err0 = SB - dSB0
	err1 = dSB1 - SB
	id_nan = np.isnan(SB)

	SB_out, R_out, out_err0, out_err1 = SB[id_nan == False], Angl_r[id_nan == False], err0[id_nan == False], err1[id_nan == False]
	dSB0, dSB1 = dSB0[id_nan == False], dSB1[id_nan == False]
	idx_nan = np.isnan(dSB1)
	out_err1[idx_nan] = 100.

	return R_out, SB_out, out_err0, out_err1, Intns, Angl_r, Intns_err

######################
dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/tot_cluster_norm_sample.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

set_z, set_ra, set_dec = z[2613: 2814], ra[2613: 2814], dec[2613: 2814]
set_x, set_y = clus_x[2613: 2814], clus_y[2613: 2814]

d_file = '/media/xkchen/My Passport/data/SDSS/tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f.fits'
##### out-lier
out_ra = ['14.079', '139.480', '147.638', '188.489', '221.544', '158.159', '197.174', '141.268', ]
out_dec = ['14.185', '8.159', '3.392', '41.144', '5.108', '29.695', '43.036', '-0.933', ]

lis_ra, lis_dec, lis_z = [], [], []
lis_x, lis_y = [], []
for ll in range(len(set_z)):
	identi = ('%.3f' % set_ra[ll] in out_ra) & ('%.3f' % set_dec[ll] in out_dec)
	if identi == True:
		continue
	else:
		lis_ra.append(set_ra[ll])
		lis_dec.append(set_dec[ll])
		lis_z.append(set_z[ll])
		lis_x.append(set_x[ll])
		lis_y.append(set_y[ll])

lis_ra = np.array(lis_ra)
lis_dec = np.array(lis_dec)
lis_z = np.array(lis_z)
lis_x = np.array(lis_x)
lis_y = np.array(lis_y)
zN = len(lis_z)

##### read the stacking image
bins = 95
with h5py.File('/home/xkchen/Downloads/test_imgs/cluster_select_BCG-stack_sub-13_test.h5', 'r') as f:
	stack_img = np.array(f['a'])

with h5py.File('/home/xkchen/Downloads/test_imgs/cluster_select_BCG-stack_sub-13_pix-cont.h5', 'r') as f:
	bcg_pix_cont = np.array(f['a'])

xn, yn = np.int(stack_img.shape[1] / 2), np.int(stack_img.shape[0] / 2)
#bcg_I, bcg_I_r, bcg_I_err = SB_pro_z0(stack_img, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), 0)[4:]
#bcg_I, bcg_I_err = bcg_I / pixel**2, bcg_I_err / pixel**2

#weit_bcg_I, weit_bcg_I_r, weit_bcg_I_err = SB_pro_z0_weit(stack_img, bcg_pix_cont, pixel, 1, 3000, xn, yn, np.int(1.22 * bins), 0)[4:]
#weit_bcg_I, weit_bcg_I_err = weit_bcg_I / pixel**2, weit_bcg_I_err / pixel**2

R_bins = np.logspace(np.log10(1), np.log10(3000), np.int(1.22 * bins) )
R_angle = 0.5 * (R_bins[1:] + R_bins[:-1]) * 0.396

### mark position
xn, yn = 2427, 1765
r_mark_0 = 50 ## 50", assuming BCG with 200kpc size and z~0.25
r_mark_1 = 100 ## beyond 100", the SB profile is more flat
short_edg = pixel * (stack_img.shape[0] / 2)
long_edg = pixel * (stack_img.shape[1] / 2)

targ_R = [r_mark_0, r_mark_1, short_edg, long_edg]
Nx = np.linspace(0, stack_img.shape[1] - 1, stack_img.shape[1])
Ny = np.linspace(0, stack_img.shape[0] - 1, stack_img.shape[0])
grd = np.array( np.meshgrid(Nx, Ny) )
cen_dR = np.sqrt( (grd[0] - xn)**2 + (grd[1] - yn)**2 )

f_stack = np.array([0])
f_arr = np.array([0])

for ll in range( 2,3 ):
	iR = targ_R[ll]
	ddr = np.abs(R_angle - iR)
	idx = np.where( ddr == ddr.min() )[0]
	edg_lo = R_bins[idx]
	edg_hi = R_bins[idx + 1]

	id_flux = (cen_dR >= edg_lo) & (cen_dR < edg_hi)
	id_nn = np.isnan(stack_img)
	id_effect = (id_nn == False) & id_flux

	sub_flux = stack_img[ id_effect ]
	f_stack = np.r_[f_stack, sub_flux]

	iuy, iux = np.where( id_effect == True)
	sub_f_arr = np.array([0])
	loop_n = len(iuy)

	for nn in range( loop_n ):
		x0, y0 = iux[nn], iuy[nn]

		for kk in range( zN ):
			ra_g, dec_g, z_g = lis_ra[kk], lis_dec[kk], lis_z[kk]
			data = fits.open(d_file % ('r', ra_g, dec_g, z_g) )
			img = data[0].data

			dev_05_x = lis_x[kk] - np.int(lis_x[kk])
			dev_05_y = lis_y[kk] - np.int(lis_y[kk])
			if dev_05_x > 0.5:
				x_img = np.int(lis_x[kk]) + 1
			else:
				x_img = np.int(lis_x[kk])

			if dev_05_y > 0.5:
				y_img = np.int(lis_y[kk]) + 1
			else:
				y_img = np.int(lis_y[kk])

			x_ori_0, y_ori_0 = x0 + x_img - xn, y0 + y_img - yn

			identy_0 = ( (x_ori_0 >= 0) & (x_ori_0 < 2048) ) & ( (y_ori_0 >= 0) & (y_ori_0 < 1489) )

			if identy_0 == True:
				sub_f_arr = np.r_[sub_f_arr, img[y_ori_0, x_ori_0] ]

	sub_f_arr = sub_f_arr[1:]
	f_arr = np.r_[f_arr, sub_f_arr]

f_arr = f_arr[1:]
f_stack = f_stack[1:]

with h5py.File('stack_radius-%.1f_flux.h5' % targ_R[ll], 'w') as f:
	f['a'] = np.array(f_stack)

with h5py.File('imgs_pix-flux_%.1f_radius.h5' % targ_R[ll], 'w') as f:
	f['a'] = np.array(f_arr)

raise
'''
fig = plt.figure( figsize = (12, 12) )
ax0 = fig.add_axes([0.05, 0.55, 0.45, 0.45])
ax1 = fig.add_axes([0.55, 0.55, 0.45, 0.45])
ax2 = fig.add_axes([0.05, 0.05, 0.45, 0.45])
ax3 = fig.add_axes([0.55, 0.05, 0.45, 0.45])

circl0 = Circle(xy = (xn, yn), radius = stack_img.shape[0] / 2, fill = False, ec = 'k', ls = '--', alpha = 0.75, )
circl1 = Circle(xy = (xn, yn), radius = stack_img.shape[1] / 2, fill = False, ec = 'k', ls = '-', alpha = 0.75, )
circl2 = Circle(xy = (xn, yn), radius = r_mark_0 / pixel, fill = False, ec = 'k', ls = ':', alpha = 0.75, )
circl3 = Circle(xy = (xn, yn), radius = r_mark_1 / pixel, fill = False, ec = 'k', ls = '-.', alpha = 0.75, )
tf = ax0.imshow(stack_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -0.02, vmax = 0.02)
plt.colorbar(tf, ax = ax0, fraction = 0.040, pad = 0.0, label = '')
ax0.add_patch(circl0)
ax0.add_patch(circl1)
ax0.add_patch(circl2)
ax0.add_patch(circl3)
ax0.set_xlim(0, stack_img.shape[1])
ax0.set_ylim(0, stack_img.shape[0])

ax1.errorbar(bcg_I_r, bcg_I, yerr = bcg_I_err, xerr = None, color = 'r', marker = 'None', ls = '-', linewidth = 1.25, 
			ecolor = 'r', elinewidth = 1.25, alpha = 0.5, label = 'No weight')
ax1.axvline(x = pixel * (stack_img.shape[0] / 2), color = 'k', linestyle = '--', linewidth = 1, label = '$ H_{img} / 2$')
ax1.axvline(x = pixel * (stack_img.shape[1] / 2), color = 'k', linestyle = '-', linewidth = 1, label = '$ W_{img} / 2$')
ax1.axvline(x = r_mark_0, color = 'k', linestyle = ':', linewidth = 1,)
ax1.axvline(x = r_mark_1, color = 'k', linestyle = '-.', linewidth = 1,)

ax1.set_xlim(1e1, 1e3)
ax1.set_ylim(3e-3, 3e-2)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_xlabel('R [arcsec]')
ax1.set_ylabel('SB [nanomaggies / $ arcsec^2 $]')
ax1.legend(loc = 1, frameon = False)
ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
ax1.tick_params(axis = 'both', which = 'both', direction = 'in')

for nn in range(4):
	ax2.hist(f_stack[nn] / pixel**2, bins = 65, density = True, color = mpl.cm.rainbow(nn / 4), alpha = 0.5,)
	ax2.axvline(np.mean(f_stack[nn]) / pixel**2, ls = '--', color = mpl.cm.rainbow(nn / 4), alpha = 0.5, )
ax2.set_xlabel('SB [nanomaggies / $arcsec^2$]')
ax2.set_ylabel('PDF')

for nn in range(4):
	ax3.hist(f_arr[nn] / pixel**2, bins = 65, density = True, color = mpl.cm.rainbow(nn / 4), alpha = 0.5,)
	ax3.axvline(np.mean(f_arr[nn]) / pixel**2, ls = '--', color = mpl.cm.rainbow(nn / 4), alpha = 0.5,)
ax3.set_xlabel('SB [nanomaggies / $arcsec^2$]')
ax3.set_ylabel('PDF')
plt.savefig('flux_hist_test.png', dpi = 300)
plt.show()
'''
