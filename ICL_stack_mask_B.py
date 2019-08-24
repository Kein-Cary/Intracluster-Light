import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import handy.scatter as hsc

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

import h5py
import numpy as np
import astropy.wcs as awc
import subprocess as subpro
import astropy.io.ascii as asc
import astropy.io.fits as fits
from scipy.interpolate import interp1d as interp
from scipy.optimize import curve_fit, minimize

from light_measure import light_measure, flux_recal
from light_measure import sigmamc
from matplotlib.patches import Circle

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
Lsun = C.L_sun.value*10**7
G = C.G.value

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

# sample catalog
with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
    catalogue = np.array(f['a'])
z = catalogue[0]
ra = catalogue[1]
dec = catalogue[2]
R0 = 1 # in unit Mpc
Angu_ref = (R0/Da_ref)*rad2asec
Rpp = Angu_ref/pixel

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'

band = ['r', 'g', 'i', 'u', 'z']
mag_add = np.array([0, 0, 0, -0.04, 0.02])

#read Redmapper catalog
goal_data = fits.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)

z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]

red_z = z_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_ra = ra_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_dec = dec_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]
red_rich = rich_eff[(z_eff <= 0.3)&(z_eff >= 0.2)]

def stack_light(band_number, stack_number, subz, subra, subdec, subrich):
	stack_N = np.int(stack_number)
	ii = np.int(band_number)
	sub_z = subz
	sub_ra = subra
	sub_dec = subdec
	sub_rich = subrich

	x0 = 2427
	y0 = 1765
	bins = 65
	Nx = np.linspace(0, 4854, 4855)
	Ny = np.linspace(0, 3530, 3531)
	sum_grid = np.array(np.meshgrid(Nx, Ny))

	sum_array_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)
	count_array_A = np.ones((len(Ny), len(Nx)), dtype = np.float) * np.nan
	p_count_A = np.zeros((len(Ny), len(Nx)), dtype = np.float)

	for jj in range(stack_N):

		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		Da_g = Test_model.angular_diameter_distance(z_g).value
		data = fits.getdata(load + 
		    'resample/resam_B/frameB-%s-ra%.3f-dec%.3f-redshift%.3f.fits' % (band[ii], ra_g, dec_g, z_g), header = True)
		img = data[0]
		xn = data[1]['CENTER_X']
		yn = data[1]['CENTER_Y']

		la0 = np.int(y0 - yn)
		la1 = np.int(y0 - yn + img.shape[0])
		lb0 = np.int(x0 - xn)
		lb1 = np.int(x0 - xn + img.shape[1])

		idx = np.isnan(img)
		idv = np.where(idx == False)
		sum_array_A[la0:la1, lb0:lb1][idv] = sum_array_A[la0:la1, lb0:lb1][idv] + img[idv]
		count_array_A[la0: la1, lb0: lb1][idv] = img[idv]
		id_nan = np.isnan(count_array_A)
		id_fals = np.where(id_nan == False)
		p_count_A[id_fals] = p_count_A[id_fals] + 1
		count_array_A[la0: la1, lb0: lb1][idv] = np.nan

	# no subtraction
	mean_array_0 = sum_array_A / p_count_A
	where_are_inf = np.isinf(mean_array_0)
	mean_array_0[where_are_inf] = np.nan
	id_zeros = np.where(p_count_A == 0)
	mean_array_0[id_zeros] = np.nan

	SB, R, Ar, error = light_measure(mean_array_0, bins, 1, Rpp, x0, y0, pixel, z_ref)[:4]
	SB_0 = SB + mag_add[ii]
	R_0 = R * 1
	Ar_0 = Ar * 1
	err_0 = error * 1

	lit_record = np.array([SB_0, R_0, Ar_0, err_0])
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_clusters_total_SB_%s_band.h5' % (stack_N, band[ii]) , 'w') as f:
		f['a'] = np.array(lit_record)
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_clusters_total_SB_%s_band.h5' % (stack_N, band[ii]) ) as f:
		for kk in range(lit_record.shape[0]):
			f['a'][kk, :] = lit_record[kk, :]

	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_Bmask_%s_band.h5' % (stack_N, band[ii]), 'w') as f:
		f['a'] = np.array(mean_array_0)

	cluster1 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', alpha = 0.5)
	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('Stacking %d clusters in %s band [B-mask]' % (stack_N, band[ii]))
	fx = ax.imshow(mean_array_0, cmap = 'Greys', vmin = 1e-4, origin = 'lower', norm = mpl.colors.LogNorm())
	plt.colorbar(fx, fraction = 0.045, pad = 0.01, label = '$flux [ 10^{-9} \, maggies]$')
	ax.scatter(x0, y0, s = 10, marker = 'X', facecolors = '', edgecolors = 'r', linewidth = 0.5, alpha = 0.5)
	ax.add_patch(cluster1)
	ax.set_xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
	ax.set_ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)

	xtick = ax.get_xticks()
	xR = (xtick - x0) * pixel * Da_ref / rad2asec
	xR = np.abs(xR)
	ax.set_xticks(xtick)
	ax.set_xticklabels(["%.2f" % uu for uu in xR])
	ax.set_xlabel('$R[Mpc]$')
	ytick = ax.get_yticks()
	yR = (ytick - y0) * pixel * Da_ref / rad2asec
	yR = np.abs(yR)
	ax.set_yticks(ytick)
	ax.set_yticklabels(["%.2f" % uu for uu in yR])
	ax.set_ylabel('$R[Mpc]$')

	plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
	plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_%d_maskB_%s_band.png' % (stack_N, band[ii]), dpi = 300)
	plt.close()


	fig = plt.figure(figsize = (16,9))
	plt.suptitle('$ Stacking \; %d \; clusters \; in \; %s \; band [B-mask] $' % (stack_N, band[ii]), fontsize = 15)
	bx = plt.subplot(111)

	bx.errorbar(R_0, SB_0, yerr = err_0, xerr = None, color = 'b', marker = 's', ls = '', linewidth = 1, markersize = 5, 
		ecolor = 'b', elinewidth = 1, alpha = 0.5, label = '$ No BL subtraction $')

	bx.set_xscale('log')
	bx.set_xlabel('$R[kpc]$')
	bx.set_ylabel('$SB[mag/arcsec^2]$')
	bx.invert_yaxis()
	bx.set_xlim(np.nanmin(R_0) + 1, np.nanmax(R_0) + 50)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')
	bx.legend( loc = 1, fontsize = 12)

	bx1 = bx.twiny()
	xtik = bx.get_xticks(minor = True)
	xR = xtik * 10**(-3) * rad2asec / Da_ref
	xR = xtik * 10**(-3) * rad2asec / Da_ref
	id_tt = xtik >= 9e1 
	bx1.set_xticks(xtik[id_tt])
	bx1.set_xticklabels(["%.2f" % uu for uu in xR[id_tt]])
	bx1.set_xlim(bx.get_xlim())
	bx1.set_xlabel('$R[arcsec]$')
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	subax = fig.add_axes([0.55, 0.35, 0.25, 0.25])
	subax.set_title('$\lambda \; distribution$')
	subax.hist(sub_rich, histtype = 'step', color = 'b')
	subax.set_xlabel('$\lambda$')
	subax.set_ylabel('$N$')

	plt.savefig(
		'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_%d_maskB_%s_band_profile.png' % (stack_N, band[ii]), dpi = 300)
	plt.close()

	return mean_array_0, SB_0, R_0, err_0

def main():
	bins = 65

	ix = red_rich >= 39
	RichN = red_rich[ix]
	zN = red_z[ix]
	raN = red_ra[ix]
	decN = red_dec[ix]
	stackn = np.int(690)

	GR_minus = []
	Mag_minus = []
	R_record = []
	x0 = 2427
	y0 = 1765

	for ii in range(2):
		with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_Bmask_%s_band.h5' % (stackn, band[ii]), 'r') as f:
			mask_B = np.array(f['a'])

		cluster1 = Circle(xy = (x0, y0), radius = Rpp, fill = False, ec = 'r', linestyle = '-', alpha = 0.5)
		cluster2 = Circle(xy = (x0, y0), radius = 0.2 * Rpp, fill = False, ec = 'g', linestyle = '--', alpha = 0.5)
		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('Stacking %d clusters in %s band [B-mask]' % (stackn, band[ii]))
		fx = ax.imshow(mask_B, cmap = 'Greys', vmin = 1e-4, origin = 'lower', norm = mpl.colors.LogNorm())
		plt.colorbar(fx, fraction = 0.045, pad = 0.01, label = '$flux [ 10^{-9} \, maggies]$')
		ax.scatter(x0, y0, s = 10, marker = 'X', facecolors = '', edgecolors = 'r', linewidth = 0.5, alpha = 0.5)
		ax.add_patch(cluster1)
		ax.add_patch(cluster2)
		ax.set_xlim(x0 - 1.2 * Rpp, x0 + 1.2 * Rpp)
		ax.set_ylim(y0 - 1.2 * Rpp, y0 + 1.2 * Rpp)

		xtick = ax.get_xticks()
		xR = (xtick - x0) * pixel * Da_ref / rad2asec
		xR = np.abs(xR)
		ax.set_xticks(xtick)
		ax.set_xticklabels(["%.2f" % uu for uu in xR])
		ax.set_xlabel('$R[Mpc]$')
		ytick = ax.get_yticks()
		yR = (ytick - y0) * pixel * Da_ref / rad2asec
		yR = np.abs(yR)
		ax.set_yticks(ytick)
		ax.set_yticklabels(["%.2f" % uu for uu in yR])
		ax.set_ylabel('$R[Mpc]$')

		plt.subplots_adjust(bottom = 0.1, right = 0.8, top = 0.9)
		plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_%d_maskB_%s_band.png' % (stackn, band[ii]), dpi = 300)
		plt.close()
	raise
	'''
	for ii in range(2):
		id_band = ii

		m_img, SB_cc, R_cc, err_cc = stack_light(id_band, stackn, zN, raN, decN, RichN)
		GR_minus.append(m_img)
		Mag_minus.append(SB_cc)
		R_record.append(R_cc)

		with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/test_h5/stack_%d_image_%s_band.h5' % (stackn, band[ii]) ) as f:
			mask_A = np.array(f['a'])

		ox = np.linspace(0, mask_A.shape[1]-1, mask_A.shape[1])
		oy = np.linspace(0, mask_A.shape[0]-1, mask_A.shape[0])
		oo_grd = np.array(np.meshgrid(ox, oy))
		cdr = np.sqrt(((2 * oo_grd[0,:] + 1)/2 - (2 * x0 + 1)/2)**2 + ((2 * oo_grd[1,:] + 1)/2 - (2 * y0 + 1)/2)**2)
		idd = (cdr > (2 * Rpp + 1)/2) & (cdr < 1.1 * (2 * Rpp + 1)/2)
		cut_region = mask_A[idd]
		id_nan = np.isnan(cut_region)
		idx = np.where(id_nan == False)
		bl_array = cut_region[idx]
		back_aft = np.mean(bl_array)
		BL_lel = 22.5 - 2.5 * np.log10(back_aft) + 2.5 * np.log10(pixel**2)
		SB, R, Ar, error = light_measure(mask_A, bins, 1, Rpp, x0, y0, pixel, z_ref)[:4]

		b0 = np.min([np.nanmin(R), np.nanmin(R_cc)])
		b1 = np.max([np.nanmax(R), np.nanmax(R_cc)])
		plt.figure(figsize = (16, 9))
		bx = plt.subplot(111)

		bx.errorbar(R, SB, yerr = error, xerr = None, color = 'g', marker = 's', ls = '', linewidth = 1, markersize = 5, 
			ecolor = 'g', elinewidth = 1, alpha = 0.5, label = '$ BCG + ICL $')
		bx.errorbar(R_cc, SB_cc, yerr = err_cc, xerr = None, color = 'r', marker = 'o', ls = '', linewidth = 1, markersize = 5, 
			ecolor = 'r', elinewidth = 1, alpha = 0.5, label = '$ Total $')
		bx.axhline(y = BL_lel, color = 'b', linestyle = '--', label = '$ background $', alpha = 0.5)
		bx.set_xscale('log')
		bx.set_xlabel('$R[kpc]$')
		bx.set_ylabel('$SB[mag/arcsec^2]$')
		bx.invert_yaxis()
		bx.set_xlim(b0 + 1, b1 + 50)
		bx.tick_params(axis = 'both', which = 'both', direction = 'in')
		bx.legend( loc = 1, fontsize = 12)

		bx1 = bx.twiny()
		xtik = bx.get_xticks(minor = True)
		xR = xtik * 10**(-3) * rad2asec / Da_ref
		xR = xtik * 10**(-3) * rad2asec / Da_ref
		id_tt = xtik >= 9e1 
		bx1.set_xticks(xtik[id_tt])
		bx1.set_xticklabels(["%.2f" % uu for uu in xR[id_tt]])
		bx1.set_xlim(bx.get_xlim())
		bx1.set_xlabel('$R[arcsec]$')
		bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

		plt.savefig(
			'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/stack_%d_img_%s_band_SB_change.png' % (stackn, band[ii]), dpi = 300)
		plt.close()

		print(ii)

	SBt = Mag_minus[1] - Mag_minus[0]
	Rt = np.nanmean(np.array(R_record), axis = 0)

	fig = plt.figure()
	plt.suptitle('$ g-r \,[stacking \, %d \, clusters] $' % stackn, fontsize = 9)
	bx = plt.subplot(111)

	bx.plot(Rt, SBt, 'r-')

	bx.set_xscale('log')
	bx.set_xlabel('$R[kpc]$')
	bx.set_ylabel('$ g-r $')
	bx.set_xlim(np.nanmin(Rt) + 1, np.nanmax(Rt) + 50)
	bx.tick_params(axis = 'both', which = 'both', direction = 'in')

	bx1 = bx.twiny()
	xtik = bx.get_xticks(minor = True)
	xR = xtik * 10**(-3) * rad2asec / Da_ref
	xR = xtik * 10**(-3) * rad2asec / Da_ref
	id_tt = xtik >= 9e1 
	bx1.set_xticks(xtik[id_tt])
	bx1.set_xticklabels(["%.2f" % uu for uu in xR[id_tt]], fontsize = 8)
	bx1.set_xlim(bx.get_xlim())
	bx1.set_xlabel('$R[arcsec]$')
	bx1.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.savefig(
		'/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/stack_img/g-r_tot_light_stack_%d_image.png' % stackn, dpi = 300)
	plt.close()
	'''

if __name__ == "__main__":
	main()
