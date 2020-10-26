import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import glob
import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import scipy.stats as sts
import astropy.units as U
import subprocess as subpro
import astropy.constants as C

from scipy.optimize import curve_fit
from scipy.stats import binned_statistic as binned
from groups import groups_find_func
from astropy import cosmology as apcy
from img_stack import stack_func
from light_measure import light_measure_Z0_weit

### cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel, z_ref = 0.396, 0.250
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # (erg/s)/cm^-2
R0 = 1 # Mpc
Angu_ref = (R0 / Da_ref)*rad2asec
Rpp = Angu_ref / pixel

##############
def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

def cc_grid_img(img_data, N_stepx, N_stepy):

	binx = img_data.shape[1] // N_stepx ## bin number along 2 axis
	biny = img_data.shape[0] // N_stepy

	beyon_x = img_data.shape[1] - binx * N_stepx ## for edge pixels divid
	beyon_y = img_data.shape[0] - biny * N_stepy

	odd_x = np.ceil(beyon_x / binx)
	odd_y = np.ceil(beyon_y / biny)

	n_odd_x = beyon_x // odd_x
	n_odd_y = beyon_y // odd_y

	d_odd_x = beyon_x - odd_x * n_odd_x
	d_odd_y = beyon_y - odd_y * n_odd_y

	# get the bin width
	wid_x = np.zeros(binx, dtype = np.float32)
	wid_y = np.zeros(biny, dtype = np.float32)
	for kk in range(binx):
		if kk == n_odd_x :
			wid_x[kk] = N_stepx + d_odd_x
		elif kk < n_odd_x :
			wid_x[kk] = N_stepx + odd_x
		else:
			wid_x[kk] = N_stepx

	for kk in range(biny):
		if kk == n_odd_y :
			wid_y[kk] = N_stepy + d_odd_y
		elif kk < n_odd_y :
			wid_y[kk] = N_stepy + odd_y
		else:
			wid_y[kk] = N_stepy

	# get the bin edge
	lx = np.zeros(binx + 1, dtype = np.int32)
	ly = np.zeros(biny + 1, dtype = np.int32)
	for kk in range(binx):
		lx[kk + 1] = lx[kk] + wid_x[kk]
	for kk in range(biny):
		ly[kk + 1] = ly[kk] + wid_y[kk]

	patch_mean = np.zeros( (biny, binx), dtype = np.float )
	patch_pix = np.zeros( (biny, binx), dtype = np.float )
	patch_S0 = np.zeros( (biny, binx), dtype = np.float )
	patch_Var = np.zeros( (biny, binx), dtype = np.float )
	for nn in range( biny ):
		for tt in range( binx ):

			sub_flux = img_data[ly[nn]: ly[nn + 1], lx[tt]: lx[tt + 1] ]
			id_nn = np.isnan(sub_flux)

			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )
			patch_S0[nn,tt] = (ly[nn + 1] - ly[nn]) * (lx[tt + 1] - lx[tt])

	return patch_mean, patch_Var, patch_pix, patch_S0

def grid_img(img_data, N_stepx, N_stepy):

	ly = np.arange(0, img_data.shape[0], N_stepy)
	ly = np.r_[ly, img_data.shape[0] - N_stepy, img_data.shape[0] ]
	lx = np.arange(0, img_data.shape[1], N_stepx)
	lx = np.r_[lx, img_data.shape[1] - N_stepx, img_data.shape[1] ]

	lx = np.delete(lx, -1)
	lx = np.delete(lx, -2)
	ly = np.delete(ly, -1)
	ly = np.delete(ly, -2)

	patch_mean = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_pix = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_Var = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	for nn in range( len(ly) ):
		for tt in range( len(lx) ):

			sub_flux = img_data[ly[nn]: ly[nn] + N_stepy, lx[tt]: lx[tt] + N_stepx]
			id_nn = np.isnan(sub_flux)
			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )

	return patch_mean, patch_Var, patch_pix

def get_cat(star_cat, gal_cat):

	## read source catalog
	cat = pds.read_csv(star_cat, skiprows = 1)
	set_ra = np.array(cat['ra'])
	set_dec = np.array(cat['dec'])
	set_mag = np.array(cat['r'])
	OBJ = np.array(cat['type'])
	xt = cat['Column1']
	flags = [str(qq) for qq in xt]

	x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)

	set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float32)

	lln = np.array([len(set_A[:,ll][set_A[:,ll] > 0 ]) for ll in range(set_A.shape[1]) ])
	lr_iso = np.array([np.max(set_A[:,ll]) for ll in range(set_A.shape[1]) ])
	sr_iso = np.array([np.max(set_B[:,ll]) for ll in range(set_B.shape[1]) ])

	# normal stars
	iq = lln >= 2 ## at lest observed in 2 band
	ig = OBJ == 6
	ie = (set_mag <= 20)

	ic = (ie & ig & iq)
	sub_x0 = x[ic]
	sub_y0 = y[ic]
	sub_A0 = lr_iso[ic] * 30
	sub_B0 = sr_iso[ic] * 30
	sub_chi0 = set_chi[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in flags]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]
	sub_A2 = lr_iso[ipx] * 75
	sub_B2 = sr_iso[ipx] * 75
	sub_chi2 = set_chi[ipx]

	comx = np.r_[sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0]]
	comy = np.r_[sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0]]
	Lr = np.r_[sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0]]
	Sr = np.r_[sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0]]
	phi = np.r_[sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0]]

	source = asc.read(gal_cat)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1

	Kron = 16
	a = Kron * A
	b = Kron * B

	tot_cx = np.r_[cx, comx]
	tot_cy = np.r_[cy, comy]
	tot_a = np.r_[a, Lr]
	tot_b = np.r_[b, Sr]
	tot_theta = np.r_[theta, phi]
	tot_Numb = Numb + len(comx)

	return tot_Numb, tot_cx, tot_cy, tot_a, tot_b, tot_theta

########################## samples (imgs) overview
band = ['r', 'g', 'i']
home = '/media/xkchen/My Passport/data/SDSS/'

### cat and select
dat = pds.read_csv('/home/xkchen/tmp/02_tot_test_change_1_selection/cluster_tot-r-band_norm-img_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

set_ra, set_dec, set_z = ra, dec, z
set_x, set_y = clus_x, clus_y

N_samp = len(set_z)
'''
cen_sigm, cen_mu = [], []
img_mu, img_sigm = [], []
### overview on img structure
for kk in range( N_samp ):

	ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]
	# original img
	file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)
	data = fits.open(file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)
	xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	# mask imgs
	res_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % ('r', ra_g, dec_g, z_g)
	res_data = fits.open(res_file)
	remain_img = res_data[0].data
	devi_img = remain_img - np.nanmean( remain_img )

	img_mu.append( np.nanmean(remain_img) )
	img_sigm.append( np.nanstd(remain_img) )

	# mask matrix
	idnn = np.isnan(remain_img)
	mask_arr = np.zeros((remain_img.shape[0], remain_img.shape[1]), dtype = np.float32)
	mask_arr[idnn == False] = 1

	ca0, ca1 = np.int(img.shape[0] / 2), np.int(img.shape[1] / 2)
	cen_D = 500
	flux_cen = remain_img[ca0 - cen_D: ca0 + cen_D, ca1 - cen_D: ca1 + cen_D]

	N_step = 200

	cen_lx = np.arange(0, 1100, N_step)
	cen_ly = np.arange(0, 1100, N_step)
	nl0, nl1 = len(cen_ly), len(cen_lx)

	sub_pock_pix = np.zeros((nl0 - 1, nl1 - 1), dtype = np.float)
	sub_pock_flux = np.zeros((nl0 - 1, nl1 - 1), dtype = np.float)
	for nn in range(nl0 - 1):
		for tt in range(nl1 - 1):
			sub_flux = flux_cen[ cen_ly[nn]: cen_ly[nn+1], cen_lx[tt]: cen_lx[tt+1] ]
			id_nn = np.isnan(sub_flux)
			sub_pock_flux[nn,tt] = np.nanmean(sub_flux)
			sub_pock_pix[nn,tt] = len(sub_flux[id_nn == False])

	## mu, sigma of center region
	id_Nzero = sub_pock_pix > 100
	mu = np.nanmean( sub_pock_flux[id_Nzero] )
	sigm = np.nanstd( sub_pock_flux[id_Nzero] )

	cen_sigm.append(sigm)
	cen_mu.append(mu)

	# grid img (for selecting flare, saturated region...)
	#block_m = grid_img(remain_img, N_step, N_step)[0]
	block_m, block_Var, block_pix, block_S0 = cc_grid_img(remain_img, N_step, N_step)

	idzo = block_pix < 1.
	pix_eta = block_pix / block_S0
	idnn = np.isnan(pix_eta)
	pix_eta[idnn] = 0.
	idnul = pix_eta < 5e-2
	block_m[idnul] = 0.

	idnn = np.isnan(remain_img)
	bin_flux = remain_img[idnn == False]
	bin_di = np.linspace(bin_flux.min(), bin_flux.max(), 51) / pixel**2

	pix_n, edgs = binned(bin_flux / pixel**2, bin_flux / pixel**2, statistic = 'count', bins = bin_di)[:2]
	pdf_pix = (pix_n / np.sum(pix_n) ) / (edgs[1] - edgs[0])
	pdf_err = (np.sqrt(pix_n) / np.sum(pix_n) ) / (edgs[1] - edgs[0])
	x_cen = 0.5 * ( edgs[1:] + edgs[:-1])

	idu = pix_n != 0.
	use_obs = pix_n[idu]
	use_err = np.sqrt(use_obs)
	use_x = x_cen[idu]
	popt, pcov = curve_fit(gau_func, use_x, pdf_pix[idu], 
		p0 = [np.mean(bin_flux / pixel**2), np.std(bin_flux / pixel**2)], sigma = pdf_err[idu],)
	e_mu, e_chi = popt[0], popt[1]
	fit_line = gau_func(x_cen, e_mu, e_chi)
	"""
	# creat compare matrix for sub-patch 2D hist (for grid_img case)
	Lx, Ly = np.int(N_step * block_m.shape[1]), np.int(N_step * block_m.shape[0])
	cc_mask = np.zeros((Ly, Lx), dtype = np.float32)
	dNy, dNx = img.shape[0] // N_step, img.shape[1] // N_step
	cc_mask[:np.int(dNy * N_step), :np.int(dNx * N_step)] = mask_arr[:np.int(dNy * N_step), :np.int(dNx * N_step)]

	cc_mask[-N_step:, -N_step:] = mask_arr[-N_step:, -N_step:]
	cc_mask[:np.int(dNy * N_step), -N_step:] = mask_arr[:np.int(dNy * N_step), -N_step:]
	cc_mask[-N_step:, :np.int(dNx * N_step)] = mask_arr[-N_step:, :np.int(dNx * N_step)]

	copy_blockm = np.zeros((Ly, Lx), dtype = np.float32)

	for ll in range(Ly):
		for mm in range(Lx):
			idy = ll // N_step
			idx = mm // N_step
			copy_blockm[ll, mm] = block_m[idy, idx]

	nlx = np.linspace(0, block_m.shape[1] - 1, cc_mask.shape[1])
	nly = np.linspace(0, block_m.shape[0] - 1, cc_mask.shape[0])
	"""
	nlx = np.linspace(-0.5, block_m.shape[1] - 0.5, mask_arr.shape[1])
	nly = np.linspace(-0.5, block_m.shape[0] - 0.5, mask_arr.shape[0])
	"""
	### applied the mask region
	star_cat = '/home/xkchen/mywork/ICL/data/corrected_star_cat/dr12/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g)
	gal_cat = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/cluster_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (ra_g, dec_g, z_g)
	tot_Numb, tot_cx, tot_cy, tot_a, tot_b, tot_theta = get_cat(star_cat, gal_cat)
	sc_x, sc_y = tot_cx / (img.shape[1] / block_m.shape[1]), tot_cy / (img.shape[0] / block_m.shape[0])
	sc_a, sc_b = tot_a * (block_m.shape[1] / img.shape[1]), tot_b * (block_m.shape[0] / img.shape[0])
	sc_x, sc_y = sc_x - 0.5, sc_y - 0.5

	fig = plt.figure( figsize = (13.12, 9.84) )
	ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.45])
	ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.45])
	ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
	ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.40])

	ax0.set_title('img ra%.3f dec%.3f z%.3f' % (ra_g, dec_g, z_g),)
	tf = ax0.imshow(img / pixel**2, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e0, norm = mpl.colors.LogNorm())
	clust = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.50,)
	ax0.add_patch(clust)
	cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)

	ax1.set_title('after masking')
	tg = ax1.imshow( remain_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-1, vmax = 4e-1,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))

	### image patch_mean case
	ax2.set_title('2D hist of sub-patch mean value')
	th = ax2.imshow( block_m / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(th, ax = ax2, fraction = 0.034, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits((0,0))

	#ax2.contour(nlx, nly, mask_arr, origin = 'lower', colors = 'k', levels = [1], alpha = 0.5, linewidths = 0.75,)
	for mm in range( tot_Numb ):
		ellips = Ellipse(xy = (sc_x[mm], sc_y[mm]), width = sc_a[mm], height = sc_b[mm], angle = tot_theta[mm], fill = True, fc = 'w', 
			ec = 'w', ls = '-', linewidth = 0.75,)
		ax2.add_patch(ellips)

	for mm in range(block_m.shape[1]):
		for nn in range(block_m.shape[0]):
			ax2.text(mm, nn, s = '%.3f' % pix_eta[nn, mm], ha = 'center', va = 'center', color = 'g', fontsize = 8, alpha = 0.5)
	ax2.set_xlabel('effective pixel ratio shown in green text')

	ax3.set_title('pixel SB PDF [after masking]')
	ax3.hist(bin_flux / pixel**2, bins = bin_di, density = True, color = 'b', alpha = 0.5,)
	ax3.plot(x_cen, fit_line, color = 'r', alpha = 0.5, label = 'Gaussian \n $\\mu=%.4f$ \n $\\sigma=%.4f$' % (e_mu, e_chi),)
	ax3.axvline(x = 0, ls = '-', color = 'k', alpha = 0.5,)
	ax3.axvline(x = e_mu - e_chi, ls = '--', color = 'k', alpha = 0.5, label = '1 $\\sigma$')
	ax3.axvline(x = e_mu + e_chi, ls = '--', color = 'k', alpha = 0.5, )
	ax3.legend(loc = 1, frameon = False,)
	ax3.set_xlabel('pixel SB [nanomaggies / $arcsec^2$]')
	ax3.set_ylabel('PDF')

	plt.savefig('result/2D_img_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
	plt.close()
	"""
cen_sigm = np.array(cen_sigm)
cen_mu = np.array(cen_mu)
img_mu = np.array(img_mu)
img_sigm = np.array(img_sigm)

keys = ['cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
values = [cen_mu, cen_sigm, img_mu, img_sigm]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('img_3100_mean_sigm.csv')


fig = plt.figure( figsize = (13.12, 9.84) )
ax0 = fig.add_axes([0.05, 0.50, 0.40, 0.40])
ax1 = fig.add_axes([0.55, 0.50, 0.40, 0.40])
ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.40])
ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.40])

ax0.hist(cen_sigm / pixel**2, bins = 25, density = True, color = 'b', alpha = 0.5,)
ax0.axvline(x = np.nanmean(cen_sigm / pixel**2), ls = '--', color = 'k', alpha = 0.5, label = 'Mean')
ax0.axvline(x = np.nanmedian(cen_sigm / pixel**2), ls = '-', color = 'k', alpha = 0.5, label = 'Median')
ax0.legend(loc = 1, frameon = False,)
ax0.set_xlabel('$\\sigma_{center}$ [nanomaggies / $arcsec^2$]')
ax0.set_ylabel('PDF')

ax1.hist(cen_mu / pixel**2, bins = 25, density = True, color = 'b', alpha = 0.5,)
ax1.axvline(x = np.nanmean(cen_mu / pixel**2), ls = '--', color = 'k', alpha = 0.5, label = 'Mean')
ax1.axvline(x = np.nanmedian(cen_mu / pixel**2), ls = '-', color = 'k', alpha = 0.5, label = 'Median')
ax1.legend(loc = 1, frameon = False,)
ax1.set_xlabel('$\\mu_{center}$ [nanomaggies / $arcsec^2$]')
ax1.set_ylabel('PDF')

ax2.hist(img_sigm / pixel**2, bins = 25, density = True, color = 'b', alpha = 0.5,)
ax2.axvline(x = np.nanmean(img_sigm / pixel**2), ls = '--', color = 'k', alpha = 0.5, label = 'Mean')
ax2.axvline(x = np.nanmedian(img_sigm / pixel**2), ls = '-', color = 'k', alpha = 0.5, label = 'Median')
ax2.legend(loc = 1, frameon = False,)
ax2.set_xlabel('$\\sigma_{img}$ [nanomaggies / $arcsec^2$]')
ax2.set_ylabel('PDF')

ax3.hist(img_mu / pixel**2, bins = 25, density = True, color = 'b', alpha = 0.5,)
ax3.axvline(x = np.nanmean(img_mu / pixel**2), ls = '--', color = 'k', alpha = 0.5, label = 'Mean')
ax3.axvline(x = np.nanmedian(img_mu / pixel**2), ls = '-', color = 'k', alpha = 0.5, label = 'Median')
ax3.legend(loc = 1, frameon = False,)
ax3.set_xlabel('$\\mu_{img}$ [nanomaggies / $arcsec^2$]')
ax3.set_ylabel('PDF')

plt.savefig('flux_hist.png', dpi = 300)
plt.close()
'''

########################## SB test part
'''
## 39 sample (2020.9.24~25)
out_ra = ['6.001', '9.238', '9.739', '21.658', '33.369', '34.610', '125.136', '132.645', '134.450',
		'137.163', '148.208', '156.075', '156.188', '158.286', '170.613', '178.530', '179.505', 
		'193.279', '199.231', '205.530', '213.029', '215.959', '217.692', '229.142', '248.456',
		'253.145', '326.315', '332.841', '27.022', 

		'324.935', '248.790', '229.742', '205.611', '208.942', '162.665', '186.550', '244.879', 
		'145.603', '172.837', ]

out_dec = ['0.502', '6.801', '-0.449', '1.430', '1.001', '-7.232', '3.695', '53.976', '59.490', 
		'6.998', '28.131', '-0.165', '21.535', '-1.860', '25.190', '10.189', '48.772', '62.822',
		'26.258', '65.739', '23.294', '18.468', '7.518', '1.510', '11.057', '17.526', '9.175', '6.071', 
		'-2.217', 

		'0.960', '28.455', '54.013', '4.844', '30.954', '9.635', '58.681', '7.299', 
		'15.281', '33.580', ]
'''
## additional sample (2020.9.25~)
#dd_dat = pds.read_csv('result/test_1000_no_select.csv')
dd_dat = pds.read_csv('result/test_1000-to-250_cat.csv')
top_ra, top_dec, top_z = np.array(dd_dat.ra), np.array(dd_dat.dec), np.array(dd_dat.z)
tt_ra = ['%.5f' % ll for ll in top_ra]
tt_dec = ['%.5f' % ll for ll in top_dec]

cen_mu, cen_sigm = [], []
img_mu, img_sigm = [], []
samp_dat = pds.read_csv('result/img_3100_mean_sigm.csv')
tmp_mu, tmp_sigm = np.array(samp_dat['img_mu']), np.array(samp_dat['img_sigma'])
tmp_cen_mu, tmp_cen_sigm = np.array(samp_dat['cen_mu']), np.array(samp_dat['cen_sigma'])

dd_ra, dd_dec, dd_z = [], [], []
dd_imgx, dd_imgy = [], []
for kk in range( N_samp ):
	if ('%.5f' % set_ra[kk] in tt_ra) & ('%.5f' % set_dec[kk] in tt_dec):
		dd_ra.append(set_ra[kk])
		dd_dec.append(set_dec[kk])
		dd_z.append(set_z[kk])
		dd_imgx.append(set_x[kk])
		dd_imgy.append(set_y[kk])

		cen_mu.append(tmp_cen_mu[kk])
		cen_sigm.append(tmp_cen_sigm[kk])
		img_mu.append(tmp_mu[kk])
		img_sigm.append(tmp_sigm[kk])
	else:
		continue

cen_sigm = np.array(cen_sigm)
cen_mu = np.array(cen_mu)
img_mu = np.array(img_mu)
img_sigm = np.array(img_sigm)

keys = ['cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
values = [cen_mu, cen_sigm, img_mu, img_sigm]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
#data.to_csv('img_test-1000_mean_sigm.csv')
data.to_csv('img_A-250_mean_sigm.csv')

raise

dd_ra = np.array(dd_ra)
dd_dec = np.array(dd_dec)
dd_z = np.array(dd_z)

dd_imgx = np.array(dd_imgx)
dd_imgy = np.array(dd_imgy)

def cat_match(ra_list, dec_list, cat_ra, cat_dec, cat_z, cat_imgx, cat_imgy, id_choice = True):

	lis_ra, lis_dec, lis_z = [], [], []
	lis_x, lis_y = [], []

	if id_choice == True:
		for kk in range( len(cat_ra) ):
			if ('%.3f' % cat_ra[kk] in ra_list) * ('%.3f' % cat_dec[kk] in dec_list):
				lis_ra.append(cat_ra[kk])
				lis_dec.append(cat_dec[kk])
				lis_z.append(cat_z[kk])
				lis_x.append(cat_imgx[kk])
				lis_y.append(cat_imgy[kk])
			else:
				continue
	else:
		for kk in range( len(cat_ra) ):
			if ('%.3f' % cat_ra[kk] in ra_list) * ('%.3f' % cat_dec[kk] in dec_list):
				continue
			else:
				lis_ra.append(cat_ra[kk])
				lis_dec.append(cat_dec[kk])
				lis_z.append(cat_z[kk])
				lis_x.append(cat_imgx[kk])
				lis_y.append(cat_imgy[kk])

	match_ra = np.array(lis_ra)
	match_dec = np.array(lis_dec)
	match_z = np.array(lis_z)
	match_x = np.array(lis_x)
	match_y = np.array(lis_y)

	return match_ra, match_dec, match_z, match_x, match_y

lis = glob.glob('/home/xkchen/tmp/20_9_26/A_norm/*.png')
name_lis = [ ll.split('/')[-1] for ll in lis ]
out_ra = [ ll.split('_')[2][2:] for ll in name_lis ]
out_dec = [ ll.split('_')[3][3:] for ll in name_lis ]
'''
## add A test
add_lis = glob.glob('/home/xkchen/tmp/20_9_26/add_A_test/*.png')
add_name = [ ll.split('/')[-1] for ll in add_lis ]
ext_ra = [ ll.split('_')[2][2:] for ll in add_name ]
ext_dec = [ ll.split('_')[3][3:] for ll in add_name ]
out_ra = out_ra + ext_ra
out_dec = out_dec + ext_dec
'''
lis_ra, lis_dec, lis_z, lis_x, lis_y = cat_match(out_ra, out_dec, dd_ra, dd_dec, dd_z, dd_imgx, dd_imgy, )
keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
values = [lis_ra, lis_dec, lis_z, lis_x, lis_y]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('result/test_1000-to-%d_cat.csv' % (len(lis_ra),) )

## the other part (img not ideal goodness)
pp_ra, pp_dec, pp_z, pp_imgx, pp_imgy = cat_match(out_ra, out_dec, dd_ra, dd_dec, dd_z, dd_imgx, dd_imgy, id_choice = False)

## seperate sub-sample case (main for "the other part" img)
B_lis = glob.glob('/home/xkchen/tmp/20_9_26/B_norm/*.png')
B_name_lis = [ ll.split('/')[-1] for ll in B_lis ]
Bpart_ra = [ ll.split('_')[2][2:] for ll in B_name_lis ]
Bpart_dec = [ ll.split('_')[3][3:] for ll in B_name_lis ]
Blis_ra, Blis_dec, Blis_z, Blis_x, Blis_y = cat_match(Bpart_ra, Bpart_dec, dd_ra, dd_dec, dd_z, dd_imgx, dd_imgy, )

C_lis = glob.glob('/home/xkchen/tmp/20_9_26/C_norm/*.png')
C_name_lis = [ ll.split('/')[-1] for ll in C_lis ]
Cpart_ra = [ ll.split('_')[2][2:] for ll in C_name_lis ]
Cpart_dec = [ ll.split('_')[3][3:] for ll in C_name_lis ]
Clis_ra, Clis_dec, Clis_z, Clis_x, Clis_y = cat_match(Cpart_ra, Cpart_dec, dd_ra, dd_dec, dd_z, dd_imgx, dd_imgy, )

D_lis = glob.glob('/home/xkchen/tmp/20_9_26/D_norm/*.png')
D_name_lis = [ ll.split('/')[-1] for ll in D_lis ]
Dpart_ra = [ ll.split('_')[2][2:] for ll in D_name_lis ]
Dpart_dec = [ ll.split('_')[3][3:] for ll in D_name_lis ]
Dlis_ra, Dlis_dec, Dlis_z, Dlis_x, Dlis_y = cat_match(Dpart_ra, Dpart_dec, dd_ra, dd_dec, dd_z, dd_imgx, dd_imgy, )
'''
keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
values = [Blis_ra, Blis_dec, Blis_z, Blis_x, Blis_y]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('result/test_1000-to-%d_cat.csv' % (len(Blis_ra),) )

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
values = [Clis_ra, Clis_dec, Clis_z, Clis_x, Clis_y]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('result/test_1000-to-%d_cat.csv' % (len(Clis_ra),) )

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
values = [Dlis_ra, Dlis_dec, Dlis_z, Dlis_x, Dlis_y]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('result/test_1000-to-%d_cat.csv' % (len(Dlis_ra),) )
'''
'''
### sample properties match
from img_cat_param_match import match_func
cat_file = '/home/xkchen/mywork/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits'
out_file = 'result/test_1000_no_select_cat-match.csv'
match_func(dd_ra, dd_dec, dd_z, cat_file, out_file)

out_file = 'result/test_1000-to-250_cat-match.csv'
match_func(lis_ra, lis_dec, lis_z, cat_file, out_file)
'''
raise
## stacking imgs
dfile = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
id_cen = 0 # BCG-stacking

tot_file = 'result/test_BCG-stack_%d.h5' % (len(dd_ra),)
tot_pix_cont = 'result/test_BCG-stack_pix-cont_%d.h5' % (len(dd_ra),)
stack_func(dfile, tot_file, dd_z, dd_ra, dd_dec, band[0], dd_imgx, dd_imgy, id_cen, rms_file = None, pix_con_file = tot_pix_cont)

sel_file = 'result/test_BCG-stack_select_%d.h5' % (len(lis_z),)
sel_pix_cont = 'result/test_BCG-stack_pix-cont_select_%d.h5' % (len(lis_z),)
stack_func(dfile, sel_file, lis_z, lis_ra, lis_dec, band[0], lis_x, lis_y, id_cen, rms_file = None, pix_con_file = sel_pix_cont)

dout_file = 'result/test_BCG-stack_dout_%d.h5' % (len(pp_ra),)
dout_pix_cont = 'result/test_BCG-stack_pix-cont_dout_%d.h5' % (len(pp_ra),)
stack_func(dfile, dout_file, pp_z, pp_ra, pp_dec, band[0], pp_imgx, pp_imgy, id_cen, rms_file = None, pix_con_file = dout_pix_cont)

Bsub_file = 'result/test_BCG-stack_B-out_%d.h5' % (len(Blis_ra),)
Bsub_pix_cont = 'result/test_BCG-stack_B-out_pix-cont_%d.h5' % (len(Blis_ra),)
stack_func(dfile, Bsub_file, Blis_z, Blis_ra, Blis_dec, band[0], Blis_x, Blis_y, id_cen, rms_file = None, pix_con_file = Bsub_pix_cont)

Csub_file = 'result/test_BCG-stack_C-out_%d.h5' % (len(Clis_ra),)
Csub_pix_cont = 'result/test_BCG-stack_C-out_pix-cont_%d.h5' % (len(Clis_ra),)
stack_func(dfile, Csub_file, Clis_z, Clis_ra, Clis_dec, band[0], Clis_x, Clis_y, id_cen, rms_file = None, pix_con_file = Csub_pix_cont)

Dsub_file = 'result/test_BCG-stack_D-out_%d.h5' % (len(Dlis_ra),)
Dsub_pix_cont = 'result/test_BCG-stack_D-out_pix-cont_%d.h5' % (len(Dlis_ra),)
stack_func(dfile, Dsub_file, Dlis_z, Dlis_ra, Dlis_dec, band[0], Dlis_x, Dlis_y, id_cen, rms_file = None, pix_con_file = Dsub_pix_cont)

## adjust star mask size (for  A, B and C(maybe) sub-sample, )
size_arr = np.array([5, 10, 15, 20, 25])
s_adj_ra, s_adj_dec, s_adj_z = np.r_[lis_ra, Blis_ra], np.r_[lis_dec, Blis_dec], np.r_[lis_z, Blis_z]
s_adj_x, s_adj_y = np.r_[lis_x, Blis_x], np.r_[lis_y, Blis_y]
for mm in range(5):
	A_file = home + '20_10_test/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_' + '%d-FWHM-ov2.fits' % (size_arr[mm])
	alt_file = 'result/test_BCG-stack_AB-%d_%d-FWHM-ov2.h5' % (len(s_adj_z), size_arr[mm])
	alt_pix_cont = 'result/test_BCG-stack_pix-cont_AB-%d_%d-FWHM-ov2.h5' % (len(s_adj_z), size_arr[mm])
	stack_func(A_file, alt_file, s_adj_z, s_adj_ra, s_adj_dec, band[0], s_adj_x, s_adj_y, id_cen, 
		rms_file = None, pix_con_file = alt_pix_cont,)

## compare (stacking img and SB profile)
with h5py.File(tot_file, 'r') as f:
	tot_img = np.array(f['a'])
with h5py.File(tot_pix_cont, 'r') as f:
	tot_cont = np.array(f['a'])

xn, yn = np.int(tot_img.shape[1] / 2), np.int(tot_img.shape[0] / 2)
id_nn = np.isnan(tot_img)
eff_y, eff_x = np.where(id_nn == False)
dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
dR_max = np.int( dR.max() ) + 1

rbins = np.logspace(0, np.log10(dR_max), 110)
# total case
sb_arr, r_arr, sb_err_arr, npix, nratio = light_measure_Z0_weit(tot_img, tot_cont, pixel, xn, yn, rbins)
idzo = npix < 1
r_arr, sb_arr, sb_err_arr = r_arr[idzo == False], sb_arr[idzo == False], sb_err_arr[idzo == False]
sb_arr, sb_err_arr = sb_arr / pixel**2, sb_err_arr / pixel**2

block_tot = cc_grid_img(tot_img, 100, 100)[0]

# goodness case
with h5py.File(sel_file, 'r') as f:
	sel_img = np.array(f['a'])
with h5py.File(sel_pix_cont, 'r') as f:
	sel_cont = np.array(f['a'])

c_sb_arr, c_r_arr, c_sb_err_arr, c_npix, c_nratio = light_measure_Z0_weit(sel_img, sel_cont, pixel, xn, yn, rbins)
idzo = c_npix < 1
c_r_arr, c_sb_arr, c_sb_err_arr = c_r_arr[idzo == False], c_sb_arr[idzo == False], c_sb_err_arr[idzo == False]
c_sb_arr, c_sb_err_arr = c_sb_arr / pixel**2, c_sb_err_arr / pixel**2

block_sel = cc_grid_img(sel_img, 100, 100)[0]

# total rule out imgs
with h5py.File(dout_file, 'r') as f:
	dout_img = np.array(f['a'])
with h5py.File(dout_pix_cont, 'r') as f:
	dout_cont = np.array(f['a'])

d_sb_arr, d_r_arr, d_sb_err_arr, d_npix, d_nratio = light_measure_Z0_weit(dout_img, dout_cont, pixel, xn, yn, rbins)
idzo = d_npix < 1
d_r_arr, d_sb_arr, d_sb_err_arr = d_r_arr[idzo == False], d_sb_arr[idzo == False], d_sb_err_arr[idzo == False]
d_sb_arr, d_sb_err_arr = d_sb_arr / pixel**2, d_sb_err_arr / pixel**2

block_dout = cc_grid_img(dout_img, 100, 100)[0]

# sub-sample of bad imgs
with h5py.File(Bsub_file, 'r') as f:
	Bsub_img = np.array(f['a'])
with h5py.File(Bsub_pix_cont, 'r') as f:
	Bsub_cont = np.array(f['a'])

Blis_sb, Blis_r, Blis_sb_err, Blis_npix, Blis_nratio = light_measure_Z0_weit(Bsub_img, Bsub_cont, pixel, xn, yn, rbins)
idzo = Blis_npix < 1
Blis_r, Blis_sb, Blis_sb_err = Blis_r[idzo == False], Blis_sb[idzo == False], Blis_sb_err[idzo == False]
Blis_sb, Blis_sb_err = Blis_sb / pixel**2, Blis_sb_err / pixel**2
block_Bsub = cc_grid_img(Bsub_img, 100, 100)[0]

with h5py.File(Csub_file, 'r') as f:
	Csub_img = np.array(f['a'])
with h5py.File(Csub_pix_cont, 'r') as f:
	Csub_cont = np.array(f['a'])

Clis_sb, Clis_r, Clis_sb_err, Clis_npix, Clis_nratio = light_measure_Z0_weit(Csub_img, Csub_cont, pixel, xn, yn, rbins)
idzo = Clis_npix < 1
Clis_r, Clis_sb, Clis_sb_err = Clis_r[idzo == False], Clis_sb[idzo == False], Clis_sb_err[idzo == False]
Clis_sb, Clis_sb_err = Clis_sb / pixel**2, Clis_sb_err / pixel**2
block_Csub = cc_grid_img(Csub_img, 100, 100)[0]

with h5py.File(Dsub_file, 'r') as f:
	Dsub_img = np.array(f['a'])
with h5py.File(Dsub_pix_cont, 'r') as f:
	Dsub_cont = np.array(f['a'])

Dlis_sb, Dlis_r, Dlis_sb_err, Dlis_npix, Dlis_nratio = light_measure_Z0_weit(Dsub_img, Dsub_cont, pixel, xn, yn, rbins)
idzo = Dlis_npix < 1
Dlis_r, Dlis_sb, Dlis_sb_err = Dlis_r[idzo == False], Dlis_sb[idzo == False], Dlis_sb_err[idzo == False]
Dlis_sb, Dlis_sb_err = Dlis_sb / pixel**2, Dlis_sb_err / pixel**2
block_Dsub = cc_grid_img(Dsub_img, 100, 100)[0]


## combine stacking image of sub-samples
cc_stack_img = np.zeros((Bsub_img.shape[0], Bsub_img.shape[1]), dtype = np.float32)
cc_pix_cont = np.zeros((Bsub_img.shape[0], Bsub_img.shape[1]), dtype = np.float32)

id_nn0 = np.isnan(Bsub_img)
weit_img0 = Bsub_img * Bsub_cont
cc_stack_img[id_nn0 == False] = cc_stack_img[id_nn0 == False] + weit_img0[id_nn0 == False]
cc_pix_cont[id_nn0 == False] = cc_pix_cont[id_nn0 == False] + Bsub_cont[id_nn0 == False]

id_nn0 = np.isnan(Csub_img)
weit_img0 = Csub_img * Csub_cont
cc_stack_img[id_nn0 == False] = cc_stack_img[id_nn0 == False] + weit_img0[id_nn0 == False]
cc_pix_cont[id_nn0 == False] = cc_pix_cont[id_nn0 == False] + Csub_cont[id_nn0 == False]

id_nn1 = np.isnan(sel_img)
weit_img1 = sel_img * sel_cont
cc_stack_img[id_nn1 == False] = cc_stack_img[id_nn1 == False] + weit_img1[id_nn1 == False]
cc_pix_cont[id_nn1 == False] = cc_pix_cont[id_nn1 == False] + sel_cont[id_nn1 == False]

id_zero = cc_pix_cont < 1.
cc_stack_img[id_zero] = np.nan
cc_pix_cont[id_zero] = np.nan
cc_stack_img = cc_stack_img / cc_pix_cont

#with h5py.File('result/combine_A-B_stack_%d-imgs.h5' % (348), 'w') as f:
with h5py.File('result/combine_A-B-C_stack_%d-imgs.h5' % (541), 'w') as f:
	f['a'] = np.array(cc_stack_img)
#with h5py.File('result/combine_A-B_stack_pix-cont_%d.h5' % (348), 'w') as f:
with h5py.File('result/combine_A-B-C_stack_pix-cont_%d.h5' % (541), 'w') as f:
	f['a'] = np.array(cc_pix_cont)

comb_sb, comb_r, comb_sb_err, comb_npix, comb_nratio = light_measure_Z0_weit(cc_stack_img, cc_pix_cont, pixel, xn, yn, rbins)
idzo = comb_npix < 1
comb_r, comb_sb, comb_sb_err = comb_r[idzo == False], comb_sb[idzo == False], comb_sb_err[idzo == False]
comb_sb, comb_sb_err = comb_sb / pixel**2, comb_sb_err / pixel**2
block_comb = cc_grid_img(cc_stack_img, 100, 100)[0]

'''
fig = plt.figure( figsize = (13.12, 9.84) )
ax0 = fig.add_axes([0.05, 0.55, 0.40, 0.45])
ax1 = fig.add_axes([0.55, 0.55, 0.40, 0.45])
ax2 = fig.add_axes([0.05, 0.05, 0.40, 0.45])
ax3 = fig.add_axes([0.55, 0.05, 0.40, 0.40])

ax0.set_title('bad sample [%d imgs]' % (len(pp_ra),),)
tf = ax0.imshow(block_dout / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits((0,0))

ax1.set_title('B type sample [%d imgs]' % (len(Blis_ra),),)
tf = ax1.imshow(block_Bsub / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tf, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits((0,0))

ax2.set_title('C type sample [%d imgs]' % (len(Clis_ra),),)
tf = ax2.imshow(block_Csub / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tf, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits((0,0))

ax3.set_title('D type sample [%d imgs]' % (len(Dlis_ra),),)
tf = ax3.imshow(block_Dsub / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tf, ax = ax3, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits((0,0))

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('the_bad_imgs.png', dpi = 300)
plt.close()
'''

fig = plt.figure( figsize = (19.84, 4.8) )
ax0 = fig.add_axes([0.03, 0.09, 0.27, 0.85])
ax1 = fig.add_axes([0.36, 0.09, 0.27, 0.85])
ax2 = fig.add_axes([0.69, 0.09, 0.27, 0.85])

ax0.set_title('total %d imgs' % (len(dd_ra),),)
tf = ax0.imshow(block_tot / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits((0,0))

#ax1.set_title('selected %d imgs' % (len(lis_ra),),)
#tg = ax1.imshow(block_sel / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
#ax1.set_title('A + B [%d imgs]' % (348), )
ax1.set_title('A + B + C [%d imgs]' % (541), )
tg = ax1.imshow(block_comb / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)

cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits((0,0))

clust = Circle(xy = (24, 17), radius = 20, fill = False, ec = 'b', ls = '--', linewidth = 1, alpha = 0.45,)
ax1.add_patch(clust)
clust = Circle(xy = (24, 17), radius = 22, fill = False, ec = 'b', ls = '-', linewidth = 1, alpha = 0.45,)
ax1.add_patch(clust)
clust = Circle(xy = (24, 17), radius = 17, fill = False, ec = 'b', ls = '-.', linewidth = 1, alpha = 0.45,)
ax1.add_patch(clust)

#ax2.set_title('total - selected')
#th = ax2.imshow( (block_tot - block_sel) / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-3, vmax = 4e-3,)
ax2.set_title('total - (A + B)')
th = ax2.imshow( (block_tot - block_comb) / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-3, vmax = 4e-3,)

cb = plt.colorbar(th, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits((0,0))

plt.savefig('bcg_2D_img.png', dpi = 300)
plt.close()


plt.figure()
ax0 = plt.subplot(111)

ax0.errorbar(r_arr, sb_arr, yerr = sb_err_arr, xerr = None, color = 'r', marker = 'None', ls = '-', ecolor = 'r',
	alpha = 0.5, label = 'A+B+C+D [%d imgs]' % (len(dd_ra),) )#'total %d imgs' % (len(dd_ra),) )
ax0.errorbar(c_r_arr, c_sb_arr, yerr = c_sb_err_arr, xerr = None, color = 'b', marker = 'None', ls = '-', ecolor = 'b',
	alpha = 0.5, label = 'A type [%d imgs]' % (len(lis_ra),) ) #'selected %d imgs' % (len(lis_ra),) )
ax0.errorbar(d_r_arr, d_sb_arr, yerr = d_sb_err_arr, xerr = None, color = 'g', marker = 'None', ls = '-', ecolor = 'g',
	alpha = 0.5, label = 'B+C+D [%d imgs]' % (len(pp_ra),) ) #'the other %d imgs' % (len(pp_ra),) )

ax0.plot(Blis_r, Blis_sb, color = 'g', ls = ':', alpha = 0.8, label = 'B type [%d imgs]' % (len(Blis_ra),) )
ax0.fill_between(Blis_r, y1 = Blis_sb - Blis_sb_err, y2 = Blis_sb + Blis_sb_err, color = 'g', alpha = 0.2,)
ax0.plot(Clis_r, Clis_sb, color = 'g', ls = '-.', alpha = 0.8, label = 'C type [%d imgs]' % (len(Clis_ra),) )
ax0.fill_between(Clis_r, y1 = Clis_sb - Clis_sb_err, y2 = Clis_sb + Clis_sb_err, color = 'g', alpha = 0.2,)
ax0.plot(Dlis_r, Dlis_sb, color = 'g', ls = '--', alpha = 0.8, label = 'D type [%d imgs]' % (len(Dlis_ra),) )
ax0.fill_between(Dlis_r, y1 = Dlis_sb - Dlis_sb_err, y2 = Dlis_sb + Dlis_sb_err, color = 'g', alpha = 0.2,)

ax0.plot(comb_r, comb_sb, color = 'c', ls = '-', alpha = 0.8, label = 'A+B+C [%d imgs]' % (541),) #label = 'A+B [%d imgs]' % (348),) 
ax0.fill_between(comb_r, y1 = comb_sb - comb_sb_err, y2 = comb_sb + comb_sb_err, color = 'c', alpha = 0.2,)
#ax0.axvline(x = 700, ls = '-.', color = 'k', alpha = 0.5,)
#ax0.axvline(x = 800, ls = '--', color = 'k', alpha = 0.5,)
#ax0.axvline(x = 900, ls = '-', color = 'k', alpha = 0.5,)

ax0.set_ylim(1e-3, 3e-2)
ax0.set_yscale('log')
ax0.set_xlim(1e1, 1e3)
ax0.set_xlabel('$ R[arcsec] $')
ax0.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax0.set_xscale('log')
ax0.legend(loc = 3, frameon = False, fontsize = 8)
ax0.grid(which = 'both', axis = 'both', alpha = 0.25)
ax0.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('SB_correct_compare.png', dpi = 300)
plt.close()
