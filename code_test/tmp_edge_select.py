import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

from astropy import cosmology as apcy
from scipy.stats import binned_statistic as binned
from scipy.optimize import curve_fit
from matplotlib.patches import Circle, Ellipse, Rectangle

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

######################
band = ['r', 'g', 'i']
home = '/media/xkchen/My Passport/data/SDSS/'
load = '/home/xkchen/mywork/ICL/data/tmp_img/'

####################
dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/tot_cluster_norm_sample.csv')
#dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/tot_random_norm_sample.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
bcg_x, bcg_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

set_z, set_ra, set_dec = z[2613: 2814], ra[2613: 2814], dec[2613: 2814]
set_x, set_y = bcg_x[2613: 2814], bcg_y[2613: 2814]

####################
## test the hist of 'over_sb' and select 'out-lier' region
def hist_divid(x_data, pdf_data, sigma_limit, title_str, file_str):

	idv = pdf_data == pdf_data.max()
	idvx = np.where(idv == True)[0]
	peak_x0 = x_data[ idvx[0] ]

	mirro_pdf = pdf_data[::-1]
	idu = mirro_pdf == mirro_pdf.max()
	idux = np.where(idu == True)[0]
	peak_x1 = x_data[ idux[0] ]
	devi_x = peak_x1 - peak_x0
	mirro_x = x_data - devi_x

	nl = len(sigma_limit)
	dtx = (mirro_x[1:] - mirro_x[:-1])
	dx, dn = sts.find_repeats(dtx)
	id_nmax = np.where(dn == dn.max() )[0]
	delta_x = dx[ id_nmax[0] ]

	eta_L = np.zeros( nl, dtype = np.float)
	eta_R = np.zeros( nl, dtype = np.float)

	for ll in range( nl ):
		## p(peak <= x <= limt sigma)
		idv = (x_data >= peak_x0) & (x_data <= sigma_limit[ll])
		bind_pdf = pdf_data[idv]
		bind_x = x_data[idv]
		popu_00 = np.sum(bind_pdf * delta_x )

		idu = (mirro_x >= peak_x0) & (mirro_x <= sigma_limit[ll])
		bind_pdf = mirro_pdf[idu]
		bind_x = mirro_x[idu]
		popu_10 = np.sum(bind_pdf * delta_x )

		eta_L[ll] = popu_10 / popu_00

		## p(x > peak_x)
		idu = mirro_x >= peak_x0
		bind_pdf = mirro_pdf[idu]
		bind_x = mirro_x[idu]
		popu_11 = np.sum(bind_pdf * delta_x )

		eta_R[ll] = popu_10 / popu_11

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title(title_str)
	ax.plot(x_data, pdf_data, color = 'r', alpha = 0.5, label = 'origin')
	ax.plot(mirro_x[mirro_x >= peak_x0], mirro_pdf[mirro_x >= peak_x0], color = 'b', ls = '--', alpha = 0.5, 
		label = 'mirrored [$ \\frac{ \\mu_{patch} - \\mu_{center} } { \\sigma_{center} } $ <= peak ]')
	ax.axvline(x = peak_x0, ls = '-', color = 'k', alpha = 0.5, label = 'peak')

	for ll in range( nl ):
		ax.axvline(x = sigma_limit[ll], ls = ':', color = mpl.cm.rainbow(ll / nl), alpha = 0.5, 
			label = '%.1f $\\sigma$ [P = %.3f, C = %.3f]' % (sigma_limit[ll], eta_L[ll], eta_R[ll]),)

	ax.set_xlabel('$[\\mu_{patch} - \\mu_{center}]$ / $ \\sigma_{center} $')
	ax.set_ylabel('PDF')
	ax.legend(loc = 2, frameon = False, fontsize = 8)
	ax.set_xlim(-10, 10)
	ax.set_ylim(0, pdf_data.max() + 0.01)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')
	plt.savefig(file_str, dpi = 300)
	plt.close()

	return
'''
## stack the histogram
central_sb = []
edge_sb = []

cut_x0, cut_x1 = 5, 15 # (initial case)
cut_y0, cut_y1 = 5, 9 # (initial case)

for kk in range( len(set_z) ):

	ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

	with h5py.File('hist_test/over-sb_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'r') as f:
	#with h5py.File('hist_test/rand_over-sb_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'r') as f:
		over_sb = np.array(f['over_sb'])

	sub_center = over_sb[cut_y0: cut_y1, cut_x0: cut_x1]
	idnn = np.isnan(sub_center)
	central_sb.append(sub_center[idnn == False])

	sub_bound = over_sb + 0.
	sub_bound[cut_y0: cut_y1, cut_x0: cut_x1] = np.nan
	idnn = np.isnan(sub_bound)
	edge_sb.append(sub_bound[idnn == False])

cen_lo = np.array([np.min(ll) for ll in central_sb])
cen_hi = np.array([np.max(ll) for ll in central_sb])

bond_lo = np.array([np.min(ll) for ll in edge_sb])
bond_hi = np.array([np.max(ll) for ll in edge_sb])

bins_cen = np.linspace(cen_lo.min(), cen_hi.max(), 550)
bins_edg = np.linspace(bond_lo.min(), bond_hi.max(), 550)

x_cen = 0.5 * (bins_cen[1:] + bins_cen[:-1])
x_edg = 0.5 * (bins_edg[1:] + bins_edg[:-1])
cen_pdf = np.ones((len(set_z), len(x_cen)), dtype = np.float)
edg_pdf = np.ones((len(set_z), len(x_edg)), dtype = np.float)

for kk in range(len(set_z)):

	ttx_cen = np.array(central_sb[kk])
	pock_n, edgs = binned(ttx_cen, ttx_cen, statistic = 'count', bins = bins_cen)[:2]
	id_zero = pock_n == 0.
	pock_n[id_zero] = np.nan
	cen_pdf[kk,:] = pock_n * 1.

	ttx_edg = np.array(edge_sb[kk])
	pock_n, edgs = binned(ttx_edg, ttx_edg, statistic = 'count', bins = bins_edg)[:2]
	id_zero = pock_n == 0.
	pock_n[id_zero] = np.nan
	edg_pdf[kk,:] = pock_n * 1.

m_cen_pdf = np.nanmean(cen_pdf, axis = 0)
ddx = x_cen[1] - x_cen[0]
m_cen_pdf = (m_cen_pdf / np.nansum(m_cen_pdf) ) / ddx
idnn = np.isnan(m_cen_pdf)
m_cen_pdf = m_cen_pdf[idnn == False]
x_cen = x_cen[idnn == False]

m_edg_pdf = np.nanmean(edg_pdf, axis = 0)
ddx = x_edg[1] - x_edg[0]
m_edg_pdf = (m_edg_pdf / np.nansum(m_edg_pdf) ) / ddx
idnn = np.isnan(m_edg_pdf)
m_edg_pdf = m_edg_pdf[idnn == False]
x_edg = x_edg[idnn == False]

bin_lim = np.array([1, 2, 3, 3.5, 4, 5])

titl_str = 'edges mean flux histogram [masked imgs]'
file_str = 'edgs_test.png'
hist_divid(x_edg, m_edg_pdf, bin_lim, titl_str, file_str)

titl_str = 'center mean flux histogram [masked imgs]'
file_str = 'center_test.png'
hist_divid(x_cen, m_cen_pdf, bin_lim, titl_str, file_str)

## combination case
com_lo, com_hi = np.min([bins_edg[0], bins_cen[0]]), np.max([bins_edg[-1], bins_cen[-1]])
bins_com = np.linspace(com_lo, com_hi, 550)
x_com = 0.5 * (bins_com[1:] + bins_com[:-1])

cen_com = np.ones((len(set_z), len(x_com)), dtype = np.float)
edg_com = np.ones((len(set_z), len(x_com)), dtype = np.float)

for kk in range(len(set_z)):

	ttx_cen = np.array(central_sb[kk])
	pock_n, edgs = binned(ttx_cen, ttx_cen, statistic = 'count', bins = bins_com)[:2]
	id_zero = pock_n == 0.
	pock_n[id_zero] = np.nan
	cen_com[kk,:] = pock_n * 1.

	ttx_edg = np.array(edge_sb[kk])
	pock_n, edgs = binned(ttx_edg, ttx_edg, statistic = 'count', bins = bins_com)[:2]
	id_zero = pock_n == 0.
	pock_n[id_zero] = np.nan
	edg_com[kk,:] = pock_n * 1.

m_cen_com = np.nanmean(cen_com, axis = 0)
m_edg_com = np.nanmean(edg_com, axis = 0)
m_com = m_cen_com + m_edg_com
ddx = x_com[1] - x_com[0]

m_com_pdf = (m_com / np.nansum(m_com) ) / ddx
idnn = np.isnan(m_com_pdf)
m_com_pdf = m_com_pdf[idnn == False]
x_com = x_com[idnn == False]

titl_str = 'total mean flux histogram [masked imgs]'
file_str = 'total_test.png'
hist_divid(x_com, m_com_pdf, bin_lim, titl_str, file_str)
'''
###################
## test the img pixel flux hist of masked imgs
def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

pix_flux_1sigm = []
pix_flux_2sigm = []

stack_1sigm = []
stack_2sigm = []

pix_remain_1sigm = []
pix_remain_2sigm = []

for kk in range(len(set_z)):
	ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

	file_0 = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_1-sigma.fits' % ('r', ra_g, dec_g, z_g)
	data_0 = fits.open(file_0)
	img_0 = data_0[0].data
	idnn = np.isnan(img_0)
	sub_flux_0 = img_0[idnn == False]

	file_1 = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_2-sigma.fits' % ('r', ra_g, dec_g, z_g)
	data_1 = fits.open(file_1)
	img_1 = data_1[0].data
	idnn = np.isnan(img_1)
	sub_flux_1 = img_1[idnn == False]

	pix_flux_1sigm.append( sub_flux_0 )
	pix_flux_2sigm.append( sub_flux_1 )
	## stack test
	stack_1sigm.append(img_0)
	stack_2sigm.append(img_1)

	eta_0 = len(sub_flux_0) / (1489 * 2048)
	eta_1 = len(sub_flux_1) / (1489 * 2048)
	pix_remain_1sigm.append(eta_0)
	pix_remain_2sigm.append(eta_1)

	pock_n, edgs = binned(sub_flux_0, sub_flux_0, statistic = 'count', bins = 60)[:2]
	id_zero = pock_n == 0.
	pock_n = pock_n[id_zero == False]
	sub_pdf0 = (pock_n / np.sum(pock_n)) / (edgs[1] - edgs[0])
	sub_x0 = 0.5 * (edgs[1:] + edgs[:-1])
	sub_x0 = sub_x0[id_zero == False]
	popt, pcov = curve_fit(gau_func, sub_x0, sub_pdf0, p0 = [np.mean(sub_flux_0), np.std(sub_flux_0)])
	e_mu0, e_sigm0 = popt[0], popt[1]
	fit_l0 = gau_func(sub_x0, e_mu0, e_sigm0,)

	pock_n, edgs = binned(sub_flux_1, sub_flux_1, statistic = 'count', bins = 60)[:2]
	id_zero = pock_n == 0.
	pock_n = pock_n[id_zero == False]
	sub_pdf1 = (pock_n / np.sum(pock_n)) / (edgs[1] - edgs[0])
	sub_x1 = 0.5 * (edgs[1:] + edgs[:-1])
	sub_x1 = sub_x1[id_zero == False]
	popt, pcov = curve_fit(gau_func, sub_x1, sub_pdf1, p0 = [np.mean(sub_flux_1), np.std(sub_flux_1)])
	e_mu1, e_sigm1 = popt[0], popt[1]
	fit_l1 = gau_func(sub_x1, e_mu1, e_sigm1,)
'''
	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('pixel flux histogram [ra%.3f dec%.3f z%.3f]' % (ra_g, dec_g, z_g), )
	ax.plot(sub_x0, sub_pdf0, color = 'r', ls = '-', alpha = 0.5, label = '1 $\\sigma$ [%.3f]' % (eta_0),)
	ax.plot(sub_x0, fit_l0, color = 'r', ls = ':', alpha = 0.5, label = 'Gaussian fitting')
	ax.plot(sub_x1, sub_pdf1, color = 'b', ls = '-', alpha = 0.5, label = '2 $\\sigma$ [%.3f]' % (eta_1),)
	ax.plot(sub_x1, fit_l1, color = 'b', ls = ':', alpha = 0.5,)

	ax.annotate(s = '$\\mu= %.5f, \\sigma=$%.5f' % (e_mu0, e_sigm0), xy = (0.03, 0.9), xycoords = 'axes fraction', color = 'r', alpha = 0.5,)
	ax.annotate(s = '$\\mu= %.5f, \\sigma=$%.5f' % (e_mu1, e_sigm1), xy = (0.03, 0.85), xycoords = 'axes fraction', color = 'b', alpha = 0.5,)
	ax.axvline(x = 0, ls = '--', color = 'k', alpha = 0.5)
	ax.axvline(x = e_mu0 - e_sigm0, color = 'r', alpha = 0.5, ls = '--',)
	ax.axvline(x = e_mu0 + e_sigm0, color = 'r', alpha = 0.5, ls = '--',)
	ax.axvline(x = e_mu1 - e_sigm1, color = 'b', alpha = 0.5, ls = '--',)
	ax.axvline(x = e_mu1 + e_sigm1, color = 'b', alpha = 0.5, ls = '--',)

	ax.legend(loc = 1, frameon = False)
	ax.set_xlabel('pixel flux [nanomaggies]')
	ax.set_ylabel('PDF')
	plt.savefig('flux_hist_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
	plt.close()

pix_remain_1sigm = np.array(pix_remain_1sigm)
pix_remain_2sigm = np.array(pix_remain_2sigm)
bin_area = np.linspace(0, 1, 26)
plt.figure()
ax = plt.subplot(111)
ax.hist(pix_remain_1sigm, bins = bin_area, density = True, color = 'r', label = '1 $\\sigma$', alpha = 0.5,)
ax.hist(pix_remain_2sigm, bins = bin_area, density = True, color = 'b', label = '2 $\\sigma$', alpha = 0.5,)
ax.set_xlabel('unmasked area / total image area')
ax.set_ylabel('PDF')
ax.legend(loc = 2, frameon = False)
plt.savefig('unmasked_area_hist.png', dpi = 300)
plt.close()
'''

stack_1sigm = np.array(stack_1sigm)
stack_1sigm = np.nanmean(stack_1sigm, axis = 0)

stack_2sigm = np.array(stack_2sigm)
stack_2sigm = np.nanmean(stack_2sigm, axis = 0)

pix_lo = np.array( [np.min(ll) for ll in pix_flux_2sigm] )
pix_hi = np.array( [np.max(ll) for ll in pix_flux_1sigm] )
bind_pix = np.linspace(pix_lo.min(), pix_hi.max(), 200)

x_pix = 0.5 * (bind_pix[1:] + bind_pix[:-1])
pix_pdf_1 = np.zeros( (len(set_z), len(x_pix)), dtype = np.float)
pix_pdf_2 = np.zeros( (len(set_z), len(x_pix)), dtype = np.float)

for kk in range(len(set_z)):

	ttx = np.array( pix_flux_1sigm[kk] )
	pock_n, edgs = binned(ttx, ttx, statistic = 'count', bins = bind_pix)[:2]
	id_zero = pock_n == 0.
	pock_n[id_zero] = np.nan
	pix_pdf_1[kk,:] = pock_n * 1.

	ttx = np.array( pix_flux_2sigm[kk] )
	pock_n, edgs = binned(ttx, ttx, statistic = 'count', bins = bind_pix)[:2]
	id_zero = pock_n == 0.
	pock_n[id_zero] = np.nan
	pix_pdf_2[kk,:] = pock_n * 1.

m_pix_pdf1 = np.nanmean(pix_pdf_1, axis = 0)
ddx = x_pix[1] - x_pix[0]
m_pix_pdf1 = (m_pix_pdf1 / np.nansum(m_pix_pdf1) ) / ddx
idnn = np.isnan(m_pix_pdf1)
m_pdf1 = m_pix_pdf1[idnn == False]
x_pix_1 = x_pix[idnn == False]

m_pix_pdf2 = np.nanmean(pix_pdf_2, axis = 0)
ddx = x_pix[1] - x_pix[0]
m_pix_pdf2 = (m_pix_pdf2 / np.nansum(m_pix_pdf2) ) / ddx
idnn = np.isnan(m_pix_pdf2)
m_pdf2 = m_pix_pdf2[idnn == False]
x_pix_2 = x_pix[idnn == False]

## fitting with gaussian
ddx = x_pix[1] - x_pix[0]
mu_1 = np.sum(m_pdf1 * x_pix_1 * ddx)
sigm_1 = np.sum(m_pdf1 * ddx * (x_pix_1 - mu_1)**2 )
popt, pcov = curve_fit(gau_func, x_pix_1, m_pdf1,)
e_mu_1, e_sigm_1 = popt[0], popt[1]
fit_l_1 = gau_func(x_pix_1, e_mu_1, e_sigm_1,)

mu_2 = np.sum(m_pdf2 * x_pix_2 * ddx)
sigm_2 = np.sum(m_pdf2 * (x_pix_2 - mu_2) * ddx)
popt, pcov = curve_fit(gau_func, x_pix_2, m_pdf2,)
e_mu_2, e_sigm_2 = popt[0], popt[1]
fit_l_2 = gau_func(x_pix_2, e_mu_2, e_sigm_2,)

plt.figure()
gs = gridspec.GridSpec(2,1, height_ratios = [4, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.set_title('pixel flux PDF [masked, rule out regions brighter than $ A \\sigma_{center}$]')
ax0.plot(x_pix_1, m_pdf1 + 0.5, color = 'r', alpha = 0.5, label = 'A = 1 [shifted, +0.5]')
ax0.plot(x_pix_1, fit_l_1 + 0.5, color = 'r', ls = ':', alpha = 0.5)

ax0.plot(x_pix_2, m_pdf2, color = 'b', alpha = 0.5, label = 'A = 2')
ax0.plot(x_pix_2, fit_l_2, color = 'b', alpha = 0.5, ls = ':', label = 'Gaussian fitting')

ax0.axvline(x = 0, color = 'k', alpha = 0.5, ls  ='--', label = 'flux = 0')
ax0.annotate(s = 'fit,$\\mu = %.5f, \\sigma = %.5f$' % (e_mu_1, e_sigm_1), xy = (0.05, 0.90), xycoords = 'axes fraction', color = 'r', alpha = 0.5)
ax0.annotate(s = 'fit,$\\mu = %.5f, \\sigma = %.5f$' % (e_mu_2, e_sigm_2), xy = (0.05, 0.85), xycoords = 'axes fraction', color = 'b', alpha = 0.5)
ax0.set_ylabel('PDF')
ax0.legend(loc = 1, frameon = False)
ax0.set_xlim(-0.2, 0.2)

ax1.plot(x_pix, m_pix_pdf1 - m_pix_pdf2, color = 'r', alpha = 0.5, ls = '-',)
ax1.axhline(y = 0, color = 'k', alpha = 0.5, ls = '--',)
ax1.set_xlim(ax0.get_xlim())
ax1.set_xlabel('pixel flux [nanomaggies]')
ax1.set_ylabel('$PDF_{A=1} - PDF_{A=2}$')
ax0.set_xticks([])

plt.subplots_adjust(hspace = 0.02,) #left = 0.2, bottom = 0.1, right = 0.95, top = 0.9, wspace = None,)
plt.savefig('pixel_flux_hist.png', dpi = 300)
plt.close()

idvx = np.isnan(stack_1sigm)
ttx0 = stack_1sigm[idvx == False]
pock_n, edgs = binned(ttx0, ttx0, statistic = 'count', bins = 200)[:2]
id_zero = pock_n == 0.
pock_n = pock_n[id_zero == False]
ddx = edgs[1] - edgs[0]
pock_pdf0 = (pock_n / np.sum(pock_n) ) / ddx
dtx0 = 0.5 * (edgs[1:] + edgs[:-1])
dtx0 = dtx0[id_zero == False]

popt, pcov = curve_fit(gau_func, dtx0, pock_pdf0, p0 = [np.mean(ttx0), np.std(ttx0)])
e_mu0, e_sigm0 = popt[0], popt[1]
fit_l0 = gau_func(dtx0, e_mu0, e_sigm0,)

idux = np.isnan(stack_2sigm)
ttx1 = stack_2sigm[idux == False]
pock_n, edgs = binned(ttx1, ttx1, statistic = 'count', bins = 200)[:2]
id_zero = pock_n == 0.
pock_n = pock_n[id_zero == False]
ddx = edgs[1] - edgs[0]
pock_pdf1 = (pock_n / np.sum(pock_n) ) / ddx
dtx1 = 0.5 * (edgs[1:] + edgs[:-1])
dtx1 = dtx1[id_zero == False]

popt, pcov = curve_fit(gau_func, dtx1, pock_pdf1, p0 = [np.mean(ttx1), np.std(ttx1)])
e_mu1, e_sigm1 = popt[0], popt[1]
fit_l1 = gau_func(dtx1, e_mu1, e_sigm1,)

plt.figure()
ax = plt.subplot(111)
ax.set_title('stacking img pixel flux PDF [centered on img center]')
ax.plot(dtx0, pock_pdf0, color = 'r', alpha = 0.5, label = 'rule out above 1 $\\sigma_{center}$ patches')
ax.plot(dtx0, fit_l0, color = 'r', ls = ':', alpha = 0.5, label = 'Gaussian fitting')

ax.plot(dtx1, pock_pdf1, color = 'b', alpha = 0.5, label = 'rule out above 2 $\\sigma_{center}$ patches')
ax.plot(dtx1, fit_l1, color = 'b', ls = ':', alpha = 0.5, )

ax.axvline(x = 0, ls = '--', color = 'k', alpha = 0.5)
ax.axvline(x = e_mu0 - e_sigm0, color = 'r', alpha = 0.5, ls = '-.', label = '1 $\\sigma$')
ax.axvline(x = e_mu0 + e_sigm0, color = 'r', alpha = 0.5, ls = '-.',)
ax.axvline(x = e_mu1 - e_sigm1, color = 'b', alpha = 0.5, ls = '-.',)
ax.axvline(x = e_mu1 + e_sigm1, color = 'b', alpha = 0.5, ls = '-.',)

ax.annotate(s = '$\\mu = %.5f, \\sigma = %.5f$' % (e_mu0, e_sigm0), xy = (0.65, 0.90), xycoords = 'axes fraction', color = 'r', alpha = 0.5)
ax.annotate(s = '$\\mu = %.5f, \\sigma = %.5f$' % (e_mu1, e_sigm1), xy = (0.65, 0.85), xycoords = 'axes fraction', color = 'b', alpha = 0.5)

ax.set_xlabel('pixel flux [nanomaggies]')
ax.set_ylabel('PDF')
ax.set_xlim(-0.015, 0.015)
ax.legend(loc = 2, frameon = False, fontsize = 8)
plt.savefig('stack-img_flux_hist.png', dpi = 300)
plt.close()
