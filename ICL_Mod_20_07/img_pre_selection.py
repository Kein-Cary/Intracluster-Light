import matplotlib as mpl
mpl.use('Agg')
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
from astropy import cosmology as apcy

## my-module
from fig_out_module import cc_grid_img


def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

def get_cat(star_cat, gal_cat, pixel, wcs_lis):

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

def map_mu_sigma_func(cat_file, img_file, band, L_cen, N_step, out_file,):
	"""
	cat_file : img catalog, including : ra, dec, z, bcg location in image frame.(.csv files)
	img_file : imgs will be analysis, have applied masking ('XX/XX/xx.fits')
	L_cen : half length of centeral region box
	N_step : grid size 
	out_file : out-put file.(.csv files)
	band : filter imformation (eg. r, g, i, u, z), str type
	"""

	dat = pds.read_csv(cat_file)
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	N_samp = len(set_z)

	cen_sigm, cen_mu = [], []
	img_mu, img_sigm = [], []

	for kk in range( N_samp ):

		ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]
		xn, yn = clus_x[kk], clus_y[kk]

		# mask imgs
		res_file = img_file % (band, ra_g, dec_g, z_g)
		res_data = fits.open(res_file)
		remain_img = res_data[0].data

		# mask matrix
		idnn = np.isnan(remain_img)
		mask_arr = np.zeros((remain_img.shape[0], remain_img.shape[1]), dtype = np.float32)
		mask_arr[idnn == False] = 1

		ca0, ca1 = np.int(img.shape[0] / 2), np.int(img.shape[1] / 2)
		cen_D = L_cen
		flux_cen = remain_img[ca0 - cen_D: ca0 + cen_D, ca1 - cen_D: ca1 + cen_D]

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

		## grid img (for selecting flare, saturated region...)
		block_m, block_Var, block_pix, block_S0 = cc_grid_img(remain_img, N_step, N_step)

		idzo = block_pix < 1.
		pix_eta = block_pix / block_S0
		idnn = np.isnan(pix_eta)
		pix_eta[idnn] = 0.
		idnul = pix_eta < 5e-2
		block_m[idnul] = 0.

		img_mu.append( np.nanmean( block_m[idnul == False] ) )
		img_sigm.append( np.nanstd( block_m[idnul == False] ) )

	cen_sigm = np.array(cen_sigm)
	cen_mu = np.array(cen_mu)
	img_mu = np.array(img_mu)
	img_sigm = np.array(img_sigm)

	keys = ['ra', 'dec', 'z', 'cen_mu', 'cen_sigma', 'img_mu', 'img_sigma',]
	values = [ra, dec, z, cen_mu, cen_sigm, img_mu, img_sigm]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return

def hist_analysis_func():

	return

def 

