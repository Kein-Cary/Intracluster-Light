import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import random
import numpy as np
import pandas as pds

import astropy.wcs as awc
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits

from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy.optimize import curve_fit, minimize

## pipe part
from img_jack_stack import jack_main_func
from light_measure import cc_grid_img, jack_SB_func
from light_measure import flux_recal

#from light_measure import light_measure_Z0_weit
from light_measure_tmp import light_measure_Z0_weit

## constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
pc2cm = U.pc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

band = ['r', 'g', 'i', 'u', 'z']
sky_SB = [21.04, 22.01, 20.36, 22.30, 19.18]
## gain value, use for CCD mock
gain = np.array([ [4.71, 4.6, 4.72, 4.76, 4.725, 4.895], 
				[3.32,  3.855, 3.845, 3.995, 4.05, 4.035], 
				[5.165, 6.565, 4.86,  4.885, 4.64, 4.76], 
				[1.62,  1.71,  1.59,  1.6,   1.47, 2.17], 
				[4.745, 5.155, 4.885, 4.775, 3.48, 4.69] ])

load = '/home/xkchen/mywork/ICL/data/tmp_img/'
home = '/media/xkchen/My Passport/data/SDSS/'

############
def sersic_pro(r, I0, r_e, n_dex):
	b_n = n_dex * 2 - 0.324
	sb_pro = I0 * np.exp( -1 * b_n * (r / r_e)**(1 / n_dex) + b_n)
	return sb_pro

def SB_fit_func(r, e_I0, e_I1, r_e0, r_e1, n_dex0, n_dex1, sb_C,):
	b_n0 = n_dex0 * 2 - 0.324
	b_n1 = n_dex1 * 2 - 0.324
	mock_sb0 = e_I0 * np.exp( -1 * b_n0 * (r / r_e0)**(1 / n_dex0) + b_n0)
	mock_sb1 = e_I1 * np.exp( -1 * b_n1 * (r / r_e1)**(1 / n_dex1) + b_n1)
	mock_SB = mock_sb0 + mock_sb1 + sb_C
	return mock_SB

def fit_SB_pro():

	#with h5py.File('clust_jack_SB-pro_30-FWHM-ov2_V-cut-Mean-img_100.h5', 'r') as f:
	with h5py.File(home + '20_10_test/jack_test/A_clust_BCG-stack_Mean_jack_SB-pro_30-FWHM-ov2_z-ref.h5', 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	idvx = (c_r_arr >= 70) & (c_r_arr <= 2e3)
	r_fit = c_r_arr[idvx]
	sb_fit = c_sb_arr[idvx]
	err_fit = c_sb_err[idvx]

	mu_e0, mu_e1 = 1e-2, 5e-3
	Re_0, Re_1 = 100, 500
	ndex0, ndex1 = 5, 3 
	BG = 1.8e-3

	po = np.array([mu_e0, mu_e1, Re_0, Re_1, ndex0, ndex1, BG])
	popt, pcov = curve_fit(SB_fit_func, r_fit, sb_fit, p0 = po, sigma = err_fit, method = 'trf')

	mu_fit0, mu_fit1, re_fit0, re_fit1, ndex_fit0, ndex_fit1, BG_fit = popt
	fit_line = SB_fit_func(c_r_arr, mu_fit0, mu_fit1, re_fit0, re_fit1, ndex_fit0, ndex_fit1, BG_fit)

	put_r = np.logspace(0, 3.6, 150)
	put_sb = SB_fit_func(put_r, mu_fit0, mu_fit1, re_fit0, re_fit1, ndex_fit0, ndex_fit1, BG_fit)
	keys = ['r', 'sb']
	value = [put_r, put_sb]
	fill = dict( zip(keys, value) )
	data = pds.DataFrame(fill)
	data.to_csv('phy_input_SB.csv')

	fit_0 = sersic_pro(c_r_arr, mu_fit0, re_fit0, ndex_fit0,)
	fit_1 = sersic_pro(c_r_arr, mu_fit1, re_fit1, ndex_fit1,)

	plt.figure()
	ax = plt.subplot(111)

	ax.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = 'g', label = 'A-250')
	ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'g', alpha = 0.2,)

	ax.plot(c_r_arr, fit_line, ls = '-', color = 'r', alpha = 0.5, label = 'fitting')
	#ax.plot(c_r_arr, fit_0, ls = '-', color = 'b', alpha = 0.5,)
	#ax.plot(c_r_arr, fit_1, ls = '--', color = 'b', alpha = 0.5,)
	#ax.axhline(y = BG_fit, ls = ':', color = 'b', alpha = 0.5,)

	ax.set_ylim(1e-3, 1e-1)
	ax.set_yscale('log')
	ax.set_xlim(5e1, 2.1e3)
	ax.set_xlabel('$ R[kpc] $')
	ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax.set_xscale('log')
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in')

	plt.subplots_adjust(left = 0.15, right = 0.95,)
	plt.savefig('SB_fit_test.png', dpi = 300)
	plt.close()

def mock_ccd(band_id, z_set, ra_set, dec_set, info_file, mask_file, out_file):

	kk = np.int(band_id)
	Nz = len(z_set)
	SB_bl = sky_SB[kk]

	y0 = np.linspace(0, 1488, 1489)
	x0 = np.linspace(0, 2047, 2048)
	Nx0, Ny0 = len(x0), len(y0)
	pxl = np.meshgrid(x0, y0)

	pdt = pds.read_csv('input_SB_parameter.txt')
	(e_I0, e_I1, re_0, re_1, ndex_0, ndex_1, L_BG) = (pdt['I_0'][0], 
		pdt['I_1'][0], pdt['re_0'][0], pdt['re_1'][0], pdt['ndex_0'][0], pdt['ndex_1'][0], pdt['BG'][0])

	for jj in range(Nz):

		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		## info_file
		ref_data = fits.open(info_file % (band[kk], ra_g, dec_g, z_g),)
		Head = ref_data[0].header
		cx0 = Head['CRPIX1']
		cy0 = Head['CRPIX2']
		RA0 = Head['CRVAL1']
		DEC0 = Head['CRVAL2']
		NMGY = Head['NMGY']

		CAMCOL = ref_data[3].data['CAMCOL'][0]
		camcol = np.int(CAMCOL - 1)
		Gain = gain[kk, camcol]

		wcs = awc.WCS(Head)
		xc, yc = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		xc, yc = np.int(xc), np.int(yc)

		## mock CCD 
		N_sky = 10**( (22.5 - SB_bl + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY

		dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )

		angl_dr = dr * pixel
		mock_flux = SB_fit_func(angl_dr, e_I0, e_I1, re_0, re_1, ndex_0, ndex_1, L_BG,) ## in unit 'nanomaggy / arcsec^2'
		mock_flux = mock_flux * pixel**2 ## in unit of nanomaggy
		mock_DN = mock_flux / NMGY ## change flux into DN (data counts)

		ix = np.abs(x0 - xc)
		iy = np.abs(y0 - yc)
		ix0 = np.where(ix == np.min(ix))[0][0]
		iy0 = np.where(iy == np.min(iy))[0][0]
		Nois = np.zeros((Ny0, Nx0), dtype = np.float)

		for pp in range(Ny0):
			for qq in range(Nx0):

				lam_x = mock_DN[pp, qq]* Gain
				N_e = lam_x + N_sky * Gain
				rand_x = np.random.poisson( N_e )
				Nois[pp, qq] += rand_x # electrons number

		N_sub = Nois / Gain - N_sky

		## read the mask img [ 30 * (FWHM / 2) for normal stars ]
		data = fits.getdata(mask_file % (band[kk], ra_g, dec_g, z_g), header = True)
		img = data[0]

		id_nan = np.isnan(img) # applied the masked pixels
		N_sub[id_nan] = np.nan
		mock_signal = N_sub * NMGY # change DN to nmaggy

		## save the mock img
		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE', 'CAMCOL', 'NMGY']
		value = ['T', 32, 2, Nx0, Ny0, cx0, cy0, xc, yc, RA0, DEC0, ra_g, dec_g, z_g, pixel, CAMCOL, NMGY]
		ff = dict( zip(keys, value) )
		fil = fits.Header(ff)
		fits.writeto(out_file % (band[kk], ra_g, dec_g, z_g), mock_signal, header = fil, overwrite=True)

	return

def phy_mock_ccd(band_id, z_ref, z_set, ra_set, dec_set, info_file, mask_file, out_file):

	kk = np.int(band_id)
	Nz = len(z_set)
	SB_bl = sky_SB[kk]

	y0 = np.linspace(0, 1488, 1489)
	x0 = np.linspace(0, 2047, 2048)
	Nx0, Ny0 = len(x0), len(y0)
	pxl = np.meshgrid(x0, y0)

	pdt = pds.read_csv('phy_input_SB.csv')
	phy_r, phy_sb = np.array(pdt.r), np.array(pdt.sb)
	r_sc = phy_r / 3e3 ## mock region within 3 Mpc
	r_min = np.min(r_sc)
	r_max = np.max(r_sc)

	for jj in range(Nz):

		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		Da_g = Test_model.angular_diameter_distance(z_g).value ## in unit Mpc
		Angu_r = (3 * rad2asec / Da_g) ## angle size of 3 Mpc
		R_pixel = Angu_r / pixel

		## info_file
		ref_data = fits.open(info_file % (band[kk], ra_g, dec_g, z_g),)
		Head = ref_data[0].header
		cx0 = Head['CRPIX1']
		cy0 = Head['CRPIX2']
		RA0 = Head['CRVAL1']
		DEC0 = Head['CRVAL2']
		NMGY = Head['NMGY']

		CAMCOL = ref_data[3].data['CAMCOL'][0]
		camcol = np.int(CAMCOL - 1)
		Gain = gain[kk, camcol]

		wcs = awc.WCS(Head)
		xc, yc = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
		xc, yc = np.int(xc), np.int(yc)

		## mock CCD 
		N_sky = 10**( (22.5 - SB_bl + 2.5*np.log10(pixel**2)) / 2.5 ) / NMGY

		dr = np.sqrt( ( (2 * pxl[0] + 1) / 2 - (2 * xc + 1) / 2)**2 + 
			( ( 2 * pxl[1] + 1)/2 - (2 * yc + 1) / 2)**2 )
		dr_sc = dr / R_pixel

		shift_sb = phy_sb * (1 + z_ref)**4 / (1 + z_g)**4 ## shift sb profile to z_g, in unit 'nanomaggies'
		sub_flux = shift_sb * pixel**2
		sub_DN = sub_flux / NMGY
		DN_min = np.min(sub_DN)
		mock_DN = interp.interp1d(r_sc, sub_DN, kind = 'cubic')

		ix = np.abs(x0 - xc)
		iy = np.abs(y0 - yc)
		ix0 = np.where(ix == np.min(ix))[0][0]
		iy0 = np.where(iy == np.min(iy))[0][0]
		Nois = np.zeros((Ny0, Nx0), dtype = np.float)

		for pp in range(Ny0):
			for qq in range(Nx0):
				if (dr_sc[pp, qq] >= r_max ) | ( dr_sc[pp, qq] <= r_min ):

					lam_x = DN_min * Gain / 10
					N_e = lam_x + N_sky * Gain
					rand_x = np.random.poisson( N_e )
					Nois[pp, qq] += rand_x # electrons number

				else:

					lam_x = mock_DN( dr_sc[pp, qq] ) * Gain
					N_e = lam_x + N_sky * Gain
					rand_x = np.random.poisson( N_e )
					Nois[pp, qq] += rand_x # electrons number

		N_sub = Nois / Gain - N_sky

		## read the mask img [ 30 * (FWHM / 2) for normal stars ]
		data = fits.getdata(mask_file % (band[kk], ra_g, dec_g, z_g), header = True)
		img = data[0]

		id_nan = np.isnan(img) # applied the masked pixels
		N_sub[id_nan] = np.nan
		mock_signal = N_sub * NMGY # change DN to nmaggy

		## save the mock img
		keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','CENTER_X','CENTER_Y',
				'CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z', 'P_SCALE', 'CAMCOL', 'NMGY']
		value = ['T', 32, 2, Nx0, Ny0, cx0, cy0, xc, yc, RA0, DEC0, ra_g, dec_g, z_g, pixel, CAMCOL, NMGY]
		ff = dict( zip(keys, value) )
		fil = fits.Header(ff)
		fits.writeto(out_file % (band[kk], ra_g, dec_g, z_g), mock_signal, header = fil, overwrite=True)

	return

dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000-to-250_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)
band_id = 0
z_ref = 0.254 # mean of the z
"""
info_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
mask_file = home + 'tmp_stack/cluster/cluster_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'
out_file = load + 'mock_img/phy_mock-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
#mock_ccd(band_id, z, ra, dec, info_file, mask_file, out_file,)
#phy_mock_ccd(band_id, z_ref, z, ra, dec, info_file, mask_file, out_file,)

from img_resample import resamp_func
d_file = load + 'mock_img/phy_mock-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
out_file = load + 'resamp_mock/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'
#resamp_func(d_file, z, ra, dec, clus_x, clus_y, band[0], out_file, z_ref, stack_info = None, pixel = 0.396, id_dimm = True,)
"""

'''
## stacking at z_ref
load = '/home/xkchen/mywork/ICL/data/tmp_img/'
d_file = load + 'resamp_mock/resamp-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

sub_img = load + 'stack_mock/mock-A_BCG-stack_sub-%d_img_z-ref.h5'
sub_pix_cont = load + 'stack_mock/mock-A_BCG-stack_sub-%d_pix-cont_z-ref.h5'
sub_sb = load + 'stack_mock/mock-A_BCG-stack_sub-%d_SB-pro_z-ref.h5'

J_sub_img = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_img_z-ref.h5'
J_sub_pix_cont = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_pix-cont_z-ref.h5'
J_sub_sb = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_SB-pro_z-ref.h5'

jack_SB_file = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_SB-pro_z-ref.h5'
jack_img = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_img_z-ref.h5'
jack_cont_arr = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_pix-cont_z-ref.h5'

z_ref = 0.254 # mean of the z

jack_main_func(id_cen, N_bin, n_rbins, set_ra, set_dec, set_z, set_x, set_y, d_file, band[0], sub_img,
	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,
	id_cut = False, N_edg = None, id_Z0 = False, z_ref = 0.254,)
'''

##### stacking img in obs. coordinate (angle coordinate)
id_cen = 0
n_rbins = 110
N_bin = 30

d_file = load + 'mock_img/mock-%s-ra%.3f-dec%.3f-redshift%.3f.fits'

sub_img = load + 'stack_mock/mock-A_BCG-stack_sub-%d_img.h5'
sub_pix_cont = load + 'stack_mock/mock-A_BCG-stack_sub-%d_pix-cont.h5'
sub_sb = load + 'stack_mock/mock-A_BCG-stack_sub-%d_SB-pro.h5'

J_sub_img = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_img.h5'
J_sub_pix_cont = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_pix-cont.h5'
J_sub_sb = load + 'stack_mock/mock-A_BCG-stack_jack-sub-%d_SB-pro.h5'

jack_SB_file = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_SB-pro.h5'
jack_img = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_img.h5'
jack_cont_arr = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_pix-cont.h5'

#jack_main_func(id_cen, N_bin, n_rbins, ra, dec, z, clus_x, clus_y, d_file, band[0], sub_img,
#	sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)

## re-calculate SB pro. (change weight matrix or radius bins)

N_bin = 30
lim_r1 = 0

for nn in range( N_bin ):

	with h5py.File(J_sub_img % nn, 'r') as f:
		sub_jk_img = np.array(f['a'])
	xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
	id_nn = np.isnan(sub_jk_img)
	eff_y, eff_x = np.where(id_nn == False)
	dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
	dR_max = np.int( dR.max() ) + 1
	lim_r1 = np.max([lim_r1, dR_max])

n_rbins = 110
r_bins_1 = np.logspace(0, np.log10(lim_r1), n_rbins)
#n_rbins = 60
#r_bins_1 = np.linspace(1, lim_r1, n_rbins) ## linear bins

for nn in range(N_bin):

	# jack sub-sample
	with h5py.File(J_sub_img % nn, 'r') as f:
		sub_jk_img = np.array(f['a'])

	with h5py.File(J_sub_pix_cont % nn, 'r') as f:
		sub_jk_cont = np.array(f['a'])

	xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
	Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(sub_jk_img, sub_jk_cont, pixel, xn, yn, r_bins_1)
	sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
	r_arr = Intns_r

	#with h5py.File(J_sub_sb % nn, 'w') as f:
	with h5py.File(load + 'stack_mock/mock-A_jack-sub-%d_SB-pro_bin-linear.h5' % nn, 'w') as f:
		f['r'] = np.array(r_arr)
		f['sb'] = np.array(sb_arr)
		f['sb_err'] = np.array(sb_err_arr)
		f['nratio'] = np.array(nratio)
		f['npix'] = np.array(npix)

## final jackknife SB profile
tmp_sb = []
tmp_r = []

for nn in range( N_bin ):

		#with h5py.File(J_sub_sb % nn, 'r') as f:
		with h5py.File(load + 'stack_mock/mock-A_jack-sub-%d_SB-pro_bin-linear.h5' % nn, 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

			idvx = npix < 1.
			sb_arr[idvx] = np.nan
			r_arr[idvx] = np.nan

			tmp_sb.append(sb_arr)
			tmp_r.append(r_arr)

## save the sb result in unit " nanomaggies / arcsec^2 "
tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, 0, N_bin)[4:]
sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

#with h5py.File(jack_SB_file, 'w') as f:
with h5py.File(load + 'stack_mock/mock-A_Mean_jack_SB-pro_bin-linear.h5', 'w') as f:
	f['r'] = np.array(tt_jk_R)
	f['sb'] = np.array(tt_jk_SB)
	f['sb_err'] = np.array(tt_jk_err)
	f['lim_r'] = np.array(sb_lim_r)

## figure the result
sb_dat = pds.read_csv('input_SB.csv')
put_r, put_sb = np.array(sb_dat['r']), np.array(sb_dat['sb'])

with h5py.File(jack_SB_file, 'r') as f:
	c_r_arr = np.array(f['r'])
	c_sb_arr = np.array(f['sb'])
	c_sb_err = np.array(f['sb_err'])

with h5py.File(load + 'stack_mock/mock-A_Mean_jack_SB-pro_bin-linear.h5', 'r') as f:
	alt_r = np.array(f['r'])
	alt_sb = np.array(f['sb'])
	alt_sb_err = np.array(f['sb_err'])

plt.figure()
ax = plt.subplot(111)

ax.plot(c_r_arr, c_sb_arr, ls = '-', alpha = 0.8, color = 'r', label = 'log-bin',)
ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

ax.plot(alt_r, alt_sb, ls = '-', alpha = 0.8, color = 'b', label = 'weighted-bin',)
ax.fill_between(alt_r, y1 = alt_sb - alt_sb_err, y2 = alt_sb + alt_sb_err, color = 'b', alpha = 0.2,)

ax.set_ylim(4e-4, 7e-3)
ax.set_yscale('log')
ax.set_xlim(5e1, 1e3)
ax.set_xlabel('$ R[arcsec] $')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax.set_xscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('mock-img_SB_R-bin_test.png', dpi = 300)
plt.close()

raise

##### err estimation compare
"""
dat = pds.read_csv('/home/xkchen/mywork/ICL/code/SEX/result/test_1000_no_select.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)
band_id = 0

Ns = np.array([100, 400, 900, 1000])
'''
for mm in range(4):

	np.random.seed(1)
	tt0 = np.random.choice(1000, size = Ns[mm], replace = False)
	set_ra, set_dec, set_z = ra[tt0], dec[tt0], z[tt0]
	set_x, set_y = clus_x[tt0], clus_y[tt0]

	sub_img = load + 'stack_mock/Err_mock_BCG-stack_sub-%d_img.h5'
	sub_pix_cont = load + 'stack_mock/Err_mock_BCG-stack_sub-%d_pix-cont.h5'
	sub_sb = load + 'stack_mock/Err_mock_BCG-stack_sub-%d_SB-pro.h5'

	J_sub_img = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_img' + '_Ns-%d.h5' % Ns[mm]
	J_sub_pix_cont = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_pix-cont' + '_Ns-%d.h5' % Ns[mm]
	J_sub_sb = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_SB-pro' + '_Ns-%d.h5' % Ns[mm]

	jack_SB_file = load + 'stack_mock/Err_mock_BCG-stack_Mean_jack_SB-pro_test_%d.h5' % Ns[mm]
	jack_img = load + 'stack_mock/Err_mock_BCG-stack_Mean_jack_img_test_%d.h5' % Ns[mm] 
	jack_cont_arr = load + 'stack_mock/Err_mock_BCG-stack_Mean_jack_pix-cont_test_%d.h5' % Ns[mm]

	jack_main_func(id_cen, N_bin, n_rbins, set_ra, set_dec, set_z, set_x, set_y, d_file, band[0], sub_img,
		sub_pix_cont, sub_sb, J_sub_img, J_sub_pix_cont, J_sub_sb, jack_SB_file, jack_img, jack_cont_arr,)

	print('Ns = ', Ns[mm])
'''
jack_SB_file = load + 'stack_mock/Err_mock_BCG-stack_Mean_jack_SB-pro_test_%d.h5'
jack_img = load + 'stack_mock/Err_mock_BCG-stack_Mean_jack_img_test_%d.h5'
jack_cont_arr = load + 'stack_mock/Err_mock_BCG-stack_Mean_jack_pix-cont_test_%d.h5'

'''
max_R = []
for mm in range(4):

	J_sub_img = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_img' + '_Ns-%d.h5' % Ns[mm]
	J_sub_pix_cont = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_pix-cont' + '_Ns-%d.h5' % Ns[mm]
	J_sub_sb = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_SB-pro' + '_Ns-%d.h5' % Ns[mm]

	lim_r1 = 0

	for nn in range( N_bin ):

		with h5py.File(J_sub_img % nn, 'r') as f:
			sub_jk_img = np.array(f['a'])
		xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
		id_nn = np.isnan(sub_jk_img)
		eff_y, eff_x = np.where(id_nn == False)
		dR = np.sqrt((eff_y - yn)**2 + (eff_x - xn)**2)
		dR_max = np.int( dR.max() ) + 1
		lim_r1 = np.max([lim_r1, dR_max])
	max_R.append(lim_r1)

maxR = np.max(max_R)
r_bins_1 = np.logspace(0, np.log10(maxR), n_rbins)

for mm in range(4):

	J_sub_img = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_img' + '_Ns-%d.h5' % Ns[mm]
	J_sub_pix_cont = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_pix-cont' + '_Ns-%d.h5' % Ns[mm]
	J_sub_sb = load + 'stack_mock/Err_mock_BCG-stack_jack-sub-%d_SB-pro' + '_Ns-%d.h5' % Ns[mm]

	for nn in range(N_bin):

		# jack sub-sample
		with h5py.File(J_sub_img % nn, 'r') as f:
			sub_jk_img = np.array(f['a'])

		with h5py.File(J_sub_pix_cont % nn, 'r') as f:
			sub_jk_cont = np.array(f['a'])

		xn, yn = np.int(sub_jk_img.shape[1] / 2), np.int(sub_jk_img.shape[0] / 2)
		Intns, Angl_r, Intns_err, npix, nratio = light_measure_Z0_weit(sub_jk_img, sub_jk_cont, pixel, xn, yn, r_bins_1)
		sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
		r_arr = Angl_r

		with h5py.File(J_sub_sb % nn, 'w') as f:
			f['r'] = np.array(r_arr)
			f['sb'] = np.array(sb_arr)
			f['sb_err'] = np.array(sb_err_arr)
			f['nratio'] = np.array(nratio)
			f['npix'] = np.array(npix)

	## final jackknife SB profile
	tmp_sb = []
	tmp_r = []
	for nn in range( N_bin ):
		with h5py.File(J_sub_sb % nn, 'r') as f:
			r_arr = np.array(f['r'])[:-1]
			sb_arr = np.array(f['sb'])[:-1]
			sb_err = np.array(f['sb_err'])[:-1]
			npix = np.array(f['npix'])[:-1]
			nratio = np.array(f['nratio'])[:-1]

			idvx = npix < 1.
			sb_arr[idvx] = np.nan
			r_arr[idvx] = np.nan

			tmp_sb.append(sb_arr)
			tmp_r.append(r_arr)

	## save the sb result in unit " nanomaggies / arcsec^2 "
	tt_jk_R, tt_jk_SB, tt_jk_err, lim_R = jack_SB_func(tmp_sb, tmp_r, 0, N_bin)[4:]
	sb_lim_r = np.ones( len(tt_jk_R) ) * lim_R

	with h5py.File(jack_SB_file, 'w') as f:
		f['r'] = np.array(tt_jk_R)
		f['sb'] = np.array(tt_jk_SB)
		f['sb_err'] = np.array(tt_jk_err)
		f['lim_r'] = np.array(sb_lim_r)

raise
'''
rep_err = []
rep_r = []

for mm in range(4):

	with h5py.File(jack_img % Ns[mm], 'r') as f:
		stack_img = np.array(f['a'])

	id_nan = np.isnan(stack_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = stack_img[y_low: y_up+1, x_low: x_up + 1]
	img_block, grd_pix = cc_grid_img(dpt_img, 100, 100)[:2]

	with h5py.File(jack_SB_file % Ns[mm], 'r') as f:
		c_r_arr = np.array(f['r'])
		c_sb_arr = np.array(f['sb'])
		c_sb_err = np.array(f['sb_err'])

	## figure the result
	sb_dat = pds.read_csv('input_SB.csv')
	put_r, put_sb = np.array(sb_dat['r']), np.array(sb_dat['sb'])

	rep_r.append(c_r_arr)
	rep_err.append(c_sb_err)

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
	ax1 = fig.add_axes([0.55, 0.26, 0.40, 0.64])
	ax2 = fig.add_axes([0.55, 0.10, 0.40, 0.16])

	ax0.set_title('stacking mock imgs [%d]' % Ns[mm],)
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]')
	cb.formatter.set_powerlimits((0,0))

	ax1.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = 'stacking mock img',)
	ax1.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)
	ax1.plot(put_r, put_sb, color = 'b', alpha = 0.5, ls = '-', label = 'input')
	ax1.set_ylim(4e-4, 7e-3)
	ax1.set_xlim(5e1, 1e3)
	ax1.set_ylabel('SB [nanomaggies / $arcsec^2$]')
	ax1.set_xscale('log')
	ax1.legend(loc = 1, frameon = False, fontsize = 8)
	ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax1.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

	interp_sb = interp.interp1d(put_r, put_sb, kind = 'cubic')
	ref_sb = interp_sb(c_r_arr)

	ax2.plot(put_r, put_sb / put_sb, color = 'b', alpha = 0.5,)
	ax2.plot(c_r_arr, c_sb_arr / ref_sb, color = 'r', ls = '-', alpha = 0.8,)
	ax2.fill_between(c_r_arr, y1 = (c_sb_arr - c_sb_err) / ref_sb , y2 = (c_sb_arr + c_sb_err) / ref_sb,
		color = 'r', alpha = 0.2,)

	ax2.set_xlim(ax1.get_xlim())
	ax2.set_xscale('log')
	ax2.set_xlabel('$ R[arcsec] $')
	ax2.set_ylim(0.95, 1.05)
	ax2.set_ylabel('$ SB / SB_{input} $')
	ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
	ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
	ax1.set_xticks([])

	plt.savefig('stack_mock_%d-imgs.png' % Ns[mm], dpi = 300)
	plt.close()

plt.figure()
ax = plt.subplot(111)
for mm in range(4):
	ax.plot(rep_r[mm][-62:], rep_err[mm][-62:] / rep_err[0][-62:], ls = '-', color = mpl.cm.plasma(mm / 4), 
		alpha = 0.5, label = 'mock imgs [%d]' % Ns[mm])
ax.axhline( y = 1/2, ls =  '-', color = 'k', alpha = 0.5,)
ax.axhline( y = 1/3, ls =  '--', color = 'k', alpha = 0.5, label = '1/3')
ax.axhline( y = 1/np.sqrt(10), ls = ':', color = 'k', alpha = 0.5, label = '1/$\\sqrt{10}$')

ax.set_ylim(0.1, 1.1)
ax.set_yscale('log')
ax.set_xlim(1e1, 1e3)
ax.set_ylabel('$err_{SB}$ / $err_{SB, 100}$')
ax.set_xlabel('R [arcsec]')
ax.set_xscale('log')
ax.legend(loc = 4, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

plt.subplots_adjust(left = 0.15, right = 0.95,)
plt.savefig('mock_err_test.png', dpi = 300)
plt.show()
"""

######## obs. mock
pdt = pds.read_csv('phy_input_SB.csv')
phy_R, put_sb = np.array(pdt.r), np.array(pdt.sb)
interp_sb = interp.interp1d(phy_R, put_sb, kind = 'cubic')

jack_SB_file = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_SB-pro_z-ref.h5'
jack_img = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_img_z-ref.h5'
jack_cont_arr = load + 'stack_mock/mock-A_BCG-stack_Mean_jack_pix-cont_z-ref.h5'

with h5py.File( jack_img, 'r') as f:
	tt_img = np.array(f['a'])
img_block, grd_pix = cc_grid_img(tt_img, 100, 100)[:2]

with h5py.File( jack_SB_file, 'r') as f:
	c_r_arr = np.array(f['r'])
	c_sb_arr = np.array(f['sb'])
	c_sb_err = np.array(f['sb_err'])

fig = plt.figure( figsize = (19.84, 4.8) )
a_ax0 = fig.add_axes([0.03, 0.09, 0.30, 0.85])
a_ax1 = fig.add_axes([0.38, 0.09, 0.30, 0.85])
a_ax2 = fig.add_axes([0.73, 0.09, 0.25, 0.85])

tf = a_ax0.imshow(grd_pix, origin = 'lower', cmap = 'rainbow', vmin = 1, vmax = grd_pix.max(),)
cb = plt.colorbar(tf, ax = a_ax0, fraction = 0.035, pad = 0.01, label = 'pixel count',)
tf.cmap.set_under('white')
cb.formatter.set_powerlimits((0,0))

idux = (grd_pix <= 500) & (grd_pix > 0)
poy, pox = np.where(idux == True)
for ll in range( np.sum(idux)):
	a_ax0.text(pox[ll], poy[ll], s = '%d' % (grd_pix[ poy[ll],pox[ll] ]), fontsize = 4, color = 'w', ha = 'center',)

tg = a_ax1.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = a_ax1, fraction = 0.035, pad = 0.01,)
cb.formatter.set_powerlimits((0,0))

a_ax2.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8,)
a_ax2.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

a_ax2.set_ylim(1e-3, 3e-2)
a_ax2.set_yscale('log')
a_ax2.set_xlim(5e1, 4e3)
a_ax2.set_xlabel('R [kpc]')
a_ax2.set_ylabel('SB [nanomaggies / $arcsec^2$]')
a_ax2.set_xscale('log')
a_ax2.legend(loc = 1, frameon = False, fontsize = 8)
a_ax2.grid(which = 'both', axis = 'both', alpha = 0.25)
a_ax2.tick_params(axis = 'both', which = 'both', direction = 'in',)
#a_ax2.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
tick_form = mpl.ticker.LogFormatter(labelOnlyBase = False)
a_ax2.get_yaxis().set_minor_formatter(tick_form)

plt.savefig('grid_2D_SB_mock_z-ref.png', dpi = 300)
plt.close()


plt.figure()
gs = gridspec.GridSpec(2,1, height_ratios = [4,1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])

ax.plot(c_r_arr, c_sb_arr, ls = '-', color = 'r', alpha = 0.8, label = 'mock img')
ax.fill_between(c_r_arr, y1 = c_sb_arr - c_sb_err, y2 = c_sb_arr + c_sb_err, color = 'r', alpha = 0.2,)

ax.plot(phy_R, put_sb, color = 'g', ls = '-', alpha = 0.5, label = 'input',)
#ax.set_xlabel('$R[kpc]$')
ax.set_ylabel('$SB[mag / arcsec^2]$')
ax.set_ylim(1e-3, 3e-2)
ax.set_xlim(5e1, 4e3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc = 1, frameon = False,)
ax.grid(which = 'both', axis = 'both')
ax.tick_params(axis = 'both', which = 'both', direction = 'in')

r_low = np.max([ phy_R.min(), c_r_arr.min() ])
r_hig = np.min([ phy_R.max(), c_r_arr.max() ])
idvx = (c_r_arr >= r_low) & (c_r_arr <= r_hig)
ref_R = c_r_arr[idvx]
ref_sb = interp_sb(ref_R)

bx.plot(phy_R, put_sb / put_sb, color = 'g', ls = '-', alpha = 0.5,)
bx.plot(ref_R, c_sb_arr[idvx] / ref_sb, color = 'r', alpha = 0.8, )
bx.fill_between(ref_R, y1 = (c_sb_arr[idvx] - c_sb_err[idvx]) / ref_sb, 
	y2 = (c_sb_arr[idvx] + c_sb_err[idvx]) / ref_sb, color = 'r', alpha = 0.2,)
idux = (c_r_arr >= 8e1) & (c_r_arr <= 4e3)
idwx = (ref_R >= 8e1) & (ref_R <= 4e3)
bx.axhline(y = np.mean(c_sb_arr[idux] / ref_sb[idwx]), ls = '--', color = 'b', alpha = 0.5,)

ax.bar(50, height = 10, width = 60, align = 'center', color = 'k', alpha = 0.5,)
bx.bar(50, height = 10, width = 60, align = 'center', color = 'k', alpha = 0.5,)

bx.set_xlim(ax.get_xlim())
bx.set_xscale('log')
bx.set_xlabel('R [kpc]')
bx.set_ylim(0.95, 1.05)
bx.set_ylabel('$ SB / SB_{input} $')
bx.grid(which = 'both', axis = 'both')
bx.tick_params(axis = 'both', which = 'both', direction = 'in')
ax.set_xticks([])

plt.subplots_adjust(left = 0.15, right = 0.95, hspace = 0.02,)
plt.savefig('SB-ref_compare.png', dpi = 300)
plt.close()
