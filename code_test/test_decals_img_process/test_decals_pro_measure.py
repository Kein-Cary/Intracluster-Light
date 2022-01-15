import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.units as U
import astropy.constants as C
import astropy.io.fits as fits

from astropy import cosmology as apcy
from light_measure import light_measure_Z0_weit

## constant
vc = C.c.to(U.km/U.s).value
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

pixel = 0.396
## SDSS photometric radius bins (unit: arcsec)
cat_Rii = np.array([0.23,  0.68,  1.03,   1.76,   3.00, 
					4.63,  7.43,  11.42,  18.20,  28.20, 
					44.21, 69.00, 107.81, 168.20, 263.00])
########
home = '/media/xkchen/My Passport/data/SDSS/'
load = '/home/xkchen/mywork/ICL/'

dat = pds.read_csv(load + 'code/SEX/result/test_1000-to-250_cat-match.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
rich, r_mag = np.array(dat.rich), np.array(dat.r_Mag)

cp_dat = pds.read_csv(load + 'code/SEX/result/test_1000-to-250_cat.csv')
clus_x, clus_y = np.array(cp_dat.bcg_x), np.array(cp_dat.bcg_y)

D_l = Test_model.luminosity_distance(z).value
r_Mag = r_mag + 5 - 5 * np.log10(10**6 * D_l)

Ns = len(ra)
'''
## single image measurement
for kk in range( Ns ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]
	cen_x, cen_y = clus_x[kk], clus_y[kk]

	## SDSS mask img
	sdss_data = fits.open(home + 'tmp_stack/cluster/cluster_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % (ra_g, dec_g, z_g),)
	sdss_img = sdss_data[0].data

	sdss_origin = fits.open(home + 'wget_data/frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_g, dec_g, z_g),)
	ori_img = sdss_origin[0].data

	source_cat = asc.read(load + 'data/source_find/cluster_r-band_mask_ra%.3f_dec%.3f_z%.3f.cat' % (ra_g, dec_g, z_g),)
	A = np.array(source_cat['A_IMAGE'])
	B = np.array(source_cat['B_IMAGE'])
	cx = np.array(source_cat['X_IMAGE']) - 1
	cy = np.array(source_cat['Y_IMAGE']) - 1

	Kron = 16
	a = Kron * A
	b = Kron * B

	tdr = np.sqrt( (cen_x - cx)**2 + (cen_y - cy)**2 )
	idx = tdr == np.min(tdr)
	#lr = a[idx]
	lr = 1.2 * A[idx] * 4.5

	set_r = np.int(np.ceil(1.0 * lr))
	la0 = np.max( [np.int(cx[idx] - set_r), 0])
	la1 = np.min( [np.int(cx[idx] + set_r +1), sdss_img.shape[1] ] )
	lb0 = np.max( [np.int(cy[idx] - set_r), 0] )
	lb1 = np.min( [np.int(cy[idx] + set_r +1), sdss_img.shape[0] ] )

	img_with_bcg = sdss_img.copy()
	img_with_bcg[lb0: lb1, la0: la1] = ori_img[lb0: lb1, la0: la1]

	id_nan = np.isnan(img_with_bcg)
	count_arr = np.ones((sdss_img.shape[0], sdss_img.shape[1]), np.float32)
	count_arr[id_nan] = np.nan

	#xn, yn = np.around(cen_x), np.around(cen_y)
	xn, yn = cen_x, cen_y
	limR = 600 ## in unit of pixel number
	r_bins = np.logspace(0, np.log10(limR), 65)

	Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(img_with_bcg, count_arr, pixel, cen_x, cen_y, r_bins)
	sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
	r_arr = Intns_r.copy()

	id_nu = npix < 1
	r_arr[ id_nu ] = np.nan
	sb_arr[ id_nu ] = np.nan
	sb_err_arr[ id_nu ] = np.nan
	npix[ id_nu ] = np.nan
	nratio[ id_nu ] = np.nan

	with h5py.File(load + 'data/tmp_img/A-250_test_pros/SB_pro_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'w') as f:
		f['r'] = np.array(r_arr)
		f['sb'] = np.array(sb_arr)
		f['sb_err'] = np.array(sb_err_arr)
		f['nratio'] = np.array(nratio)
		f['npix'] = np.array(npix)

	## DECaLS imgs
	decals_data = fits.open(
		'/media/xkchen/My Passport/data/BASS/A_250_mask/ap_sdss_mask_r_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g),)
	decals_img = decals_data[0].data

	decals_origin = fits.open(load + 'data/tmp_img/A_250_to_SDSS/decals_cut_r_ra%.3f_dec%.3f_z%.3f.fits'%(ra_g, dec_g, z_g),)
	decals_ori_img = decals_origin[0].data

	cp_decals_img = decals_img.copy()
	cp_decals_img[lb0: lb1, la0: la1] = decals_ori_img[lb0: lb1, la0: la1]

	Intns, Intns_r, Intns_err, npix, nratio = light_measure_Z0_weit(cp_decals_img, count_arr, pixel, cen_x, cen_y, r_bins)
	sb_arr, sb_err_arr = Intns / pixel**2, Intns_err / pixel**2
	r_arr = Intns_r.copy()

	id_nu = npix < 1
	r_arr[ id_nu ] = np.nan
	sb_arr[ id_nu ] = np.nan
	sb_err_arr[ id_nu ] = np.nan
	npix[ id_nu ] = np.nan
	nratio[ id_nu ] = np.nan

	with h5py.File(load + 'data/tmp_img/A-250_test_pros/decals_SB_pro_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'w') as f:
		f['r'] = np.array(r_arr)
		f['sb'] = np.array(sb_arr)
		f['sb_err'] = np.array(sb_err_arr)
		f['nratio'] = np.array(nratio)
		f['npix'] = np.array(npix)

raise
'''
'''
fig = plt.figure(figsize = (24, 24))
fig.suptitle('BCG pros')
gs = gridspec.GridSpec(30 // 5, 5)

for kk in range( 30 ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]
	cen_x, cen_y = clus_x[kk], clus_y[kk]

	## SDSS photometry pros
	#	the band info. 0, 1, 2, 3, 4 --> u, g, r, i, z
	cat_pro = pds.read_csv(load + 'data/BCG_pros/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g), skiprows = 1)
	dat_band = np.array(cat_pro.band)
	dat_bins = np.array(cat_pro.bin)
	dat_pro = np.array(cat_pro.profMean) # in unit of nmaggy / arcsec^2
	dat_pro_err = np.array(cat_pro.profErr)

	id_band = dat_band == 2
	obs_r = cat_Rii[ dat_bins[id_band] ]
	obs_pro = dat_pro[id_band]
	obs_pro_err = dat_pro_err[id_band]

	## measured based on SDSS img 
	with h5py.File(load + 'data/tmp_img/A-250_test_pros/SB_pro_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'r') as f:
		sdss_r = np.array(f['r'])
		sdss_sb = np.array(f['sb'])
		sdss_sb_err = np.array(f['sb_err'])

	## measured based on DECaLS img 
	with h5py.File(load + 'data/tmp_img/A-250_test_pros/decals_SB_pro_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'r') as f:
		desi_r = np.array(f['r'])
		desi_sb = np.array(f['sb'])
		desi_sb_err = np.array(f['sb_err'])

	ax = plt.subplot(gs[kk // 5, kk % 5])
	ax.set_title('ra%.3f dec%.3f z%.3f [$ M_{r} = %.3f$]' % (ra_g, dec_g, z_g, r_Mag[kk]),)

	ax.plot(sdss_r, sdss_sb, color = 'r', ls = '-', alpha = 0.5, label = 'SDSS')
	ax.fill_between(sdss_r, y1 = sdss_sb - sdss_sb_err, y2 = sdss_sb + sdss_sb_err, color = 'r', alpha = 0.5,)

	ax.plot(desi_r, desi_sb, color = 'b', ls = '-', alpha = 0.5, label = 'DECaLS')
	ax.fill_between(desi_r, y1 = desi_sb - desi_sb_err, y2 = desi_sb + desi_sb_err, color = 'b', alpha = 0.5,)

	ax.plot(desi_r, desi_sb * 3, color = 'g', ls = '-', alpha = 0.5, label = 'DECaLS * 3')
	ax.fill_between(desi_r, y1 = desi_sb * 3 - desi_sb_err, y2 = desi_sb * 3 + desi_sb_err, color = 'g', alpha = 0.5,)

	ax.errorbar(obs_r, obs_pro, yerr = obs_pro_err, xerr = None, ls = '--', fmt = 'k.', alpha = 0.5, label = 'SDSS photometric catalog')

	ax.set_ylim(1e-3, 1e1)
	ax.set_xlim(2e-1, 2e1)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.legend(loc = 3, frameon = False,)
	ax.grid(which = 'both', axis = 'both', alpha = 0.25)
	ax.tick_params(axis = 'both', which = 'both', direction = 'in',)

	if kk // 5 == 5:

		ax.set_xlabel('R [arcsec]')
		ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')

plt.tight_layout()
plt.savefig('decals_pro_measure_test.pdf', dpi = 300)
plt.close()
'''

## stacked profile case
bin_Mag = np.array([r_Mag.min(), -23.5, -22.5, r_Mag.max()])

B_dex = 2
id_lim = (r_Mag >= bin_Mag[ B_dex ]) & (r_Mag <= bin_Mag[ B_dex + 1])
set_ra, set_dec, set_z = ra[id_lim], dec[id_lim], z[id_lim]
N_lim = len(set_ra)

ref_pro = []
ref_Rii = []

for kk in range( N_lim ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]
	cen_x, cen_y = clus_x[kk], clus_y[kk]

	## SDSS photometry pros
	#	the band info. 0, 1, 2, 3, 4 --> u, g, r, i, z
	cat_pro = pds.read_csv(load + 'data/BCG_pros/BCG_prof_Z%.3f_ra%.3f_dec%.3f.txt' % (z_g, ra_g, dec_g), skiprows = 1)
	dat_band = np.array(cat_pro.band)
	dat_bins = np.array(cat_pro.bin)
	dat_pro = np.array(cat_pro.profMean) # in unit of nmaggy / arcsec^2
	dat_pro_err = np.array(cat_pro.profErr)

	id_band = dat_band == 2
	obs_r = cat_Rii[ dat_bins[id_band] ]
	obs_pro = dat_pro[id_band]
	obs_pro_err = dat_pro_err[id_band]

	ref_pro.append(obs_pro)
	ref_Rii.append(obs_r)

mm_R = cat_Rii.copy()
mm_F = np.zeros((N_lim, len(cat_Rii) ), dtype = np.float)

for kk in range( N_lim ):

	put_pro = ref_pro[kk]
	put_r = ref_Rii[kk]
	Len_r = len(put_r)

	for pp in range(Len_r):
		dr = np.abs(mm_R - put_r[pp])
		idx = dr == np.nanmin(dr)
		mm_F[pp,:][idx] = put_pro[pp]

id_zero = mm_F == 0.
mm_F[id_zero == True] = np.nan

mean_flux = np.nanmean(mm_F, axis = 0)
std_flux = np.nanstd(mm_F, axis = 0)

mm_Ns = np.zeros(len(mean_flux), dtype = np.float)
for q in range( len(cat_Rii) ):
	id_nan = np.isnan(mm_F[:, q])
	mm_Ns[q] = len(mm_F[id_nan == False])
std_flux = std_flux / np.sqrt(mm_Ns - 1)

## SB pros. based on imgs
sdss_com_r = []
sdss_com_sb = []
desi_com_r = []
desi_com_sb = []

for kk in range( N_lim ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

	with h5py.File(load + 'data/tmp_img/A-250_test_pros/SB_pro_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
	sdss_com_r.append(tt_r)
	sdss_com_sb.append(tt_sb)

	with h5py.File(load + 'data/tmp_img/A-250_test_pros/decals_SB_pro_ra%.3f_dec%.3f_z%.3f.h5' % (ra_g, dec_g, z_g), 'r') as f:
		tt_r = np.array(f['r'])
		tt_sb = np.array(f['sb'])
	desi_com_r.append(tt_r)
	desi_com_sb.append(tt_sb)

sdss_com_r = np.array( sdss_com_r )
sdss_com_sb = np.array( sdss_com_sb )
sdss_R = np.nanmean(sdss_com_r, axis = 0)
sdss_SB = np.nanmedian(sdss_com_sb, axis = 0)
sdss_SB_err = np.nanstd(sdss_com_sb, axis = 0)

desi_com_r = np.array( desi_com_r )
desi_com_sb = np.array( desi_com_sb )
desi_R = np.nanmean(desi_com_r, axis = 0)
desi_SB = np.nanmedian(desi_com_sb, axis = 0)
desi_SB_err = np.nanstd(desi_com_sb, axis = 0)

plt.figure()
ax = plt.subplot(111)
ax.set_title('SB profile in angle coordinate [$ M_{r,BCG} = %.3f \\sim %.3f$]' % (bin_Mag[B_dex], bin_Mag[B_dex + 1]),)

ax.plot(sdss_R, sdss_SB, color = 'r', ls = '-', alpha = 0.5, label = 'SDSS')
ax.fill_between(sdss_R, y1 = sdss_SB - sdss_SB_err, y2 = sdss_SB + sdss_SB_err, color = 'r', alpha = 0.2)

ax.plot(desi_R, desi_SB, color = 'b', ls = '-', alpha = 0.5, label = 'DECaLS')
ax.fill_between(desi_R, y1 = desi_SB - desi_SB_err, y2 = desi_SB + desi_SB_err, color = 'b', alpha = 0.2)

ax.plot(desi_R, desi_SB * 3, color = 'g', ls = '-', alpha = 0.5, label = 'DECaLS * 3')
ax.fill_between(desi_R, y1 = desi_SB * 3 - desi_SB_err, y2 = desi_SB *3 + desi_SB_err, color = 'g', alpha = 0.2)

ax.errorbar(mm_R, mean_flux, yerr = std_flux, xerr = None, ls = '--', fmt = 'k.', alpha = 0.5, label = 'SDSS photometric catalog')

ax.set_ylim(1e-2, 2e1)
ax.set_xlim(4e-1, 2e1)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc = 3, frameon = False,)
ax.grid(which = 'both', axis = 'both', alpha = 0.25)
ax.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax.set_xlabel('R [arcsec]')
ax.set_ylabel('SB [nanomaggies / $arcsec^2$]')

plt.savefig('SB_in_obs-coordinate_B%d-sub.png' % B_dex, dpi = 300)
plt.close()

