import h5py
import pandas as pds
import numpy as np

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits

import scipy.stats as sts
import scipy.special as special
from scipy import optimize
from scipy.stats import binned_statistic as binned

# constant
vc = C.c.to(U.km/U.s).value
G = C.G.value # gravitation constant
Ms = C.M_sun.value # solar mass
kpc2m = U.kpc.to(U.m)
Msun2kg = U.M_sun.to(U.kg)

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
Lsun2erg_s = U.L_sun.to(U.erg/U.s)
rad2arcsec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

def pix_flux_set_func(x0, y0, m_cen_x, m_cen_y, set_ra, set_dec, set_z, set_x, set_y, img_file, band_str,):
	'''
	x0, y0 : the point which need to check flux (in the stacking image)
	m_cen_x, m_cen_y : the center pixel of stacking image
	set_ra, set_dec, set_z, set_x, set_y : img information of stacking sample imgs,
			including ra, dec, z, and BCG position on img (set_x, set_y)
	img_file : imgs for given catalog ('XXX/XX.fits')
	'''
	targ_f = []
	Ns = len( set_z )

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]
		img_x, img_y = set_x[kk], set_y[kk]

		data = fits.open( img_file % (band_st, ra_g, dec_g, z_g) )
		img = data[0].data

		dev_05_x = img_x - np.int( img_x )
		dev_05_y = img_y - np.int( img_y )

		if dev_05_x > 0.5:
			x_cen = np.int( img_x ) + 1
		else:
			x_cen = np.int( img_x )

		if dev_05_y > 0.5:
			y_cen = np.int( img_y ) + 1
		else:
			y_cen = np.int( img_y )

		x_ori_0, y_ori_0 = x0 + x_cen - m_cen_x, y0 + y_cen - m_cen_y

		identy_0 = ( (x_ori_0 >= 0) & (x_ori_0 < 2048) ) & ( (y_ori_0 >= 0) & (y_ori_0 < 1489) )

		if identy_0 == True:
			targ_f.append( img[y_ori_0, x_ori_0] )

	targ_f = np.array( targ_f )

	return targ_f

def radi_bin_flux_set_func(targ_R, stack_img, m_cen_x, m_cen_y, R_bins, set_ra, set_dec, set_z, set_x, set_y,
	img_file, pix_size, band_str, out_file,):
	"""
	targ_R : the radius bin in which need to check flux
	stack_img : the stacking img file (in .h5 format, img = np.array(f['a']),)
	R_bins : the radius bins will be applied on the stacking img
	set_ra, set_dec, set_z, set_x, set_y : the catalog imformation, (set_x, set_y) is BCG position
		on img
	img_file : imgs for given catalog ('XXX/XX.fits')
	pix_size : the pixel scale of imgs
	band_str : filter, str type
	out_file : out-put the data (.csv file)
	"""
	with h5py.File(stack_img, 'r') as f:
		tt_img = np.array( f['a'] )

	R_angle = 0.5 * (R_bins[1:] + R_bins[:-1]) * pix_size

	Nx = np.linspace(0, tt_img.shape[1] - 1, tt_img.shape[1] )
	Ny = np.linspace(0, tt_img.shape[0] - 1, tt_img.shape[0] )
	grd = np.array( np.meshgrid(Nx, Ny) )
	cen_dR = np.sqrt( (grd[0] - m_cen_x)**2 + (grd[1] - m_cen_y)**2 )

	ddr = np.abs( R_angle - targ_R )
	idx = np.where( ddr == ddr.min() )[0]
	edg_lo = R_bins[idx]
	edg_hi = R_bins[idx + 1]

	id_flux = (cen_dR >= edg_lo) & (cen_dR < edg_hi)
	id_nn = np.isnan( tt_img )
	id_effect = ( id_nn == False ) & id_flux
	f_stack = tt_img[ id_effect ]

	lx = np.linspace(0, 2047, 2048)
	ly = np.linspace(0, 1488, 1489)
	grd_lxy = np.array( np.meshgrid(lx, ly) )

	sub_f_arr = []
	Ns = len( set_z )

	for kk in range( Ns ):

		ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]
		img_x, img_y = set_x[kk], set_y[kk]

		data = fits.open( img_file % (band_str, ra_g, dec_g, z_g),)
		img = data[0].data

		dev_05_x = img_x - np.int( img_x )
		dev_05_y = img_y - np.int( img_y )

		if dev_05_x > 0.5:
			x_cen = np.int( img_x ) + 1
		else:
			x_cen = np.int( img_x )

		if dev_05_y > 0.5:
			y_cen = np.int( img_y ) + 1
		else:
			y_cen = np.int( img_y )

		## identy by radius edges
		dr = np.sqrt( (grd_lxy[0] - x_cen)**2 + (grd_lxy[1] - y_cen)**2)
		id_last = (dr >= edg_lo) & (dr < edg_hi)
		idnn = np.isnan( img )
		idin = id_last & (idnn == False)

		if np.sum( idin ) == 0:
			continue
		else:
			sub_f_arr.append( img[ idin ] )

	dtf = np.hstack( sub_f_arr )

	out_data = pds.DataFrame(dtf, columns = ['pix_flux'], dtype = np.float32)
	out_data.to_csv( out_file % targ_R,)

	return

def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

############## test for SB in radius bins
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

pixel = 0.396
band = ['r', 'g', 'i']

home = '/home/xkchen/'
load = '/home/xkchen/project/'

dat = pds.read_csv(load + 'random_r-band_tot_remain_mock-BCG-pos_0-rank.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)


d_file = home + 'data/SDSS/tmp_stack/random/random_mask_%s_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

n_rbins = 100
R_bins = np.logspace(0, np.log10(2520), n_rbins)

stack_img = '/home/xkchen/fig_tmp/stack/random_r-band_rand-stack_Mean_jack_img_0-rank.h5'

with h5py.File( stack_img, 'r') as f:
	tt_img = np.array( f['a'] )

xn, yn = np.int( tt_img.shape[1] / 2), np.int( tt_img.shape[0] / 2)	

out_file = load + 'stack/%.3f-arcsec_sample-img_flux.csv'

## Mean flus profile
cdat = pds.read_csv(load + 'stack/Mean_f_pros.csv')
Angl_r = np.array( cdat['R_arcsec'] )
flux_r = np.array( cdat['f_mean'] )

idNul = np.isnan( flux_r )
angl_r = Angl_r[idNul == False]
Nr = len(angl_r)

'''
## collect sample imgs flux for given radius bin
m, n = divmod( Nr, cpus)
N_sub0, N_sub1 = m * rank, (rank + 1) * m
if rank == cpus - 1:
	N_sub1 += n

tmp_R = angl_r[N_sub0 : N_sub1]

for mm in range( len(tmp_R) ):

	targ_R = tmp_R[mm]

	radi_bin_flux_set_func(targ_R, stack_img, xn, yn, R_bins, ra, dec, z, clus_x, clus_y,
		d_file, pixel, band[0], out_file,)

print('finished')
'''

fig = plt.figure( figsize = (60, 24) )
gs = gridspec.GridSpec(Nr // 6 + 1, 6)

# hist compare
tmp_r, tmp_mu, tmp_chi = [], [], []
sam_tmp_mode_f, sam_tmp_mean_f, sam_tmp_medi_f, sam_tmp_err = [], [], [], []

for mm in range( Nr ):

	tmp_r.append( angl_r[mm] )

	# flux of sample images
	f_dat = pds.read_csv(load + 'stack/%.3f-arcsec_sample-img_flux.csv' % angl_r[mm],)
	samp_flux = np.array( f_dat['pix_flux'] )

	id_nn = np.isnan(samp_flux)
	samp_flux = samp_flux[id_nn == False] / pixel**2

	bin_wide_samp = 1e-2
	bin_edgs_samp = np.arange( samp_flux.min(), samp_flux.max(), bin_wide_samp,)

	sam_tmp_mean_f.append( np.mean(samp_flux) )
	sam_tmp_medi_f.append( np.median(samp_flux) )
	sam_tmp_err.append( np.std(samp_flux) )

	# pdf of sample img flux
	N_pix, edg_f = binned(samp_flux, samp_flux, statistic = 'count', bins = bin_edgs_samp)[:2]

	pdf_pix = (N_pix / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	pdf_err = ( np.sqrt(N_pix) / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	f_cen = 0.5 * ( edg_f[1:] + edg_f[:-1])

	mean_f = np.mean( samp_flux )
	medi_f = np.median( samp_flux )

	try:
		idu = N_pix != 0.

		popt, pcov = optimize.curve_fit(gau_func, f_cen[idu], pdf_pix[idu], p0 = [mean_f, np.std(samp_flux) ], sigma = pdf_err[idu],)
		e_mu, e_chi = popt[0], popt[1]
		fit_line = gau_func(f_cen, e_mu, e_chi)

		mode_f = f_cen[ fit_line == fit_line.max() ][0]
	except:
		popt = None
		e_mu, e_chi = mean_f, np.std(samp_flux)
		mode_f = f_cen[ pdf_pix == pdf_pix.max() ][0]

	sam_tmp_mode_f.append( mode_f )
	tmp_mu.append( e_mu )
	tmp_chi.append( e_chi )

	ax0 = plt.subplot( gs[mm // 6, mm % 6] )
	ax0.hist(samp_flux, bins = bin_edgs_samp, density = False, color = 'b', alpha = 0.5,)
	ax0.errorbar(f_cen, N_pix, yerr = np.sqrt(N_pix), xerr = None, color = 'g', marker = '.', ms = 3, mec = 'g', mfc = 'none',
		ls = '', ecolor = 'g', elinewidth = 1,)
	ax0.plot(f_cen, fit_line * np.sum(N_pix) * (edg_f[1] - edg_f[0]), color = 'g', alpha = 0.5,
		label = 'G$[\\mu = %.5f, \\sigma = %.5f] $' % (e_mu, e_chi),)

	ax0.axvline(x = mean_f, ymin = 0.80, ymax = 0.95, ls = '-', color = 'r', alpha = 0.5, label = 'mean [%.5f]' % mean_f,)
	ax0.axvline(x = medi_f, ymin = 0.45, ymax = 0.60, ls = '--', color = 'r', alpha = 0.5, label = 'median [%.5f]' % medi_f,)
	ax0.axvline(x = mode_f, ymin = 0.1, ymax = 0.25, ls = ':', color = 'r', alpha = 0.5, label = 'mode [%.5f]' % mode_f,)

	ax0.annotate(s = '%d, R = %.3f arcsec' % (mm,angl_r[mm]), xy = (0.05, 0.1), xycoords = 'axes fraction', color = 'k', fontsize = 8,)	

	ax0.legend( loc = 2, fontsize = 8, frameon = False,)
	ax0.set_yscale('log')
	ax0.set_ylim(1, N_pix.max() + 10)

	if mm % 6 == 0:
		ax0.set_xlabel('SB [nanomaggies /$arcsec^2$]')
		ax0.set_ylabel('# of pixels')

	plt.figure()
	ax = plt.subplot(111)

	ax.hist(samp_flux, bins = bin_edgs_samp, density = True, color = 'b', alpha = 0.5,)
	ax.axvline(x = mean_f, ymin = 0.4, ymax = 0.55, ls = '-', color = 'r', alpha = 0.5, label = 'mean [%.5f]' % mean_f,)
	ax.axvline(x = medi_f, ymin = 0.2, ymax = 0.35, ls = '--', color = 'r', alpha = 0.5, label = 'median [%.5f]' % medi_f,)
	ax.axvline(x = mode_f, ymin = 0.0, ymax = 0.15, ls = ':', color = 'r', alpha = 0.5, label = 'mode [%.5f]' % mode_f,)

	if popt is not None:
		ax.plot(f_cen, fit_line, ls = '-', color = 'g', alpha = 0.5, label = 'G$[\\mu = %.5f, \\sigma = %.5f] $' % (e_mu, e_chi),)

	ax.set_xlabel('SB [nanomaggies /$arcsec^2$]')
	ax.set_ylabel('PDF')
	ax.legend( loc = 1,)
	ax.set_xlim( mean_f - 5 * np.std(samp_flux), mean_f + 5 * np.std(samp_flux))

	plt.savefig('/home/xkchen/fig_tmp/figs/0-rank_random_rand-stack_sample-img_R_%.3f-arcsec_flux_hist.png' % angl_r[mm], dpi = 300)
	plt.close()

plt.subplots_adjust(left = 0.03, bottom = 0.03, right = 0.98, top = 0.98, wspace = 0.07, hspace = 0.2,)
plt.savefig('0-rank_random_rand-stack_flux_hist_overview.pdf', dpi = 300)
plt.close()

tmp_r = np.array( tmp_r )

sam_tmp_mode_f = np.array( sam_tmp_mode_f )
sam_tmp_mean_f = np.array( sam_tmp_mean_f )
sam_tmp_medi_f = np.array( sam_tmp_medi_f )
sam_tmp_err = np.array( sam_tmp_err )

keys = ['R_arcsec', 'mean_sb', 'medi_sb', 'mode_sb', 'sb_std']
values = [ tmp_r, sam_tmp_mean_f, sam_tmp_medi_f, sam_tmp_mode_f, sam_tmp_err ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame(fill)
out_data.to_csv('0-rank_random_rand-stack_sample-img_SB-pros.csv')

tmp_mu = np.array( tmp_mu )
tmp_chi = np.array( tmp_chi )
keys = ['R_arcsec', 'e_mu', 'e_chi']
values = [ tmp_r, tmp_mu, tmp_chi ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame(fill)
out_data.to_csv('0-rank_random_rand-stack_sample-img_SB-hist_G-fit_record.csv')

print('fit finished !')

raise
'''
m_dat = pds.read_csv('/home/xkchen/project/ICL/random_r-band_tot_remain_img_aveg-flux.csv')
#m_dat = pds.read_csv('/home/xkchen/random_r-band_tot_remain_img_aveg-flux.csv')
aveg_f = np.array(m_dat['aveg_flux'])
N_pix = np.array(m_dat['N_eff'])
aveg_SB = np.sum(aveg_f * N_pix) / np.sum(N_pix) / pixel**2

dat_0 = pds.read_csv('/home/xkchen/project/ICL/sample-img_SB-pros.csv')
#dat_0 = pds.read_csv('/home/xkchen/sample-img_SB-pros.csv')
tmp_r0 = np.array( dat_0['R_arcsec'] )
sam_tmp_mean_f = np.array( dat_0['mean_sb'] )
sam_tmp_medi_f = np.array( dat_0['medi_sb'] )
sam_tmp_mode_f = np.array( dat_0['mode_sb'] )
sam_tmp_err = np.array( dat_0['sb_std'] )

dat_1 = pds.read_csv('/home/xkchen/project/ICL/SB-pros.csv')
#dat_1 = pds.read_csv('/home/xkchen/SB-pros.csv')
tmp_r1 = np.array( dat_1['R_arcsec'] )
tmp_mean_f = np.array( dat_1['mean_sb'] )
tmp_medi_f = np.array( dat_1['medi_sb'] )
tmp_mode_f = np.array( dat_1['mode_sb'] )


plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.plot(tmp_r1, tmp_mean_f, color = 'b', ls = ':', alpha = 0.5, label = 'stacking img, mean')
ax0.plot(tmp_r1, tmp_medi_f, color = 'b', ls = '--', alpha = 0.5, label = 'stacking img, median')
ax0.plot(tmp_r1, tmp_mode_f, color = 'b', ls = '-', alpha = 0.5, label = 'stacking img, mode')

ax0.plot(tmp_r0, sam_tmp_mean_f, color = 'r', ls = ':', alpha = 0.5, label = 'sample imgs')
ax0.plot(tmp_r0, sam_tmp_medi_f, color = 'r', ls = '--', alpha = 0.5,)
ax0.plot(tmp_r0, sam_tmp_mode_f, color = 'r', ls = '-', alpha = 0.5,)

Medi_SB = 0.00032615662 / pixel**2
ax0.axhline(y = aveg_SB, ls = '-', color = 'k', alpha = 0.5, label = 'mean SB of total imgs')
ax0.axhline(y = Medi_SB, ls = '--', color = 'k', alpha = 0.5, label = 'median SB of total imgs')

ax0.set_ylim(-2e-3, 8e-3)
ax0.set_xlim(1e1, 1e3)
ax0.set_ylabel('SB [nanomaggies / $arcsec^2$]')
ax0.set_xscale('log')
#ax0.set_xlabel('R [arcsec]')
ax0.legend(loc = 'upper center', fontsize = 8,)
ax0.grid(which = 'both', axis = 'both', alpha = 0.25,)
ax0.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax0.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)

ax1.plot(tmp_r1, tmp_mean_f - sam_tmp_mean_f, ls = ':', alpha = 0.5, color = 'g',)
ax1.plot(tmp_r1, tmp_medi_f - sam_tmp_medi_f, ls = '--', alpha = 0.5, color = 'g',)
ax1.plot(tmp_r1, tmp_mode_f - sam_tmp_mode_f, ls = '-', alpha = 0.5, color = 'g',)

ax1.set_ylim(-5e-3, 5e-3)
ax1.set_xlim( ax0.get_xlim() )
ax1.set_xscale( 'log' )
ax1.set_xlabel('R [arcsec]')
ax1.set_ylabel('$ SB_{stack} - SB_{sample} $')
ax1.grid(which = 'both', axis = 'both', alpha = 0.25)
ax1.tick_params(axis = 'both', which = 'both', direction = 'in',)
ax1.ticklabel_format(axis = 'y', style = 'scientific', scilimits = (0,0),)
ax0.set_xticklabels( labels = [], fontsize = 0.005,)

plt.subplots_adjust(hspace = 0.09)
#plt.savefig('SB-pros_compare.png', dpi = 300)
plt.savefig('/home/xkchen/fig_tmp/figs/SB-pros_compare.png', dpi = 300)
plt.close()
'''

'''
## hist compare (~50, 100, 500, 900, and the last radius)
targ_r = np.array([50, 100, 500, 900, tmp_r0[-1] ])

line_c = ['b', 'g', 'c', 'm', 'r']

r_id = []

for mm in range( 5 ):

	DR = np.abs(tmp_r0 - targ_r[mm])
	idr = np.where( DR == DR.min() )[0][0]

	r_id.append( idr )

fig = plt.figure()
ax = plt.subplot(111)
add_ax = fig.add_axes([0.15, 0.60, 0.30, 0.20])

for jj in range( 5 ):

	order = r_id[jj]

	f_dat = pds.read_csv(load + 'stack/%.3f-arcsec_sample-img_flux.csv' % tmp_r0[order],)
	samp_flux = np.array( f_dat['pix_flux'] )

	id_nn = np.isnan(samp_flux)
	samp_flux = samp_flux[id_nn == False] / pixel**2

	bin_wide = 1e-2
	bin_edgs = np.arange( samp_flux.min(), samp_flux.max(), bin_wide)

	mean_f = np.mean( samp_flux )
	medi_f = np.median( samp_flux )

	ax.hist(samp_flux, bins = bin_edgs, density = True, histtype = 'step', color = line_c[jj], alpha = 0.5,
		label = '$%.3f arcsec$' % tmp_r0[order],)

	if jj == 0:
		ax.axvline(x = mean_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '-', color = line_c[jj], alpha = 0.5, label = 'mean',)
		ax.axvline(x = medi_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '--', color = line_c[jj], alpha = 0.5, label = 'median',)
	else:
		ax.axvline(x = mean_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '-', color = line_c[jj], alpha = 0.5,)
		ax.axvline(x = medi_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '--', color = line_c[jj], alpha = 0.5,)

	add_ax.hist(samp_flux, bins = bin_edgs, density = True, histtype = 'step', color = line_c[jj], alpha = 0.5,)
	add_ax.axvline(x = mean_f, ls = '-', color = line_c[jj], alpha = 0.5, linewidth = 1,)
	add_ax.axvline(x = medi_f, ls = '--', color = line_c[jj], alpha = 0.5,linewidth = 1,)

print('line 489')

add_ax.set_xlim(-2e-3, 8e-3)
add_ax.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0,0),)
add_ax.set_yticks([])

ax.set_xlabel('SB [nanomaggies /$arcsec^2$]')
ax.set_ylabel('PDF')
ax.legend( loc = 1,)
ax.set_xlim(-0.5, 0.5)

plt.savefig('/home/xkchen/fig_tmp/figs/sample-img_differ-R_SB-hist_compare.png', dpi = 300)
plt.close()
'''
