import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.ascii as asc
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from astropy.coordinates import SkyCoord

import scipy.stats as sts
import scipy.special as special
from scipy import optimize
from scipy.stats import binned_statistic as binned

from light_measure import light_measure_Z0_weit
from light_measure import light_measure_weit
from light_measure import jack_SB_func

# cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396
band = ['r', 'g', 'i']

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

##### R-bins flux hist and pdf compare
load = '/home/xkchen/fig_tmp/'
home = '/home/xkchen/data/SDSS/'
z_ref = 0.25

kk = 0

id_cen = 0
n_rbins = 100
N_bin = 30

### analysis for radius bin pixel flux
def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

aveg_pros = ['/home/xkchen/project/stack/random_BCG-stack_%.3f-arcsec_flux-arr.csv',
			'/home/xkchen/project/stack/cluster_BCG-stack_%.3f-arcsec_flux-arr.csv']

samp_pros = ['/home/xkchen/project/stack/random_BCG-stack_%.3f-arcsec_sample-img_flux.csv',
			'/home/xkchen/project/stack/cluster_BCG-stack_%.3f-arcsec_sample-img_flux.csv']

mean_pros = ['/home/xkchen/project/stack/random_BCG-stack_Mean_f_pros.csv',
			'/home/xkchen/project/stack/cluster_BCG-stack_Mean_f_pros.csv']

line_name = ['random', 'cluster']
'''
## stack
for tt in range( 2 ):

	cdat = pds.read_csv( mean_pros[tt] )
	Angl_r = np.array( cdat['R_arcsec'] )
	flux_r = np.array( cdat['f_mean'] )

	idNul = np.isnan( flux_r )
	angl_r = Angl_r[idNul == False]
	Nr = len(angl_r)

	tmp_r = []
	tmp_mean_sb, tmp_medi_sb, tmp_mode_sb = [], [], []
	tmp_mu, tmp_chi = [], []

	for qq in range( len(angl_r) ):

		dat = pds.read_csv( aveg_pros[tt] % angl_r[qq])
		bin_flux = np.array( dat['flux'] )
		bin_pix_cont = np.array( dat['pix_count'] )

		id_nan = np.isnan(bin_flux)

		bin_flux = bin_flux[id_nan == False]
		bin_pix_cont = bin_pix_cont[id_nan == False]
		bin_pix_cont = bin_pix_cont.astype( np.int32 )

		lis_f = []
		for dd in range( len(bin_flux) ):
			rep_arr = np.ones( bin_pix_cont[dd], dtype = np.float32) * bin_flux[dd]
			lis_f.append( rep_arr)

		F_arr = np.hstack(lis_f) / pixel**2
		mean_f = np.mean( F_arr )
		medi_f = np.median( F_arr )

		bin_wide = 5e-4
		bin_edgs = np.arange( F_arr.min(), F_arr.max(), bin_wide)

		N_pix, edg_f = binned(F_arr, F_arr, statistic = 'count', bins = bin_edgs)[:2]

		pdf_pix = (N_pix / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
		pdf_err = ( np.sqrt(N_pix) / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
		f_cen = 0.5 * ( edg_f[1:] + edg_f[:-1])

		try:
			idu = N_pix != 0.

			popt, pcov = optimize.curve_fit(gau_func, f_cen[idu], pdf_pix[idu], p0 = [mean_f, np.std(F_arr)], sigma = pdf_err[idu],)
			e_mu, e_chi = popt[0], popt[1]
			fit_line = gau_func(f_cen, e_mu, e_chi)

			mode_f = f_cen[ fit_line == fit_line.max() ][0]

			tmp_mu.append( e_mu )
			tmp_chi.append( e_chi )
		except:
			popt = None
			tmp_mu.append( np.nan )
			tmp_chi.append( np.nan )

			mode_f = f_cen[ pdf_pix == pdf_pix.max() ][0]

		tmp_mean_sb.append( mean_f )
		tmp_medi_sb.append( medi_f )
		tmp_mode_sb.append( mode_f )
		tmp_r.append( angl_r[qq] )

		# figs
		fig = plt.figure()
		ax = plt.subplot(111)
		ax.set_title('R = %.3f arcsec' % angl_r[qq])

		ax.hist(F_arr, bins = bin_edgs, density = True, color = 'b', alpha = 0.5)

		ax.axvline(x = mean_f, ymin = 0.4, ymax = 0.55, ls = '-', color = 'r', alpha = 0.5, label = 'mean [%.5f]' % mean_f,)
		ax.axvline(x = medi_f, ymin = 0.2, ymax = 0.35, ls = '--', color = 'r', alpha = 0.5, label = 'median [%.5f]' % medi_f,)
		ax.axvline(x = mode_f, ymin = 0.0, ymax = 0.15, ls = ':', color = 'r', alpha = 0.5, label = 'mode [%.5f]' % mode_f,)
		if popt is not None:
			ax.plot(f_cen, fit_line, ls = '-', color = 'g', alpha = 0.5, label = 'G$[\\mu = %.5f, \\sigma = %.5f] $' % (e_mu, e_chi),)

		ax.set_xlabel('SB [nanomaggies /$arcsec^2$]')
		ax.set_ylabel('PDF')
		ax.legend( loc = 1,)
		ax.set_xlim(mean_f - 5 * np.std(F_arr), mean_f + 5 * np.std(F_arr) )

		plt.savefig('/home/xkchen/fig_tmp/figs/%s_BCG-stack_SB_hist_%.3f-bin.png' % (line_name[tt], angl_r[qq]), dpi = 300)
		plt.close()

	tmp_mean_sb = np.array( tmp_mean_sb )
	tmp_medi_sb = np.array( tmp_medi_sb )
	tmp_mode_sb = np.array( tmp_mode_sb )
	tmp_r = np.array( tmp_r )

	tmp_mu = np.array( tmp_mu )
	tmp_chi = np.array( tmp_chi )

	keys = ['R_arcsec', 'mean_sb', 'medi_sb', 'mode_sb']
	values = [ tmp_r, tmp_mean_sb, tmp_medi_sb, tmp_mode_sb ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv('%s_BCG-stack_SB-pros.csv' % line_name[tt])

	keys = ['R_arcsec', 'e_mu', 'e_chi']
	values = [ tmp_r, tmp_mu, tmp_chi ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv('%s_BCG-stack_SB-pros_fit-record.csv' % line_name[tt])

	print('%s finished !' % line_name[tt])
print('aveged pros finished !')
'''

## sample image case
for tt in range( 2 ):

	cdat = pds.read_csv( mean_pros[tt] )
	Angl_r = np.array( cdat['R_arcsec'] )
	flux_r = np.array( cdat['f_mean'] )

	idNul = np.isnan( flux_r )
	angl_r = Angl_r[idNul == False]
	Nr = len(angl_r)

	tmp_r, tmp_mu, tmp_chi = [], [], []
	sam_tmp_mode_f, sam_tmp_mean_f, sam_tmp_medi_f, sam_tmp_err = [], [], [], []

	fig = plt.figure( figsize = (60, 24) )
	gs = gridspec.GridSpec(Nr // 6 + 1, 6)

	for mm in range( Nr ):

		tmp_r.append( angl_r[mm] )

		f_dat = pds.read_csv( samp_pros[tt] % angl_r[mm],)
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
			tmp_mu.append( e_mu )
			tmp_chi.append( e_chi )
		except:
			popt = None
			mode_f = f_cen[ pdf_pix == pdf_pix.max() ][0]
			tmp_mu.append( np.nan )
			tmp_chi.append( np.nan )

		sam_tmp_mode_f.append( mode_f )

		#figs
		ax0 = plt.subplot( gs[mm // 6, mm % 6] )
		ax0.hist(samp_flux, bins = bin_edgs_samp, density = False, color = 'b', alpha = 0.5,)
		ax0.errorbar(f_cen, N_pix, yerr = np.sqrt(N_pix), xerr = None, color = 'g', marker = '.', ms = 3, mec = 'g', mfc = 'none',
			ls = '', ecolor = 'g', elinewidth = 1,)
		ax0.plot(f_cen, fit_line * np.sum(N_pix) * (edg_f[1] - edg_f[0]), color = 'g', alpha = 0.5,
			label = 'G$[\\mu = %.5f, \\sigma = %.5f] $' % (e_mu, e_chi),)

		ax0.axvline(x = mean_f, ymin = 0.80, ymax = 0.95, ls = '-', color = 'r', alpha = 0.5, label = 'mean [%.5f]' % mean_f,)
		ax0.axvline(x = medi_f, ymin = 0.45, ymax = 0.60, ls = '--', color = 'r', alpha = 0.5, label = 'median [%.5f]' % medi_f,)
		ax0.axvline(x = mode_f, ymin = 0.1, ymax = 0.25, ls = ':', color = 'r', alpha = 0.5, label = 'mode [%.5f]' % mode_f,)

		ax0.annotate(s = '%d, R = %.3f arcsec' % (mm, angl_r[mm]), xy = (0.05, 0.1), xycoords = 'axes fraction', color = 'k', fontsize = 8,)	

		ax0.legend( loc = 2, fontsize = 8, frameon = False,)
		ax0.set_yscale('log')
		ax0.set_ylim(1, N_pix.max() + 10)

		if mm % 6 == 0:
			ax0.set_xlabel('SB [nanomaggies /$arcsec^2$]')
			ax0.set_ylabel('# of pixels')

		plt.figure()
		ax = plt.subplot(111)
		ax.set_title('%d, R = %.3f arcsec' % (mm, angl_r[mm]),)

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

		plt.savefig('/home/xkchen/fig_tmp/figs/' + 
			'%s_BCG-stack_sample-img_R_%.3f-arcsec_flux_hist.png' % (line_name[tt], angl_r[mm]), dpi = 300)
		plt.close()

	plt.subplots_adjust(left = 0.03, bottom = 0.03, right = 0.98, top = 0.98, wspace = 0.07, hspace = 0.2,)
	plt.savefig('%s_BCG-stack_flux_hist_overview.pdf' % line_name[tt], dpi = 300)
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
	out_data.to_csv('%s_BCG-stack_sample-img_SB-pros.csv' % line_name[tt])

	tmp_mu = np.array( tmp_mu )
	tmp_chi = np.array( tmp_chi )
	keys = ['R_arcsec', 'e_mu', 'e_chi']
	values = [ tmp_r, tmp_mu, tmp_chi ]
	fill = dict(zip( keys, values) )
	out_data = pds.DataFrame(fill)
	out_data.to_csv('%s_BCG-stack_sample-img_SB-hist_G-fit_record.csv' % line_name[tt])

	print('%s fit finished !' % line_name[tt])

print('part 2 finished !')

raise

## compare flux histogram in large scale
cdat_0 = pds.read_csv( mean_pros[0] )
Angl_r_0 = np.array( cdat_0['R_arcsec'] )
flux_r_0 = np.array( cdat_0['f_mean'] )

idNul = np.isnan( flux_r_0 )
angl_r_0 = Angl_r_0[idNul == False]

idmx = angl_r_0 >= 100 # radius large than 100 arcsec

cdat_1 = pds.read_csv( mean_pros[1] )
Angl_r_1 = np.array( cdat_1['R_arcsec'] )
flux_r_1 = np.array( cdat_1['f_mean'] )

idNul = np.isnan( flux_r_1 )
angl_r_1 = Angl_r_1[idNul == False]

## common radius
com_r0 = angl_r_0[ idmx ]
com_r1 = angl_r_1[ idmx ]

Nr = np.sum( idmx )

fig = plt.figure( figsize = (30, 25) )
rows = np.int( np.ceil(Nr / 5) )
gs = gridspec.GridSpec(rows, 5)

for ll in range( np.sum( idmx ) ):
	# random
	f0_dat = pds.read_csv( samp_pros[0] % com_r0[ll],)
	samp_flux_0 = np.array( f0_dat['pix_flux'] )

	id_nn = np.isnan(samp_flux_0)
	samp_flux_0 = samp_flux_0[id_nn == False] / pixel**2

	# cluster
	f1_dat = pds.read_csv( samp_pros[1] % com_r1[ll],)
	samp_flux_1 = np.array( f1_dat['pix_flux'] )

	id_nn = np.isnan(samp_flux_1)
	samp_flux_1 = samp_flux_1[id_nn == False] / pixel**2

	mini_sb = np.min([samp_flux_0.min(), samp_flux_1.min() ])
	maxi_sb = np.max([samp_flux_0.max(), samp_flux_1.max() ])

	bin_wide = 3e-2
	bin_edgs = np.arange( mini_sb, maxi_sb, bin_wide,)

	#figs
	ax0 = plt.subplot( gs[ll // 5, ll % 5] )
	ax0.hist(samp_flux_0, bins = bin_edgs, density = True, color = 'b', alpha = 0.5, label = 'random',)
	ax0.axvline(x = np.mean(samp_flux_0), ymin = 0.80, ymax = 1.0, ls = '-', color = 'b', alpha = 0.5, label = 'mean [%.5f]' % np.mean(samp_flux_0),)
	ax0.axvline(x = np.median(samp_flux_0), ymin = 0.80, ymax = 1.0, ls = '--', color = 'b', alpha = 0.5, label = 'median [%.5f]' % np.median(samp_flux_0),)

	ax0.annotate(s = 'R = %.3f arcsec' % com_r0[ll], xy = (0.05, 0.1), xycoords = 'axes fraction', color = 'k', fontsize = 8,)

	ax0.hist(samp_flux_1, bins = bin_edgs, density = True, color = 'r', alpha = 0.5, label = 'cluster',)
	ax0.axvline(x = np.mean(samp_flux_1), ymin = 0, ymax = 0.20, ls = '-', color = 'r', alpha = 0.5, label = 'mean [%.5f]' % np.mean(samp_flux_1),)
	ax0.axvline(x = np.median(samp_flux_1), ymin = 0, ymax = 0.20, ls = '--', color = 'r', alpha = 0.5, label = 'median [%.5f]' % np.median(samp_flux_1),)

	ax0.legend( loc = 2, fontsize = 8, frameon = False,)
	ax0.set_xlim( -1.5, 1.5)
	if ll % 5 == 0:
		ax0.set_xlabel('SB [nanomaggies /$arcsec^2$]')
		ax0.set_ylabel('PDF')

plt.subplots_adjust(left = 0.03, bottom = 0.03, right = 0.98, top = 0.98, wspace = 0.07, hspace = 0.2,)
plt.savefig('BCG-stack_large-scale_flux-hist_compare.pdf', dpi = 300)
plt.close()

