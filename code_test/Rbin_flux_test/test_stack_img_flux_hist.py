import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.units as U
import astropy.constants as C
import scipy.stats as sts
import scipy.special as special
from astropy import cosmology as apcy
from scipy import optimize
from scipy.stats import binned_statistic as binned

# pipe-code
from light_measure_tmp import SB_measure_Z0_weit_func

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

# constant
rad2asec = U.rad.to(U.arcsec)
band = ['r', 'g', 'i',]
mag_add = np.array([0, 0, 0])
pixel = 0.396

def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

def log_norm_func(x, mu, sigma):

	pf0 = -0.5 * (np.log(x) - mu)**2 / sigma**2
	pf1 = sigma * x * np.sqrt(2 * np.pi)
	pf = ( 1 / pf1) * np.exp( pf0 )

	return pf

def skew_t_pdf_func(x, eta, lamda):

	gama_f0 = special.gamma( (eta + 1) / 2 )
	gama_f1 = np.sqrt( np.pi * (eta - 2) ) * special.gamma( eta / 2 )
	pc = gama_f0 / gama_f1
	pa = 4 * lamda * pc * (eta - 2) / (eta - 1)
	pb = 1 + 3 * lamda**2 - pa**2

	idx0 = x < ( -1 * pa / pb)
	idx1 = x >= ( -1 * pa / pb)

	p_x0 = (1 / (eta - 2) ) * ( (pa + pb * x[idx0]) / (1 - lamda) )**2
	f_x0 = pb * pc * ( 1 + p_x0)**(-0.5 * (eta + 1) )

	p_x1 = (1 / (eta - 2) ) * ( (pa + pb * x[idx1]) / (1 + lamda) )**2
	f_x1 = pb * pc * ( 1 + p_x1)**(-0.5 * (eta + 1) )

	f_x = np.r_[ f_x0, f_x1 ]

	return f_x

def err_func(p, x, y):

	eta, lamda = p[0], p[1]
	f_model = skew_t_pdf_func(x, eta, lamda)

	return f_model - y

load = '/home/xkchen/Downloads/random_img_test/'

id_cen = 0
n_rbins = 100
N_bin = 30
'''
### test for the medi and mode SB pros?
with h5py.File( load + 'tmp_test/random_r-band_rand-stack_Mean_jack_img_0-rank.h5', 'r') as f:
	aveg_rnd_img = np.array( f['a'] )

with h5py.File( load + 'tmp_test/random_r-band_rand-stack_Mean_jack_pix-cont_0-rank.h5', 'r') as f:
	aveg_rnd_cont = np.array( f['a'] )
xn, yn = np.int( aveg_rnd_img.shape[1] / 2), np.int(aveg_rnd_img.shape[0] / 2)
'''
R_bins = np.logspace(0, np.log10(2520), n_rbins)
bin_flux_file = load + 'tmp_test/random_rand-stack_%.3f-arcsec_flux-arr.csv'

'''
mean_intens, Angl_r, intens_err, N_pix, nsum_ratio = SB_measure_Z0_weit_func(aveg_rnd_img, aveg_rnd_cont, pixel, xn, yn, R_bins, bin_flux_file,)

keys = ['f_mean', 'R_arcsec',]
values = [ mean_intens, Angl_r ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame(fill)
out_data.to_csv('Mean_f_pros.csv')
'''

cdat = pds.read_csv(load + 'Mean_f_pros.csv')
Angl_r = np.array( cdat['R_arcsec'] )
mean_f = np.array( cdat['f_mean'] )

idNul = np.isnan(mean_f)
angl_r = Angl_r[idNul == False]

tmp_r = []
tmp_mean_sb, tmp_medi_sb, tmp_mode_sb = [], [], []
tmp_mu, tmp_chi = [], []

for qq in range( len(angl_r) ):
#for qq in range( len(angl_r) - 1, -1, -1):
	# stacking pixel flux
	dat = pds.read_csv(bin_flux_file % angl_r[qq])
	bin_flux = np.array( dat['flux'], dtype = np.float32)
	bin_pix_cont = np.array( dat['pix_count'],dtype = np.int32)

	id_nan = np.isnan(bin_flux)

	bin_flux = bin_flux[id_nan == False]
	bin_pix_cont = bin_pix_cont[id_nan == False]
	#bin_pix_cont = bin_pix_cont.astype( np.int32 )

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
	abs_min_f = np.abs( F_arr.min() )

	try:
		idu = N_pix != 0.
		'''
		popt, pcov = optimize.curve_fit(gau_func, f_cen[idu], pdf_pix[idu], p0 = [mean_f, np.std(F_arr)], sigma = pdf_err[idu],)
		e_mu, e_chi = popt[0], popt[1]
		fit_line = gau_func(f_cen, e_mu, e_chi)

		'''
		eta_0, lamda_0 = 3, -0.5
		p0 = [ eta_0, lamda_0 ]
		popt, pcov = optimize.curve_fit(skew_t_pdf_func, f_cen[idu], pdf_pix[idu], p0 = p0, bounds = ([2, -1], [np.inf, 1]),
			method = 'trf',)#sigma = pdf_err[idu],)

		e_eta, e_lamda = popt
		fit_line = skew_t_pdf_func(f_cen, e_eta, e_lamda)


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
	ax.legend( loc = 1, fontsize = 8,)
	ax.set_xlim(mean_f - 5 * np.std(F_arr), mean_f + 5 * np.std(F_arr) )

	plt.savefig('SB_hist_%.3f-bin.png' % angl_r[qq], dpi = 300)
	plt.close()

	raise

tmp_mean_sb = np.array( tmp_mean_sb )
tmp_medi_sb = np.array( tmp_medi_sb )
tmp_mode_sb = np.array( tmp_mode_sb )
tmp_r = np.array( tmp_r )

tmp_mu = np.array( tmp_mu )
tmp_chi = np.array( tmp_chi )
'''
keys = ['R_arcsec', 'mean_sb', 'medi_sb', 'mode_sb']
values = [ tmp_r, tmp_mean_sb, tmp_medi_sb, tmp_mode_sb ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame(fill)
out_data.to_csv('stacking-img_SB-pros.csv')

keys = ['R_arcsec', 'e_mu', 'e_chi']
values = [ tmp_r, tmp_mu, tmp_chi ]
fill = dict(zip( keys, values) )
out_data = pds.DataFrame(fill)
out_data.to_csv('stacking-img_SB-pros_fit-record.csv')
'''
raise

## hist compare (~50, 100, 500, 900, and the last radius)
# SB-pros file
dat = pds.read_csv('SB-pros.csv')
angl_r = np.array(dat['R_arcsec'])
mean_sb = np.array(dat['mean_sb'])
medi_sb = np.array(dat['medi_sb'])
mode_sb = np.array(dat['mode_sb'])

targ_r = np.array([50, 100, 500, 900, angl_r[-1] ])

line_c = ['b', 'g', 'c', 'm', 'r']

r_id = []

for mm in range( 5 ):

	DR = np.abs(angl_r - targ_r[mm])
	idr = np.where( DR == DR.min() )[0][0]

	r_id.append( idr )

fig = plt.figure()
ax = plt.subplot(111)
add_ax = fig.add_axes([0.15, 0.40, 0.30, 0.20])

for jj in range( 5 ):

	order = r_id[jj]
	dat = pds.read_csv( bin_flux_file % angl_r[order] )
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
	mode_f = mode_sb[ order ]

	bin_edgs = np.arange( F_arr.min(), F_arr.max(), 1e-4)

	ax.hist(F_arr, bins = bin_edgs, density = True, histtype = 'step', color = line_c[jj], alpha = 0.5,
		label = '$%.3f arcsec$' % angl_r[order],)

	if jj == 0:
		ax.axvline(x = mean_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '-', color = line_c[jj], alpha = 0.5, label = 'mean',)
		ax.axvline(x = medi_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '--', color = line_c[jj], alpha = 0.5, label = 'median',)
		ax.axvline(x = mode_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = ':', color = line_c[jj], alpha = 0.5, label = 'mode',)
	else:
		ax.axvline(x = mean_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '-', color = line_c[jj], alpha = 0.5,)
		ax.axvline(x = medi_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = '--', color = line_c[jj], alpha = 0.5,)
		ax.axvline(x = mode_f, ymin = 0.05 * jj, ymax = 0.05 * (jj + 0.8), ls = ':', color = line_c[jj], alpha = 0.5,)

	add_ax.hist(F_arr, bins = bin_edgs, density = True, histtype = 'step', color = line_c[jj], alpha = 0.5,)
	add_ax.axvline(x = mean_f, ls = '-', color = line_c[jj], alpha = 0.5, linewidth = 1,)
	add_ax.axvline(x = medi_f, ls = '--', color = line_c[jj], alpha = 0.5,linewidth = 1)
	add_ax.axvline(x = mode_f, ls = ':', color = line_c[jj], alpha = 0.5, linewidth = 1)

add_ax.set_xlim(-2e-3, 8e-3)
add_ax.ticklabel_format(axis = 'x', style = 'scientific', scilimits = (0,0),)
add_ax.set_yticks([])

ax.set_xlabel('SB [nanomaggies /$arcsec^2$]')
ax.set_ylabel('PDF')
ax.legend( loc = 1,)
ax.set_xlim(-0.2, 0.2)

plt.savefig('differ-R_SB-hist_compare.png', dpi = 300)
plt.close()


## 2D imgs
R_angle = 0.5 * (R_bins[1:] + R_bins[:-1]) * pixel

ddr = np.abs( R_angle - angl_r[87] )
idx = np.where( ddr == ddr.min() )[0]
edg_lo = R_bins[idx]
edg_hi = R_bins[idx + 1]

'''
Nx = np.linspace(0, aveg_rnd_img.shape[1] - 1, aveg_rnd_img.shape[1] )
Ny = np.linspace(0, aveg_rnd_img.shape[0] - 1, aveg_rnd_img.shape[0] )
grd = np.array( np.meshgrid(Nx, Ny) )
cen_dR = np.sqrt( (grd[0] - xn)**2 + (grd[1] - yn)**2 )

id_flux = (cen_dR >= edg_lo) & (cen_dR < edg_hi)
f_stack = aveg_rnd_img[ id_flux ]

plt.figure()
ax = plt.subplot(111)
tf = ax.imshow(aveg_rnd_img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-2, vmax = 5e-2,)
cb = plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $ arcsec^2 $]')
cb.formatter.set_powerlimits((0,0))

clust = Circle(xy = (xn, yn), radius = edg_lo, fill = False, ec = 'k', ls = '--', linewidth = 1, alpha = 0.5,)
ax.add_patch(clust)
clust = Circle(xy = (xn, yn), radius = edg_hi, fill = False, ec = 'k', ls = '-', linewidth = 1, alpha = 0.5,)
ax.add_patch(clust)

plt.savefig('2D_SB_hist.png', dpi = 300)
plt.close()
'''
