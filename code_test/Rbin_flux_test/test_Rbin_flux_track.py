import h5py
import pandas as pds
import numpy as np

import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits

from scipy import optimize
from scipy.stats import binned_statistic as binned
import scipy.stats as sts
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

def Rbin_flux_track(targ_R, R_lim, flux_lim, img_file, set_ra, set_dec, set_z, set_x, set_y, out_file,):
	"""
	targ_R : the radius bin in which the flux will be collected
	R_lim : the limited radius edges, R_lim[0] is the inner one, and R_lim[1] is the outer one
	flux_lim : the flux range in which those pixels will be collected,
				flux_lim[0] is the smaller one and flux_lim[1] is the larger one
	set_ra, set_dec, set_z, set_x, set_y : the catalog information, including the BCG position (set_x, set_y)
				in image region
	img_file : .fits files
	out_file : out put data of the collected flux, .csv file
	"""
	Ns = len(set_z)

	lx = np.linspace(0, 2047, 2048)
	ly = np.linspace(0, 1488, 1489)
	grd_lxy = np.array( np.meshgrid(lx, ly) )

	for mm in range( Ns ):

		ra_g, dec_g, z_g = set_ra[mm], set_dec[mm], set_z[mm]
		cen_x, cen_y = set_x[mm], set_y[mm]

		data = fits.open( img_file % (ra_g, dec_g, z_g),)
		img = data[0].data

		pix_dR = np.sqrt( (grd_lxy[0] - cen_x)**2 + (grd_lxy[1] - cen_y)**2)

		id_rx = (pix_dR >= R_lim[0]) & (pix_dR < R_lim[1])
		idnn = np.isnan( img )

		id_vlim = ( img >= flux_lim[0] ) & ( img <= flux_lim[1] )

		id_set = id_rx & (idnn == False) & id_vlim

		my, mx = np.where( id_set == True )

		flux_in = img[ id_set ]

		# save the flux info.
		keys = ['pix_flux', 'pos_x', 'pos_y',]
		values = [ flux_in, mx, my ]
		fill = dict( zip(keys,values) )
		out_data = pds.DataFrame(fill)
		out_data.to_csv( out_file % (targ_R, ra_g, dec_g, z_g),)

	return

def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

########
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

home = '/home/xkchen/'
load = '/home/xkchen/project/'
pixel = 0.396

#cdat = pds.read_csv(load + 'stack/Mean_f_pros.csv' )
cdat = pds.read_csv('/home/xkchen/Downloads/random_img_test/Mean_f_pros.csv')
Angl_r = np.array( cdat['R_arcsec'] )
flux_r = np.array( cdat['f_mean'] )

idNul = np.isnan( flux_r )
angl_r = Angl_r[idNul == False]

n_rbins = 100
R_bins = np.logspace(0, np.log10(2520), n_rbins)
R_angle = 0.5 * (R_bins[1:] + R_bins[:-1]) * pixel

ddr = np.abs( R_angle - angl_r[-1] )
idx = np.where( ddr == ddr.min() )[0]
edg_lo = R_bins[idx]
edg_hi = R_bins[idx + 1]

radi_lim = np.array( [edg_lo, edg_hi] )
flux_lim = np.array( [0.0210, 0.021108] ) * pixel**2

#dat = pds.read_csv(load + 'random_r-band_tot_remain_mock-BCG-pos_0-rank.csv')
dat = pds.read_csv('/home/xkchen/Downloads/random_r-band_tot_remain_mock-BCG-pos_0-rank.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
img_x, img_y = np.array(dat['bcg_x']), np.array(dat['bcg_y'])
'''
dR0 = np.sqrt( img_x**2 + img_y**2 )
dR1 = np.sqrt( (2047 - img_x)**2 + img_y**2 )
dR2 = np.sqrt( img_x**2 + (1488 - img_y)**2 )
dR3 = np.sqrt( (2047 - img_x)**2 + (1488 - img_y)**2 )

idx0 = dR0 >= radi_lim
idx1 = dR1 >= radi_lim
idx2 = dR2 >= radi_lim
idx3 = dR3 >= radi_lim

idx = idx0 | idx1 | idx2 | idx3

set_ra, set_dec, set_z = ra[idx], dec[idx], z[idx]
set_x, set_y = img_x[idx], img_y[idx]

dRlb = dR0[idx]
dRlu = dR2[idx]
dRrb = dR1[idx]
dRru = dR3[idx]

keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y', 'dR_lb', 'dR_lu', 'dR_rb', 'dR_ru']
values = [ set_ra, set_dec, set_z, set_x, set_y, dRlb, dRlu, dRrb, dRlu ]
fill = dict( zip(keys,values) )
bin_data = pds.DataFrame(fill)
bin_data.to_csv('R_%.3f-arcsec_flux-lim_img-track.csv' % angl_r[-1],)

out_file = load + 'R_%.3f-arcsec_ra%.3f_dec%.3f_z%.3f_lim-flux.csv'
img_file = home + 'data/SDSS/tmp_stack/random/random_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits'

Rbin_flux_track(angl_r[-1], radi_lim, flux_lim, img_file, set_ra, set_dec, set_z, set_x, set_y, out_file,)

print('collection finished !')
'''

### hist of the flux-lim pixel
dat = pds.read_csv('/home/xkchen/Downloads/R_bin_flux_track/R_937.830-arcsec_flux-lim_img-track.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
img_x, img_y = np.array(dat['bcg_x']), np.array(dat['bcg_y'])

N_pix = []
tmp_flux = []
for mm in range( len(z) ):

	ra_g, dec_g, z_g = ra[mm], dec[mm], z[mm]
	cen_x, cen_y = img_x[mm], img_y[mm]

	bin_dat = pds.read_csv('/home/xkchen/Downloads/R_bin_flux_track/Rbin_flim_pixel/' +
		'R_937.830-arcsec_ra%.3f_dec%.3f_z%.3f_lim-flux.csv' % (ra_g, dec_g, z_g),)
	sam_flux = np.array( bin_dat['pix_flux'] )
	pos_x, pos_y = np.array(bin_dat['pos_x']), np.array(bin_dat['pos_y'])

	if len(pos_x) < 1:
		N_pix.append( 0 )
		continue

	else:
		N_pix.append( len(sam_flux) )
		tmp_flux.append( sam_flux )
'''
		data = fits.open('/media/xkchen/My Passport/data/SDSS/tmp_stack/random/' +
			'random_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % (ra_g, dec_g, z_g),)
		img = data[0].data

		plt.figure()
		ax = plt.subplot(111)
		tf = ax.imshow( img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)
		cb = plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $ arcsec^2 $]')
		cb.formatter.set_powerlimits((0,0))

		clust = Circle(xy = (cen_x, cen_y), radius = edg_lo, fill = False, ec = 'k', ls = '--', linewidth = 1, alpha = 0.5,)
		ax.add_patch(clust)
		ax.scatter(cen_x, cen_y, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', alpha = 0.5,)
		ax.set_xlim(0, 2048)
		ax.set_ylim(0, 1489)
		plt.savefig('2D_SB_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
		plt.close()
'''
X = np.linspace(0, len(z) - 1, len(z))
N_pix = np.array(N_pix)

# the img of N_pix.max()
idvx = N_pix == N_pix.max()
ra_g, dec_g, z_g = ra[idvx], dec[idvx], z[idvx]
cen_x, cen_y = img_x[idvx], img_y[idvx]

bin_dat = pds.read_csv('/home/xkchen/Downloads/R_bin_flux_track/Rbin_flim_pixel/' +
	'R_937.830-arcsec_ra%.3f_dec%.3f_z%.3f_lim-flux.csv' % (ra_g, dec_g, z_g),)
pos_x, pos_y = np.array(bin_dat['pos_x']), np.array(bin_dat['pos_y'])

data = fits.open('/media/xkchen/My Passport/data/SDSS/tmp_stack/random/' +
	'random_mask_r_ra%.3f_dec%.3f_z%.3f_cat-corrected.fits' % (ra_g, dec_g, z_g),)
img = data[0].data

# capture those pixels
lx = np.linspace(0, 2047, 2048)
ly = np.linspace(0, 1488, 1489)
grd_lxy = np.array( np.meshgrid(lx, ly) )

pix_dR = np.sqrt( (grd_lxy[0] - cen_x)**2 + (grd_lxy[1] - cen_y)**2)
idmx = pix_dR >= edg_lo
idnn = np.isnan( img )

bin_wide = 5e-2

flux_in = img[ idmx & (idnn == False)] / pixel**2
bin_edgs_0 = np.arange( flux_in.min(), flux_in.max(), bin_wide,)

norm_flux = img[idnn == False] / pixel**2
bin_edgs_1 = np.arange( norm_flux.min(), norm_flux.max(), bin_wide,)

# total image sample flux
order = [32, -3, -2, -1]
colors = ['r', 'c', 'g', 'b']
plt.figure()

for tt in range( len(order) ):

	dat = pds.read_csv('/home/xkchen/Downloads/random_img_test/Rbin_sample_img_flux/%.3f-arcsec_sample-img_flux.csv' % angl_r[ order[tt] ])
	sam_flux = np.array(dat['pix_flux'])
	idnn = np.isnan( sam_flux )
	sam_flux = sam_flux[idnn == False] / pixel**2

	bin_wide_samp = 3e-2
	#bin_edgs_samp = np.arange( sam_flux.min(), sam_flux.max(), bin_wide_samp,)
	bin_edgs_samp = np.arange( -1.25, 1.25, bin_wide_samp,)

	N_pix, edg_f = binned(sam_flux, sam_flux, statistic = 'count', bins = bin_edgs_samp)[:2]

	pdf_pix = (N_pix / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	pdf_err = ( np.sqrt(N_pix) / np.sum(N_pix) ) / (edg_f[1] - edg_f[0])
	f_cen = 0.5 * ( edg_f[1:] + edg_f[:-1])

	idu = N_pix != 0.
	popt, pcov = optimize.curve_fit(gau_func, f_cen[idu], pdf_pix[idu], p0 = [np.median(sam_flux), np.std(sam_flux)], sigma = pdf_err[idu],)
	e_mu, e_chi = popt[0], popt[1]
	fit_line = gau_func(f_cen, e_mu, e_chi)
	mode_p = f_cen[fit_line == fit_line.max()]

	plt.plot(f_cen, pdf_pix, color = colors[tt], label = 'R%.3f' % angl_r[ order[tt] ] )
	plt.axvline( x = np.median(sam_flux), color = colors[tt], )
	plt.axvline( x = np.mean(sam_flux), color = colors[tt], ls = '--',)

plt.legend(loc = 1)
plt.show()

raise

plt.figure()
gs = gridspec.GridSpec(2, 1, height_ratios=[2,3])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])

ax0.hist(sam_flux, bins = bin_edgs_samp, density = False, histtype = 'step', color = 'g', alpha = 0.5, label = 'total sample imgs',)
ax0.axvline( x = np.mean(sam_flux), ymin = 0.90, ymax = 1, ls = '-', color = 'g', alpha = 0.5, label = 'mean')
ax0.axvline( x = np.median(sam_flux), ymin = 0.80, ymax = 0.9, ls = '--', color = 'g', alpha = 0.5, label = 'median')
ax0.axvline( x = np.median(sam_flux), ymin = 0.70, ymax = 80, ls = ':', color = 'g', alpha = 0.5, label = 'mode [%.5f]' % mode_p)
ax0.errorbar(f_cen, pdf_pix * np.sum(N_pix) * (edg_f[1] - edg_f[0]), yerr = np.sqrt(N_pix), xerr = None, color = 'g', marker = '.', ms = 3, mec = 'g', mfc = 'none', 
	ls = '', ecolor = 'g', elinewidth = 1,)
ax0.plot(f_cen, fit_line * np.sum(N_pix) * (edg_f[1] - edg_f[0]), ls = '-', color = 'b', alpha = 0.5, label = 'G$[\\mu = %.5f, \\sigma = %.5f] $' % (e_mu, e_chi),)
ax0.legend(loc = 'lower center', fontsize = 6)
ax0.set_yscale('log')
ax0.set_ylim(100, 1e6)
#ax0.xlabel('SB [nanomaggies /$arcsec^2$]')

frac_devi = (pdf_pix - fit_line) / fit_line

ax1.plot(f_cen, frac_devi, ls = '-', color = 'g', alpha = 0.5,)
ax1.axhline( y = 0, ls = ':', color = 'k', alpha = 0.5,)
ax1.fill_between(f_cen, y1 = -1 * pdf_err / fit_line, y2 = pdf_err / fit_line, color = 'k', alpha = 0.5,)

ax1.set_ylim(-0.5, 0.5)
ax1.set_xlim( ax0.get_xlim())
ax1.set_xlabel('SB [nanomaggies /$arcsec^2$]')
ax0.set_xticklabels(labels = [])

plt.subplots_adjust( hspace = 0.05,)
plt.savefig('R_%.3f-arcsec_SB_hist_with-devi-to-fit.png' % angl_r[ order ], dpi = 300)
plt.close()



raise

plt.figure()
plt.hist( flux_in, bins = bin_edgs_0, density = True, histtype = 'step', color = 'r', alpha = 0.5, label = 'R bin',)
plt.axvline( x = np.mean(flux_in), ymin = 0.90, ymax = 1, ls = '-', color = 'r', alpha = 0.5, label = 'mean',)
plt.axvline( x = np.median(flux_in), ymin = 0.90, ymax = 1, ls = '--', color = 'r', alpha = 0.5, label = 'median',)

plt.hist( norm_flux, bins = bin_edgs_1, density = True, histtype = 'step', color = 'b', alpha = 0.5, label = 'full img',)
plt.axvline( x = np.mean(norm_flux), ymin = 0.90, ymax = 1, ls = '-', color = 'b', alpha = 0.5,)
plt.axvline( x = np.median(norm_flux), ymin = 0.90, ymax = 1, ls = '--', color = 'b', alpha = 0.5,)

plt.hist(sam_flux, bins = bin_edgs_samp, density = True, histtype = 'step', color = 'g', alpha = 0.5, label = 'total sample imgs',)
plt.axvline( x = np.mean(sam_flux), ymin = 0.90, ymax = 1, ls = '-', color = 'g', alpha = 0.5,)
plt.axvline( x = np.median(sam_flux), ymin = 0.90, ymax = 1, ls = '--', color = 'g', alpha = 0.5,)
#plt.plot(f_cen, fit_line, color = 'g', ls = '-', alpha = 0.5, )

plt.xlabel('SB [nanomaggies /$arcsec^2$]')
plt.savefig('hist_compare.png', dpi = 300)
plt.close()


raise

fig = plt.figure( figsize = (13.12, 4.8) )
ax0 = fig.add_axes([0.05, 0.10, 0.40, 0.80])
ax1 = fig.add_axes([0.55, 0.10, 0.40, 0.80])

tf = ax0.imshow( img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)
cb = plt.colorbar(tf, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $ arcsec^2 $]')
cb.formatter.set_powerlimits((0,0))

clust = Circle(xy = (cen_x, cen_y), radius = edg_lo, fill = False, ec = 'k', ls = '--', linewidth = 1, alpha = 0.5,)
ax0.add_patch(clust)
ax0.scatter(cen_x, cen_y, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', alpha = 0.5,)
ax0.set_xlim(0, 2048)
ax0.set_ylim(0, 1489)

tf = ax1.imshow( img / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -5e-1, vmax = 5e-1,)

clust = Circle(xy = (cen_x, cen_y), radius = edg_lo, fill = False, ec = 'k', ls = '--', linewidth = 1, alpha = 0.5,)
ax1.add_patch(clust)
ax1.scatter(cen_x, cen_y, s = 25, marker = 'X', edgecolors = 'b', facecolors = 'none', alpha = 0.5,)
ax1.scatter(pos_x, pos_y, s = 2.5, marker = 'o', edgecolors = 'k', facecolors = 'none', alpha = 0.5,)

ax1.set_xlim(0, pos_x.max() )
ax1.set_ylim(pos_y.min(), 1489)

plt.savefig('majority_img_2D_pix-pos.png', dpi = 300)
plt.close()

