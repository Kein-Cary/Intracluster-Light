import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
import astropy.io.ascii as asc
from astropy import cosmology as apcy

import time
import subprocess as subpro

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

rad2asec = U.rad.to(U.arcsec)
pixel = 0.45
R0 = 2 ## Mpc / h

#ra_g, dec_g, z_g = 253.551, 40.922, 0.326
#ra_g, dec_g, z_g = 155.681, 32.310, 0.293
ra_g, dec_g, z_g = 113.253, 39.418, 0.169

## img with white line
#ra_g, dec_g, z_g = 352.727, 34.169, 0.194

wcs_dat = fits.open('bass_img_ra%.3f_dec%.3f_z%.3f_g_band.fits' % (ra_g, dec_g, z_g) )
img_g = wcs_dat[0].data
head = wcs_dat[0].header
w = awc.WCS(wcs_dat[0].header)
cx, cy = w.all_world2pix(head['CRVAL1'] * U.deg, head['CRVAL2'] * U.deg, 1)
eta = 0.5 / 0.45

cat = fits.open('desi_tractor-cat_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g) )
ra_sor = cat[1].data['ra']
dec_sor = cat[1].data['dec']
source_type = cat[1].data['type']
pox, poy = w.all_world2pix(ra_sor * U.deg, dec_sor * U.deg, 1)
FWHM_g = (cat[1].data['psfsize_g'] / pixel) / 2 ## take half value as radius
raise
## gaia sources
stars = cat[1].data['gaia_pointsource']
star_mag = cat[1].data['gaia_phot_g_mean_mag']
r_star = cat[1].data.gaia_astrometric_sigma5d_max
id_sda = stars == True
Mag = star_mag[id_sda]
R_star = r_star[id_sda]
x_star, y_star = pox[id_sda], poy[id_sda]

## use flux / SB (to select point sources)
apf_flux = cat[1].data.apflux_r[:, 0]
apf_mag = 22.5 - 2.5 * np.log10(apf_flux) + 2.5 * np.log10(np.pi * 0.5**2)

exp_r = cat[1].data.shapeexp_r
exp_r_a = cat[1].data.shapeexp_e1
exp_r_b = cat[1].data.shapeexp_e2
exp_e = np.sqrt(exp_r_a**2 + exp_r_b**2)
exp_b2a = (1 - exp_e) / (1 + exp_e)
exp_angle = 0.5 * np.arctan(exp_r_b / exp_r_a) * (180 / np.pi) # + 90

dev_r = cat[1].data.shapedev_r
dev_r_a = cat[1].data.shapedev_e1
dev_r_b = cat[1].data.shapedev_e2
dev_e = np.sqrt(dev_r_a**2 + dev_r_b**2)
dev_b2a = (1 - dev_e) / (1 + dev_e)
dev_angle = 0.5 * np.arctan(dev_r_b / dev_r_a) * (180 / np.pi) # + 90

## divide sources into: point source, galaxy
idx_pont = source_type == 'PSF'
pont_x, pont_y = pox[idx_pont], poy[idx_pont]
star_mag = apf_mag[[idx_pont]]
sub_FWHM = FWHM_g[[idx_pont]]

idx0 = star_mag <= 19
pont_x0, pont_y0 = pont_x[idx0], pont_y[idx0]
pont_r0 = 15 * sub_FWHM[idx0]

idx1 = (star_mag > 19) & (star_mag < 24)
pont_x1, pont_y1 = pont_x[idx1], pont_y[idx1]
pont_r1 = 5 * sub_FWHM[idx1]

idx2 = (star_mag > 24)
pont_x2, pont_y2 = pont_x[idx2], pont_y[idx2]
pont_r2 = 2 * sub_FWHM[idx2]

plt.figure()
ax = plt.subplot(111)
ax.set_title('cluster ra%.3f dec%.3f z%.3f' % (ra_g, dec_g, z_g) )
ax.imshow(img_g, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e1, norm = mpl.colors.LogNorm())

ax.scatter(cx, cy, s = 10, color = '', edgecolors = 'r', alpha = 0.5, )
ax.set_xlim(0, img_g.shape[1])
ax.set_ylim(0, img_g.shape[0])
plt.savefig('clust_ra%.3f_dec%.3f_z%.3f.png' % (ra_g, dec_g, z_g), dpi = 300)
plt.close()

plt.figure()
ax = plt.subplot(111)
ax.set_title('cluster ra%.3f dec%.3f z%.3f [pixscale = 0.45]' % (ra_g, dec_g, z_g) )
ax.imshow(img_g, cmap = 'Greys', origin = 'lower', vmin = 1e-5, vmax = 1e1, norm = mpl.colors.LogNorm())

#for ll in range(len(x_star)):
#	circl = Circle(xy = (x_star[ll], y_star[ll]), radius = R_star[ll], fill = False, ec = 'r', ls = '-', linewidth = 0.5, alpha = 0.5,)
#	ax.add_patch(circl)

for ll in range(len(pont_x0)):
	circl = Circle(xy = (pont_x0[ll], pont_y0[ll]), radius = pont_r0[ll], fill = False, ec = 'm', ls = '-', linewidth = 0.5, alpha = 0.5,)
	ax.add_patch(circl)

for ll in range(len(pont_x1)):
	circl = Circle(xy = (pont_x1[ll], pont_y1[ll]), radius = pont_r1[ll], fill = False, ec = 'g', ls = '-', linewidth = 0.5, alpha = 0.5,)
	ax.add_patch(circl)

for ll in range(len(pont_x2)):
	circl = Circle(xy = (pont_x2[ll], pont_y2[ll]), radius = pont_r2[ll], fill = False, ec = 'b', ls = '-', linewidth = 0.5, alpha = 0.5,)
	ax.add_patch(circl)

ax.scatter(cx, cy, s = 10, color = '', edgecolors = 'r', alpha = 0.5, )
ax.set_xlim(0, img_g.shape[1])
ax.set_ylim(0, img_g.shape[0])
#plt.savefig('source_region_ra%.3f_dec%.3f_z%.3f_pix-045.png' % (ra_g, dec_g, z_g), dpi = 300)
plt.savefig('source_region_ra%.3f_dec%.3f_z%.3f_pix-045_stars.png' % (ra_g, dec_g, z_g), dpi = 300)
plt.close()
