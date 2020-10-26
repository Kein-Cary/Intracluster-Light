import time
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
from matplotlib.patches import Circle, Ellipse

### cosmology model
rad2asec = U.rad.to(U.arcsec)
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

band = ['r', 'g', 'i']
home = '/media/xkchen/My Passport/data/SDSS/'
load = '/home/xkchen/mywork/ICL/data/tmp_img/'

##### add photo-Galaxy cat.
gal_file = '/media/xkchen/My Passport/data/SDSS/redmapper/dr8photoz_deblended_as_moving_zwshao.fit'
gal_cat = fits.open(gal_file)
gal_z = gal_cat[1].data['z']
gal_ra = gal_cat[1].data['ra']
gal_dec = gal_cat[1].data['dec']
#gal_Mag = gal_cat[1].data['absMagR']
gal_Mag = gal_cat[1].data['deext_r']

dat = pds.read_csv('/home/xkchen/mywork/ICL/r_band_sky_catalog.csv')
set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
'''
dat = pds.read_csv('/home/xkchen/mywork/ICL/rand_r_band_catalog.csv')
set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
'''
N_samp = len(set_z)

### galaxies match
for kk in range( N_samp ):

	ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]

	file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)
	#file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_g, dec_g, z_g)

	data = fits.open(file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)
	xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

	hdu = fits.PrimaryHDU()
	hdu.data = img
	hdu.header = head
	hdu.writeto('test.fits', overwrite = True)

	param_A = 'default_mask_A.sex'
	out_cat = 'default_mask_A.param'

	out_load_A = 'test.cat'
	file_source = 'test.fits'

	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s' % (param_A, out_load_A, out_cat)
	a = subpro.Popen(cmd, shell = True)
	a.wait()

	source = asc.read(out_load_A)
	Numb = np.array(source['NUMBER'][-1])
	A = np.array(source['A_IMAGE'])
	B = np.array(source['B_IMAGE'])
	theta = np.array(source['THETA_IMAGE'])
	cx = np.array(source['X_IMAGE']) - 1
	cy = np.array(source['Y_IMAGE']) - 1
	p_type = np.array(source['CLASS_STAR'])

	Kron = 16
	a = Kron * A
	b = Kron * B

	### list of sources which close to img edges
	id_xin = (cx > 300) & ( cx < 1748)
	id_yin = (cy > 300) & ( cy < 1189)
	id_in = id_xin & id_yin
	cx_edg, cy_edg = cx[ id_in == False ], cy[ id_in == False ]

	### photo-G cat. add (select gal_cat source)
	cen_ra = head['CRVAL1']
	cen_dec = head['CRVAL2']

	lef_ra, rit_ra = cen_ra - 0.42, cen_ra + 0.42
	lo_dec, up_dec = cen_dec - 0.42, cen_dec + 0.42

	id_ra = (gal_ra >= lef_ra) & (gal_ra <= rit_ra)
	id_dec = (gal_dec >= lo_dec) & (gal_dec <= up_dec)
	id_gal = (id_ra) & (id_dec)

	if lef_ra < 0:
		id_ra = (gal_ra <= rit_ra) | (gal_ra >= 360 + (cen_ra - 0.42) )
		id_gal = id_ra & id_dec

	if rit_ra > 360:
		id_ra = (gal_ra >= lef_ra) | (gal_ra <= 0.42 + (cen_ra - 360) )
		id_gal = id_ra & id_dec

	gal_ra_in = gal_ra[id_gal]
	gal_dec_in = gal_dec[id_gal]
	gal_in_Mag = gal_Mag[id_gal]
	gal_in_z = gal_z[id_gal]

	gal_x, gal_y = wcs_lis.all_world2pix(gal_ra_in * U.deg, gal_dec_in * U.deg, 1)

	id_xedg = ( np.abs(gal_x) <= 300) | ( np.abs(gal_x - 2048) <= 300)
	sub_yedg = (gal_y >= -300) & (gal_y <= 1789)
	sx_edg = id_xedg & sub_yedg

	id_yedg = ( np.abs(gal_y) <= 300) | ( np.abs(gal_y - 1489) <= 300)	
	sub_xedg = (gal_x >= -300) & (gal_x <= 2348)
	sy_edg = id_yedg & sub_xedg

	id_edg = sx_edg | sy_edg

	gcat_x, gcat_y = gal_x[id_edg], gal_y[id_edg]
	gcat_ra, gcat_dec = gal_ra_in[id_edg], gal_dec_in[id_edg]
	gcat_Mag = gal_in_Mag[id_edg]
	gcat_z = gal_in_z[id_edg]

	n_gal = len(gcat_x)
	### rule out those have been detected
	add_gal_x, add_gal_y, add_gal_Mag = [], [], []
	add_gal_ra, add_gal_dec, add_gal_z = [], [], []
	for nn in range(n_gal):
		devi_x = np.abs(gcat_x[nn] - cx_edg)
		devi_y = np.abs(gcat_y[nn] - cy_edg)

		identi = (devi_x <= 10) & (devi_y <= 10)

		if np.sum(identi) > 0: ## have been detected
			continue
		else:
			add_gal_x.append( gcat_x[nn] )
			add_gal_y.append( gcat_y[nn] )
			add_gal_ra.append( gcat_ra[nn] )
			add_gal_dec.append( gcat_dec[nn] )
			add_gal_z.append( gcat_z[nn] )
			add_gal_Mag.append( gcat_Mag[nn] )

	add_gal_x = np.array( add_gal_x )
	add_gal_y = np.array( add_gal_y )
	add_gal_ra = np.array( add_gal_ra )
	add_gal_dec = np.array( add_gal_dec )
	add_gal_z = np.array( add_gal_z )
	add_gal_Mag = np.array( add_gal_Mag )

	### save the selection galaxies
	keys = ['ra', 'dec', 'z', 'imgx', 'imgy', 'MagR',]
	values = [add_gal_ra, add_gal_dec, add_gal_z, add_gal_x, add_gal_y, add_gal_Mag ]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(load + 'source_find/clus_photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_g, dec_g, z_g),)
	#data.to_csv(load + 'source_find/photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_g, dec_g, z_g),)

raise
### over-view of these source

dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/clust-1000-select_remain_test.csv')
dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_clus-1000-match_remain_test.csv')
set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
N_samp = len(set_z)

lis_N = []
lis_in = []
for kk in range( N_samp ):

	ra_g, dec_g, z_g = set_ra[kk], set_dec[kk], set_z[kk]
	#dat = pds.read_csv(load + 'source_find/clus_photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_g, dec_g, z_g),)
	dat = pds.read_csv(load + 'source_find/photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_g, dec_g, z_g),)
	tt_x, tt_y = np.array(dat.imgx), np.array(dat.imgy)

	tt_ra = np.array(dat.ra)
	lis_N.append( len(tt_ra) )

	id_xin = (tt_x >= 0) & (tt_x <= 2048)
	id_yin = (tt_y >= 0) & (tt_y <= 1489)
	id_in = id_xin & id_yin
	lis_in.append( np.sum(id_in) )

lis_in = np.array(lis_in)
lis_N = np.array(lis_N)

idx = lis_in >= 20 # 10 ## add most galaxies in frame region
lis_ra, lis_dec, lis_z = set_ra[idx], set_dec[idx], set_z[idx]

for nn in range( len(lis_ra) ):

	ra_n, dec_n, z_n = lis_ra[nn], lis_dec[nn], lis_z[nn]
	#file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_n, dec_n, z_n)
	file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % ('r', ra_n, dec_n, z_n)
	data = fits.open(file)
	img = data[0].data
	head = data[0].header
	wcs_lis = awc.WCS(head)
	xn, yn = wcs_lis.all_world2pix(ra_n * U.deg, dec_n * U.deg, 1)

	#dat = pds.read_csv(load + 'source_find/clus_photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_n, dec_n, z_n),)
	dat = pds.read_csv(load + 'source_find/photo-G_match_ra%.3f_dec%.3f_z%.3f.csv' % (ra_n, dec_n, z_n),)
	tt_ra, tt_dec = np.array(dat.ra), np.array(dat.dec)
	tt_mag = np.array(dat.MagR)
	tt_x, tt_y = wcs_lis.all_world2pix(tt_ra * U.deg, tt_dec * U.deg, 1)

	fig = plt.figure( figsize = (13.12, 4.8) )
	ax0 = fig.add_axes([0.03, 0.09, 0.45, 0.80])
	ax1 = fig.add_axes([0.55, 0.09, 0.40, 0.80])

	ax0.set_title('ra%.3f dec%.3f z%.3f' % (ra_n, dec_n, z_n), )
	ax0.imshow(img, origin = 'lower', cmap = 'Greys', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm(),)

	id_sort = np.argsort(tt_mag)
	p_mag = tt_mag[id_sort]
	p_imgx = tt_x[id_sort]
	p_imgy = tt_y[id_sort]
	colo = np.linspace(0, len(tt_x) - 1, len(tt_x), )
	colo = colo + 1
	colo = 1 / colo

	ax0.scatter(p_imgx, p_imgy, s = 20, marker = 'o', facecolors = 'None', edgecolors = mpl.cm.rainbow( colo ), alpha = 0.5,)
	ax0.set_xlim(-320, 2368)
	ax0.set_ylim(-320, 1809)

	ax1.set_title('r-band apparent magnitude PDF [%d]' % (len(tt_ra) ), )
	ax1.hist(tt_mag, bins = 20, color = 'b', density = True, alpha = 0.5,)
	ax1.set_xlabel('$ m_{r} $')

	#plt.savefig('clust_ra%.3f_dec%.3f_z%.3f.png' % (ra_n, dec_n, z_n), dpi = 300)
	plt.savefig('random_ra%.3f_dec%.3f_z%.3f.png' % (ra_n, dec_n, z_n), dpi = 300)
	plt.close()

raise
