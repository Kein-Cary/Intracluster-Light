import matplotlib as mpl
mpl.use('Agg')
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
from light_measure import light_measure_Z0
from matplotlib.patches import Circle, Ellipse, Rectangle

# constant
kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
kpc2m = U.kpc.to(U.m)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Msun2kg = U.M_sun.to(U.kg)
G = C.G.value

# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0 / 100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.25
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

R0 = 1 # in unit Mpc
Angu_ref = (R0 / Da_ref) * rad2asec
Rpp = Angu_ref / pixel

######################
load = '/home/xkchen/mywork/ICL/data/tmp_img/'
dat = pds.read_csv('/home/xkchen/mywork/ICL/r_band_sky_catalog.csv')
ra, dec, z = dat.ra, dat.dec, dat.z

### trouble images
lis_ra = np.array([231.20355251, 150.89168502,  26.22923052, 191.24271753])
lis_dec = np.array([11.30799735, 42.54243936, 14.20254774, 61.31027512])
lis_z = np.array([0.27726924, 0.2639949 , 0.26970404, 0.230333])

### random images
rnd_ra = np.array([197.25518847828275, 174.0337913017225, 359.75253625, 191.0523974, 17.2688937, 197.56986305, 130.3783955])
rnd_dec = np.array([21.219788684716207, 3.9099808755364465, 28.05588246, 16.85510289, 21.10405202, 19.96462519, 21.40908432])
rnd_z = np.array([0.2825178205966949, 0.2936766445636749, 0.23799674, 0.26528448, 0.25037688, 0.25795066, 0.24528672])

ra_g = rnd_ra[3]
dec_g = rnd_dec[3]
z_g = rnd_z[3]

file = load + 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (ra_g, dec_g, z_g)
data = fits.open(file)
img = data[0].data
head = data[0].header
wcs_lis = awc.WCS(head)
xn, yn = wcs_lis.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)

hdu = fits.PrimaryHDU()
hdu.data = img
hdu.header = head
hdu.writeto('test.fits', overwrite = True)

plt.figure()
ax = plt.subplot(111)
tf = ax.imshow(img, cmap = 'Greys', origin = 'lower', vmin = 1e-4, vmax = 1e1, norm = mpl.colors.LogNorm())
ellips = Circle(xy = (xn, yn), radius = Rpp, fill = False, ec = 'g', ls = '-', linewidth = 1, alpha = 0.5,)
ax.add_patch(ellips)
plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux[nanomaggies]')
plt.savefig('origin_img.png', dpi = 300)
plt.close()

### times of Kron radius
N_binx = 36
N_biny = 30
### for sources find by SExTractor
#k_r = np.array([5, 6, 7, 8, 9, 10, 12, 14, 16, 17, 18]) ## for galaxy
k_r = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 110, 120]) # 65, 70, 75, 80]) ## for stars
#k_r = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]) ## for stars
"""
for kk in range( len(k_r) ):

	param_A = 'default_mask_A.sex'
	#param_A = 'default_adjust.sex'  ## detection for small "structure"
	out_cat = 'default_mask_A.param'
	out_load_A = '/home/xkchen/mywork/ICL/code/SEX/result/A_mask.cat'

	file_source = 'test.fits'
	cmd = 'sex '+ file_source + ' -c %s -CATALOG_NAME %s -PARAMETERS_NAME %s'%(param_A, out_load_A, out_cat)
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

	### test the concentration of sources
	Area_img = np.array(source['ISOAREAF_IMAGE'])
	S_l0 = np.array(source['ISO0'])
	S_l2 = np.array(source['ISO2'])
	S_l4 = np.array(source['ISO4'])
	S_l6 = np.array(source['ISO6'])

	concen = S_l4 / S_l0

	Kron = 16 # k_r[kk] ## twice of kron factor, since A, B are semi-axis
	a = Kron * A
	b = Kron * B
	'''
	### divide sources based on area (pixel**2)
	idx0 = concen > 0.6
	trac_x0 = cx[idx0]
	trac_y0 = cy[idx0]
	trac_a0 = A[idx0] * 8
	trac_b0 = B[idx0] * 8
	trac_phi0 = theta[idx0]

	trac_x1 = cx[idx0 == False]
	trac_y1 = cy[idx0 == False]
	trac_a1 = A[idx0 == False] * Kron
	trac_b1 = B[idx0 == False] * Kron
	trac_phi1 = theta[idx0 == False]

	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('source location')
	ax.imshow(img, cmap = 'Greys', origin = 'lower', norm = mpl.colors.LogNorm())
	for mm in range( len(trac_x0) ):
		ellips = Ellipse(xy = (trac_x0[mm], trac_y0[mm]), width = trac_a0[mm], height = trac_b0[mm], angle = trac_phi0[mm], fill = False, 
			ec = 'g', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(ellips)
	for mm in range( len(trac_x1) ):
		ellips = Ellipse(xy = (trac_x1[mm], trac_y1[mm]), width = trac_a1[mm], height = trac_b1[mm], angle = trac_phi1[mm], fill = False, 
			ec = 'r', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(ellips)
	ax.set_xlim(-500, img.shape[1] + 500)
	ax.set_ylim(-500, img.shape[0] + 500)
	plt.show()

	cx = np.r_[trac_x0, trac_x1]
	cy = np.r_[trac_y0, trac_y1]
	a = np.r_[trac_a0, trac_a1]
	b = np.r_[trac_b0, trac_b1]
	theta = np.r_[trac_phi0, trac_phi1]
	'''
	##### stars
	mask = load + 'source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'%(z_g, ra_g, dec_g)
	cat = pds.read_csv(mask, skiprows = 1)
	set_ra = np.array(cat['ra'])
	set_dec = np.array(cat['dec'])
	set_mag = np.array(cat['r'])
	OBJ = np.array(cat['type'])
	xt = cat['Column1']
	x, y = wcs_lis.all_world2pix(set_ra * U.deg, set_dec * U.deg, 1)
	tau = k_r[kk] / 2 # initial case: 10

	set_A = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
	set_B = np.array( [ cat['psffwhm_r'] , cat['psffwhm_g'], cat['psffwhm_i']]) * tau / pixel
	set_chi = np.zeros(set_A.shape[1], dtype = np.float)

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
	sub_A0 = lr_iso[ic] # * 1.5
	sub_B0 = sr_iso[ic] # * 1.5
	sub_chi0 = set_chi[ic]

	# saturated source(may not stars)
	xa = ['SATURATED' in qq for qq in xt]
	xv = np.array(xa)
	idx = xv == True
	ipx = (idx)

	sub_x2 = x[ipx]
	sub_y2 = y[ipx]
	sub_A2 = lr_iso[ipx] # * 3
	sub_B2 = sr_iso[ipx] # * 3
	sub_chi2 = set_chi[ipx]

	comx = np.r_[ sub_x0[sub_A0 > 0], sub_x2[sub_A2 > 0] ]
	comy = np.r_[ sub_y0[sub_A0 > 0], sub_y2[sub_A2 > 0] ]
	Lr = np.r_[ sub_A0[sub_A0 > 0], sub_A2[sub_A2 > 0] ]
	Sr = np.r_[ sub_B0[sub_A0 > 0], sub_B2[sub_A2 > 0] ]
	phi = np.r_[ sub_chi0[sub_A0 > 0], sub_chi2[sub_A2 > 0] ]
	'''
	plt.figure()
	ax = plt.subplot(111)
	ax.set_title('source location')
	ax.imshow(img, cmap = 'Greys', origin = 'lower', norm = mpl.colors.LogNorm())
	for mm in range( len(sub_x0) ):
		ellips = Ellipse(xy = (sub_x0[mm], sub_y0[mm]), width = sub_A0[mm], height = sub_B0[mm], angle = sub_chi0[mm], fill = False, 
			ec = 'g', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(ellips)
	for mm in range( len(sub_x2) ):
		ellips = Ellipse(xy = (sub_x2[mm], sub_y2[mm]), width = sub_A2[mm], height = sub_B2[mm], angle = sub_chi2[mm], fill = False, 
			ec = 'r', ls = '-', linewidth = 1, alpha = 0.5,)
		ax.add_patch(ellips)
	ax.set_xlim(-200, img.shape[1] + 200)
	ax.set_ylim(-200, img.shape[0] + 200)
	plt.savefig('source_location.png', dpi = 300)
	plt.show()
	'''

	cx = np.r_[cx, comx]
	cy = np.r_[cy, comy]
	a = np.r_[a, Lr]
	b = np.r_[b, Sr]
	theta = np.r_[theta, phi]
	Numb = Numb + len(comx)

	mask_path = np.ones((img.shape[0], img.shape[1]), dtype = np.float)
	ox = np.linspace(0, img.shape[1] - 1, img.shape[1])
	oy = np.linspace(0, img.shape[0] - 1, img.shape[0])
	basic_coord = np.array(np.meshgrid(ox, oy))
	major = a / 2
	minor = b / 2
	senior = np.sqrt(major**2 - minor**2)

	for k in range(Numb):
		xc = cx[k]
		yc = cy[k]

		lr = major[k]
		sr = minor[k]
		cr = senior[k]
		chi = theta[k] * np.pi / 180

		set_r = np.int(np.ceil(1.2 * lr))
		la0 = np.max( [np.int(xc - set_r), 0])
		la1 = np.min( [np.int(xc + set_r + 1), img.shape[1] ] )
		lb0 = np.max( [np.int(yc - set_r), 0] ) 
		lb1 = np.min( [np.int(yc + set_r + 1), img.shape[0] ] )

		df1 = (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.cos(chi) + (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.sin(chi)
		df2 = (basic_coord[1,:][lb0: lb1, la0: la1] - yc)* np.cos(chi) - (basic_coord[0,:][lb0: lb1, la0: la1] - xc)* np.sin(chi)
		fr = df1**2 / lr**2 + df2**2 / sr**2
		jx = fr <= 1

		iu = np.where(jx == True)
		iv = np.ones((jx.shape[0], jx.shape[1]), dtype = np.float)
		iv[iu] = np.nan
		mask_path[lb0: lb1, la0: la1] = mask_path[lb0: lb1, la0: la1] * iv

	mask_img = mask_path * img

	hdu = fits.PrimaryHDU()
	hdu.data = mask_img
	hdu.header = head
	#hdu.writeto(load + 'mask/mask_ra%.3f_dec%.3f_z%.3f_%.1fR-kron.fits' % (ra_g, dec_g, z_g, k_r[kk] / 2), overwrite = True)
	#hdu.writeto(load + 'mask/mask_ra%.3f_dec%.3f_z%.3f_%.1fR-kron_divid-source.fits' % (ra_g, dec_g, z_g, k_r[kk] / 2), overwrite = True)
	hdu.writeto(load + 'mask/mask_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.fits' % (ra_g, dec_g, z_g, k_r[kk] / 2), overwrite = True)

	plt.figure()
	ax = plt.subplot(111)
	#ax.set_title('$ source \; masking \; [%.1f R_{Kron}] $' % (k_r[kk] / 2) )
	ax.set_title('$ star \; masking \; [%.1f(FWHM / 2)] $' % (k_r[kk] / 2) )
	tf = ax.imshow(mask_img, cmap = 'coolwarm', origin = 'lower', vmin = -0.05, vmax = 0.05)
	plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies]')
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_xlim(-20, img.shape[1] + 20)
	ax.set_ylim(-20, img.shape[0] + 20)
	plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.8, top = 0.9, wspace = None, hspace = None)
	#plt.savefig('source_mask_%.1f.png' % (k_r[kk] / 2), dpi = 300)
	plt.savefig('star_mask_%.1f.png' % (k_r[kk] / 2), dpi = 300)
	plt.close()

	### divide img into smaller patches
	def sub_blocks(img_divi, nx, ny):
		id_nn = np.isnan(img_divi)
		flux_in = img_divi[id_nn == False]
		brok_y, brok_x = np.where(id_nn == False)

		pos_x = []
		pos_y = []
		flux_bin = []

		step_y = len(brok_y) // ny
		for nn in range(ny):

			if nn == ny - 1:
				da = np.int(nn * step_y)
				pos_x.append(brok_x[da:])
				pos_y.append(brok_y[da:])
				flux_bin.append(flux_in[da:])
			else:
				da0 = np.int( nn * step_y )
				da1 = np.int( (nn + 1) * step_y )
				pos_x.append(brok_x[da0: da1])
				pos_y.append(brok_y[da0: da1])
				flux_bin.append(flux_in[da0: da1])

		sub_x = []
		sub_y = []
		sub_flux = []

		for nn in range(ny):

			step_x = len(pos_x[nn]) // nx

			sort_indx = np.argsort(pos_x[nn])
			ttx = pos_x[nn][ sort_indx ]
			tty = pos_y[nn][ sort_indx ]
			tt_flux = flux_bin[nn][ sort_indx ]

			for mm in range(nx):

				if mm == nx - 1:
					da = np.int(mm * step_x)
					sub_x.append(ttx[da:])
					sub_y.append(tty[da:])
					sub_flux.append(tt_flux[da:])
				else:
					da0 = np.int( mm * step_x )
					da1 = np.int( (mm + 1) * step_x )
					sub_x.append(ttx[da0: da1])
					sub_y.append(tty[da0: da1])
					sub_flux.append(tt_flux[da0: da1])
		return sub_x, sub_y, sub_flux, flux_in

	sub_x, sub_y, sub_flux, flux_in = sub_blocks(mask_img, N_binx, N_biny)

	aveg = np.array([np.mean(ll) for ll in sub_flux])
	sigma = np.array([np.std(ll) for ll in sub_flux])

	##### check out-lier
	percen = 97.5
	dtx = np.percentile(aveg, percen)
	idu = aveg >= dtx
	idx = np.where(idu == True)[0]

	dty = np.percentile(sigma, percen)
	idv = sigma >= dty
	idy = np.where(idv == True)[0]

	res_img = aveg.reshape(N_biny, N_binx)
	## initial case
	#with h5py.File(load + 'sub_mean_ra%.3f_dec%.3f_z%.3f_%.1fR_1R-star.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
	with h5py.File(load + 'flux_bin/sub_mean_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
		f['sub_mean'] = np.array(res_img)
	'''
	plt.figure()
	ax = plt.subplot(111)
	#ax.set_title('out-lier of average $ [%.1f R_{Kron}] $' % (k_r[kk] / 2) )
	ax.set_title('out-lier of average [%.1f(FWHM / 2)]' % (k_r[kk] / 2) )
	tf = ax.imshow(mask_img, cmap = 'coolwarm', origin = 'lower', vmin = -0.05, vmax = 0.05)
	plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies]')
	for nn in range( np.sum(idu) ):
		a0, a1 = sub_x[ idx[nn] ].min(), sub_x[ idx[nn] ].max()
		b0, b1 = sub_y[ idx[nn] ].min(), sub_y[ idx[nn] ].max()
		region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, ec = 'r', linewidth = 1, alpha = 0.5)
		ax.add_patch(region)
	ax.set_xlim(-20, img.shape[1] + 20)
	ax.set_ylim(-20, img.shape[0] + 20)
	plt.savefig('out_mean_blocks_%d.png' % kk, dpi = 300)
	plt.close()

	plt.figure()
	ax = plt.subplot(111)
	#ax.set_title('out-lier of sigma $ [%.1f R_{Kron}] $' % (k_r[kk] / 2) )
	ax.set_title('out-lier of sigma [%.1f(FWHM / 2)]' % (k_r[kk] / 2) )
	tf = ax.imshow(mask_img, cmap = 'coolwarm', origin = 'lower', vmin = -0.05, vmax = 0.05)
	plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux [nanomaggies]')
	for nn in range( np.sum(idv) ):
		a0, a1 = sub_x[ idy[nn] ].min(), sub_x[ idy[nn] ].max()
		b0, b1 = sub_y[ idy[nn] ].min(), sub_y[ idy[nn] ].max()
		region = Rectangle(xy = (a0, b0), width = a1 - a0, height = b1 - b0, fill = False, ec = 'r', linewidth = 1, alpha = 0.5)
		ax.add_patch(region)
	ax.set_xlim(-20, img.shape[1] + 20)
	ax.set_ylim(-20, img.shape[0] + 20)
	plt.savefig('out_sigma_blocks_%d.png' % kk, dpi = 300)
	plt.close()
	'''

	#with h5py.File(load + 'flux_bin/sub_patch_aveg_ra%.3f_dec%.3f_z%.3f_%.1f.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
	#with h5py.File(load + 'flux_bin/sub_patch_aveg_ra%.3f_dec%.3f_z%.3f_%.1f_divid-source.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
	with h5py.File(load + 'flux_bin/sub_patch_aveg_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
		f['aveg'] = np.array(aveg)

	#with h5py.File(load + 'flux_bin/sub_patch_sigma_ra%.3f_dec%.3f_z%.3f_%.1f.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
	#with h5py.File(load + 'flux_bin/sub_patch_sigma_ra%.3f_dec%.3f_z%.3f_%.1f_divid-source.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
	with h5py.File(load + 'flux_bin/sub_patch_sigma_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
		f['sigma'] = np.array(sigma)

	#with h5py.File(load + 'flux_bin/tot_flux_ra%.3f_dec%.3f_z%.3f_%.1f.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
	#with h5py.File(load + 'flux_bin/tot_flux_ra%.3f_dec%.3f_z%.3f_%.1f_divid-source.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
	with h5py.File(load + 'flux_bin/tot_flux_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'w') as f:
		f['flux'] = np.array(flux_in)

raise
"""
############## test for the flux hist
def gau_func(x, mu, sigma):
	return sts.norm.pdf(x, mu, sigma)

mean_arry = []
scatter_arry = []
tot_flux = []

### percentile
mean_percen = []
sigm_percen = []
tot_percen = []
percen = 97.5 #100 # for in-percen

for kk in range( len(k_r) ):

	#with h5py.File(load + 'flux_bin/sub_patch_aveg_ra%.3f_dec%.3f_z%.3f_%.1f.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
	#with h5py.File(load + 'flux_bin/sub_patch_aveg_ra%.3f_dec%.3f_z%.3f_%.1f_divid-source.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
	with h5py.File(load + 'flux_bin/sub_patch_aveg_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
		aveg = np.array(f['aveg'])
	mean_arry.append(aveg)
	dtx = np.percentile(aveg, percen)
	idu = aveg <= dtx
	mean_percen.append(aveg[idu])

	#with h5py.File(load + 'flux_bin/sub_patch_sigma_ra%.3f_dec%.3f_z%.3f_%.1f.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
	#with h5py.File(load + 'flux_bin/sub_patch_sigma_ra%.3f_dec%.3f_z%.3f_%.1f_divid-source.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
	with h5py.File(load + 'flux_bin/sub_patch_sigma_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
		sigma = np.array(f['sigma'])
	scatter_arry.append(sigma)
	dtx = np.percentile(sigma, percen)
	idu = sigma <= dtx
	sigm_percen.append(sigma[idu])

	#with h5py.File(load + 'flux_bin/tot_flux_ra%.3f_dec%.3f_z%.3f_%.1f.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
	#with h5py.File(load + 'flux_bin/tot_flux_ra%.3f_dec%.3f_z%.3f_%.1f_divid-source.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
	with h5py.File(load + 'flux_bin/tot_flux_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
		res_flux = np.array(f['flux'])
	tot_flux.append(res_flux)
	dtx = np.percentile(res_flux, percen)
	idu = res_flux <= dtx
	tot_percen.append(res_flux[idu])

aveg_sb = np.array( [np.mean(ll) for ll in mean_arry] )
aveg_rms = np.array( [np.sqrt(np.sum(ll**2) / (N_binx * N_biny)) for ll in mean_arry] )
aveg_err = aveg_rms / np.sqrt(N_binx * N_biny - 1)

per_aveg_sb = np.array( [np.mean(ll) for ll in mean_percen] )
per_aveg_rms = np.array( [np.sqrt(np.sum(ll**2) / len(ll) ) for ll in mean_percen] )
per_aveg_err = aveg_rms / np.array( [np.sqrt(len(ll) - 1) for ll in mean_percen] )

mean_tot = np.array( [np.mean(ll) for ll in tot_flux] )

plt.figure()
ax = plt.subplot(111)
ax.errorbar(k_r / 2, aveg_sb, yerr = aveg_err, xerr = None, color = 'g', marker = '.', ls = 'None', ecolor = 'g', elinewidth = 1, alpha = 0.5)
#ax.plot(k_r / 2, mean_tot, 'rs', alpha = 0.5)
ax.axhline(y = np.min(aveg_sb), ls = '--', color = 'g', alpha = 0.5)
ax.set_ylabel(' mean flux of sub-patches [nanomaggies]')
#ax.set_xlabel('$ R_{mask} / R_{kron}$')
ax.set_xlabel('$ R_{mask} / [FWHM / 2]$')
plt.subplots_adjust(left = 0.2, bottom = 0.1, right = 0.95, top = 0.9, wspace = None, hspace = None)
plt.savefig('flux_check.png', dpi = 300)
plt.close()

la0 = np.array([np.min(ll) for ll in mean_arry])
la1 = np.array([np.max(ll) for ll in mean_arry])
bins_0 = np.linspace(la0.min(), la1.max(), 60)

lb0 = np.array([np.min(ll) for ll in scatter_arry])
lb1 = np.array([np.max(ll) for ll in scatter_arry])
bins_1 = np.linspace(lb0.min(), lb1.max(), 60)

lc0 = np.array([np.min(ll) for ll in tot_flux])
lc1 = np.array([np.max(ll) for ll in tot_flux])
bins_2 = np.linspace(lc0.min(), lc1.max(), 60)

chi_test = []
nu_on_chi = []
for kk in range( len(k_r) ):
	## parameter fit
	tt_data = mean_percen[kk]
	bin_di = bins_0

	mu = np.mean(tt_data)
	sqr_var = np.std(tt_data)

	pix_n, edgs = binned(tt_data, tt_data, statistic = 'count', bins = bin_di)[:2]
	pdf_pix = (pix_n / np.sum(pix_n) ) / (edgs[1] - edgs[0])
	pdf_err = (np.sqrt(pix_n) / np.sum(pix_n) ) / (edgs[1] - edgs[0])
	x_cen = 0.5 * ( edgs[1:] + edgs[:-1])

	idu = pix_n != 0.
	use_obs = pix_n[idu]
	use_err = np.sqrt(use_obs)
	use_x = x_cen[idu]
	popt, pcov = curve_fit(gau_func, use_x, pdf_pix[idu], p0 = [mu, sqr_var], sigma = pdf_err[idu],)
	#	bounds = ([mu - 0.1 * mu, sqr_var - 0.1 * sqr_var], [mu + 0.1 * mu, sqr_var + 0.1 * sqr_var]),)
	e_mu, e_chi = popt[0], popt[1]
	f_gau = gau_func(x_cen, 0, e_chi)

	### chi-square
	fit_line = gau_func(use_x, e_mu, e_chi)
	use_expt = fit_line * (edgs[1] - edgs[0]) * len(tt_data)
	Xi = np.sum( (use_obs - use_expt)**2 / (use_err)**2 )

	freedom = np.sum(idu) - 2 - 1
	chi_test.append(Xi)
	nu_on_chi.append(freedom)

	plt.figure()
	ax = plt.subplot(111)
	#ax.set_title('$ R_{mask} / R_{Kron}$ = %.1f' % (k_r[kk] / 2) )
	ax.set_title('$ R_{mask} $ / [FWHM / 2] = %.1f' % (k_r[kk] / 2) )
	ax.errorbar(x_cen, pdf_pix, yerr = pdf_err, xerr = None, color = 'b', marker = '.', ls = 'None', ecolor = 'b', elinewidth = 1, )
	ax.hist(tt_data, bins = bin_di, histtype = 'step', density = True, color = 'b', alpha = 0.5, linestyle = '-', label = 'data')
	ax.plot(x_cen, f_gau, color = 'r', alpha = 0.5, label = 'Gaussian $ [\\mu = 0] $')
	ax.plot(use_x, fit_line, color = 'g', alpha = 0.5, label = 'data fit $[\\mu = %.5f]$' % e_mu)

	ax.axvline(x = 0, color = 'r', ls = '--', alpha = 0.5)
	ax.axvline(x = e_mu, color = 'g', ls = '--', alpha = 0.5)
	ax.axvline(x = e_mu - e_chi, color = 'g', ls = '-.', alpha = 0.5)
	ax.axvline(x = e_mu + e_chi, color = 'g', ls = '-.', alpha = 0.5)

	ax.set_xlabel('flux [nanomaggies]')
	ax.set_ylabel('PDF')
	ax.legend(loc = 2, frameon = False)
	ax.set_xlim(mu - 5 * sqr_var, mu + 5 * sqr_var)
	plt.savefig('percen_hist_%.1fR_kron.png' % (k_r[kk] / 2), dpi = 300)
	plt.close()

raise

## initial case
with h5py.File(load + 'sub_mean_ra%.3f_dec%.3f_z%.3f_8.0R_1R-star.h5' % (ra_g, dec_g, z_g), 'r') as f:
	sub_mean_ini = np.array(f['sub_mean'])
	cc_ini = np.abs(sub_mean_ini).max()

res_img = []
for kk in range( len(k_r) ):

	with h5py.File(load + 'flux_bin/sub_mean_ra%.3f_dec%.3f_z%.3f_8.0R-kron_%.1fR-fwhm.h5' % (ra_g, dec_g, z_g, k_r[kk] / 2), 'r') as f:
		sub_mean = np.array(f['sub_mean'])
	res_img.append(sub_mean)

	plt.figure( figsize = (18, 6) )
	ax0 = plt.subplot(131)
	ax1 = plt.subplot(132)
	ax2 = plt.subplot(133)

	ax0.set_title('initial case')
	tf = ax0.imshow(sub_mean_ini / cc_ini, cmap = 'coolwarm', origin = 'lower', vmin = -1, vmax = 1)
	plt.colorbar(tf, ax = ax0, fraction = 0.040, pad = 0.01, label = 'flux / %.5f' % cc_ini)

	ax1.set_title('$ Adjust: R_{mask} $ / [FWHM / 2] = %.1f' % (k_r[kk] / 2) )
	tf = ax1.imshow(sub_mean / cc_ini, cmap = 'coolwarm', origin = 'lower', vmin = -1, vmax = 1)
	plt.colorbar(tf, ax = ax1, fraction = 0.040, pad = 0.01, label = 'flux / %.5f' % cc_ini)

	ax2.set_title('Adjust - initial case')
	tf = ax2.imshow(sub_mean - sub_mean_ini, cmap = 'bwr', origin = 'lower', vmin = -4e-3, vmax = 4e-3)
	plt.colorbar(tf, ax = ax2, fraction = 0.040, pad = 0.01, label = 'flux [nanomaggy]')	

	plt.tight_layout()
	plt.savefig('blocks_mean_compare_%d.png' % kk, dpi = 300)
	plt.close()
