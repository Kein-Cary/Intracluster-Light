import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Ellipse, Rectangle

import h5py
import numpy as np
import pandas as pds
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy

#constant
rad2arcsec = U.rad.to(U.arcsec)

# cosmology model
vc = C.c.to(U.km/U.s).value
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0

# band information of SDSS
band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
mag_add = np.array([0, 0, 0, -0.04, 0.02])

### img grid
def cc_grid_img( img_data, N_stepx, N_stepy):

	binx = img_data.shape[1] // N_stepx ## bin number along 2 axis
	biny = img_data.shape[0] // N_stepy

	beyon_x = img_data.shape[1] - binx * N_stepx ## for edge pixels divid
	beyon_y = img_data.shape[0] - biny * N_stepy

	odd_x = np.ceil(beyon_x / binx)
	odd_y = np.ceil(beyon_y / biny)

	n_odd_x = beyon_x // odd_x
	n_odd_y = beyon_y // odd_y

	d_odd_x = beyon_x - odd_x * n_odd_x
	d_odd_y = beyon_y - odd_y * n_odd_y

	# get the bin width
	wid_x = np.zeros(binx, dtype = np.float32)
	wid_y = np.zeros(biny, dtype = np.float32)
	for kk in range(binx):
		if kk == n_odd_x :
			wid_x[kk] = N_stepx + d_odd_x
		elif kk < n_odd_x :
			wid_x[kk] = N_stepx + odd_x
		else:
			wid_x[kk] = N_stepx

	for kk in range(biny):
		if kk == n_odd_y :
			wid_y[kk] = N_stepy + d_odd_y
		elif kk < n_odd_y :
			wid_y[kk] = N_stepy + odd_y
		else:
			wid_y[kk] = N_stepy

	# get the bin edge
	lx = np.zeros(binx + 1, dtype = np.int32)
	ly = np.zeros(biny + 1, dtype = np.int32)
	for kk in range(binx):
		lx[kk + 1] = lx[kk] + wid_x[kk]
	for kk in range(biny):
		ly[kk + 1] = ly[kk] + wid_y[kk]

	patch_mean = np.zeros( (biny, binx), dtype = np.float )
	patch_pix = np.zeros( (biny, binx), dtype = np.float )
	patch_S0 = np.zeros( (biny, binx), dtype = np.float )
	patch_Var = np.zeros( (biny, binx), dtype = np.float )
	for nn in range( biny ):
		for tt in range( binx ):

			sub_flux = img_data[ly[nn]: ly[nn + 1], lx[tt]: lx[tt + 1] ]
			id_nn = np.isnan(sub_flux)

			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )
			patch_S0[nn,tt] = (ly[nn + 1] - ly[nn]) * (lx[tt + 1] - lx[tt])

	return patch_mean, patch_pix, patch_Var, patch_S0, lx, ly

def grid_img( img_data, N_stepx, N_stepy):

	ly = np.arange(0, img_data.shape[0], N_stepy)
	ly = np.r_[ly, img_data.shape[0] - N_stepy, img_data.shape[0] ]
	lx = np.arange(0, img_data.shape[1], N_stepx)
	lx = np.r_[lx, img_data.shape[1] - N_stepx, img_data.shape[1] ]

	lx = np.delete(lx, -1)
	lx = np.delete(lx, -2)
	ly = np.delete(ly, -1)
	ly = np.delete(ly, -2)

	patch_mean = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_pix = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	patch_Var = np.zeros( (len(ly), len(lx) ), dtype = np.float )
	for nn in range( len(ly) ):
		for tt in range( len(lx) ):

			sub_flux = img_data[ly[nn]: ly[nn] + N_stepy, lx[tt]: lx[tt] + N_stepx]
			id_nn = np.isnan(sub_flux)
			patch_mean[nn,tt] = np.mean( sub_flux[id_nn == False] )
			patch_pix[nn,tt] = len( sub_flux[id_nn == False] )
			patch_Var[nn,tt] = np.std( sub_flux[id_nn == False] )

	return patch_mean, patch_pix, patch_Var, lx, ly

def zref_BCG_pos_func( cat_file, z_ref, out_file):
	"""
	this part use for calculate BCG position after pixel resampling. 
	"""
	dat = pds.read_csv( cat_file )
	ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)
	clus_x, clus_y = np.array(dat.bcg_x), np.array(dat.bcg_y)

	Da_z = Test_model.angular_diameter_distance(z).value
	Da_ref = Test_model.angular_diameter_distance(z_ref).value

	L_ref = Da_ref * pixel / rad2asec
	L_z = Da_z * pixel / rad2asec
	eta = L_ref / L_z

	ref_bcgx = np.array( [np.int(ll) for ll in clus_x / eta] )
	ref_bcgy = np.array( [np.int(ll) for ll in clus_y / eta] )

	keys = ['ra', 'dec', 'z', 'bcg_x', 'bcg_y']
	values = [ra, dec, z, ref_bcgx, ref_bcgy]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv( out_file )

	return

### stacking region select based on the grid variance
def SN_lim_region_select( stack_img, lim_img_id, lim_set, grd_len,):
	"""
	lim_img_id : use for marker selected region, 
				0 -- use blocks' mean value
				1 -- use blocks' Variance
				2 -- use blocks' effective pixel number 
				3 -- use blocks' effective pixel fraction
				(check the order of return array of cc_grid_img function)
	lim_set : threshold for region selection
	grd_len : block edges for imgs division
	"""
	id_nan = np.isnan( stack_img )
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = stack_img[y_low: y_up+1, x_low: x_up + 1]
	grid_patch = cc_grid_img(dpt_img, grd_len, grd_len,)
	lx, ly = grid_patch[-2], grid_patch[-1]

	lim_img = grid_patch[ lim_img_id ]
	id_region = lim_img >= lim_set

	Nx, Ny = lim_img.shape[1], lim_img.shape[0]
	tp_block = np.ones((Ny, Nx), dtype = np.float32)
	tp_block[id_region] = np.nan

	copy_img = dpt_img.copy()

	for ll in range( Ny ):
		for pp in range( Nx ):

			idx_cen = (pp >= 17) & (pp <= 25)
			idy_cen = (ll >= 12) & (ll <= 17)
			id_cen = idx_cen & idy_cen

			if id_cen == True:
				continue
			else:
				lim_x0, lim_x1 = lx[pp], lx[ pp + 1 ]
				lim_y0, lim_y1 = ly[ll], ly[ ll + 1 ]

				id_NAN = np.isnan( tp_block[ll,pp] )

				if id_NAN == True:
					copy_img[lim_y0: lim_y1, lim_x0: lim_x1] = np.nan
				else:
					continue

	SN_mask = np.ones((dpt_img.shape[0], dpt_img.shape[1]), dtype = np.float32)
	id_nn = np.isnan(copy_img)
	SN_mask[id_nn] = np.nan

	all_mask = np.ones((stack_img.shape[0], stack_img.shape[1]), dtype = np.float32)
	id_nn = np.isnan(stack_img)
	all_mask[id_nn] = np.nan
	all_mask[y_low: y_up+1, x_low: x_up + 1] = SN_mask

	return all_mask

#??????????????

"""
### 2D img compare
for mm in range( 3 ):

	with h5py.File(
		'/home/xkchen/jupyter/stack/Bro-mode-select_Mean_jack_img_%.1f-sigma_%d-FWHM-ov2_z-ref.h5' % (tt_sigma[mm], size_arr[0]), 'r') as f:
		tt_img = np.array(f['a'])

	id_nan = np.isnan(tt_img)
	idvx = id_nan == False
	idy, idx = np.where(idvx == True)
	x_low, x_up = np.min(idx), np.max(idx)
	y_low, y_up = np.min(idy), np.max(idy)

	dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
	img_block = cc_grid_img(dpt_img, 100, 100,)[0]

	block_arr = []

	for kk in range( 1,3):

		if kk == 2:
			with h5py.File(
			load + '20_10_test_jack/Bro-mode-select_Mean_jack_img_selected-by-tot_%.1f-sigma_z-ref.h5' % tt_sigma[mm], 'r') as f:
				tk_img = np.array(f['a'])
		else:
			with h5py.File(
			'/home/xkchen/jupyter/stack/Bro-mode-select_Mean_jack_img_%.1f-sigma_%d-FWHM-ov2_z-ref.h5' % (tt_sigma[mm], size_arr[kk]), 'r') as f:
				tk_img = np.array(f['a'])

		dpk_img = tk_img[y_low: y_up+1, x_low: x_up + 1]
		patch_mean = cc_grid_img(dpk_img, 100, 100,)[0]
		block_arr.append( patch_mean )

	fig = plt.figure( figsize = (19.84, 4.8) )
	ax0 = fig.add_axes([0.02, 0.09, 0.28, 0.85])
	ax1 = fig.add_axes([0.35, 0.09, 0.28, 0.85])
	ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.85])

	ax0.set_title('10 (FWHM/2)')
	tg = ax0.imshow(img_block / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax1.set_title('20 (FWHM/2)')
	tg = ax1.imshow(block_arr[0] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	ax2.set_title('30 (FWHM/2)')
	tg = ax2.imshow(block_arr[1] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
	cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
	cb.formatter.set_powerlimits( (0,0) )

	plt.savefig('2D-img_compare_%.1f-sigma.png' % (tt_sigma[mm]), dpi = 300)
	plt.close()

tot_img_lis = [ '/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_img_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-tot_Mean_jack_img_20-FWHM-ov2_z-ref.h5',
				load + '20_10_test_jack/T1000-tot_BCG-stack_Mean_jack_img_z-ref.h5']

A250_img_lis = ['/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_img_10-FWHM-ov2_z-ref.h5',
				'/home/xkchen/jupyter/stack/T1000-to-A250_Mean_jack_img_20-FWHM-ov2_z-ref.h5',
				load + '20_10_test_jack/A_clust_BCG-stack_Mean_jack_img_30-FWHM-ov2_z-ref.h5']

tot_patch = []
Asub_patch = []

with h5py.File(tot_img_lis[0], 'r') as f:
	tt_img = np.array(f['a'])

id_nan = np.isnan(tt_img)
idvx = id_nan == False
idy, idx = np.where(idvx == True)
x_low, x_up = np.min(idx), np.max(idx)
y_low, y_up = np.min(idy), np.max(idy)

dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
img_block = cc_grid_img(dpt_img, 100, 100,)[0]
tot_patch.append( img_block )

with h5py.File(A250_img_lis[0], 'r') as f:
	tt_img = np.array(f['a'])
dpt_img = tt_img[y_low: y_up+1, x_low: x_up + 1]
img_block = cc_grid_img(dpt_img, 100, 100,)[0]
Asub_patch.append( img_block )

### 2D img compare
for mm in range( 1,3 ):

	with h5py.File(tot_img_lis[mm], 'r') as f:
		tk_img = np.array(f['a'])

	dpk_img = tk_img[y_low: y_up+1, x_low: x_up + 1]
	patch_mean = cc_grid_img(dpk_img, 100, 100,)[0]
	tot_patch.append( patch_mean )	

	with h5py.File(A250_img_lis[mm], 'r') as f:
		tk_img = np.array(f['a'])

	dpk_img = tk_img[y_low: y_up+1, x_low: x_up + 1]
	patch_mean = cc_grid_img(dpk_img, 100, 100,)[0]
	Asub_patch.append( patch_mean )		

fig = plt.figure( figsize = (19.84, 4.8) )
#fig.suptitle('A-250')
fig.suptitle('tot-1000')

ax0 = fig.add_axes([0.02, 0.09, 0.28, 0.80])
ax1 = fig.add_axes([0.35, 0.09, 0.28, 0.80])
ax2 = fig.add_axes([0.68, 0.09, 0.28, 0.80])

ax0.set_title('10 (FWHM/2)')
#tg = ax0.imshow(Asub_patch[0] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
tg = ax0.imshow(tot_patch[0] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax0, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax1.set_title('20 (FWHM/2)')
#tg = ax1.imshow(Asub_patch[1] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
tg = ax1.imshow(tot_patch[1] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax1, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

ax2.set_title('30 (FWHM/2)')
#tg = ax2.imshow(Asub_patch[2] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
tg = ax2.imshow(tot_patch[2] / pixel**2, origin = 'lower', cmap = 'seismic', vmin = -4e-2, vmax = 4e-2,)
cb = plt.colorbar(tg, ax = ax2, fraction = 0.035, pad = 0.01, label = 'SB [nanomaggies / $arcsec^2$]',)
cb.formatter.set_powerlimits( (0,0) )

#plt.savefig('A250_2D-img_compare.png', dpi = 300)
plt.savefig('T1000_2D-img_compare.png', dpi = 300)
plt.close()
"""

