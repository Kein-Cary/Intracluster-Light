import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import pandas as pds
import astropy.io.fits as fits

import astropy.wcs as awc
import astropy.io.fits as fits
import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy
from resample_modelu import sum_samp, down_samp
from matplotlib.patches import Circle, Ellipse
from light_measure import light_measure, light_measure_Z0

import mechanize
from io import StringIO

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()

kpc2cm = U.kpc.to(U.cm)
Mpc2pc = U.Mpc.to(U.pc)
Mpc2cm = U.Mpc.to(U.cm)
rad2asec = U.rad.to(U.arcsec)
pc2cm = U.pc.to(U.cm)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Rpp = (rad2asec / Da_ref) / pixel
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631 * Jy # zero point in unit (erg/s)/cm^-2

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/'
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'
band = ['r', 'g', 'i', 'u', 'z']

url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'
r_select = 0.167 # centered at BCG, radius = 10 arcmin (1515.15 pixel)

def phot_z_sample():
	dat = fits.open(load + 
		'data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
	cat = dat[1].data
	RA = cat.RA
	DEC = cat.DEC
	Rich = cat.LAMBDA

	u_mag = cat.MODEL_MAG_U
	g_mag = cat.MODEL_MAG_G
	r_mag = cat.MODEL_MAG_R
	i_mag = cat.MODEL_MAG_I
	z_mag = cat.MODEL_MAG_Z	

	Z0 = cat.Z_SPEC
	idx = (Z0 >= 0.2) & (Z0 <= 0.3)
	ra0 = RA[idx]
	dec0 = DEC[idx]
	spec_z = Z0[idx]
	rich0 = Rich[idx]
	mag_U0 = u_mag[idx]
	mag_G0 = g_mag[idx]
	mag_R0 = r_mag[idx]
	mag_I0 = i_mag[idx]
	mag_Z0 = z_mag[idx]

	Z1 = cat.Z_LAMBDA
	idy = (Z1 >= 0.2) & (Z1 <= 0.3)
	ra1 = RA[idy]
	dec1 = DEC[idy]
	phot_z = Z1[idy]
	rich1 = Rich[idy]
	mag_U1 = u_mag[idy]
	mag_G1 = g_mag[idy]
	mag_R1 = r_mag[idy]
	mag_I1 = i_mag[idy]
	mag_Z1 = z_mag[idy]

	ra0_S = ['%.3f'% ll for ll in ra0]
	dec0_S = ['%.3f'% ll for ll in dec0]
	ra1_S = ['%.3f'% ll for ll in ra1]
	dec1_S = ['%.3f'% ll for ll in dec1]

	## save the difference samples in photo_z
	pho_z = np.zeros(len(phot_z), dtype = np.float)
	pho_ra = np.zeros(len(phot_z), dtype = np.float)
	pho_dec = np.zeros(len(phot_z), dtype = np.float)
	pho_rich = np.zeros(len(phot_z), dtype = np.float)
	pho_u_mag = np.zeros(len(phot_z), dtype = np.float)
	pho_g_mag = np.zeros(len(phot_z), dtype = np.float)
	pho_r_mag = np.zeros(len(phot_z), dtype = np.float)
	pho_i_mag = np.zeros(len(phot_z), dtype = np.float)	
	pho_z_mag = np.zeros(len(phot_z), dtype = np.float)

	for kk in range(len(phot_z)):
		if (ra1_S[kk] in ra0_S) & (dec1_S[kk] in dec0_S):
			continue
		else:
			pho_z[kk] = phot_z[kk]
			pho_ra[kk] = ra1[kk]
			pho_dec[kk] = dec1[kk]
			pho_rich[kk] = rich1[kk]
			pho_u_mag[kk] = mag_U1[kk]
			pho_g_mag[kk] = mag_G1[kk]
			pho_r_mag[kk] = mag_R1[kk]
			pho_i_mag[kk] = mag_I1[kk]
			pho_z_mag[kk] = mag_Z1[kk]

	idu = pho_z != 0
	pho_ra = pho_ra[idu]
	pho_dec = pho_dec[idu]
	pho_rich = pho_rich[idu]
	pho_u_mag = pho_u_mag[idu]
	pho_g_mag = pho_g_mag[idu]
	pho_r_mag = pho_r_mag[idu]
	pho_i_mag = pho_i_mag[idu]
	pho_z_mag = pho_z_mag[idu]
	pho_z = pho_z[idu]

	## save the data 
	keys = ['ra', 'dec', 'z', 'rich', 'r_Mag', 'g_Mag', 'i_Mag', 'u_Mag', 'z_Mag']
	values = [pho_ra, pho_dec, pho_z, pho_rich, pho_r_mag, pho_g_mag, pho_i_mag, pho_u_mag, pho_z_mag]
	fill = dict(zip(keys, values))
	data = pds.DataFrame(fill)
	data.to_csv(load + 'data/selection/photo_z_difference_sample.csv')

	## save h5py for mpirun
	sub_array = np.array(values)
	with h5py.File(load + 'data/mpi_h5/photo_z_difference_sample.h5', 'w') as f:
		f['a'] = np.array(sub_array)

	with h5py.File(load + 'data/mpi_h5/photo_z_difference_sample.h5') as f:
		for tt in range( len(sub_array) ):
			f['a'][tt,:] = sub_array[tt,:]

def photo_z_fig(band_id, sub_z, sub_ra, sub_dec):

	for mm in range(len(sub_ra)):
		z_g = sub_z[mm]
		ra_g = sub_ra[mm]
		dec_g = sub_dec[mm]
		try:
			file = load + 'photo_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[band_id], ra_g, dec_g, z_g)
			dat = fits.open(file)
			img = dat[0].data
			head = dat[0].header
			wcs = awc.WCS(head)
			cx, cy = wcs.all_world2pix(ra_g * U.deg, dec_g * U.deg, 1)
			R_pix = (rad2asec / (Test_model.angular_diameter_distance(z_g).value)) / pixel

			plt.figure()
			ax = plt.subplot(111)
			ax.set_title('photo_z %s_band ra%.3f dec%.3f z%.3f' % (band[band_id], ra_g, dec_g, z_g) )
			clust = Circle(xy = (cx, cy), radius = R_pix, fill = False, ec = 'b', alpha = 0.5)
			tf = ax.imshow(img, origin = 'lower', cmap = 'Greys', vmin = 1e-5, vmax = 1e2, norm = mpl.colors.LogNorm())
			ax.add_patch(clust)
			plt.colorbar(tf, ax = ax, fraction = 0.035, pad = 0.01, label = 'flux[nmaggy]')
			ax.set_xlim(0, img.shape[1])
			ax.set_ylim(0, img.shape[0])
			plt.savefig(load + 
				'fig_class/photo_z/photo_z_%s_band_ra%.3f_dec%.3f_z%.3f.png' % (band[band_id], ra_g, dec_g, z_g), dpi = 300)
			plt.close()
		except FileNotFoundError:
			print('rank', rank)
			print('ra%.3f = ' % ra_g)
			print('dec%.3f = ' % dec_g)
			print('z%.3f = ' % z_g)
			continue

def photo_star_sql(z_set, ra_set, dec_set):

	Nz = len(z_set)
	for q in range(Nz):
		time = z_set[q]
		ra = ra_set[q]
		dec = dec_set[q]
		set_r = r_select
		c_ra0 = str(ra - set_r)
		c_dec0 = str(dec - set_r)
		c_ra1 = str(ra + set_r)
		c_dec1 = str(dec + set_r)

		data_set = """
		SELECT ALL
			p.ra,p.dec,p.u,p.g,p.r,p.i,p.z,p.type,  
			p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
			p.petroR90_u, p.petroR90_g, p.petroR90_r, p.petroR90_i, p.petroR90_z,

			p.deVRad_u, p.deVRad_g, p.deVRad_r, p.deVRad_i, p.deVRad_z,
			p.deVAB_u, p.deVAB_g, p.deVAB_r, p.deVAB_i, p.deVAB_z,
			p.deVPhi_u, p.deVPhi_g, p.deVPhi_r, p.deVPhi_i, p.deVPhi_z,

			p.expRad_u, p.expRad_g, p.expRad_r, p.expRad_i, p.expRad_z,
			p.expAB_u, p.expAB_g, p.expAB_r, p.expAB_i, p.expAB_z,
			p.expPhi_u, p.expPhi_g, p.expPhi_r, p.expPhi_i, p.expPhi_z,
			p.flags, dbo.fPhotoFlagsN(p.flags)
		FROM PhotoObj AS p
		WHERE
			p.ra BETWEEN %s AND %s
			AND p.dec BETWEEN %s AND %s
			AND (p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0)
		ORDER by p.r
		""" % (c_ra0, c_ra1, c_dec0, c_dec1)
		
		br = mechanize.Browser()
		resp = br.open(url)
		resp.info()
		#print(data_set)
		
		br.select_form(name = "sql")
		br['cmd'] = data_set
		br['format'] = ['csv']
		response = br.submit()
		s = str(response.get_data(), encoding = 'utf-8')
		doc = open(load + 'data/photo_z/star_cat/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'%(time, ra, dec), 'w')
		print(s, file = doc)
		doc.close()

def main():
	#phot_z_sample()
	'''
	## image data
	for tt in range(3):
		with h5py.File(load + 'data/mpi_h5/photo_z_difference_sample.h5', 'r') as f:
			dat = np.array(f['a'])
		ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
		zN = len(z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n 
		photo_z_fig(tt, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		commd.Barrier()
	'''
	## star catalogue
	with h5py.File(load + 'data/mpi_h5/photo_z_difference_sample.h5', 'r') as f:
		dat = np.array(f['a'])
	ra, dec, z = dat[0,:], dat[1,:], dat[2,:]
	zN = len(z)
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n
	photo_star_sql(z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
	commd.Barrier()	

if __name__ == "__main__":
	main()
