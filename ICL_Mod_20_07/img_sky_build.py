import h5py
import numpy as np
import pandas as pds
import astropy.wcs as awc
import astropy.io.fits as fits
from scipy.ndimage import map_coordinates as mapcd

def sky_build_func(d_file, z_set, ra_set, dec_set, band, out_file):
	"""
	d_file : path where save the masked data (include file-name structure:'/xxx/xxx/xxx.xxx')
	z_set, ra_set, dec_set : ra, dec, z of will be resampled imgs
	band : the band of imgs, 'str' type
	out_file : path where to save the resampling img
	"""
	Nz = len(z_set)

	for jj in range(Nz):

		ra_g = ra_set[jj]
		dec_g = dec_set[jj]
		z_g = z_set[jj]

		file = d_file % (band, ra_g, dec_g, z_g)

		try:
			data = fits.open(file)
			img = data[0].data
			sky0 = data[2].data['ALLSKY'][0]
			sky_x = data[2].data['XINTERP'][0]
			sky_y = data[2].data['YINTERP'][0]
			inds = np.array(np.meshgrid(sky_x, sky_y))
			t_sky = mapcd(sky0, [inds[1,:], inds[0,:]], order = 1, mode = 'nearest')
			sky_bl = t_sky * (data[0].header['NMGY'])

			## save the sky img
			hdu = fits.PrimaryHDU()
			hdu.data = sky_bl
			hdu.header = data[0].header
			hdu.writeto(out_file % (band, ra_g, dec_g, z_g), overwrite = True)

		except:
			continue

	return

def main():

	from mpi4py import MPI
	commd = MPI.COMM_WORLD
	rank = commd.Get_rank()
	cpus = commd.Get_size()

	band = ['r', 'g', 'i']
	home = '/home/xkchen/data/SDSS/'
	'''
	## random imgs
	dat = pds.read_csv(home + 'selection/redMapper_rand_cat.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	out_file = home + 'random_cat/sky_img/random_sky_%s-band_ra%.3f-dec%.3f-z%.3f.fits'

	zN = len( set_z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	sky_build_func(d_file, set_z[N_sub0:N_sub1], set_ra[N_sub0:N_sub1], set_dec[N_sub0:N_sub1], band[0], out_file)

	print('finished random imgs')
	commd.Barrier()
	'''

	## cluster imgs
	dat = pds.read_csv(home + 'selection/target_BCG_catalogue.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z_spec)

	zN = len( set_z )
	m, n = divmod(zN, cpus)
	N_sub0, N_sub1 = m * rank, (rank + 1) * m
	if rank == cpus - 1:
		N_sub1 += n

	d_file = home + 'wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	out_file = home + 'sky/origin_sky/sky-%s-band-ra%.3f-dec%.3f-z%.3f.fits'

	for kk in range( 3 ):

		sky_build_func(d_file, set_z[N_sub0:N_sub1], set_ra[N_sub0:N_sub1], set_dec[N_sub0:N_sub1], band[kk], out_file)

	print('finished !')

	raise

if __name__ == "__main__":
	main()
