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
		hdu.writeto(out_file % (ra_g, dec_g, z_g, band), overwrite = True)

	return

def main():
	home = '/media/xkchen/My Passport/data/SDSS/'

	dat = pds.read_csv('/home/xkchen/Downloads/test_imgs/random_clus-1000-match_cat.csv')
	set_ra, set_dec, set_z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

	d_file = home + 'redMap_random/rand_img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	out_file = '/media/xkchen/My Passport/data/SDSS/random_cat/sky_img/random_sky-ra%.3f-dec%.3f-z%.3f-%s-band.fits'

	band = 'r'
	sky_build_func(d_file, set_z, set_ra, set_dec, band, out_file)

	raise

if __name__ == "__main__":
	main()
