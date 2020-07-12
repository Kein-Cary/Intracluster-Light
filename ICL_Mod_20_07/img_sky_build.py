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
		hdu.writeto(out_file + 'sky-%s-band-ra%.3f-dec%.3f-z%.3f.fits' % (band, ra_g, dec_g, z_g,), overwrite = True)

	return

def main():

	dat = pds.read_csv('/home/xkchen/mywork/ICL/r_band_sky_catalog.csv')
	ra, dec, z = dat.ra, dat.dec, dat.z
	Nz = 10
	set_ra, set_dec, set_z = ra[:10], dec[:10], z[:10]
	d_file = '/home/xkchen/mywork/ICL/data/sdss_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	out_file = '/home/xkchen/mywork/ICL/data/tmp_img/source_find/'
	band = 'r'
	sky_build_func(d_file, set_z, set_ra, set_dec, band, out_file)

if __name__ == "__main__":
	main()
