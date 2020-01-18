import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import numpy as np
import astropy.wcs as awc
import astropy.io.fits as fits

from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
from extinction_redden import A_wave

from mpi4py import MPI
commd = MPI.COMM_WORLD
rank = commd.Get_rank()
cpus = commd.Get_size()
# introduce dust map
import sfdmap
Rv = 3.1
sfd = SFDQuery()

band = ['r', 'g', 'i', 'u', 'z']
l_wave = np.array([6166, 4686, 7480, 3551, 8932])
dfile = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/' ## save the catalogue data
load = '/mnt/ddnfs/data_users/cxkttwl/ICL/data/'  ## save the process data
tmp = '/mnt/ddnfs/data_users/cxkttwl/PC/'

def extinction_correct(band_id, sub_z, sub_ra, sub_dec):
	tot_N = len(sub_z)
	ii = np.int(band_id)
	for jj in range(tot_N):
		ra_g = sub_ra[jj]
		dec_g = sub_dec[jj]
		z_g = sub_z[jj]

		file = dfile + 'frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2' % (band[ii], ra_g, dec_g, z_g)
		data = fits.open(file)
		img = data[0].data
		head_inf = data[0].header
		wcs = awc.WCS(head_inf)

		x0 = np.linspace(0, img.shape[1] - 1, img.shape[1])
		y0 = np.linspace(0, img.shape[0] - 1, img.shape[0])
		img_grid = np.array(np.meshgrid(x0, y0))
		ra_img, dec_img = wcs.all_pix2world(img_grid[0,:], img_grid[1,:], 1)
		pos = SkyCoord(ra_img, dec_img, frame = 'fk5', unit = 'deg')
		BEV = sfd(pos)
		Av = Rv * BEV
		Al = A_wave(l_wave[ii], Rv) * Av
		correct_img = img * 10**(Al / 2.5)

		## save the extinction correct data
		hdu = fits.PrimaryHDU()
		hdu.data = correct_img
		hdu.header = head_inf
		hdu.writeto(tmp + 'test/Extinction_correct_%s_ra%.3f_dec%.3f_z%.3f.fits'%(band[ii], ra_g, dec_g, z_g), overwrite = True)

def main():
	## here the image data have been selected for rule out defects imgs

	#for kk in range(len(band)):
	for kk in range( 3 ):
		with h5py.File(load + 'mpi_h5/%s_band_sample_catalog.h5' % band[kk], 'r') as f:
			cat = np.array(f['a'])
		ra, dec, z = cat[0,:], cat[1,:], cat[2,:]

		zN = len(z)
		m, n = divmod(zN, cpus)
		N_sub0, N_sub1 = m * rank, (rank + 1) * m
		if rank == cpus - 1:
			N_sub1 += n

		rich_divid(kk, z[N_sub0 :N_sub1], ra[N_sub0 :N_sub1], dec[N_sub0 :N_sub1])
		commd.Barrier()

if __name__ == "__main__":
	main()
