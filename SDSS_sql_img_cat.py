import h5py
import numpy as np
import astropy.io.fits as fits

load = '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/'

with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
	catalog = np.array(f['a'])

z = catalog[0]
ra = catalog[1]
dec = catalog[2]
Nz = len(z)
def img_cat():
	img_cat = np.zeros((Nz, 2), dtype = np.float)
	tim = np.zeros(Nz, dtype = np.float)
	for q in range(Nz):
		Ra = ra[q]
		Dec = dec[q]
		tim[q] = z[q]
		file = 'frame-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%(Ra, Dec, z[q])
		data = fits.getdata(load + file, header = True)
		img_cat[q, 0] = data[1]['CRVAL1']
		img_cat[q, 1] = data[1]['CRVAL2']
		print(q/Nz)
	pos = np.array([tim, img_cat[:,0], img_cat[:,1]])
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sdss_sql_img_catalog.h5', 'w') as f:
		f['a'] = np.array(pos)
	with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sdss_sql_img_catalog.h5', ) as f:
		for k in range(len(pos)):
			f['a'][k,:] = pos[k,:]

	np.savetxt('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sdss_sql_img_catalog.txt', pos)

	return

def main():
	img_cat()

if __name__ == "__main__":
	main()
