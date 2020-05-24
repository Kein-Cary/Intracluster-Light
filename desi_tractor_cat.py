import numpy as np
import pandas as pds
import astropy.io.fits as fits
import wget as wt

dat = pds.read_csv('/home/xkchen/mywork/ICL/BASS_cat/sdss_z-lambda_cat.csv')
ra = np.array(dat.ra)
dec = np.array(dat.dec)
z = np.array(dat.z)

band = ['g', 'r', 'z']
angl_r = 0.25 ## deg, the brick size of DESI

doc = open('/home/xkchen/mywork/ICL/BASS_cat/err_tractor_cat.txt', 'w')

for jj in range(len(z)):
	ra_g = ra[jj]
	dec_g = dec[jj]
	z_g = z[jj]

	ralo = ra_g - angl_r
	rahi = ra_g + angl_r
	declo = dec_g - angl_r
	dechi = dec_g + angl_r

	try:
		url_load = 'http://legacysurvey.org/viewer/dr8/cat.fits?ralo=%f&rahi=%f&declo=%f&dechi=%f' % (ralo, rahi, declo, dechi)

		out_file = '/home/xkchen/mywork/ICL/BASS_cat/desi_tractor-cat_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)
		wt.download(url_load, out_file)

		print('**********-----')
		print('finish--', jj / len(z))
	except:
		s = '%s, %d, %.3f, %.3f, %.3f' % (band[kk], jj, ra_g, dec_g, z_g)
		print(s, file = doc, )

doc.close()

print('Done!')
