import numpy as np
import pandas as pds
import astropy.io.fits as fits
import threading
import wget
import subprocess as subpro

#*******************
# download DECaLS imgs based on SDSS imgs' center coordinate (ra, dec)
#*******************
dat = pds.read_csv('/home/xkchen/mywork/ICL/code/A250_LRG_match_cat.csv')
ra, dec, z = np.array(dat.ra), np.array(dat.dec), np.array(dat.z)

ref_ra, ref_dec = [], []
Nt = len(ra)

load = '/home/xkchen/mywork/ICL/data/'
for kk in range( Nt ):

	ra_g, dec_g, z_g = ra[kk], dec[kk], z[kk]

	dfile = load + 'tmp_img/A250_LRG_match/SDSS_LRG-img-r-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
	data = fits.open(dfile % (ra_g, dec_g, z_g),)
	Head = data[0].header
	cx0 = Head['CRPIX1']
	cy0 = Head['CRPIX2']
	RA0 = Head['CRVAL1']
	DEC0 = Head['CRVAL2']

	ref_ra.append( RA0 )
	ref_dec.append( DEC0 )

ref_ra = np.array(ref_ra)
ref_dec = np.array(ref_dec)


keys = ['bcg_ra', 'bcg_dec', 'bcg_z', 'ref_ra', 'ref_dec']
values = [ra, dec, z, ref_ra, ref_dec]
fill = dict(zip(keys, values))
data = pds.DataFrame(fill)
data.to_csv('A250_LRG_ref-coord_cat.csv')

raise

dat = pds.read_csv('A250_LRG_ref-coord_cat.csv')
ra, dec, z = np.array(dat['bcg_ra']), np.array(dat['bcg_dec']), np.array(dat['bcg_z'])
ref_ra, ref_dec = np.array(dat.ref_ra), np.array(dat.ref_dec)

def fits_url2load(ra, dec, z, cen_ra, cen_dec):

	url_name = []
	out_name = []

	for ll in range(len(z)):
		ra_g = ra[ll]
		dec_g = dec[ll]
		z_g = z[ll]

		id_ra = cen_ra[ll]
		id_dec = cen_dec[ll]

		http = 'http://legacysurvey.org/viewer/fits-cutout?ra=%f&dec=%f&layer=dr8&pixscale=0.396&bands=r&size=3000' % (id_ra, id_dec)
		out = 'A_250/desi_r_ra%.3f_dec%.3f_z%.3f.fits' % (ra_g, dec_g, z_g)

		url_name.append(http)
		out_name.append(out)

	return url_name, out_name

downs = fits_url2load(ra, dec, z, ref_ra, ref_dec)
url_lis, out_lis = downs[0], downs[1]

doc = open('err_stop_pix-SDSS.txt', 'w')

for kk in range(len(z)):
	ra_g = ra[kk]
	dec_g = dec[kk]
	z_g = z[kk]
	try:
		wget.download(url_lis[kk], out_lis[kk])
		print('\n finish--', kk / len(z))
	except:
		s = '%d, %.3f, %.3f, %.3f' % (kk, ra_g, dec_g, z_g)
		print(s, file = doc, )

doc.close()

