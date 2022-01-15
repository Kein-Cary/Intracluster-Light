import h5py
import random
import numpy as np
import pandas as pds
import astropy.units as U
import astropy.constants as C
import astroquery.sdss as asds
import astropy.io.fits as fits

from io import StringIO
import changds
import wget as wt
#####


dat = pds.read_csv('A250_LRG_match_cat.csv')
ra, dec, z = np.array(dat['ra']), np.array(dat['dec']), np.array(dat['z'])
Ns = len(ra)

# calculate the angular size 
size_cluster = 2. # assumptiom: cluster size is 2.Mpc/h
from angular_diameter_reshift import mark_by_self
from angular_diameter_reshift import mark_by_plank
A_size, A_d= mark_by_self(z, size_cluster)
view_d = A_size * U.rad

#### section 2: cite the data and save fits figure
from astroquery.sdss import SDSS
from astropy import coordinates as coords
#from astroML.plotting import setup_text_plots
from astropy.table import Table
R_A = 0.5 * view_d.to(U.arcsec) # angular radius in angular second unit

#band = ['g','r','i']
band = ['r']

## download imgs

doc = open('no_match_LRG.txt', 'w')

for k in range( Ns ):

	pos = coords.SkyCoord('%fd %fd'%(ra[k],dec[k]), frame='icrs')
	try:
	# set for solve the ReadtimeOut error
		xid = SDSS.query_region(pos, spectro = False, radius = R_A[k], timeout = None)
		# for galaxy, don't take spectra into account
		name = xid.colnames
		aa = np.array([xid['ra'],xid['dec']])
		da = np.sqrt((aa[0,:]-ra[k])**2+(aa[1,:]-dec[k])**2)

		# select the goal region
		dl = da.tolist()
		pl = dl.index(np.min(da))
		s_bgn = changds.chidas_int(xid[pl][3],6)
		s_end = changds.chidas_int(xid[pl][6],4)

		# change the list as the astropy table type
		for q in range( len(band) ):
			url_road = 'http://data.sdss.org/sas/dr12/boss/photoObj/frames/%.0f/%.0f/%.0f/frame-%s-%s-%.0f-%s.fits.bz2'%\
			(xid[pl][4], xid[pl][3], xid[pl][5], band[q], s_bgn, xid[pl][5], s_end)

			out_file = '/home/xkchen/mywork/ICL/data/tmp_img/A250_LRG_match/SDSS_LRG-img-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
			(band[q], ra[k], dec[k], z[k])

			wt.download(url_road, out_file)
		print('**********-----')
		print('finish--', k/len(z))
	except KeyError:
		# save the "exception" case
		s = '\n k = %d, ra%.3f, dec%.3f, z%.3f \n' % (k, ra[k], dec[k], z[k])
		print(s, file = doc)

		continue

doc.close()

raise

## query star catalogs
import mechanize
load = '/home/xkchen/mywork/ICL/data/'
url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'

dat = pds.read_csv('A250_LRG_ref-coord_cat.csv')
ra, dec, z = np.array(dat['bcg_ra']), np.array(dat['bcg_dec']), np.array(dat['bcg_z'])
ref_ra, ref_dec = np.array(dat.ref_ra), np.array(dat.ref_dec)

Ns = len(z)
r_select = 0.42 ## 1.5 * diagonal line length
out_file = load + 'tmp_img/LRG_stars/source_SQL_Z%.3f_ra%.3f_dec%.3f.txt'

for q in range( Ns ):

	z_g = z[q]
	ra_g = ra[q]
	dec_g = dec[q]

	cen_ra = ref_ra[q]
	cen_dec = ref_dec[q]

	c_ra0 = cen_ra - r_select
	c_dec0 = cen_dec - r_select
	c_ra1 = cen_ra + r_select
	c_dec1 = cen_dec + r_select

	if ra_g + r_select > 360:

		l_ra_0 = ra_g
		l_ra_1 = 0

		r_ra_0 = 360
		r_ra_1 = r_select - (360 - ra_g)

		data_set = """
		SELECT ALL
			p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
			p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
			p.flags, dbo.fPhotoFlagsN(p.flags)
		FROM PhotoObj AS p
		WHERE
			( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
			AND ( p.dec BETWEEN %.5f AND %.5f )
			AND ( p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0 )
		ORDER by p.r
		""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)

	elif ra_g - r_select < 0:

		l_ra_0 = 0
		l_ra_1 = 360 + (ra_g - r_select)

		r_ra_0 = ra_g
		r_ra_1 = 360

		data_set = """
		SELECT ALL
			p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
			p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
			p.flags, dbo.fPhotoFlagsN(p.flags)
		FROM PhotoObj AS p
		WHERE
			( (p.ra BETWEEN %.5f AND %.5f) OR (p.ra BETWEEN %.5f AND %.5f) )
			AND ( p.dec BETWEEN %.5f AND %.5f )
			AND ( p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0 )
		ORDER by p.r
		""" % (l_ra_0, r_ra_0, l_ra_1, r_ra_1, c_dec0, c_dec1)

	else:
		data_set = """
		SELECT ALL
			p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.type,  
			p.psffwhm_u, p.psffwhm_g, p.psffwhm_r, p.psffwhm_i, p.psffwhm_z,
			p.flags, dbo.fPhotoFlagsN(p.flags)
		FROM PhotoObj AS p
		WHERE
			p.ra BETWEEN %.5f AND %.5f
			AND p.dec BETWEEN %.5f AND %.5f
			AND (p.type = 6 OR (p.flags & dbo.fPhotoFlags('SATURATED')) > 0)
		ORDER by p.r
		""" % (c_ra0, c_ra1, c_dec0, c_dec1)

	br = mechanize.Browser()
	resp = br.open(url)
	resp.info()

	br.select_form(name = "sql")
	br['cmd'] = data_set
	br['format'] = ['csv']
	response = br.submit()
	s = str(response.get_data(), encoding = 'utf-8')
	doc = open( out_file % (z_g, ra_g, dec_g), 'w')
	print(s, file = doc)
	doc.close()

print('finished!')

