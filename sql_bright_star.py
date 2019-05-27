import h5py
import numpy as np
import astropy.io.fits as fits

import mechanize
import pandas as pd 
from io import StringIO

import astropy.units as U
import astropy.constants as C
from astropy import cosmology as apcy

url = 'http://skyserver.sdss.org/dr12/en/tools/search/sql.aspx'
def bright_star():
	data_set = (
	"""
	SELECT ALL
		p.ra,p.dec,p.u,p.g,p.r,p.i,p.z,p.type
	FROM PhotoObj AS p
	WHERE 
		p.type = 6
		AND p.r BETWEEN 8.0 AND 8.2
	ORDER by p.r
	""" )

	br = mechanize.Browser()
	resp = br.open(url)
	resp.info()
	print(data_set)

	br.select_form(name = "sql")
	br['cmd'] = data_set
	br['format'] = ['csv']
	response = br.submit()
	s = str(response.get_data(), encoding = 'utf-8')
	doc = open( 
		'/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/sdss_bright_star.txt', 'w')
	print(s, file = doc)
	doc.close()
	return

def main():
	bright_star()

if __name__ == "__main__":
	main()