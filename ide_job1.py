# this part use for read and change the SEXtractor file parameter "default.parameter"
import matplotlib as mpl
import matplotlib.pyplot as plt
import handy.scatter as hsc
import astropy.io.fits as fits
import numpy as np
detect = np.loadtxt('/home/xkchen/tool/SExtractor/result/test_circle.cat')
# sources coordinates
Ns = detect[:,0][-1] # number of source

sx = detect[:,1]-1
sy = detect[:,2]-1

A = detect[:,17]
B = detect[:,18]
angula = detect[:,19]
data = fits.getdata('/home/xkchen/mywork/ICL/data/test_data/test_tot_1.fits',header = True)

plt.imshow(data[0],vmin=1e-5,cmap='Greys',origin='lower',norm=mpl.colors.LogNorm())
hsc.ellipses(sx,sy,w=A,h=B,rot=angula,ec='r',fc='',ls='-',lw=0.5)
plt.savefig('source.png',dpi=600)
# result with WCS system
import astropy.wcs as awc
from astropy.coordinates import SkyCoord
import astropy.units as U
import astropy.constants as C
sra = detect[:,13]
sdec = detect[:,14]
