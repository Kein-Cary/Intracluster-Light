# this file record how to find source, produce mask metrix, and save the masked data
import matplotlib as mpl
import matplotlib.pyplot as plt
import handy.scatter as hsc
import astropy.io.fits as fits
import numpy as np
# result with WCS system
import astropy.wcs as awc
from astropy.coordinates import SkyCoord
import astropy.units as U
import astropy.constants as C
detect = np.loadtxt(
        '/home/xkchen/tool/SExtractor/result/mask_B/mask_B_test.cat')
# sources coordinates
Soce_Numb = np.int0(detect[:,0][-1]) # number of source

MAG_ISO = detect[:,3] #isophotal magnitude
Kron = detect[:,7] # Kron radius
sx0 = detect[:,8]-1
sy0 = detect[:,9]-1 # peak x,y coordinate (in pixel)   
sx1 = detect[:,10]-1
sy1 = detect[:,11]-1# object x,y coordinate (in pixel)
sx2 = detect[:,12]
sy2 = detect[:,13] # source center ra,dec,(WCS)
sx3 = detect[:,14]
sy3 = detect[:,15] # source center ra,dec,(J2000) __ used one
A = detect[:,19]
B = detect[:,20]
THETA = detect[:,21]

cxx = detect[:,16]
cxx = detect[:,17]
stellarity = detect[:,24] # probability of a source be a star
plt.hist(stellarity,bins=50);
plt.xlabel('stellarity')
plt.ylabel('#source')
plt.savefig('source_class.png',dpi=600)

data = fits.getdata(
        '/home/xkchen/mywork/ICL/data/test_data/frame-r-ra234.901-dec49.666-redshift0.299.fits',
        header = True)
wcs = awc.WCS(data[1])
pox,poy = wcs.all_world2pix(sx2*U.deg,sy2*U.deg,1)
plt.imshow(data[0],vmin=1e-5,cmap='Greys',origin='lower',
           norm=mpl.colors.LogNorm())
hsc.ellipses(pox,poy,w=A*6,h=B*6,rot=THETA,ec='r',fc='',ls='-',lw=0.5)
#hsc.ellipses(pox,poy,w=A*Kron,h=B*Kron,rot=THETA,ec='r',fc='',ls='-',lw=0.5)
# here use A*6, B*6 as the major and minor axis of ellipes
plt.xlim(0,2048)
plt.ylim(0,1489)
plt.savefig('source_find_test.png',dpi = 600)
### mask these source 
mirro = data[0]*1
Metrix = np.ones((data[0].shape[0],data[0].shape[1]),dtype = np.float)
# the mask metrix (only 0,1 element)
ox = np.linspace(0,2047,2048)
oy = np.linspace(0,1488,1489)
basic_coord = np.array(np.meshgrid(ox,oy))
major = A*Kron
minor = B*Kron
for k in range(Soce_Numb):
    xc = pox[k]
    yc = poy[k] # center coordinate in pixel
    lr = major[k]
    sr = minor[k]
    idr = (basic_coord[0,:]-xc)**2/lr**2+(basic_coord[1,:]-yc)**2/sr**2
    jx = idr<=1
    jx = (-1)*jx+1
    Metrix = Metrix*jx
mirro = mirro*Metrix
plt.imshow(Metrix,cmap = 'Greys',origin = 'lower')
plt.savefig('mask_metrix.png',dpi = 600)
plt.imshow(mirro,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
plt.savefig('maks_source_test.png', dpi = 600)
# save the masked data as fits
hdu = fits.PrimaryHDU()
hdu.data = mirro
hdu.header = data[1]
hdu.writeto('mask_test.fits',overwrite = True)

