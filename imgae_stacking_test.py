"""
# image stacking test: use square cut region
"""
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits
import astropy.wcs as awc
from scipy.interpolate import interp2d as inter2
from scipy.interpolate import RectBivariateSpline as spline
import find
c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
c3 = U.L_sun.to(U.erg/U.s)
c4 = U.rad.to(U.arcsec)
Lsun = C.L_sun.value*10**7
# cosmology model
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311)
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)

pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
select_a = 500 # in unit pixel
Da_ref = Test_model.angular_diameter_distance(z_ref).value
R_ref = Da_ref*select_a*pixel/c4
scal_ref = select_a/R_ref # 1Mpc = scal_ref pixel
Metri = np.int0(2*np.ceil(scal_ref)) # the scale metrix size

data1 = fits.getdata(
        '/home/xkchen/mywork/ICL/data/test_data/frame-r-ra36.455-dec-5.896-redshift0.233.fits',header=True)
wcs1 = awc.WCS(data1[1])
transf1 = data1[1]['NMGY']
z1 = 0.233
ra1 = 36.455
dec1 = -5.896
Da1 = Test_model.angular_diameter_distance(z1).value
R1 = Da1*select_a*pixel/c4
cx1, cy1 = wcs1.all_world2pix(ra1*U.deg, dec1*U.deg, 1)
scal1 = select_a/R1

data2 = fits.getdata(
        '/home/xkchen/mywork/ICL/data/test_data/frame-r-ra234.901-dec49.666-redshift0.299.fits',header=True)
wcs2 = awc.WCS(data2[1])
transf2 = data2[1]['NMGY']
z2 = 0.299
ra2 = 234.901
dec2 = 49.666
Da2 = Test_model.angular_diameter_distance(z2).value
R2 = Da2*select_a*pixel/c4
cx2, cy2 = wcs2.all_world2pix(ra2*U.deg, dec2*U.deg, 1)
scal2 = select_a/R2

###
def flux_scale(data,z,zref):
    obs = data+1000
    z0 = z
    z_stak = zref
    ref_data = obs*(1+z0)**4/(1+z_stak)**4 -1000*(1+z0)**4/(1+z_stak)**4
    return ref_data

def extractor(data,x0,x1,y0,y1):
    a0 = np.int0(x0)
    a1 = np.int0(x1)
    b0 = np.int0(y0)
    b1 = np.int0(y1)
    M = data
    n0 = np.max([a0,0])
    n1 = np.min([a1,2047])
    m0 = np.max([b0,0])
    m1 = np.min([b1,1488])
    M_use = M[m0:m1+1,n0:n1+1]
    return M_use

###
z = np.array([z1,z2])
R = np.array([R1,R2])
ra = np.array([ra1,ra2])
dec = np.array([dec1,dec2])
cen = np.array([[cx1,cy1],[cx2,cy2]])
data = np.array([data1[0],data2[0]])
header = [data1[1],data2[1]]
scal = np.array([scal1,scal2])
image = {}
pos_record = np.zeros((len(z),2),dtype = np.float)
x0 = np.linspace(0,2047,2048)
y0 = np.linspace(0,1488,1489)
pix_id = np.array(np.meshgrid(x0,y0)) # set a data grid
R_stack = np.zeros((len(z),2),dtype = np.float)
NMGY = np.array([transf1, transf2])
for k in range(len(z)):
    a0 = np.floor(cen[k,0] - select_a)
    a1 = np.ceil(cen[k,0] + select_a)
    b0 = np.floor(cen[k,1] - select_a)
    b1 = np.ceil(cen[k,1] + select_a)
    idr = ((pix_id[0]<=a1)&(pix_id[0]>=a0))&((pix_id[1]<=b1)&(pix_id[1]>=b0))
    # region select
    mirro = data[k]*(idr*1)
    mirr1 = extractor(data[k],a0,a1,b0,b1)
    oa = (cen[k,0] - select_a) - np.floor(cen[k,0] - select_a)
    ob = (cen[k,1] - select_a) - np.floor(cen[k,1] - select_a)
    rx, ry = np.array([select_a+oa,select_a+ob])
    # flux re-calculate
    interf = flux_scale(mirr1, z[k], z_ref)
    scal_data = interf/(scal_ref/scal[k])
    image['%.0f'%k] = scal_data       
    R_stack[k,0] = (mirr1.shape[1]/2)/scal_ref
    R_stack[k,1] = (mirr1.shape[0]/2)/scal_ref
    pos_record[k,0] = rx
    pos_record[k,1] = ry
shape_a,shape_b = image['0'].shape  
sum_f = np.zeros((len(z),shape_a,shape_b),dtype = np.float)
for k in range(len(image)):
    sum_f[k,:] = image['%.0f'%k]
stac_data = sum_f.sum(axis=0)/len(image) 
#### the follow part record: fits file creat with given header
keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2','NMGY',
        'CRVAL1','CRVAL2','ORIGN_Z','Z_REF','SAMP_N']
value = ['T', -32, 2, shape_a, shape_b, rx, ry, NMGY[1], ra1, dec1, z1, z_ref, 706]
ff = dict(zip(keys,value))
fil = fits.Header(ff)
fits.writeto('/home/xkchen/Meeting/stacking_image_%s_ra%.3f_dec%.3f.fits'%(
        'r',ra1,dec1),scal_data,header = fil,overwrite=True) 
