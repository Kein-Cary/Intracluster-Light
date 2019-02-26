# this file use to comfir the stacking process
import matplotlib as mpl
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

data1 = fits.getdata(
        '/home/xkchen/mywork/ICL/data/test_data/frame-r-ra36.455-dec-5.896-redshift0.233.fits',header=True)
wcs1 = awc.WCS(data1[1])
transf1 = data1[1]['NMGY']
z1 = 0.233
ra1 = 36.455
dec1 = -5.896
Da1 = Test_model.angular_diameter_distance(z1).value
Alpha1 = (1/h)*c4/Da1 # the observational size
R1 = Alpha1/pixel # the radius in pixel number
cx1, cy1 = wcs1.all_world2pix(ra1*U.deg, dec1*U.deg, 1)

data2 = fits.getdata(
        '/home/xkchen/mywork/ICL/data/test_data/frame-r-ra234.901-dec49.666-redshift0.299.fits',header=True)
wcs2 = awc.WCS(data2[1])
transf2 = data2[1]['NMGY']
z2 = 0.299
ra2 = 234.901
dec2 = 49.666
Da2 = Test_model.angular_diameter_distance(z2).value
Alpha2 = (1/h)*c4/Da2 # the observational size
R2 = Alpha2/pixel # the radius in pixel number
cx2, cy2 = wcs2.all_world2pix(ra2*U.deg, dec2*U.deg, 1)

def flux_scale(data,z,zref):
    obs = data+1000
    z0 = z
    z_stak = zref
    ref_data = obs*(1+z0)**4/(1+z_stak)**4 -1000*(1+z0)**4/(1+z_stak)**4
    return ref_data

def pixel_scale_compa(z, zref):
    z0 = z
    z_stak = zref
    Da_0 = Test_model.angular_diameter_distance(z0).value
    L_0 = Da_0*pixel/c4
    Da_ref = Test_model.angular_diameter_distance(z_stak).value
    L_ref = Da_ref*pixel/c4
    pix_ratio = L_ref/L_0
    return pix_ratio

def R_angl(z):
    z = z
    Da = Test_model.angular_diameter_distance(z).value
    R = ((1/h)*c4/Da)/pixel
    return R

R = np.array([R1,R2])
z = np.array([z1,z2])
cx = np.array([cx1,cx2])
cy = np.array([cy1,cy2])
transf = np.array([transf1,transf2])
data = np.array([data1[0],data2[0]])
x0 = np.linspace(0,2047,2048)
y0 = np.linspace(0,1488,1489)
pix_id = np.array(np.meshgrid(x0,y0)) # set a data grid
refdata = {}
sum_f = {}
R_ref = R_angl(0.25)
scal_f = np.zeros((len(R),1),dtype = np.float)
pos_record = np.zeros((len(z),2),dtype = np.float) # record the new cluster center pos in pixel
# un-squre cut-out region
for k in range(2):
    dr = np.sqrt((pix_id[0]-cx[k])**2+(pix_id[1]-cy[k])**2)
    idr = dr <= R[k] 
    mirro = data[k]*(idr*1) # get the select region, for comparation
    '''
    im = plt.imshow(mirro,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
    plt.colorbar(im, label = 'flux', fraction = 0.035,pad = 0.003)
    plt.savefig('step_1_select.png',dpi = 600)
    '''
    inter_data = flux_scale(mirro, z[k], z_ref)
    '''
    im = plt.imshow(inter_data,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
    plt.colorbar(im, label = 'flux', fraction = 0.035,pad = 0.003)
    plt.savefig('step_2_flux_scale.png',dpi = 600)
    '''    
    size_vers = pixel_scale_compa(z[k], z_ref)
    new_size = 1/size_vers
    
    