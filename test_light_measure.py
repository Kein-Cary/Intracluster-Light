# this file use to comfir the stacking process
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.io.fits as fits
import astropy.wcs as awc
# resample part
from ICL_up_resampling import sum_samp
from ICL_down_resampling import down_samp
from resamp import gen

c0 = U.kpc.to(U.cm)
c1 = U.Mpc.to(U.pc)
c2 = U.Mpc.to(U.cm)
c3 = U.L_sun.to(U.erg/U.s)
c4 = U.rad.to(U.arcsec)
c5 = U.pc.to(U.cm)
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
Da_ref = Test_model.angular_diameter_distance(z_ref).value
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2
'''
data2 = fits.getdata(
        '/home/xkchen/mywork/ICL/data/test_data/frame-r-ra234.901-dec49.666-redshift0.299.fits',header=True)
wcs2 = awc.WCS(data2[1])
z2 = 0.299
ra2 = 234.901
dec2 = 49.666
Da2 = Test_model.angular_diameter_distance(z2).value
Alpha2 = (1/h)*c4/Da2 # the observational size
R2 = Alpha2/pixel # the radius in pixel number
cx2, cy2 = wcs2.all_world2pix(ra2*U.deg, dec2*U.deg, 1)
'''
data2 = fits.getdata(
        '/home/xkchen/mywork/ICL/data/test_data/frame-u-ra203.834-dec41.001-redshift0.228.fits',header=True)
wcs2 = awc.WCS(data2[1])
z2 = 0.228
ra2 = 203.834
dec2 = 41.001
Da2 = Test_model.angular_diameter_distance(z2).value
Alpha2 = (1/h)*c4/Da2 # the observational size
R2 = Alpha2/pixel # the radius in pixel number
cx2, cy2 = wcs2.all_world2pix(ra2*U.deg, dec2*U.deg, 1)

def flux_scale(data, s0, z0, zref):
    obs = data
    s0 = s0
    z0 = z0
    z_stak = zref
    ref_data = (obs/s0)*(1+z0)**4/(1+z_stak)**4 
    return ref_data

def flux_recal(data, z0, zref):
    obs = data
    z0 = z0
    z1 = zref
    Da0 = Test_model.angular_diameter_distance(z0).value
    Da1 = Test_model.angular_diameter_distance(z1).value
    flux = obs*((1+z0)**2*Da0)**2/((1+z1)**2*Da1)**2
    return flux

def angu_area(s0, z0, zref):
    s0 = s0
    z0 = z0
    z1 = zref
    Da0 = Test_model.angular_diameter_distance(z0).value
    Da1 = Test_model.angular_diameter_distance(z1).value
    angu_S = s0*Da0**2/Da1**2
    return angu_S

from light_measure import light_measure
light_1, R_1, r0_1 = light_measure(data2[0], 25, 2, R2, cx2, cy2, pixel)

x0 = np.linspace(0,2047,2048)
y0 = np.linspace(0,1488,1489)
pix_id = np.array(np.meshgrid(x0,y0)) #data grid for original data 
f_data = data2[0]
Nbins = 25

r = np.logspace(-2, np.log10(R2), Nbins) # in unit: pixel number
ia = r<= 2
ib = np.array(np.where(ia == True))
ic = ib.shape[1]
R = (r/R2)*10**3 # in unit kpc
R = R[np.max(ib):]
r0 = r[np.max(ib):]
Ar1 = ((R/10**3)/Da_ref)*c4 # in unit arcsec

dr = np.sqrt((pix_id[0]-cx2)**2+(pix_id[1]-cy2)**2)
light = np.zeros(len(r)-ic+1, dtype = np.float)
zrefl = np.zeros(len(r)-ic+1, dtype = np.float)
dim_l = np.zeros(len(r)-ic+1, dtype = np.float)
thero_l = np.zeros(len(r)-ic+1, dtype = np.float)
for k in range(1,len(r)):
        if r[k] <= 2:
            ig = r <= 2
            ih = np.array(np.where(ig == True))
            im = np.max(ih)
            ir = dr < r[im]
            io = np.where(ir == True)
            iy = io[0]
            ix = io[1]
            num = len(ix)
            tot_flux = np.sum(f_data[iy,ix])/num
            tot_area = pixel**2
            light[0] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
            k = im+1 
        else:
            ir = (dr >= r[k-1]) & (dr < r[k])
            io = np.where(ir == True)
            iy = io[0]
            ix = io[1]
            num = len(ix)
            tot_flux = np.sum(f_data[iy,ix])/num
            tot_area = pixel**2
            light[k-im] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area) # mag/arcsec^2

thero_l = light +10*np.log10((1+z_ref)/(1+z2))
f_dim = flux_scale(f_data, pixel**2, z2, z_ref)
s_new = angu_area(pixel**2, z2, z_ref)
d_new = np.sqrt(s_new)
f_dim = f_dim*s_new
f_ref = flux_recal(f_data, z2, z_ref)

for k in range(1,len(r)):
    if r[k] <= 2:
        ig = r <= 2
        ih = np.array(np.where(ig == True))
        im = np.max(ih)
        ir = dr <= r[im]
        io = np.where(ir == True)
        iy = io[0]
        ix = io[1]
        num = len(ix)
        
        tot_flux1 = np.sum(f_dim[iy,ix])/num
        tot_area1 = s_new
        dim_l[0] = 22.5-2.5*np.log10(tot_flux1)+2.5*np.log10(tot_area1)
        
        tot_flux2 = np.sum(f_ref[iy,ix])/num
        tot_area2 = s_new
        zrefl[0] = 22.5-2.5*np.log10(tot_flux2)+2.5*np.log10(tot_area2)        
        k = im+1
    else:
        ir = (dr >= r[k-1]) & (dr < r[k])
        io = np.where(ir == True)
        iy = io[0]
        ix = io[1]
        num = len(ix)
        
        tot_flux1 = np.sum(f_dim[iy,ix])/num
        tot_area1 = s_new
        dim_l[k-im] = 22.5-2.5*np.log10(tot_flux1)+2.5*np.log10(tot_area1)
        
        tot_flux2 = np.sum(f_ref[iy,ix])/num
        tot_area2 = s_new
        zrefl[k-im] = 22.5-2.5*np.log10(tot_flux2)+2.5*np.log10(tot_area2)
# resample compare
data_test = fits.getdata('/home/xkchen/Meeting/New_resamp/resamp_image_ra203.834_dec41.001_z0.228.fits',header = True)
test_f = data_test[0]

Rref = ((1/h)*c4/Da_ref)/pixel
cx_t = data_test[1]['CENTER_X']  
cy_t = data_test[1]['CENTER_Y']
x0_t = data_test[0].shape[1]
y0_t = data_test[0].shape[0]
x_t = np.linspace(0,x0_t-1,x0_t)
y_t = np.linspace(0,y0_t-1,y0_t)
pi_t = np.array(np.meshgrid(x_t,y_t))
r_t = np.logspace(-2, np.log10(Rref), Nbins)
ia_t = r_t <= 2
ib_t = np.array(np.where(ia_t == True))
ic_t = ib_t.shape[1]
R_t = (r_t/Rref)*10**3 # in unit kpc
R_t = R_t[np.max(ib_t):]
r0_t = r_t[np.max(ib_t):]
dr_t = np.sqrt((pi_t[0]-cx_t)**2+(pi_t[1]-cy_t)**2)
test_l = np.zeros(len(r_t)-ic_t+1, dtype = np.float)
Ar_t = ((R_t/10**3)/Da_ref)*c4

for k in range(1,len(r_t)):
        if r_t[k] <= 2:
            ig_t = r_t <= 2
            ih_t = np.array(np.where(ig_t == True))
            im_t = np.max(ih_t)
            ir_t = dr_t < r_t[im_t]
            io_t = np.where(ir_t == True)
            iy_t = io_t[0]
            ix_t = io_t[1]
            num_t = len(ix_t)
            tot_flux = np.sum(test_f[iy_t,ix_t])/num_t
            tot_area = pixel**2
            test_l[0] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area)
            k = im_t+1 
        else:
            ir_t = (dr_t >= r_t[k-1]) & (dr_t < r_t[k])
            io_t = np.where(ir_t == True)
            iy_t = io_t[0]
            ix_t = io_t[1]
            num_t = len(ix_t)
            tot_flux = np.sum(test_f[iy_t,ix_t])/num_t
            tot_area = pixel**2
            test_l[k-im_t] = 22.5-2.5*np.log10(tot_flux)+2.5*np.log10(tot_area) # mag/arcsec^2 

plt.plot(r0*pixel, light, 'k-', label = 'SB_ini')
plt.plot(r0_1*pixel, light_1, 'r--', label = 'test')
'''
plt.plot(Ar1, zrefl, 'r-*', label = 'SB_ref')                
plt.plot(Ar1, dim_l, 'b-', label = 'SB_dim')
plt.plot(Ar1, thero_l, 'g--', label = 'SB_intr')
plt.plot(Ar_t, test_l, 'ro-', label = 'SB_t01')
plt.xlabel('R [arcsec]')
'''
'''
plt.plot(R, light, 'k-', label = 'SB_ini')                
plt.plot(R, zrefl, 'r-*', label = 'SB_ref', alpha = 0.5)                
plt.plot(R, dim_l, 'b-', label = 'SB_dim', alpha = 0.5)
plt.plot(R, thero_l, 'g--', label = 'SB_intr')
plt.plot(R_t, test_l, 'ro-', label = 'SB_t01')
plt.text(2, 27, '$\chi^2 = %.3f$'%sigma)
plt.xlabel('R [kpc]')
'''
plt.legend(loc = 1)
plt.ylabel(r'$SB [mag/arcsec^2]$')
plt.gca().invert_yaxis()
plt.xscale('log')
#plt.savefig('test_the_one_in.png',dpi=600)
#plt.savefig('test_the_one_in1.png',dpi=600)
#plt.savefig('test_new_resamp_up.png',dpi=600) # label 'up' for pixel become bigger
#plt.savefig('test_new_resamp_up1.png',dpi=600)
plt.show()
plt.close()