"""
# image stacking test: use square cut region
"""
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
select_a = 50 # in unit pixel
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
    obs = data 
    z0 = z
    z_stak = zref
    ref_data = obs*(1+z0)**4/(1+z_stak)**4 
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

###
z = np.array([z1,z2])
R = np.array([R1,R2])
ra = np.array([ra1,ra2])
dec = np.array([dec1,dec2])
cen = np.array([[cx1,cy1],[cx2,cy2]])
cx = np.array([cx1,cx2])
cy = np.array([cy1,cy2])
data = np.array([data1[0],data2[0]])
header = [data1[1],data2[1]]
scal = np.array([scal1,scal2])

pos_record = np.zeros((len(z),2),dtype = np.float)
x0 = np.linspace(0,2047,2048)
y0 = np.linspace(0,1488,1489)
pix_id = np.array(np.meshgrid(x0,y0)) # set a data grid
R_stack = np.zeros((len(z),2),dtype = np.float)
NMGY = np.array([transf1, transf2])
'''
import matplotlib.gridspec as grid
f = plt.figure(figsize = (20,20))
f.suptitle('calculate process', fontsize = 20) # set the figure title
spc = grid.GridSpec(ncols = 3,nrows = 2,figure = f)
'''
k = 0
a0 = np.floor(cen[k,0] - select_a)
a1 = np.ceil(cen[k,0] + select_a)
b0 = np.floor(cen[k,1] - select_a)
b1 = np.ceil(cen[k,1] + select_a)
idr = ((pix_id[0]<=a1)&(pix_id[0]>=a0))&((pix_id[1]<=b1)&(pix_id[1]>=b0))
# region select
mirr1 = extractor(data[k],a0,a1,b0,b1)
# pixel change
size_vers = pixel_scale_compa(z[k], z_ref)
new_size = 1/size_vers
mt = np.float('%.3f'%size_vers)
inter_data = flux_scale(mirr1, z[k], z_ref) 
minus_data = mirr1/inter_data
if size_vers > 1:
    sum_data, cpos = sum_samp(mt, mt, inter_data, select_a, select_a)
else:
    sum_data, cpos = down_samp(mt, mt, inter_data, select_a, select_a)
resam = gen(inter_data, 1, mt) # compare the resample way variance
tt = resam[1:,1:]
sigma = np.sum((tt-sum_data)**2/sum_data)
print('chi = ', sigma)
'''
ax1 = f.add_subplot(spc[0,0])
im1 = ax1.imshow(mirr1,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
plt.colorbar(im1, label = 'flux [nMgy]', fraction = 0.048,pad = 0.003)
ax1.scatter(select_a, select_a, facecolors = '',marker='o',edgecolors='r')
ax1.set_title('flux of z0')

ax4 = f.add_subplot(spc[1,0])
im4 = ax4.imshow(mirr1,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
plt.colorbar(im4, label = 'flux [nMgy]', fraction = 0.048,pad = 0.003)
ax4.scatter(select_a, select_a, facecolors = '',marker='o',edgecolors='r')
ax4.set_title('original pixel image')

ax2 = f.add_subplot(spc[0,1])        
im2 = ax2.imshow(inter_data,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
plt.colorbar(im2, label = 'flux [nMgy]', fraction = 0.048, pad = 0.003)
ax2.scatter(select_a, select_a, facecolors = '',marker='o',edgecolors='r')
ax2.set_title('flux at z_ref')

ax3 = f.add_subplot(spc[0,2])
im3 = ax3.imshow(minus_data,cmap = plt.get_cmap('Greys', 10),vmin=5e-1, vmax = 1,origin = 'lower',norm = mpl.colors.LogNorm())
plt.colorbar(im3,label = 'ratio', fraction = 0.048, pad = 0.003)
ax3.scatter(select_a, select_a, facecolors = '',marker='o',edgecolors='r')
ax3.set_title('ratio of z0 to flux at z_ref')

ax5 = f.add_subplot(spc[1,1])
im5 = ax5.imshow(mirr1,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
plt.colorbar(im5,label = 'flux [nMgy]', fraction = 0.048, pad = 0.003)
ax5.axes.imshow(inter_data,cmap='rainbow',vmin=1e-5,origin='lower',extent=(0,100*new_size,0,100*new_size),
                norm = mpl.colors.LogNorm())
ax5.scatter(select_a*new_size,select_a*new_size,facecolors = '',marker='o',edgecolors='b')
ax5.set_title('pixel size change(the color part is z0 image)')

ax6 = f.add_subplot(spc[1,2])
im6 = ax6.imshow(sum_data,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
plt.colorbar(im6, label = 'flux [nMgy]', fraction = 0.048,pad = 0.003)
ax6.scatter(cpos[0],cpos[1],facecolors = '',marker='o',edgecolors='r')
ax6.set_title('resample')

plt.savefig('resample_test.pdf',dpi = 600)
plt.close(f)
'''