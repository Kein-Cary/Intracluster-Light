# this file use to creat the resample data and svae as fits
"""
in this file, assume all the pixel scale is '1' compared to the reference redshift
"""
import matplotlib as mpl
mpl.use('Agg')
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
band = ['u','g','r','i','z']

goal_data = fits.getdata(
        '/mnt/ddnfs/data_users/cxkttwl/ICL/data/redmapper/redmapper_dr8_public_v6.3_catalog.fits')
RA = np.array(goal_data.RA)
DEC = np.array(goal_data.DEC)
redshift = np.array(goal_data.Z_SPEC)
richness = np.array(goal_data.LAMBDA)
# except the part with no spectra redshift
z_eff = redshift[redshift != -1]
ra_eff = RA[redshift != -1]
dec_eff = DEC[redshift != -1]
rich_eff = richness[redshift != -1]
# select the nearly universe
z = z_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
ra = ra_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
dec = dec_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
rich = rich_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
# inter def
def flux_scale(data,z,zref):
    obs = data 
    z0 = z
    z_stak = zref
    ref_data = obs*(1+z0)**4/(1+z_stak)**4 
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
# resample process
x0 = np.linspace(0,2047,2048)
y0 = np.linspace(0,1488,1489)
pix_id = np.array(np.meshgrid(x0,y0)) #data grid for original data 
for k in range(1):
    for q in range(len(band)):
        cut_data = fits.getdata(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'%\
                (band[q],ra[k],dec[k],z[k]),header = True)
        wcs1 = awc.WCS(cut_data[1])
        transf = cut_data[1]['NMGY']
        cx, cy = wcs1.all_world2pix(ra[k]*U.deg, dec[k]*U.deg, 1)
        cx = np.float(cx)
        cy = np.float(cy)
        # get the observational size
        Da = Test_model.angular_diameter_distance(z[k]).value
        Alpha = (1/h)*c4/Da 
        R = Alpha/pixel
        # get the select region, for comparation
        dr = np.sqrt((pix_id[0]-cx)**2+(pix_id[1]-cy)**2)
        idr = dr <= R 
        mirro = cut_data[0]*(idr*1)
        
        plt.figure()
        im0 = plt.imshow(mirro,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
        plt.colorbar(im0, label = 'flux', fraction = 0.035,pad = 0.003)
        plt.scatter(cx,cy,facecolors = '',marker='o',edgecolors='r')
        plt.title('select_ra%.3f_dec%.3f_z%.3f_rich%.3f.png'%(ra[k],dec[k],z[k],rich[k]))
        plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/region_cut/\
                    select_ra%.3f_dec%.3f_z%.3f_rich%.3f.png'%(ra[k],dec[k],z[k],rich[k]),dpi = 600)  
        plt.close()
        
        x = mirro.shape[1]
        y = mirro.shape[0]
        keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
                'CENTER_X','CENTER_Y','NMGY','CRVAL1','CRVAL2',
                'CENTER_RA','CENTER_DEC','ORIGN_Z','Z_REF',]
        value = ['T', 32, 2, x, y,cut_data[1]['CRPIX1'],cut_data[1]['CRPIX2'],
                 cx, cy, transf, cut_data[1]['CRVAL1'],cut_data[1]['CRVAL2'],
                 ra[k], dec[k], z[k], z_ref]
        ff = dict(zip(keys,value))
        fil = fits.Header(ff)
        fits.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/cut_sample/cut_record/cut_image_%s_ra%.3f_dec%.3f_z%.3f_rich%.3f.fits'%(
                band[q],ra[k],dec[k],z[k],rich[k]), mirro, header = fil, overwrite=True) 
        # flux reset
        inter_data = flux_scale(mirro, z[k], z_ref)
        size_vers = pixel_scale_compa(z[k], z_ref) 
        if size_vers > 1:
            resam_data, cpos = sum_samp(size_vers, size_vers, inter_data, cx, cy)
        else:
            resam_data, cpos = down_samp(size_vers, size_vers, inter_data, cx, cy)
        cx1 = cpos[0]
        cy1 = cpos[1]
        
        plt.figure()
        im1 = plt.imshow(resam_data,cmap = 'Greys',vmin = 1e-5,origin = 'lower',norm = mpl.colors.LogNorm())
        plt.colorbar(im1, label = 'flux', fraction = 0.035,pad = 0.003)
        plt.scatter(cx1,cy1,facecolors = '',marker='o',edgecolors='r')
        plt.title('resampl_ra%.3f_dec%.3f_z%.3f_rich%.3f.png'%(ra[k],dec[k],z[k],rich[k]))
        plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/resample/\
                    resampl_ra%.3f_dec%.3f_z%.3f_rich%.3f.png'%(ra[k],dec[k],z[k],rich[k]),dpi = 600)  
        plt.close()
        
        x1 = resam_data.shape[1]
        y1 = resam_data.shape[0]
        keys1 = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
                'CENTER_X','CENTER_Y','NMGY','CRVAL1','CRVAL2',
                'CENTER_RA','CENTER_DEC','ORIGN_Z','Z_REF',]
        intx = np.ceil(cut_data[1]['CRPIX1']/size_vers)
        inty = np.ceil(cut_data[1]['CRPIX2']/size_vers)
        value1 = ['T', 32, 2, x1, y1, intx, inty,
                 cx1, cy1, transf, cut_data[1]['CRVAL1'],cut_data[1]['CRVAL2'],
                 ra[k], dec[k], z[k], z_ref]
        ff1 = dict(zip(keys1,value1))
        fil1 = fits.Header(ff1)
        fits.writeto(
                '/mnt/ddnfs/data_users/cxkttwl/ICL/data/cut_sample/resamp_record/cut_image_%s_ra%.3f_dec%.3f_z%.3f_rich%.3f.fits'%(
                band[q],ra[k],dec[k],z[k],rich[k]), mirro, header = fil, overwrite=True) 
  