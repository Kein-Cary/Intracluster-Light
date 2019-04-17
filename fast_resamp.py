# this file use for resample and data save
import matplotlib as mpl
mpl.use( 'agg' )
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
import h5py
import numpy as np
import astropy.io.fits as fits
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
import astropy.wcs as awc

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
# constant
pixel = 0.396 # the pixel size in unit arcsec
z_ref = 0.250 
Jy = 10**(-23) # (erg/s)/cm^2
f0 = 3631*10**(-23) # zero point in unit (erg/s)/cm^-2

def get_index( i, t ):
    ii = i * t
    t = int( ii )
    di = ii - t
    ii = t
    return ( ii, di )

def get_index2( i, j, t ):
    ii, di = get_index( i, t )
    jj, dj = get_index( j, t )
    ii1, di1 = get_index( i+1, t )
    jj1, dj1 = get_index( j+1, t )
    return ( [ii, jj, ii1, jj1], [di, dj, di1, dj1] )

def check_index( i, M, sig ):
    if i >=M:
        print( "[%i] bug???"%sig )
        i = M-1
    return i

def check_index2( ii, M, N, sig ):
    t = [ M, N, M, N ]
    for i in range(4):
        ii[0] = check_index( ii[0], t[i], sig*10+i )
    return ii

def get_r( d, r, flag ):
    if flag == 1:
        t1 = ( 1-d[1] ) * r
        t2 = d[3] * r
        return ( t1, t2 )

    if flag == 2:
        t1 = ( 1-d[0] ) * r
        t2 = d[2] * r
        return ( t1, t2 )

    if flag == 3:
        t1 = (1-d[0]) * (1-d[1])
        t2 =  d[2] * d[3]
        t3 = (1-d[0]) * d[3]
        t4 = d[2] * (1-d[1])
        return ( t1, t2, t3, t4 )

def gen1( d1, res1, res2 ):

    M1, N1 = d1.shape
    M2 = int(M1 * res1 / res2) - 2
    N2 = int(N1 * res1 / res2) - 2
    d2 = np.zeros( ( M2, N2 ) )

    print( "res: %.2f -> %.2f"%(res1, res2) )
    print( "(%i, %i) -> (%i,%i)"%(M1, N1, M2, N2) )

    t21 = res2 / res1
    r21 = res2 * res2 / ( res1 * res1 )

    sig = M2 * N2 // 10
    for i in range( M2 ):
        for j in range( N2 ):

            if (i*N2+j) % sig == 0:
                #print( i*M2+j )
                print( "%3.0f%%"%( (i*N2+j) / (M2*N2) * 100 ) )

            iii, dii = get_index2( i+1, j+1, t21 )
            iii = check_index2( iii, M1, N1, 1 )

            ii = iii[0]
            jj = iii[1]
            ii1 = iii[2]
            jj1 = iii[3]

            if (ii == ii1 and jj == jj1):
                d2[i,j] +=  d1[ii,jj] * r21
                continue


            if ii == ii1:
                t1, t2 = get_r( dii, t21, 1 )
                d2[i,j] += d1[ii,jj] * t1 + d1[ii,jj1] * t2
                #print( t1+t2 )
                continue

            if jj == jj1:
                t1, t2 = get_r( dii, t21, 2 )
                d2[i,j] += d1[ii,jj] * t1 + d1[ii1,jj] * t2
                #print( t1+t2 )
                continue


            t1, t2, t3, t4 = get_r( dii, t21, 3 )
            #print( t1+t2+t3+t4 )
            d2[i,j] += d1[ii,jj]   * t1
            d2[i,j] += d1[ii1,jj1] * t2
            d2[i,j] += d1[ii,jj1]  * t3
            d2[i,j] += d1[ii1,jj]  * t4

    return d2

def gen2( d1, res1, res2 ):

    M1, N1 = d1.shape
    M2 = int(M1 * res1 / res2) + 2
    N2 = int(N1 * res1 / res2) + 2
    d2 = np.zeros( ( M2, N2 ) )

    print( "res: %.2f -> %.2f"%(res1, res2) )
    print( "(%i, %i) -> (%i,%i)"%(M1, N1, M2, N2) )

    t12 = res1 / res2
    r21 = res2 * res2 / ( res1 * res1 )

    sig = M1 * N1 // 10
    for i in range( M1 ):
        for j in range( N1 ):

            if (i*N1+j) % sig == 0:
                #print( i*M2+j )
                print( "%3.0f%%"%( (i*N1+j) / (M1*N1) * 100 ) )

            iii, dii = get_index2( i, j, t12 )
            iii = check_index2( iii, M2, N2, 2 )

            ii  = iii[0] + 1
            jj  = iii[1] + 1
            ii1 = iii[2] + 1
            jj1 = iii[3] + 1

            if ii == ii1 and jj == jj1:
                d2[ii,jj] +=  d1[i,j]
                continue

            if ii == ii1:
                t1, t2 = get_r( dii, t12, 1 )
                d2[ii,jj]  += d1[i,j] * t1 * r21
                d2[ii,jj1] += d1[i,j] * t2 * r21
                continue


            if jj == jj1:
                t1, t2 = get_r( dii, t12, 2 )
                d2[ii,jj] += d1[i,j]  * t1 * r21
                d2[ii1,jj] += d1[i,j] * t2 * r21
                continue

            t1, t2, t3, t4 = get_r( dii, t12*r21, 3 )
            #print( t1+t2+t3+t4 )
            d2[ii,jj]   += d1[i,j] * t1 * r21
            d2[ii1,jj1] += d1[i,j] * t2 * r21
            d2[ii,jj1]  += d1[i,j] * t3 * r21
            d2[ii1,jj]  += d1[i,j] * t4 * r21
        continue
        a = 4
        if i >= a:
            a, dii = get_index2( a+10, j, t12 )
            for j in range( M2 ):
                if j % 2 == 0:
                    d2[a[0]:,j] = 2
                else:
                    d2[a[0]:,j] = -2
            break

    return d2

def flux_recal(data, z0, zref):
    obs = data
    z0 = z0
    z1 = zref
    Da0 = Test_model.angular_diameter_distance(z0).value
    Da1 = Test_model.angular_diameter_distance(z1).value
    flux = obs*((1 +z0)**2*Da0)**2/((1 +z1)**2*Da1)**2
    return flux

def gen( d, res1, res2, cx, cy ):
    if res1 > res2:
        xn = cx /res2
        yn = cy /res2
        return xn,yn,gen1( d, res1, res2)
    if res1 < res2:
        xn = cx /res2
        yn = cy /res2
        return xn,yn,gen2( d, res1, res2)
    if res1 == res2:
        xn = cx*1
        yn = cy*1
        return xn, yn,gen2( d, res1, res2)
    #print( "res1 == res2 !!!!" )
    #exit()

def resamp():
    band = ['u', 'g', 'r', 'i', 'z']
    D_ref = Test_model.angular_diameter_distance(z_ref).value
    L_ref = D_ref*pixel/c4
    '''
    # catalog     
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
    Ra = ra_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
    Dec = dec_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
    rich = rich_eff[(z_eff >= 0.2)&(z_eff <= 0.3)]
    catalog = np.array([z, Ra, Dec, rich])
    with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5', 'w') as f:
        f['a'] = catalog
    with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
        for u in range(len(catalog)):
            f['a'][u,:] = catalog[u,:]
    '''
    with h5py.File('/mnt/ddnfs/data_users/cxkttwl/ICL/data/sample_catalog.h5') as f:
        catalogue = np.array(f['a'])
    z = catalogue[0]
    Ra = catalogue[1]
    Dec = catalogue[2]
    lamb = catalogue[3]
    for k in range(len(z)):
        for q in range(len(band)):
            f = fits.getdata('/mnt/ddnfs/data_users/cxkttwl/ICL/wget_data/frame-%s-ra%.3f-dec%.3f-redshift%.3f.fits.bz2'
                             %(band[q], Ra[k], Dec[k], z[k]), header=True)
            wcs = awc.WCS(f[1])
            d1 = f[0]
            sum_data = flux_recal(f[0], zf, z_ref)
            zf = z[k]
            D_z = Test_model.angular_diameter_distance(zf).value
            L_z = D_z*pixel/c4        
            ra = Ra[k]
            dec = Dec[k]
            #Alpha = (1/h)*c4/D_z
            #R[k] = Alpha/pixel
            cx, cy = wcs.all_world2pix(ra*U.deg, dec*U.deg, 1)
            b = L_ref/L_z
            b = np.float('%.4f'%b)
            print('b = ', b)
            xn, yn, resam = gen(sum_data, 1, b, cx, cy)
            
            plt.figure()
            ax1 = plt.subplot(121)
            ax1.imshow(f[0], cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mplc.LogNorm())
            ax1.scatter(cx,cy,facecolors = '',marker='o',edgecolors='r')
            ax2 = plt.subplot(122)
            ax2.imshow(resam, cmap = 'Greys', vmin = 1e-5, origin = 'lower', norm = mplc.LogNorm())
            ax2.scatter(xn,yn,facecolors = '',marker='o',edgecolors='r')
            plt.savefig('/mnt/ddnfs/data_users/cxkttwl/ICL/fig_cut/resample/position_%s_ra%.3f_dec%.3f_z%.3f.pdf'
                        %(band[q], ra, dec, zf), dpi = 600)
            plt.close()
            
            x1 = resam.shape[1]
            y1 = resam.shape[0]
            intx = np.ceil(f[1]['CRPIX1'] // b)
            inty = np.ceil(f[1]['CRPIX2'] // b)
            keys = ['SIMPLE','BITPIX','NAXIS','NAXIS1','NAXIS2','CRPIX1','CRPIX2',
            'CENTER_X','CENTER_Y','CRVAL1','CRVAL2','CENTER_RA','CENTER_DEC','ORIGN_Z','Z_REF',]
            value = ['T', 32, 2, x1, y1, intx, inty, xn, yn, f[1]['CRVAL1'], f[1]['CRVAL2'],
            ra, dec, zf, z_ref ]
            head = dict(zip(keys, value))
            file_s = fits.Header(head)
            fits.writeto(
                    '/mnt/ddnfs/data_users/cxkttwl/ICL/data/cut_sample/resamp_record/resamp_image_%s_ra%.3f_dec%.3f_z%.3f.fits'
                    %(band[q], ra, dec, zf), resam, header = file_s, overwrite = True) 
def main():
    resamp()
    
if __name__ == '__main__':
    main()