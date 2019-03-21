# use for light measure
import numpy as np
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
# constant
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

def light_measure(data, Nbin, small, Rp, cx, cy, psize):
	"""
	data: data used to measure
	Nbin: number of bins will devide
	Rp: radius in unit pixel number
	cx, cy: cluster central position in image frame (in inuit pixel)
	psize: pixel size
	"""
	cx = cx
	cy = cy
	Nbins = Nbin
	f_data = data
	cen_close = small
	pixel = psize
	R_pixel = Rp

	Nx = data.shape[1]
	Ny = data.shape[0]
	x0 = np.linspace(0, Nx-1, Nx)
	y0 = np.linspace(0, Ny-1, Ny)
	pix_id = np.array(np.meshgrid(x0,y0)) #data grid for original data 
	
	r = np.logspace(-2, np.log10(R_pixel), Nbins) # in unit: pixel number
	ia = r<= cen_close
	ib = np.array(np.where(ia == True))
	ic = ib.shape[1]
	R = (r/R_pixel)*10**3 # in unit kpc
	R = R[np.max(ib):]
	r0 = r[np.max(ib):]

	dr = np.sqrt((pix_id[0]-cx)**2+(pix_id[1]-cy)**2)
	light = np.zeros(len(r)-ic+1, dtype = np.float)
	for k in range(1,len(r)):
	        if r[k] <= cen_close:
	            ig = r <= cen_close
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

	return light, R, r0

def main():
	light_measure()

if __name__ == '__main__':
	main()