#this file used to find the angular diameter in different redshift
import matplotlib as mpl
import numpy as np
import astropy.io.fits as asft
import matplotlib.pyplot as plt
import astropy.constants as C
import astropy.units as U
from astropy import cosmology as apcy
from scipy import interpolate as interp
from scipy.integrate import quad as scq
import scipy.integrate as sit
# get the velocity of light and in unit Km/s
vc = C.c.to(U.km/U.s).value
## test: control the calculation and choose value :1,2,3
test = 2
#Test_model = apcy.Planck15 ##use the cosmology model Plank 2015 to test
Test_model = apcy.Planck15.clone(H0 = 67.74, Om0 = 0.311) ##use the cosmology model Plank 2018 to analysis
H0 = Test_model.H0.value
h = H0/100
Omega_m = Test_model.Om0
Omega_lambda = 1.-Omega_m
Omega_k = 1.- (Omega_lambda + Omega_m)
DH = vc/H0
def mark_by_self(z_in,goal_size):
    z_array = z_in
    size_cluster = goal_size
    Alpha = np.zeros(len(z_array), dtype = np.float)
    Da = np.zeros(len(z_array), dtype = np.float)
    Dc = np.zeros(len(z_array), dtype = np.float)
    fEz1 = lambda sysz: 1./np.sqrt(Omega_m*(1+sysz)**3+Omega_lambda+Omega_k*(1+sysz)**2)
    for k in range(len(z_array)):
        if z_array[k] == 0:
            Da[k] = 0.
        else:
            Dc[k] = DH*scq(fEz1,0.,z_array[k])[0]
    if Omega_k == 0.:
        Dm = Dc*1.
    elif Omega_k > 0.:
        Dm = DH*np.sinh(np.asrt(Omega_k)*Dc/DH)*1/np.sqrt(Omega_k)
    else:
        Dm = DH*np.sin(np.asrt(np.abs(Omega_k))*Dc/DH)*1/np.sqrt(np.abs(Omega_k))
    DA = Dm/(1+z_array)
    alpha = (size_cluster/h)/DA
    Alpha = alpha
    Da = DA 
    return Alpha, Da

def mark_by_plank(z_in,goal_size):
    z = z_in
    size = goal_size
    Alpha = np.zeros(len(z), dtype = np.float)
    Da = np.zeros(len(z), dtype = np.float)
    for k in range(len(z)):
        if z[k] == 0:
            Da[k] = 0.
            Alpha[k] = np.inf
        else:
            DA_conference = Test_model.angular_diameter_distance(z[k]).value
            alpha_conference = (size/h)/DA_conference
            Alpha[k] = alpha_conference
            Da[k] = DA_conference
    return Alpha, Da

def mark_by_self_Noh(z_in,goal_size):
    z_array = z_in
    size_cluster = goal_size
    Alpha = np.zeros(len(z_array), dtype = np.float)
    Da = np.zeros(len(z_array), dtype = np.float)
    Dc = np.zeros(len(z_array), dtype = np.float)
    fEz1 = lambda sysz: 1./np.sqrt(Omega_m*(1+sysz)**3+Omega_lambda+Omega_k*(1+sysz)**2)
    for k in range(len(z_array)):
        if z_array[k] == 0:
            Da[k] = 0.
        else:
            Dc[k] = DH*scq(fEz1,0.,z_array[k])[0]
    if Omega_k == 0.:
        Dm = Dc*1.
    elif Omega_k > 0.:
        Dm = DH*np.sinh(np.asrt(Omega_k)*Dc/DH)*1/np.sqrt(Omega_k)
    else:
        Dm = DH*np.sin(np.asrt(np.abs(Omega_k))*Dc/DH)*1/np.sqrt(np.abs(Omega_k))
    DA = Dm/(1+z_array)
    alpha = size_cluster/DA
    Alpha = alpha
    Da = DA 
    return Alpha, Da

def mark_by_plank_Noh(z_in,goal_size):
    z = z_in
    size = goal_size
    Alpha = np.zeros(len(z), dtype = np.float)
    Da = np.zeros(len(z), dtype = np.float)
    for k in range(len(z)):
        if z[k] == 0:
            Da[k] = 0.
            Alpha[k] = np.inf
        else:
            DA_conference = Test_model.angular_diameter_distance(z[k]).value
            alpha_conference = size/DA_conference
            Alpha[k] = alpha_conference
            Da[k] = DA_conference
    return Alpha, Da