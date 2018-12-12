# see the angular diameter distance and angular size
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as C
import astropy.units as U
import astroquery.sdss as asds
import astropy.io.fits as aft
from ICL_angular_diameter_reshift import mark_by_self
z = np.linspace(0,1,1001)
D = 4.
A_size, A_d= mark_by_self(z,D)
plt.plot(z,A_size)
plt.ylabel(r'$\alpha$')
plt.xlabel(r'$z$')
plt.show()
plt.figure()
plt.plot(z,A_d)
plt.ylabel(r'$D_A$')
plt.xlabel(r'$z$')
plt.show()