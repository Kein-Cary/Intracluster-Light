######## test the result and fitting
with h5py.File('cluster_record.h5') as f:
    tot_sub = np.array(f['a'][0])
    inr_sub = np.array(f['a'][1])
    sub_ratio = np.array(f['a'][2])
    area_ratio = np.array(f['a'][3])
    reference_ratio = np.array(f['a'][4])
    rich = np.array(f['a'][5])
'''
sampl_tot = tot_sub[(z<=0.3) & (z>=0.2)]
sampl_sub = inr_sub[(z<=0.3) & (z>=0.2)]  # the two number just the galaxies number, the later is those in the 1Mpc/h circle
'''
sampl_S_ratio = area_ratio[(z<=0.3) & (z>=0.2)]
sampl_refer = reference_ratio[(z<=0.3) & (z>=0.2)]
sampl_z = z[(z<=0.3) & (z>=0.2)]
sampl_rich = rich[(z<=0.3) & (z>=0.2)]
# how change is the S/S0 along z variables
from scipy.stats import binned_statistic_2d as spd
zx = z[(z<=0.3) & (z>=0.2)]
it = sampl_S_ratio>=0.75
ts = sampl_S_ratio[it]
tr = sampl_rich[it]
tm = np.median(tr)
miu = np.mean(sampl_S_ratio)
miu1 = np.median(sampl_S_ratio)
std = np.std(sampl_S_ratio)
fg, ax = plt.subplots()
n, bins, patches = ax.hist(sampl_S_ratio,bins = 50,density = True,alpha = 0.75,
                           label = 'origin_data')
y = ((1 / (np.sqrt(2 * np.pi) * std)) *np.exp(-0.5 * (1 / std * (bins - miu))**2))
y1 = ((1 / (np.sqrt(2 * np.pi) * std)) *np.exp(-0.5 * (1 / std * (bins - miu1))**2))
ax.plot(bins,y,'r--',label = r'$fit-with-\bar{S/S0}-\sigma$')
ax.plot(bins,y1,'g--',label = r'$fit-with-Median(S/S0)-\sigma$')
ax.legend(loc = 1,fontsize = 7.5)
ax.text(0.3,2.5,r'$\mu=%f$'%miu)
ax.text(0.3,2.4,r'$\sigma=%f$'%std)
ax.text(0.3,2.3,r'$Median=%f$'%miu1)
ax.set_title(r'$S/S0$')
ax.set_xlabel(r'$S/S0$')
ax.set_ylabel(r'$Probability-density$')
## this part shows the PDF fitting

######## get the coordinate all of the pixel from the from itself
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
#file = get_pkg_data_filename('test_tot.fits') # k = 0
file = get_pkg_data_filename('area_test.fits') # k =1
hdu = fits.open(file)[0]
wcs = WCS(hdu.header)
cir_data = aft.getdata('area_test.fits',header = True)
y = np.linspace(0,1488,1489)
x = np.linspace(0,2047,2048)
vx, vy = np.meshgrid(x,y)
cx, cy = wcs.all_pix2world(vx,vy,1)
a0 = cir_data[1]['CD1_1']
a1 = cir_data[1]['CD1_2']
b0 = cir_data[1]['CD2_1']
b1 = cir_data[1]['CD2_2']
rax = vx - 1025
ray = vy - 745
x0 = cir_data[1]['CRVAL1']
y0 = cir_data[1]['CRVAL2']
# from SDSS EDR paper
New_Ra = x0 + (rax*a0 + ray*a1) / np.cos(y0)
New_Dec = y0 + rax*b0 + ray*b1
# ????
plt.pcolormesh(New_Ra,New_Dec,cir_data[0],cmap='Greys',vmin=1e-5,norm=mpl.colors.LogNorm())
plt.axis('square')
plt.pcolormesh(cx,cy,cir_data[0],cmap='Greys',vmin=1e-5,norm=mpl.colors.LogNorm())
# ????
# comparation (cx,cy) and (New_RA,New_Dec) 
from astropy.coordinates import SkyCoord
POS1 = SkyCoord(cx*U.deg,cy*U.deg,frame = 'icrs')
POS2 = SkyCoord(New_Ra*U.deg,New_Dec*U.deg,frame = 'icrs')
POS1 = POS1.galactic;
POS2 = POS2.galactic; # ???????? shape problem
plt.pcolormesh(POS1.galactic.l.value,POS1.galactic.b.value,cir_data[0],
               cmap='Greys',vmin=1e-5,norm=mpl.colors.LogNorm())
plt.colorbar()
plt.pcolormesh(POS2.galactic.l.value,POS2.galactic.b.value,cir_data[0],
               cmap='Greys',vmin=1e-5,norm=mpl.colors.LogNorm())
plt.colorbar()

# distance calculate: calculate the pixel distance or the distance in ra-dec coordinate
from astropy.coordinates import SkyCoord
c0 = SkyCoord(Ra[k]*U.deg,Dec[k]*U.deg,frame = 'icrs')
c1 = SkyCoord(cx*U.deg,cy*U.deg,frame = 'icrs')
sep0 = c0.separation(c1)
aa = R_A[1].value/3600
ik = sep.value <= aa
inrg = sep[ik]
reference = len(inrg)

t1, t2 = wcs.all_world2pix(Ra[k]*U.deg,Dec[k]*U.deg,1)
dep = np.sqrt((vx-t1)**2+(vy-t2)**2)
ad = R_A[k].value/0.396
ig = dep <= ad
al = dep[ig]
npixel = len(al)

######## test 6: change the coordinate system
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(111, projection = wcs)
im = ax.imshow(hdu.data, cmap='viridis',vmin=1e-5,origin = 'lower',
               norm=mpl.colors.LogNorm())
ax.set_xlabel("Right Ascension", fontsize = 16)
ax.set_ylabel("Declination", fontsize = 16)
ax.grid(color = 'red', ls = 'dotted', lw = 2)
overlay = ax.get_coords_overlay('galactic')
overlay.grid(color='blue', ls='dotted', lw=1)
overlay[0].set_axislabel('Galactic Longitude', fontsize=14)
overlay[1].set_axislabel('Galactic Latitude', fontsize=14)

############## test 5: fix the Ra-Dec coordinate
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
#file = get_pkg_data_filename('test_tot.fits') # k = 0
file = get_pkg_data_filename('area_test.fits') # k =1
hdu = fits.open(file)[0]
wcs = WCS(hdu.header)
# change pixel to world coordinate
y = np.linspace(0,1488,1489)
x = np.linspace(0,2047,2048)
vx, vy = np.meshgrid(x,y)
cx, cy = wcs.all_pix2world(vx,vy,1)
plt.pcolormesh(cx,cy,hdu.data,cmap='Greys',vmin=1e-5,norm=mpl.colors.LogNorm())
##### point the BCG
import mice_symbol 
mice_symbol.mice_symble(Ra[k],Dec[k],2/3600,8/3600,8/3600,8/3600,8/3600,c='r')
hsc.circles(Ra[k],Dec[k],s = R_A[k].value/3600,fc = '',ec = 'r')
plt.plot(cx[0,0],cy[0,0],'r*')
#plt.axis('scaled')
# select the pixels 
ff = np.abs(cx-Ra[1])
gg = np.abs(cy-Dec[1])
hh = np.sqrt((cx-Ra[1])**2+(cy-Dec[1])**2)
goal = find.find2d(hh,np.min(hh))
xg = goal[1]
yg = goal[0]
# select with ra,dec
Dpixel = np.sqrt((cx-Ra[k])**2+(cy-Dec[k])**2)
inpixel = Dpixel[Dpixel <= R_A[k].value/3600]
npixel = len(inpixel)
# select with pixel distance (this not correct!!!!!)
Dpixel = np.sqrt((vx-xg)**2+(vy-yg)**2)
inpixel = Dpixel[Dpixel <= R_A[k].value/0.396]
npixel = len(inpixel)

############## coord test4
import aplpy
## k =1 
f = aplpy.FITSFigure('area_test.fits')
cir_data = fits.getdata('area_test.fits',header = True)
wcs = WCS(cir_data[1])
x0,y0 = wcs.all_world2pix(Ra[1]*U.deg, Dec[1]*U.deg, 1)
r0 = R_A[1].value/0.396
f.set_xaxis_coord_type('longitude')
f.set_yaxis_coord_type('latitude')
f.tick_labels.set_xformat('ddd.ddd')
f.tick_labels.set_yformat('ddd.ddd')
#f.tick_labels.set_xformat('hh:mm:ss')
#f.tick_labels.set_yformat('hh:mm:ss')
#f.add_scalebar(r0, color = 'r',label = '1Mpc/h',loc = 2) # add a scale ruler
hsc.circles(x0,y0,s = r0,fc = '',ec = 'r')
plt.scatter(x0,y0,marker = 'P')
f.show_grayscale()
## add colorbar
f.frame.set_color('grey')
f.add_colorbar()
f.colorbar.show()
#f.show_grid()
#### hard to point out BCG, but it's good for showing the data!!!!

############## coord test 3: main body
cir_data = aft.getdata('area_test.fits',header = True) # k = 1
#cir_data = aft.getdata('test_tot.fits',header = True) # k = 0
size = cir_data[0].shape
h = size[0]
w = size[1]
# get the mirror data
import copy 
ILR = copy.deepcopy(cir_data[0])
for p in range(h):
    for q in range(w):
        ILR[p,w-1-q] = cir_data[0][p,q]
########
wcs = awc.WCS(cir_data[1])
fig = plt.figure(figsize = (10,10))
y = np.linspace(0,1488,1489)
x = np.linspace(0,2047,2048)
vx, vy = np.meshgrid(x,y)
cx, cy = wcs.all_pix2world(vx,vy,1)
hh = np.sqrt((cx-Ra[1])**2+(cy-Dec[1])**2)
goal = find.find2d(hh,np.min(hh))
ax = fig.add_axes([0.1,0.1,0.8,0.8],projection = wcs) # initial
#ax = fig.add_axes([0.1,0.1,0.8*h/w,0.8],projection = wcs) # transpose
#ax = fig.add_axes([0.1,0.1,0.8,0.8*h/w],projection = wcs) # mirror
ax.set_xlabel('RA-(spacing-[0.05deg])')
ax.set_ylabel('DEC-(spacing-[0.05deg])')
im = ax.imshow(cir_data[0],cmap = 'Greys',aspect = 'auto',vmin=1e-5,
               origin = 'lower',norm=mpl.colors.LogNorm()) #initial
#plt.colorbar(im,fraction = 0.05,pad = 0.03,label = r'$f_{flux}[nanoMaggy]$') # colorbar adjust
#ax.imshow(np.transpose(cir_data[0]),cmap = 'viridis',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm()) #transpose
#ax.imshow(ILR,cmap = 'viridis',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm()) #mirror
#ax.imshow(np.transpose(ILR),cmap = 'viridis',aspect = 'auto',vmin=1e-5,origin = 'lower',norm=mpl.colors.LogNorm()) #transpose and mirror
ra = ax.coords[0]
#ra.set_major_formatter('dd:mm:ss')
ra.set_major_formatter('d.ddddd')
ra.grid(color = 'red',alpha = 0.45)
dec = ax.coords[1]
#dec.set_major_formatter('dd:mm:ss')
dec.set_major_formatter('d.ddddd')
dec.grid(color = 'green',alpha = 0.45)
xx = Ra[k]
yy = Dec[k]
ax.scatter(xx,yy,facecolors = '', marker = 'P',edgecolors = 'b',
           transform=ax.get_transform('world'))
#ax.scatter(goal[1],goal[0],yy,facecolors = 'b', marker = '+',edgecolors = 'b')
aa = ax.get_xlim()
bb = ax.get_ylim()
'''
poa = member_pos[0][sum_IA:sum_IA+rept_ID[1][IA]]
pob = member_pos[1][sum_IA:sum_IA+rept_ID[1][IA]]
ak = find.find1d(poa,Ra[k])
posx = poa[poa!=poa[ak]]
posy = pob[pob!=pob[ak]] # out the BCG, and then mark the satellite 
ax.scatter(posx,posy,facecolors = '',marker = 'o',edgecolors = 'r',
           transform=ax.get_transform('world'))
'''
from matplotlib.patches import Circle
#r1 = Circle((xx,yy),radius = R_A[k].value/3600,facecolor = 'None',edgecolor = 'r',transform=ax.get_transform('world')) # just circle 
'''
r1 = Circle((xx,yy),radius = R_A[k].value/3600,alpha = 0.25,transform=ax.get_transform('world'))
ax.add_patch(r1)
'''
hsc.circles(goal[1],goal[0],s = R_A[k].value/0.396,fc = '',ec = 'r')
ra.set_ticks(spacing = 0.05*U.deg)
ra.set_ticklabel(color = 'red')
dec.set_ticks(spacing = 0.05*U.deg) 
dec.set_ticklabel(color = 'green')# set the spacing of ra-dec grid, and must in unit 'degree'
#cc = ax.get_xlim()
#dd = ax.get_ylim()s
# comparation aa, bb, cc, dd
ax.axis('scaled')
ax.set_xlim(aa[0],aa[1])
ax.set_ylim(bb[0],bb[1])
plt.colorbar(im,fraction = 0.035,pad = 0.03,label = r'$f_{flux}[nanoMaggy]$') # colorbar adjust
## calculate the shadow area in the cluster radius : 1Mpc/h
scale_L = (R_A[k].value/0.396)*0.5
A = [[100,100],[100+scale_L,100]]
ax.plot(A[1],A[0],'b',)
ax.text(x=112,y=112,s = '500kpc/h',color = 'b')
#ax.set_title('initial')
#ax.set_title('transpose')
#ax.set_title('mirror')
### the two "get xilm and ylim" says that we can get all the axis-limit and then set a suitable one.
# hsc.circles(xx,yy,s = R_A[k].value/3600,fc = '',ec = 'r',transform=ax.get_transform('world')) # cannot use

##### test 2: figure in different 
file = get_pkg_data_filename('area_test.fits') # k =1
hdu = fits.open(file)[0]
wcs = WCS(hdu.header)
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111, projection= wcs)
im = ax.contour(hdu.data,cmap = 'viridis',vmin=1e-5,norm=mpl.colors.LogNorm())
ax.set_xlabel("RA", fontsize = 16)
ax.set_ylabel("Dec", fontsize = 16)
ax.grid(color = 'red', ls = 'dotted', lw = 2)
cbar = plt.colorbar(im, pad=.07)
cbar.set_label(''.join('flux-[nanomaggy]'),size = 16) # colorbar name, size, and directiion
overlay = ax.get_coords_overlay('galactic')
overlay.grid(color='blue', ls='dotted', lw=1)
overlay[0].set_axislabel('Galactic Longitude', fontsize=14)
overlay[1].set_axislabel('Galactic Latitude', fontsize=14) ## multi-coordinate figure

################ coord test1
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
file = get_pkg_data_filename('test_tot.fits')
hdu = fits.open(file)[0]
wcs = WCS(hdu.header)
fig = plt.figure(figsize = (8,8))
fig.add_subplot(111,projection = wcs)
plt.imshow(hdu.data, origin = 'lower',cmap = plt.cm.viridis, norm = mpl.colors.LogNorm())
plt.xlabel('RA')
plt.ylabel('DEC')
plt.grid(color = 'blue')
# plt.savefig('coord_adjust3.png',dpi = 600)
plt.show()
#### hard to point out BCG