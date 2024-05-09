import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr 
import cartopy.crs as ccrs 
import cartopy.feature as cfeature
import seaborn as sns
from calculate_eof import eof, nonsquare_eof

sns.set(style="whitegrid")
pal = sns.color_palette('Paired')
proj = ccrs.Mercator()

g = 9.80665

ds = xr.open_dataset(r'path/file.txt')

## ----- Extract values from dataset
lats = np.array(ds['latitude'])
lons = np.array(ds['longitude'])
geopot = np.array(ds['z'].reduce(np.nansum, 'expver'))/g
level = np.array(ds['level'])

## ----- Get number of months fron length of dataset
times = np.linspace(1, geopot.shape[0], geopot.shape[0])


vals, vecs, pcas = nonsquare_eof(geopot)

## ----- Make plots of eigvals, eigvecs, principal components

plt.figure(figsize=(8,5))
plt.loglog(vals, 'o', color=pal[9])
plt.title('Normalised eigenvalues of geopotential height')
plt.xlabel('Eigenvalue number')
plt.ylabel('Magnitude')
plt.grid(visible=True, linestyle='--')
plt.show()


plt.figure(figsize=(8,5))
data = vecs[0,:,:,0]
ax = plt.axes(projection=proj)
ax.set_extent([-60, 20, 65, 45], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', zorder=10)
ax.gridlines(draw_labels=True, xlocs=np.arange(-60,20,20), ylocs=np.arange(45,65,5))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
plt.contour(lons, lats, data, transform=ccrs.PlateCarree(), zorder=11, alpha=0.5, cmap='gnuplot')
plt.colorbar(location='bottom', label='Magnitude')
plt.title('First eigenvector contours for the '+str(level[0])+'hPa level')
plt.savefig('Transparent/unfiltmode1.png', transparent=True)


labs = ['1979', '1980', '1981', '1982', '1983']


plt.figure(figsize=(8,5))
plt.plot(pcas[0:60,0], 'r')
plt.title('Time evolution of first mode of geopotential height, ERA-5')
plt.grid(visible=True, linestyle='--')
plt.xticks(np.arange(0,60,12), labs)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

