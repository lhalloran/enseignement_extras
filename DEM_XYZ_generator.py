# -*- coding: utf-8 -*-
"""
DEM_XYZ_generator.py 
Landon Halloran
landon.halloran AT unine.ch
23.oct.2022

Script to generate a synthetic metre-based DEM and export it in text format (i.e., columns of X,Y,Z coordinates).
You can combine the output files to generate hydrological stratigraphy for import into FEFLOW (in that case, add column "Slice").

Script written for Hydrogeological Modelling couse, autumn 2022 (MSc in Hydrogeology and Geothermics @ Uni Neuchatel.)
"""
import numpy as np
import matplotlib.pyplot as plt
#%%
file_out = 'generated_xyz.dat'
extent_x = np.array([0,1000])
extent_y = np.array([0,1000])
resolution = 10 

# we define a 2nd-degree polynomial of 2 variables (x and y) + Gaussian-distributed noise
coeffs = np.array([0.00014,0.0002,0.00006,0.020,0.014,75]) #coefficients
noise = 2
def z_gen(x,y,cf,sigma):
    z = cf[0]*x**2 + cf[1]*x*y + cf[2]*y**2 + cf[3]*x + cf[4]*y + cf[5]
    z = z + np.random.normal(scale=sigma)
    return z

x0=extent_x[0]
y0=extent_y[0]

# make independent arrays of x and y coordinates 
xarr = np.arange(extent_x[0], extent_x[1]+0.001, resolution)    
yarr = np.arange(extent_y[0], extent_y[1]+0.001, resolution)

#%% Make the elevation map using the function we defined above.
topo = []

# make the elevation map based on z_gen. note that we pass all coordinates as x-x0 and y-y0
for x in xarr:
    for y in yarr:
        z = z_gen(x-x0, y-y0, coeffs, noise)
        topo.append([x, y, z])
topo = np.array(topo)

#%%
# For fun, we can also do things like add little hills and depressions:
nhills = 15
hills_height_factor = 3
hills_size_factor = 90
min_hills_size = 20
for i in np.arange(nhills):
    # make random locations and randomised extents and orientations
    x_hill = np.random.uniform(low=extent_x[0], high=extent_x[1])
    y_hill = np.random.uniform(low=extent_y[0], high=extent_y[1])
    sigmaA = np.max([min_hills_size,np.abs(hills_size_factor + np.random.normal(scale=hills_size_factor/4))])
    sigmaB = np.max([min_hills_size,np.abs(hills_size_factor + np.random.normal(scale=hills_size_factor/4))])
    angle = np.random.uniform(low=0, high=2*np.pi)
    # To understand this math, see:  https://en.wikipedia.org/wiki/Gaussian_function#Meaning_of_parameters_for_the_general_equation
    a = np.cos(angle)**2/(2*sigmaA**2) + np.sin(angle)**2/(2*sigmaB**2)
    b = np.sin(2*angle)/(4*sigmaA**2) - np.sin(2*angle)/(4*sigmaB**2)
    c = np.sin(angle)**2/(2*sigmaA**2) + np.cos(angle)**2/(2*sigmaB**2)
    
    # note np.random.normal is a gaussian distribution with sigma=scale... to make only "hills" add np.abs()       
    topo[:,2] = topo[:,2] + np.random.normal(scale=hills_height_factor)*np.exp(-( a*(topo[:,0]-x_hill)**2 + 2*b*(topo[:,0]-x_hill)*(topo[:,1]-y_hill) + c*(topo[:,1]-y_hill)**2 ) )


print('range of z: ' + str(np.min(topo[:,2])) + ' to ' + str(np.max(topo[:,2])) +  ' m')

#%%
# VIEW RESULT
plt.figure()
nlevels = 16
levels = np.linspace(np.min(topo[:,2]),np.max(topo[:,2]),nlevels)
plt.tricontourf(topo[:,0],topo[:,1],topo[:,2],levels=levels)
plt.colorbar()
plt.tricontour(topo[:,0],topo[:,1],topo[:,2],colors='k',levels=levels)
plt.xlabel('x [m]'); plt.ylabel('y [m]')

#%%
# EXPORT TO FILE
np.savetxt(file_out,topo,header = 'X Y Z',comments='')
print('data exported to: '+ file_out)