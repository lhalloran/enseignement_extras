# -*- coding: utf-8 -*-
"""
SMC_Day5.py
Landon Halloran @ Uni Neuchatel
ljsh.ca

Part of the "figures" lecture in the course:
Scientific Method and Communication 
(part of the MSc in Hydrogeology and Geothermics programme)
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
#import datetime as dt

#%% GENERATE DATA
nval=20
steps= np.random.randint(30,size=nval-1)+1
steps[4]=1;steps[5]=1;steps[6]=1;steps[8]=1;steps[9]=1
t0 = np.datetime64('2021-01-02')
t = np.tile(t0,(nval))
for i in range(nval-1):
    tnow = t[i] + np.timedelta64(steps[i],'D')
    t[i+1]=tnow

Conc1 = (5 + 1*((t-t0)/np.timedelta64(1,'D')))*0.1
Conc1 = Conc1 + np.random.normal(loc=0.00,scale=0.5,size=nval)
#%% MAKE BAD FIG (bar)
plt.figure(figsize=(10,8))
plt.bar(np.arange(nval),Conc1)
plt.xticks(np.arange(nval),t,rotation=90)
plt.xlabel('Date')
plt.ylabel('Concentration [mg/l]')

#%% MAKE BAD FIG (= default "plot" from Excel)
plt.figure(figsize=(10,8))
plt.plot(Conc1)
plt.xticks(np.arange(nval),t,rotation=90)
plt.xlabel('Date')
plt.ylabel('Concentration [mg/l]')

#%% MAKE CORRECTED FIG
plt.figure(figsize=(10,8))
plt.plot(t,Conc1,'ko')
plt.grid()
#plt.xticks(np.arange(nval),t,rotation=90)
plt.xlabel('Date')
plt.ylabel('Concentration [mg/l]')


##########################################################################################
#%% colour maps ( from https://pbett.wordpress.com/datafun/python-matplotlib-2d/ )
import scipy
import scipy.stats
import matplotlib.cm as mpl_cm    # For Brewer etc. colourmaps
import matplotlib as mpl
# http://matplotlib.org/api/pyplot_api.html#module-matplotlib.pyplot
#----------------------------------------------------------------------
 #%%
# Make some data:
nx = 200   # i.e. number of columns
ny = 300   # i.e. number of rows
x = np.linspace(-3,3, nx)
y = np.linspace(-3,3, ny)
X,Y = np.meshgrid(x,y)
# X and Y both have shape (ny,nx) i.e. (nrow,ncol)
 
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
# pos has shape (ny,nx,2)
 
meanpos1 = [1.2, -0.5]
cov1 = [[2.0, 0.3], [0.3, 0.5]]
binorm1 = scipy.stats.multivariate_normal(meanpos1, cov1)
binormpdf1 = binorm1.pdf(pos)
 
meanpos2 = [-0.2,0.3]
cov2 = [[2.0, 0.1], [0.2, 1.0]]
binorm2 = scipy.stats.multivariate_normal(meanpos2, cov2)
binormpdf2 = binorm2.pdf(pos)
 
z = 100.0 * (binormpdf1 - binormpdf2)
# z has shape (ny,nx)

#%% Now set up the colour bars 

vmin = -10
vmax =  10
vstep=   1
levels = np.arange(vmin, vmax+vstep, vstep)
levels2= np.arange(vmin, vmax+2, 2)
 
cpal = "Reds"
cmap_cont = mpl_cm.get_cmap(cpal)
cmap_disc = mpl_cm.get_cmap(cpal, len(levels)-1)
 
# Add values for bad/over/under:
#if badcol:   cmap.set_bad(  color=badcol,  alpha=1)
#if overcol:  cmap.set_over( color=overcol, alpha=1)
#if undercol: cmap.set_under(color=undercol,alpha=1)
#and initialise the figure:

sizecm = (33,17) # (w,h)
sizein = [l/2.54 for l in sizecm]
fig = plt.figure(figsize=sizein, dpi=96 )
#First Axes object: pcolormesh using continuous colours.

ax1 = fig.add_subplot(2,3,1)
#ax1.set_title("pcolormesh, continuous")
ax1.pcolormesh(X, Y, z,
               cmap=cmap_cont,
               vmin=vmin,vmax=vmax )
#Second Axes object: pcolormesh, using discrete colours:

ax2 = fig.add_subplot(2,3,2)
#ax2.set_title("pcolormesh, discrete")
ax2.pcolormesh(X, Y, z,
               cmap=cmap_disc,
               vmin=vmin,vmax=vmax ),
#Third Axes object: contourf (filled contours, discrete colours only):

ax3 = fig.add_subplot(2,3,3)
#ax3.set_title("contourf, discrete")
ax3.contourf(X, Y, z,
             cmap=cmap_disc,
             levels=levels,
             extend="both")
#OK, onto the next row, looking at contour lines.

#Fourth Axes: colour-graded contours.

ax4 = fig.add_subplot(2,3,4)
#ax4.set_title("Contours, coloured")
ax4.contour(X, Y, z,
            cmap=cmap_disc,
            levels=levels)
#Fifth Axes: Styling and clabelling of contours:

ax5 = fig.add_subplot(2,3,5)
#ax5.set_title("Contours, styled")
# For monochrome lines, negatives are dashed by default;
# you can set them to be 'solid', but not 'dotted'??
mpl.rcParams['contour.negative_linestyle'] = 'dashed'
conts = ax5.contour(X, Y, z,
                    colors="black",
                    levels=levels2 )
ax5.clabel(conts, inline=1, fontsize=8, fmt="%2d")
#Sixth Axes: Combining continuous shading (pcolormesh) and discrete contours.

ax6 = fig.add_subplot(2,3,6)
ax6.pcolormesh(X, Y, z,
               cmap=cmap_cont,
               vmin=vmin,vmax=vmax )
mpl.rcParams['contour.negative_linestyle'] = 'solid'
ax6.contour(X, Y, z,
            colors="black",
            levels=levels)
#Finally, finish off the plot.
#%% extra 3d plot
from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, z, cmap=mpl_cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlabel('f(x,y)')
ax.set_xlabel('x')
ax.set_ylabel('y')


##################################################################################
#%% LH demo with pandas
import pandas as pd

filein='example_borehole_data.xlsx' 

rawdata=pd.read_excel(filein)
#%%
print(rawdata.columns)

#%% plot 1 BH with depth on x axis
fig,axs = plt.subplots(4,1,figsize=(10,8))
axs[0].plot(rawdata['Depth [m]'][rawdata['Borehole number']==1],rawdata['Cl [mg/L]'][rawdata['Borehole number']==1])
axs[1].plot(rawdata['Depth [m]'][rawdata['Borehole number']==1],rawdata['Temperature [C]'][rawdata['Borehole number']==1])
axs[2].plot(rawdata['Depth [m]'][rawdata['Borehole number']==1],rawdata['Turbidity [NTU]'][rawdata['Borehole number']==1])
axs[3].plot(rawdata['Depth [m]'][rawdata['Borehole number']==1],rawdata['Presence of bacteria?'][rawdata['Borehole number']==1])


#%% plot 1 BH
fig,axs = plt.subplots(1,4,figsize=(10,8),sharey=True)
axs[0].plot(rawdata['Cl [mg/L]'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1])
axs[1].plot(rawdata['Temperature [C]'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1])
axs[2].plot(rawdata['Turbidity [NTU]'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1])
axs[3].plot(rawdata['Presence of bacteria?'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1])

#%%
fig,axs = plt.subplots(1,4,figsize=(10,8),sharey=True)
axs[0].plot(rawdata['Cl [mg/L]'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1],'o-')
axs[1].plot(rawdata['Temperature [C]'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1],'o-')
axs[2].plot(rawdata['Turbidity [NTU]'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1],'o-')
axs[3].plot(rawdata['Presence of bacteria?'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1],'o-')

#%% step binary data
axs[3].step(rawdata['Presence of bacteria?'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1],)

#%% maybe better to not connect points here
axs[3].plot(rawdata['Presence of bacteria?'][rawdata['Borehole number']==1],rawdata['Depth [m]'][rawdata['Borehole number']==1],'*')

#%% plot all bore holes = make loops
fig,axs = plt.subplots(1,4,figsize=(10,8),sharey=True)
for i in range(4):
    axs[0].plot(rawdata['Cl [mg/L]'][rawdata['Borehole number']==i+1],rawdata['Depth [m]'][rawdata['Borehole number']==i+1],'o-')
    axs[1].plot(rawdata['Temperature [C]'][rawdata['Borehole number']==i+1],rawdata['Depth [m]'][rawdata['Borehole number']==i+1],'o-')
    axs[2].plot(rawdata['Turbidity [NTU]'][rawdata['Borehole number']==i+1],rawdata['Depth [m]'][rawdata['Borehole number']==i+1],'o-')
    axs[3].plot(rawdata['Presence of bacteria?'][rawdata['Borehole number']==i+1],rawdata['Depth [m]'][rawdata['Borehole number']==i+1],'*')

#%% better way to do loop:
colnames=rawdata.columns[[1,2,4,5]]
print(colnames)
fig,axs = plt.subplots(1,4,figsize=(10,8),sharey=True)

for i in range(4): # BOREHOLE LOOP
    for j in range(3): # TYPES OF DATA LOOP
        axs[j].plot(rawdata[colnames[j]][rawdata['Borehole number']==i+1],rawdata['Depth [m]'][rawdata['Borehole number']==i+1],'o-')
    axs[3].plot(rawdata[colnames[3]][rawdata['Borehole number']==i+1],rawdata['Depth [m]'][rawdata['Borehole number']==i+1],'*')

#%% invert y axis
axs[0].invert_yaxis()
#%% make yes on right, no on left
axs[3].invert_xaxis()

#%% labels
axs[0].set_ylabel('Depth [m]')
for i in range(4):
    axs[i].set_xlabel(colnames[i])

#%% legend
axs[1].legend(['BH1','BH2','BH3','BH4'],loc=3)
#%% import seaborn and run again 
# https://seaborn.pydata.org/tutorial/aesthetics.html
import seaborn as sb
sb.set_style("darkgrid")

#%% 
plt.figure()
for i in range(4):
    plt.plot(rawdata['Cl [mg/L]'][rawdata['Borehole number']==i+1],rawdata['Temperature [C]'][rawdata['Borehole number']==i+1],'o')
