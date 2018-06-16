# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 19:08:45 2018

@author: Otm
"""

### Import libraries 
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches

### Secretary_dilemma 

### Parameters initialization 
N = 6
T = 7

### Main function 
def Throw_dice(T,N):
    ### Parameters initialization 
    index_val = np.arange(N)
    x= np.repeat(1+np.arange(T),N) # x-axis values 
    y= np.tile(1+index_val,T) # y-axis values 
    D = np.zeros((N)*(T)) # decision vector
    U = np.zeros((N)*(T)) #  U_n values
    ### Final constraint
    U[-N:] = 1+index_val
    D[-N:] = 2 
    
    
    ### Loop
    for j in range(T-1): # j = 0
        start_index = N*(T-2-j)
        end_index = start_index +N
        u_2 = (1)/(N)*U[end_index:end_index+N].sum()
        u_1 = 1+index_val
        U[start_index:end_index] = np.maximum(u_2,u_1)
        index_c = (u_2 > u_1) 
        D[start_index + index_val[index_c]] = 1
        index_s = (u_2 <= u_1)
        D[start_index + index_val[index_s]] = 2
    
    return [x,y,U,D]

### Test the function 
### Fonction to plot values
res = Throw_dice(T,N)
x,y,U,D = res
c_values = ['w','g','r']
c = np.zeros(N*T,dtype=str)
c[D == 0] = c_values[0]
c[D == 1] = c_values[1]
c[D == 2] = c_values[2]

def Plot_(x, y, c, v ,i,j,titre,xLegend='',yLegend='',option_save=False,filename='',
          xmin_0 = 0., xmax_0 = 40):
    plt.figure(figsize=(10,8), dpi=80)
    ax = plt.subplot(111)
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.set_xticks(range(i,j+1))
    ax.set_yticks(range(i,j+1))
    ax.axis(xmin=xmin_0,xmax=xmax_0)
    ax.grid(True, color='0.9', linestyle='-')

    for label, _x, _y, _c in zip(v, x, y, c):
        #ax.plot(x,y,'bo')
        ax.annotate(
            label, 
            xy = (_x, _y), xytext = (0, 15),
            textcoords = 'offset points', 
            bbox = dict(boxstyle = 'round,pad=0.05', fc = _c, alpha = 0.5),fontsize=8)

    ax.scatter(x, y, s=100, c=c, marker='o', alpha=0.8)
    plt.xlabel(xLegend)
    plt.ylabel(yLegend)
    plt.title(titre,fontsize=12)
    plt.tight_layout()
    red_patch = mpatches.Patch(color='red', label='région arrêt')
    green_patch = mpatches.Patch(color='green', label='région continuation')
    plt.legend(handles=[red_patch,green_patch],bbox_to_anchor=(1.05, 1),loc=2)
    if option_save :
        plt.savefig(filename)
    plt.show()

option_save=True
filename='D:\\etude\\ColoVacsTDs\\Program_dyn_vf\\images\\Throw_dice.pdf'
xmin_0 = 0.; xmax_0 = T+1
Plot_(x, y, c, U.round(1),1,N,'graphe jeu lancé de dé','nombre de lancés','valeur dé',option_save,filename,xmin_0,xmax_0)
