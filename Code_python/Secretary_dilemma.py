# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 19:08:45 2018

@author: Otm
"""

### Import libraries 
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import pylab

### Secretary_dilemma 

### Parameters initialization 
N = 15
 
### Main function 
def Secretary_dilemma(N):
    ### Parameters initialization 
    index_val = np.arange(N)
    x=np.repeat(index_val+1,N) # x-axis values 
    y=np.tile(index_val+1,N) # y-axis values 
    D = np.zeros(int(N*N)) # decision vector
    U =np.zeros(int(N*N)) #  U_n values 
    
    ### Final constraint
    U[-N]=1
    D[-N+1:]=1
    D[-N]=2
    
    
    ### Loop
    for j in range(N-1): # j= 0
        start_index = N*(N-2-j) 
        end_index = (N-j-1)
        u_2 = (1/(N-j))*(U[N*(N-1-j):N*(N-j)].sum())
        u_1 = np.zeros(end_index) 
        u_1[0] = (N-1-j)/N
        U[start_index:start_index+end_index] = np.maximum(u_2,u_1)
        index_c = (u_2 > u_1) 
        aux_val = np.arange(end_index)
        D[start_index + aux_val[index_c]] = 1
        index_s = np.ones(end_index,dtype=bool) # index of element where leave the game is the best decision
        index_s[index_c] = 0
        D[start_index + aux_val[index_s]] = 2
    
    return [x,y,U,D]

### Test the function 
### Fonction to plot values
res = Secretary_dilemma(N)
x,y,U,D = res
c_values = ['w','g','r']
c = np.zeros(int(N*N),dtype=str)
c[D == 0] = c_values[0]
c[D == 1] = c_values[1]
c[D == 2] = c_values[2]

def Plot_(x, y, c, v ,i,j,titre,xLegend='',yLegend='',option_save=False,filename=''):
    plt.figure(figsize=(10,8), dpi=80)
    ax = plt.subplot(111)
    
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.set_xticks(range(i,j+1))
    ax.set_yticks(range(i,j+1))
    ax.axis(xmin=0.,xmax=N+1)
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
filename='D:\\etude\\ColoVacsTDs\\Program_dyn_vf\\images\\Secretary_dilemma.pdf'
Plot_(x, y, c, U.round(3),1,N,'graphe sélection candidats','entretiens effectués','rang candidats',option_save,filename)
