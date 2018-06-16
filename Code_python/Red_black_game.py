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
N = 8
T = 2*N

### Main function 
def Red_black_game(T,N):
    ### Parameters initialization 
    index_val = np.arange(N+1)
    x= np.repeat(np.arange(T+1),N+1) # x-axis values 
    y= np.tile(index_val,T+1) # y-axis values 
    D = np.zeros((N+1)*(T+1)) # decision vector
    U = np.zeros((N+1)*(T+1)) #  U_n values
    u_2 = np.zeros(N+1)
    ### Final constraint
    D[-1]=2 
    
    
    ### Loop
    for j in range(T): # j = 2
        start_index = (N+1)*(T-1-j)
        end_index = start_index +(N+1)
        u_2[:-1] = (N-index_val[:-1])/(j+1)*U[end_index+1:end_index+(N+1)]  + ((N-(T-1-j)+index_val[:-1])/(j+1))*U[end_index:end_index+N] 
        u_2[-1] = U[end_index+N]
        u_1 = 2*index_val - (T-1-j)
        index_poss = ((T-1-j)-N <= index_val ) & ((T-1-j) >= index_val )
        U[start_index + index_val[index_poss]] = np.maximum(u_2,u_1)[index_val[index_poss]]
        index_c = (u_2 > u_1) & ((T-1-j)-N <= index_val ) & ((T-1-j) >= index_val )
        D[start_index + index_val[index_c]] = 1
        index_s= (u_2 <= u_1) & ((T-1-j)-N <= index_val ) & ((T-1-j) >= index_val )
        D[start_index + index_val[index_s]] = 2
    
    return [x,y,U,D]

### Test the function 
### Fonction to plot values
res = Red_black_game(T,N)
x,y,U,D = res
c_values = ['w','g','r']
c = np.zeros((N+1)*(T+1),dtype=str)
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
filename='D:\\etude\\ColoVacsTDs\\Program_dyn_vf\\images\\Red_black_game.pdf'
xmin_0 = -1.; xmax_0 = T+2
Plot_(x, y, c, U.round(1),0,T,'graphe jeu red/black','nombre total carte tirées','cartes noires tirées',option_save,filename,xmin_0,xmax_0)
