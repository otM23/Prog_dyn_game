# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 19:08:45 2018

@author: Otm
"""

### Import libraries 
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

### Plot 3d 
class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

### Annotate the point xyz with the text 's'
def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
        
### The collector problem

### Parameters initialization 
N = 2
T = 3
p = 1
c_1 = 0.2
c_2 = 0.6
r = int(np.ceil(c_2 / c_1))
T_c = r*T
pen_val = -T*c_2

### Main function 
def Collector_pbm(T,N,T_c,p,c_1,c_2,pen_val = -100):
    ### Parameters initialization
    r = int(np.ceil(c_2 / c_1))
    index_val_x = np.arange(N+1)
    index_val_z = np.arange(T_c)
    x= np.repeat(np.arange(T+1),(N+1)*T_c) # x-axis values 
    y= np.tile(index_val_x,(T+1)*T_c) # y-axis values 
    z= np.tile(np.repeat(index_val_z,N+1),T+1)*c_1
    D = np.zeros((N+1)*T_c*(T+1)) # decision vector
    U = np.zeros((N+1)*T_c*(T+1)) #  U_n values
    ### Final constraint
    start_index = (N+1)*T_c*T
    U[start_index + (N+1)*index_val_z + N] = p
    U[start_index:] -= z[start_index:]
    D[start_index:] = 2 
    
    
    ### Loop
    values_x = np.tile(index_val_x,T_c)
    values_z = np.repeat(index_val_z,N+1)
    index_values = np.arange((N+1)*T_c)
    values1 = np.tile(index_val_x[:-1],T_c-1)
    index_values_0 = values1 + (N+1)*np.repeat(index_val_z[:-1],N)
    index_values_1 = values1 + (N+1)*np.repeat(index_val_z[1:],N)
    index_values_2 = np.tile(index_val_x[1:],T_c-1) + (N+1)*np.repeat(index_val_z[1:],N)
    index_values_3 = np.tile(index_val_x[:-1],T_c-r) + (N+1)*np.repeat(index_val_z[:-r],N)
    index_values_4 = np.tile(index_val_x[1:],T_c-r) + (N+1)*np.repeat(index_val_z[r:],N)
    u_2 = np.zeros((N+1)*T_c)
    u_1 = np.zeros((N+1)*T_c)
    for j in range(T): # j = 0
        start_index = (N+1)*T_c*(T-1-j)
        end_index = start_index + (N+1)*T_c
        u_2[index_values_0] = (N-values1)/(N)*U[end_index+index_values_2]  + (values1/N)*U[end_index+index_values_1]
        u_2[(N+1)*(T_c-1):] = pen_val # U[end_index + (N+1)*(T_c-1): end_index + (N+1)*T_c] 
        u_2[(N+1)*index_val_z+N] = U[end_index + (N+1)*index_val_z+N] 
        u_1[index_values_3] = U[end_index + index_values_4]
        u_1[(N+1)*(T_c-r):] = pen_val # U[end_index + (N+1)*(T_c-r): end_index + (N+1)*T_c] 
        u_1[(N+1)*index_val_z + N] = U[end_index + (N+1)*index_val_z + N]
        index_poss = (values_x <= (T-1-j)) & (values_z <= (T-1-j)*r) & (values_z >= (T-1-j))
        U[start_index + index_values[index_poss]] = np.maximum(u_2,u_1)[index_poss]
        index_c = (u_2 > u_1) & (values_x <= (T-1-j)) & (values_z <= (T-1-j)*r) & (values_z >= (T-1-j))
        D[start_index + index_values[index_c]] = 1
        index_s = (u_2 <= u_1) & (values_x <= (T-1-j)) & (values_z <= (T-1-j)*r) & (values_z >= (T-1-j))
        D[start_index + index_values[index_s]] = 2
        D[start_index + index_values]
    
    return [x,y,z,U,D]

### Test the function 
x,y,z,U,D = Collector_pbm(T,N,T_c,p,c_1,c_2)

### Plot the result
c_values = ['w','g','r']
c = np.zeros((N+1)*T_c*(T+1),dtype=str)
c[D == 0] = c_values[0]
c[D == 1] = c_values[1]
c[D == 2] = c_values[2]

### Fonction to plot values
def Plot_(x, y, z, c, v ,i,j_x,j_y,title,xLegend='',yLegend='',zLegend='',option_save=False,filename='',
          elev0 = 30, azim0 = -100, dist0 = 12,epsilon_x = 0.2, epsilon_y = 0.4, epsilon_z = -0.3):
    fig = plt.figure(figsize=(10,8), dpi=80)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.view_init(elev= elev0,azim=azim0)
    ax.dist=dist0  
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks(range(i,j_x+1))
    ax.set_yticks(range(i,j_y+1))
    ax.grid(True, color='0.9', linestyle='-')
    
    index_poss = c != c_values[0]
    v_ = v[index_poss]
    x_ = x[index_poss]
    y_ = y[index_poss]
    z_ = z[index_poss]
    c_ = c[index_poss]
    for label, _x, _y, _z, _c in zip(v_, x_, y_, z_, c_):
        annotate3D(ax, s=label, xyz=(_x + epsilon_x,_y + epsilon_y,_z + epsilon_z), fontsize=8, xytext=(0, 15),
                   bbox = dict(boxstyle = 'round,pad=0.05', fc = _c, alpha = 0.5),
                   textcoords='offset points') #, ha='right',va='bottom')   

    ax.scatter(x, y, z, s=100, c=c, marker='o', alpha=0.8)
    plt.xlabel(xLegend)
    plt.ylabel(yLegend)
    ax.set_zlabel(zLegend)
    plt.title(title,fontsize=12)
    plt.tight_layout()
    red_patch = mpatches.Patch(color='red', label='région arrêt')
    green_patch = mpatches.Patch(color='green', label='région continuation')
    plt.legend(handles=[red_patch,green_patch],bbox_to_anchor=(1, 1),loc='best')
    if option_save :
        plt.savefig(filename)
    plt.show()


option_save=True
filename = 'D:\\etude\\ColoVacsTDs\\Program_dyn_vf\\images\\Collector_dilemma.pdf'
elev0 = 30; azim0 = -100; dist0 = 12
j_x = T; j_y = N
Plot_(x, y, z,c, U.round(2),1,j_x,j_y,'graphe problème collectionneur','nombre de jours', 'Images collectionnées','Cout total',option_save,filename)
