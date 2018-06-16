# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 13:06:57 2018

@author: Otm
"""

import pylab
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
fig = pylab.figure()
ax = fig.add_subplot(111, projection = '3d')
x = y = z = [1, 2, 3]
sc = ax.scatter(x,y,z)
# now try to get the display coordinates of the first point

x2, y2, _ = proj3d.proj_transform(1,1,1, ax.get_proj())
