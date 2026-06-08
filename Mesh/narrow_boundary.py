import numpy as np
import pdb
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import scipy
from scipy.io import loadmat
from scipy.io import savemat

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, DataLoader
from torch import tensor
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../pycamotk')
from pycamotk.pyCaMOtk.setup_ins_base_handcode import setup_ins_base_handcode
# from pyCaMOtk.setup_ins_base_handcode_NSf import setup_ins_base_handcode_f
from pycamotk.pyCaMOtk.create_femsp_cg import create_femsp_cg_mixed2  #,create_femsp_cg_mixed2_f
from pycamotk.pyCaMOtk.create_dbc_strct import create_dbc_strct
from pycamotk.pyCaMOtk.solve_fem import solve_fem_f
from pycamotk.pyCaMOtk.visualize_fem import visualize_fem
from pycamotk.pyCaMOtk.mesh import Mesh

sys.path.insert(0, '../source')

U0=None
rho=1
nu=0.01
pltit=True
ndim=2
nvar=2
porder=2

etype='simplex'
'''求解区域[0,1]x[-0.25,0]'''
xcg_u=loadmat('D:/Matlab2019b/bin/plot_triangle_data/u1=0u2=1/25 elements/xcg.mat')
e2vcg_u_gmsh=loadmat('D:/Matlab2019b/bin/plot_triangle_data/u1=0u2=1/25 elements/e2vcg.mat')
xcg2_p=loadmat('D:/Matlab2019b/bin/plot_triangle_data/u1=0u2=1/25 elements/xcg2.mat')
e2vcg2_p=loadmat('D:/Matlab2019b/bin/plot_triangle_data/u1=0u2=1/25 elements/e2vcg2.mat')
e2bnd_up = loadmat('D:/Matlab2019b/bin/plot_triangle_data/u1=0u2=1/25 elements/e2bnd.mat')
e2vcg_u_gmsh=e2vcg_u_gmsh['e2vcg']-1
e2vcg=np.zeros((e2vcg_u_gmsh.shape[0],e2vcg_u_gmsh.shape[1]))
e2vcg[0,:] = e2vcg_u_gmsh[0,:]
e2vcg[1,:] = e2vcg_u_gmsh[3,:]
e2vcg[2,:] = e2vcg_u_gmsh[1,:]
e2vcg[3,:] = e2vcg_u_gmsh[5,:]
e2vcg[4,:] = e2vcg_u_gmsh[4,:]
e2vcg[5,:] = e2vcg_u_gmsh[2,:]
e2vcg=e2vcg.astype(int)#这部分操作是为了将Gmsh生成网格全局索引按照程序中的f2n进行转换

# e2vcg=e2vcg_u['e2vcg']-1
xcg=xcg_u['xcg']
xcg2 = xcg2_p['xcg2']
e2vcg2=e2vcg2_p['e2vcg2']-1
e2bnd = e2bnd_up['e2bnd']
msh_ = Mesh(etype,xcg2,e2vcg2,e2bnd,ndim)#p
msh = Mesh(etype,xcg,e2vcg,e2bnd,ndim)#u
nnode = xcg.shape[1]
nnode_p = xcg2.shape[1]
print("xcg=",xcg)
print("xcg2=",xcg2)
print("e2vcg=",e2vcg)
print("e2vcg2=",e2vcg2)
print("e2bnd=",e2bnd)

# 施加边界
ndofU = ndim * nnode
ndofUP = ndofU + xcg2.shape[1]

L1=1
L2=1
dbc_idx1 = []
for i in range(nnode):
	if i in dbc_idx1:
		continue
	if xcg[0, i] < 1e-12 or xcg[0, i] > (L1 - 1e-12) or xcg[1,i]<(-1e-12) or xcg[1,i] > (L2 - 1e-12):
		dbc_idx1.append(i)
print("idx1=",dbc_idx1)
dbc_idx=dbc_idx1

# ReDefine Mesh
xcg_=msh_.xcg
shrinkScalar=lambda y :(1-s*np.cos(np.pi*(y-L1/2)))
s=0.4
for i in range(xcg.shape[1]):
	xcg[0,i]=(xcg[0,i]-L1/2)*shrinkScalar(xcg[1,i])+L1/2
for i in range(xcg_.shape[1]):
	xcg_[0,i]=(xcg_[0,i]-L1/2)*shrinkScalar(xcg_[1,i])+L1/2

msh=Mesh(etype,xcg,e2vcg,msh.e2bnd,2)
msh_=Mesh(etype,xcg_,e2vcg2,msh_.e2bnd,2)
e2vcg2=msh_.e2vcg
xcg=msh.xcg
e2vcg=msh.e2vcg
nnode=xcg.shape[1]

xcg_bnd=[]
xcg_bnd=xcg[:,dbc_idx1]
print("xcg_bnd",xcg_bnd)
savemat('xcg_bnd.mat', {'xcg_bnd': xcg_bnd})