import numpy as np
import pdb
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import scipy
from scipy.io import loadmat

import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, DataLoader
from torch import tensor

# sys.path.insert(0, '../pycamotk')
from pycamotk.pyCaMOtk.create_mesh_hcube import mesh_hcube
from pycamotk.pyCaMOtk.setup_NS_base_handcode_f import setup_ins_base_handcode_NSf
# from pycamotk.pyCaMOtk.setup_ins_base_handcode import setup_ins_base_handcode_NSf
from pycamotk.pyCaMOtk.create_femsp_cg import create_femsp_cg_mixed2
from pycamotk.pyCaMOtk.create_dbc_strct import create_dbc_strct
from pycamotk.pyCaMOtk.solve_fem import solve_fem_f
from pycamotk.pyCaMOtk.visualize_fem import visualize_fem
from pycamotk.pyCaMOtk.mesh import Mesh

sys.path.insert(0, '../source')
from source import TensorFEMCore_up
from source.GCNNModel import e2vcg2connectivity,Ns_Chebnet
from source.TensorFEMCore_up import Double,solve_fem_GCNN, ReshapeFix
from source import setup_prob_eqn_handcode
from source.FEM_ForwardModel import analyticalNS_u1, analyticalNS_u2,analyticalNS_p

U0=None
rho=1
nu=1
pltit=True
ndim=2
nvar=2
porder=2
etype='simplex'
# xcg_u=loadmat('D:/Matlab2019b/bin/mesh/Copy_of_NS/xcg.mat')
# e2vcg_u_gmsh=loadmat('D:/Matlab2019b/bin/mesh/Copy_of_NS/e2vcg.mat')
# xcg2_p=loadmat('D:/Matlab2019b/bin/mesh/Copy_of_NS/xcg2.mat')
# e2vcg2_p=loadmat('D:/Matlab2019b/bin/mesh/Copy_of_NS/e2vcg2.mat')
# e2bnd_up = loadmat('D:/Matlab2019b/bin/mesh/Copy_of_NS/e2bnd.mat')

xcg_u=loadmat('D:/Matlab2019b/bin/mesh/NS/xcg.mat')
e2vcg_u_gmsh=loadmat('D:/Matlab2019b/bin/mesh/NS/e2vcg.mat')
xcg2_p=loadmat('D:/Matlab2019b/bin/mesh/NS/xcg2.mat')
e2vcg2_p=loadmat('D:/Matlab2019b/bin/mesh/NS/e2vcg2.mat')
e2bnd_up = loadmat('D:/Matlab2019b/bin/mesh/NS/e2bnd.mat')
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


# 真解
extU=[]
for i in range(nnode):
	u_1 =[Double(analyticalNS_u1(xcg[:, i].reshape(-1, 1)).flatten().reshape(1, -1))]
	extU.append(u_1)
	u_2 = [Double(analyticalNS_u2(xcg[:, i].reshape(-1, 1)).flatten().reshape(1, -1))]
	extU.append(u_2)

for i in range(nnode_p):
	P = [Double(analyticalNS_p(xcg2[:, i].reshape(-1,1)).flatten().reshape(1,-1))]
	# extp.append(P)
	extU.append(P)

extUU=[]
for item in extU:
	extUU.append(item[0].item())
analyU=np.array(extUU)
uu = np.reshape(analyU[0:ndim * nnode], [ndim, nnode], order='F')
uuabs = np.sqrt(uu[0, :] ** 2 + uu[1, :] ** 2)
result_p = analyU[ndim*nnode:]

# #设置方程参数和自然边界条件
tb=lambda x,n,bnd,el,fc:np.zeros([ndim+1,1])#这个与弱形式的边界积分项有关
bnd2nbc=[0,1,2,3,4]#取值不影响，因为在计算残差时仅用来判断是否为边界即可
prob=setup_ins_base_handcode_NSf(ndim,lambda x,el:rho,
									lambda x,el:nu,
									tb,bnd2nbc)
# 施加边界
ndofU = ndim * nnode
ndofUP = ndofU + xcg2.shape[1]


'''单位方域'''
# L=1
# dbc_idx1 = []
# for i in range(nnode):
# 	if i in dbc_idx1:
# 		continue
# 	if xcg[0, i] < 1e-12 or xcg[0, i] > (L - 1e-12) or xcg[1,i]< 1e-12 or xcg[1,i] > (L- 1e-12):
# 		dbc_idx1.append(i)
# print("idx1=",dbc_idx1)
'''求解区域[0,1]x[-0.25,0]'''
L1=1
L2=-0.25
dbc_idx1 = []
for i in range(nnode):
	if i in dbc_idx1:
		continue
	if xcg[0, i] < 1e-12 or xcg[0, i] > (L1 - 1e-12) or xcg[1,i]>(-1e-12) or xcg[1,i] <(L2+ 1e-12):
		dbc_idx1.append(i)
print("idx1=",dbc_idx1)

dbc_val1 = []
for i in range(nnode):
	if i in dbc_idx1:
		dbc_val11 = [Double(analyticalNS_u1(xcg[:, i].reshape(-1,1)).flatten().reshape(1,-1))]
		dbc_val1.append(dbc_val11)

dbc_val2=[]
for i in range(nnode):
	if i in dbc_idx1:
		dbc_val22 = [Double(analyticalNS_u2(xcg[:, i].reshape(-1,1)).flatten().reshape(1,-1))]
		dbc_val2.append(dbc_val22)

#将所得数据转换为numpy数组
# u1
result_dbc_val1 = []
for item in dbc_val1:
	result_dbc_val1.append(item[0].item())

# u2
result_dbc_val2 = []
for item in dbc_val2:
	result_dbc_val2.append(item[0].item())

# 生成新边界编号及数值
dbc_idx=[2*i for i in dbc_idx1]
dbc_val=[i for i in result_dbc_val1]
for i in range(len(dbc_val1)):
	dbc_idx.append(2 * dbc_idx1[i] + 1)
	dbc_val.append(result_dbc_val2[i])

dbc_idx,I=np.unique(np.asarray(dbc_idx),return_index=True)
dbc_idx=[i for i in dbc_idx]
dbc_val=np.asarray(dbc_val)
dbc_val=dbc_val[I]
dbc_val=[i for i in dbc_val]

dbc_idx=np.asarray(dbc_idx)
dbc_val=np.asarray(dbc_val)
dbc=create_dbc_strct(ndofUP,dbc_idx,dbc_val)

neqn1 = ndim; neqn2 = 1
nvar1 = ndim; nvar2 = 1
femsp=create_femsp_cg_mixed2(prob,msh,
								neqn1,nvar1,
								porder,porder,
								e2vcg,e2vcg,
								neqn2,nvar2,
								porder-1,porder-1,
								e2vcg2,e2vcg2)
ldof2gdof = femsp.ldof2gdof_var.ldof2gdof
femsp.dbc=dbc

tol = 1.0e-8
maxit = 20
[U, info] = solve_fem_f('cg', msh.transfdatacontiguous,
					  femsp.elem, femsp.elem_data,
					  femsp.ldof2gdof_eqn.ldof2gdof,
					  femsp.ldof2gdof_var.ldof2gdof,
					  msh.e2e, femsp.spmat, dbc, U0,
					  tol, maxit)
print("U=",U)
print("information=",info)
idx_free = [i for i in range(len(U)) if i not in dbc_idx]
U0 = U[idx_free].reshape([-1, 1])
uv = np.reshape(U[0:ndim * nnode], [ndim, nnode], order='F')
p = U[ndim * nnode:]
uabs = np.sqrt(uv[0, :] ** 2 + uv[1, :] ** 2)
"""GCN"""
connectivity_uv = e2vcg2connectivity(e2vcg, 'ele')
connectivity_p = e2vcg2connectivity(e2vcg2, 'ele')
connectivity = torch.cat([connectivity_uv, connectivity_uv,
							  connectivity_p], axis=1)

prob = setup_prob_eqn_handcode.setup_ins_base_handcode_f \
		(ndim, lambda x, el: rho, lambda x, el: nu, tb, bnd2nbc)

femsp_gcnn = create_femsp_cg_mixed2(prob, msh,
										neqn1, nvar1,
										porder, porder,
										e2vcg, e2vcg,
										neqn2, nvar2,
										porder - 1, porder - 1,
										e2vcg2, e2vcg2)
LossF = []
fcn = lambda u_: TensorFEMCore_up.create_fem_resjac_f('cg',
												 u_, msh.transfdatacontiguous,
												 femsp_gcnn.elem, femsp_gcnn.elem_data,
												 femsp_gcnn.ldof2gdof_eqn.ldof2gdof,
												 femsp_gcnn.ldof2gdof_var.ldof2gdof,
												 msh.e2e, femsp_gcnn.spmat, dbc)
LossF.append(fcn)
ii = 0
Graph = []
# Ue=Double(analyU.flatten().reshape(-1,1))
Ue=Double(U.flatten().reshape(-1,1))
fcn_id = Double(np.asarray([ii]))
Ue_aug = torch.cat((fcn_id, Ue), axis=0)
#print("Ue_aug",Ue_aug)
xcg_gcnn = np.zeros((2, 2 * xcg.shape[1] + msh_.xcg.shape[1]))
for i in range(xcg.shape[1]):
	xcg_gcnn[:, 2 * i] = xcg[:, i]
	xcg_gcnn[:, 2 * i + 1] = xcg[:, i]
for i in range(msh_.xcg.shape[1]):
	xcg_gcnn[:, 2 * xcg.shape[1] + i] = msh_.xcg[:, i]


Uin = Double(xcg_gcnn.T)
graph = Data(x=Uin, y=Ue_aug, edge_index=connectivity)
Graph.append(graph)
DataList = [[Graph[0]]]
TrainDataloader = DataLoader(DataList, batch_size=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
split = [xcg.shape[1], msh_.xcg.shape[1], connectivity_uv.shape[1]]


model = Ns_Chebnet(split).to(device)
model = model.double()
[model, info,erlist_u,erlist_p] = solve_fem_GCNN(msh,TrainDataloader, LossF, model, tol, maxit)


torch.save(model, './Model.pth')
np.save('modelTrain.npy', info)
solution = model(Graph[0].to('cuda'))
solution = ReshapeFix(torch.clone(solution), [len(solution.flatten()), 1], 'C')
solution[dbc.dbc_idx] = Double(dbc.dbc_val.reshape([len(dbc.dbc_val), 1]))
solution = solution.detach().cpu().numpy()


uv_GCNN = np.reshape(solution[0:ndim * nnode], [ndim, nnode], order='F')
uabs_GCNN = np.sqrt(uv_GCNN[0, :] ** 2 + uv_GCNN[1, :] ** 2)
pGCNN = solution[ndim * nnode:]

# #计算GCN和FEM得绝对误差
# uabserror=[]
# uabsL2error=[]
# for i in range(nnode):
# 	uabs_error = abs(uabs_GCNN[i]-uabs[i])
# 	uabserror.append(uabs_error)
# 	uabs_L2_error =  (uabs_GCNN[i]-uabs[i])**2
# 	uabsL2error.append(uabs_L2_error)
# uabserror = np.array(uabserror)
# uabs_L2_er = np.sqrt (sum(uabsL2error))
# print("uL2=",uabs_L2_er)
# #abserror=np.max(abs_error)
# #print("abserror=",uabserror)
#
# pabserror=[]
# pabsL2error=[]
# for i in range(nnode_p):
# 	pabs_error = abs(pGCNN[i] - p[i])
# 	pabserror.append(pabs_error)
# 	pabs_L2_error = (pGCNN[i] - p[i]) ** 2
# 	pabsL2error.append(pabs_L2_error)
# pabserror = np.array(pabserror)
# pabs_L2_er = np.sqrt(sum(pabsL2error))
# print("pL2=",pabs_L2_er)


#计算GCN和真解得绝对误差
uabserror=[]
uabsL2error=[]
for i in range(nnode):
	uabs_error = abs(uabs_GCNN[i]-uuabs[i])
	uabserror.append(uabs_error)
	uabs_L2_error = (uabs_GCNN[i]-uuabs[i])**2
	uabsL2error.append(uabs_L2_error)
uabserror = np.array(uabserror)
uabs_L2_er = np.sqrt (sum(uabsL2error))
print("uL2=",uabs_L2_er)

pabserror=[]
pabsL2error=[]
for i in range(nnode_p):
	pabs_error = abs(pGCNN[i] - result_p[i])
	pabserror.append(pabs_error)
	pabs_L2_error = (pGCNN[i] - result_p[i]) ** 2
	pabsL2error.append(pabs_L2_error)
pabserror = np.array(pabserror)
pabs_L2_er = np.sqrt(sum(pabsL2error))
print("pL2=",pabs_L2_er)

#相对误差
print("erlist_u=",erlist_u)
np.savetxt('erlist_u.txt', erlist_u)
print("erlist_p=",erlist_p)
np.savetxt('erlist_p.txt', erlist_p)
#无穷范数
b_u = np.max(abs(uuabs-uabs_GCNN))
print("inf error u=",b_u)

b_p = np.max(abs(result_p-pGCNN))
print("inf error p=",b_p)

fig = plt.figure()
ax1 = plt.subplot(2, 2, 1)
visualize_fem(ax1, msh, uabs[e2vcg], {"plot_elem": False, "nref": 1}, [])
ax1.set_title('FEM U ')
# visualize_fem(ax1, msh, uuabs[e2vcg], {"plot_elem": False, "nref": 4}, [])
# ax1.set_title(' u')
ax2 = plt.subplot(2,2,2)
visualize_fem(ax2, msh, uuabs[e2vcg], {"plot_elem": False, "nref": 1}, [])
ax2.set_title('analytical U')
ax3 = plt.subplot(2, 2 ,3)
visualize_fem(ax3, msh, uabs_GCNN[e2vcg], {"plot_elem": False, "nref": 1}, [])
ax3.set_title('GCNN U ')
ax4 = plt.subplot(2, 2, 4)
#plt.xscale('symlog')
visualize_fem(ax4, msh, uabserror[e2vcg], {"plot_elem": False, "nref": 1}, [])
ax4.set_title('U abs error')
#ax3.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')
fig.tight_layout(pad=2)
plt.savefig('StenosisNSU.eps', bbox_inches='tight')
plt.savefig('StenosisNSU.pdf', bbox_inches='tight')

fig = plt.figure()
ax1 = plt.subplot(2, 2, 1)
visualize_fem(ax1, msh_, p[e2vcg2], {"plot_elem": False, "nref": 1}, [])
ax1.set_title('FEM P ')
# visualize_fem(ax1, msh_, result_p[e2vcg2], {"plot_elem": False, "nref": 4}, [])
# ax1.set_title(' p')
ax2 = plt.subplot(2,2,2)
visualize_fem(ax2, msh_, result_p[e2vcg2], {"plot_elem": False, "nref": 1}, [])
ax2.set_title('analytical P')
ax3 = plt.subplot(2, 2, 3)
visualize_fem(ax3, msh_, pGCNN[e2vcg2], {"plot_elem": False, "nref": 1}, [])
ax3.set_title('GCNN P ')
ax4 = plt.subplot(2, 2, 4)
visualize_fem(ax4, msh_, pabserror[e2vcg2], {"plot_elem": False, "nref": 1}, [])
ax4.set_title('P abs error')
fig.tight_layout(pad=2)
plt.savefig('StenosisNSP.eps', bbox_inches='tight')
plt.savefig('StenosisNSP.pdf', bbox_inches='tight')
