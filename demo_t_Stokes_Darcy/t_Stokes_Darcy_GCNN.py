import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.linalg import solve
import matplotlib
matplotlib.use('TkAgg')
from scipy.sparse import lil_matrix,coo_matrix
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset, DataLoader
from torch import tensor
import os
import sys
# 获取当前文件的上一级目录，也就是项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
sys.path.insert(0, '../Stokes_Ab')
from Stokes_Ab.Stokes_t_functions_data import Functions

sys.path.insert(0, '../matrixAb')
from matrixAb.generate_Gauss_reference_1D import generate_Gauss_reference_1D
from matrixAb.generate_M_T_triangle import generate_M_T_triangle
sys.path.insert(0, '../GCNN')
from GCNN.basis_Ab import Double,ReshapeFix

from GCNN.GATModel import T_connectivity
from GCNN.GATModel import Ns_Darcy_Chebnet
from GCNN.t_Stokes_functions_data_torch import Functions_torch_t
from GCNN.generate_residual_SD import generate_residual_u
# from GCNN.solve_GCNN import solve_GCNN
from GCNN.solve_GCNN_loss_tol_strong import solve_GCNN
sys.path.insert(0, os.path.abspath('../demo_t_Stokes_Darcy'))
from demo_t_Stokes_Darcy.t_Stokes_Darcy_fem import unsteady_Stokes_Darcy_fem

left_S=0;right_S=2;bottom_S=-2;top_S=0
left_D=0;right_D=2;bottom_D=0;top_D=2
h_partition_S=(2,2);h_partition_D=(2,2)
initial_t=0;end_t=1;dt=0.5
fix_pressure=1;Dirichlet_switch=1;Darcy_scaling_constant=1
theta =0.5

coef_data = Functions()
[M_partition_D, T_partition_D] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 1)
[M_basis_phi, T_basis_phi] = generate_M_T_triangle(left_D, right_D, bottom_D, top_D, h_partition_D, 2)
[M_basis_u, T_basis_u] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 2)
[M_basis_p, T_basis_p] = generate_M_T_triangle(left_S, right_S, bottom_S, top_S, h_partition_S, 1)
M_partition_S = M_basis_p
T_partition_S = T_basis_p
number_of_local_basis_phi = 6
number_of_local_basis_u = 6
savemat('M_partition_D.mat',{'M_partition_D':M_partition_D})
savemat('T_partition_D.mat',{'T_partition_D':T_partition_D})
savemat('M_basis_phi.mat',{'M_basis_phi':M_basis_phi})
savemat('T_basis_phi.mat',{'T_basis_phi':T_basis_phi})
savemat('M_basis_u.mat',{'M_basis_u':M_basis_u})
savemat('T_basis_u.mat',{'T_basis_u':T_basis_u})
savemat('M_basis_p.mat',{'M_basis_p':M_basis_p})
savemat('T_basis_p.mat',{'T_basis_p':T_basis_p})

#Step 1: Get some basic quantities which will be used in the code.
N_t=(end_t-initial_t)/dt
N1_partition_D = (right_D - left_D) / h_partition_D[0]
N2_partition_D = (top_D - bottom_D) / h_partition_D[1]
N1_partition_S = (right_S - left_S) / h_partition_S[0]
N2_partition_S = (top_S - bottom_S) / h_partition_S[1]

N1_basis_phi = N1_partition_D * 2
N2_basis_phi = N2_partition_D * 2
N1_basis_u = N1_partition_S * 2
N2_basis_u = N2_partition_S * 2
N1_basis_p = N1_partition_S
N2_basis_p = N2_partition_S

number_of_unknowns_Darcy = int((N1_basis_phi + 1) * (N2_basis_phi + 1))#M.shape[1]
number_of_FE_nodes_u = int((N1_basis_u + 1) * (N2_basis_u + 1))
number_of_FE_nodes_p = int((N1_basis_p + 1) * (N2_basis_p + 1))
number_of_unknowns_Stokes = int(2 * number_of_FE_nodes_u + number_of_FE_nodes_p)
number_of_all_unknowns = int(number_of_unknowns_Stokes + number_of_unknowns_Darcy)

# l2gT_D = create_ldof2gdof_cg(1,T_basis_phi)#每个节点对应一个自由度phi
#l2gT_D = T_basis_phi事实上两者相的
# l2gT_S = create_ldof2gdof_cg_mixed2(2,T_basis_u,1,T_basis_p)#速度场每个节点对应2个自由度u1,u2,压力场一个自由度p
# l2gT_DS = np.vstack((T_basis_phi,np.max(T_basis_phi)+1+l2gT_S.ldof2gdof))
'''FEM'''
phih, uh1,uh2,ph, r_D_boundary, r_Stokes_boundary, boundary_nodes_D, boundary_nodes_S\
	= unsteady_Stokes_Darcy_fem(theta, initial_t, end_t, dt,
    left_S, right_S, bottom_S, top_S,
    left_D, right_D, bottom_D, top_D,
    h_partition_S, h_partition_D,
    fix_pressure, Dirichlet_switch,
    Darcy_scaling_constant)

U_fem =np.vstack((phih, uh1, uh2, ph))
u = np.sqrt(uh1**2 + uh2**2)
savemat('phih.mat',{'phih':phih})
savemat('u.mat',{'u':u})
savemat('uh1.mat',{'uh1':uh1})
savemat('uh2.mat',{'uh2':uh2})
savemat('ph.mat',{'ph':ph})
savemat('U_fem.mat',{'U_fem':U_fem})


'''真解'''
# phih_D_1=[]
for n in range(int(N_t)):
    current_time = initial_t + dt * (n+1)
    coef_data_torch = Functions_torch_t()
    phih_D = Double(coef_data_torch.phi_exact_solution(Double(M_basis_phi[0, :]), Double(M_basis_phi[1, :]), current_time))
    phih_D = phih_D.cpu().numpy().reshape(-1, 1)
    phih_D = phih_D.tolist()  # 转换为Python列表
    phih_D.append(phih_D)  # 使用append
    u_1 = coef_data_torch.u1_exact_solution(Double(M_basis_u[0, :]), Double(M_basis_u[1, :]))
    u_1 = u_1.cpu().numpy().reshape(-1, 1)
    u_1 = u_1.tolist()
    u_1.append(u_1.copy)
    u_2 = coef_data_torch.u2_exact_solution(Double(M_basis_u[0, :]), Double(M_basis_u[1, :]), current_time)
    u_2 = u_2.cpu().numpy().reshape(-1, 1)
    u_2 = u_2.tolist()
    u_2.append(u_2.copy)
    p = coef_data_torch.p_exact_solution(Double(M_basis_u[0, :]), Double(M_basis_u[1, :]), current_time)
    # p = np.zeros((number_of_FE_nodes_p, 1)).cpu().numpy().reshape(-1, 1)
    p = p.cpu().numpy().reshape(-1, 1)
    p = p.tolist()
    p.append(p.copy)

#此时压力场真解=0 ，使用函数调用返回值维度为[1，1]，可以在真解为函数表达式时使用以下代码
# p = coef_data_torch.p_exact_solution(Double(M_basis_p[0,:]), Double(M_basis_p[1,:]))
# p = p.cpu().numpy().reshape(-1, 1)
analyU = np.vstack((phih_D, u_1, u_2,p))
u_exact = np.sqrt(u_1**2 + u_2**2)
savemat('phih_D.mat',{'phih_D':phih_D})
savemat('u_exact.mat',{'u_exact':u_exact})
savemat('u_1.mat',{'u_1':u_1})
savemat('u_2.mat',{'u_2':u_2})
savemat('p.mat',{'p':p})
savemat('analyU.mat',{'analyU':analyU})
U_boundary_D = analyU[boundary_nodes_D[1, :].astype(int), :]
U_boundary_S = analyU[boundary_nodes_S[2, :].astype(int), :]

'''计算FEM与真解之间的误差'''
D_L2_FEM = np.linalg.norm(phih_D - phih)
u_L2_FEM = np.linalg.norm(u_exact - u)#直接计算
p_L2_FEM = np.linalg.norm(p - ph)
D_Linf_FEM = np.max(abs(phih_D-phih))
u_Linf_FEM = np.max(abs(u_exact - u))
p_Linf_FEM = np.max(abs(p - ph))

'''GCNN'''
startime=time.time()
connectivity_phi = T_connectivity(T_basis_phi, 'ele')
connectivity_uv = T_connectivity(T_basis_u, 'ele')
connectivity_p = T_connectivity(T_basis_p, 'ele')
connectivity = torch.cat([connectivity_phi, connectivity_uv, connectivity_uv,
							  connectivity_p], axis=1)
#此处写入残差
LossF = []
fcn = lambda u_:generate_residual_u(u_,
    left_S, right_S, bottom_S, top_S,
    left_D, right_D, bottom_D, top_D,
    h_partition_S, h_partition_D,
    fix_pressure, Dirichlet_switch,
    Darcy_scaling_constant,U_boundary_D, U_boundary_S)#使用真解投入求解区域边界
LossF.append(fcn)

ii=0
Graph=[]
# Ue=Double(U_fem.flatten().reshape(-1,1))#fem解训练
Ue=Double(analyU.flatten().reshape(-1,1))
fcn_id=Double(np.asarray([ii]).reshape(-1,1))  # 将 fcn_idx 转换为形状为 (1, 1) 的张量
Ue_aug=torch.cat((fcn_id,Ue),axis=0)

xcg_gcnn=np.zeros((2,M_basis_phi.shape[1] + 2*M_basis_u.shape[1] + M_basis_p.shape[1]))
# 填充 M_basis_phi 的坐标
for i in range(M_basis_phi.shape[1]):
    xcg_gcnn[:, i] = M_basis_phi[:, i]
# 填充 M_basis_u 的坐标
for i in range(M_basis_u.shape[1]):
    xcg_gcnn[:, M_basis_phi.shape[1] + i] = M_basis_u[:, i]
    xcg_gcnn[:, M_basis_phi.shape[1] + M_basis_u.shape[1] + i] = M_basis_u[:, i]
# 填充 M_basis_p 的坐标
for i in range(M_basis_p.shape[1]):
    xcg_gcnn[:, M_basis_phi.shape[1] + 2 * M_basis_u.shape[1] + i] = M_basis_p[:, i]

Uin=Double(xcg_gcnn.T)
graph=Data(x=Uin,y=Ue_aug,edge_index=connectivity)
Graph.append(graph)
DataList=[[Graph[0]]]
TrainDataloader=DataLoader(DataList,batch_size=1)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
split=[M_basis_phi.shape[1],connectivity_phi.shape[1],
	   M_basis_u.shape[1], M_basis_p.shape[1], connectivity_uv.shape[1]]
model=Ns_Darcy_Chebnet(split).to(device)
model=model.double()

tol=1e-08
maxit_1=5000
# maxit_2=3
[model_1,Loss,Er_D,Er_u,Er_u1,Er_u2,Er_p,Erlist_D,Erlist_u,Erlist_u1,Erlist_u2,Erlist_p,erlist_u, erlist_u1, erlist_u2, erlist_p, erlist_D] = \
                                  solve_GCNN(number_of_unknowns_Darcy,number_of_FE_nodes_u,
                                             number_of_FE_nodes_p,TrainDataloader,
                                             LossF, model, tol, maxit_1)#这部分取消画图

# [model,Loss_MBGD,Er_D,Er_u,Er_u1,Er_u2,Er_p,Erlist_D,Erlist_u,Erlist_u1,Erlist_u2,Erlist_p,erlist_u, erlist_u1, erlist_u2, erlist_p, erlist_D] = \
#                                   solve_GCNN_MBGD(number_of_unknowns_Darcy,number_of_FE_nodes_u,
#                                              number_of_FE_nodes_p,TrainDataloader,
#                                              LossF, model_1, tol, maxit_2)
torch.save(model, './Model.pth')
solution = model(Graph[0].to('cuda'))
print('wallclock time = ',time.time()-startime)
solution = ReshapeFix(torch.clone(solution), [len(solution.flatten()), 1], 'C')

solution = solution.detach().cpu().numpy()
phih_GCNN = solution[:number_of_unknowns_Darcy]
uh1_GCNN = solution[number_of_unknowns_Darcy:number_of_unknowns_Darcy + number_of_FE_nodes_u]
uh2_GCNN = solution[number_of_unknowns_Darcy + number_of_FE_nodes_u:number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u]
ph_GCNN = solution[number_of_unknowns_Darcy + 2 * number_of_FE_nodes_u:number_of_all_unknowns]
uh_GCNN = np.sqrt(uh1_GCNN**2 + uh2_GCNN**2)
savemat('solution.mat',{'solution':solution})
savemat('phih_GCNN.mat',{'phih_GCNN':phih_GCNN})
savemat('uh1_GCNN.mat',{'uh1_GCNN':uh1_GCNN})
savemat('uh2_GCNN.mat',{'uh2_GCNN':uh2_GCNN})
savemat('ph_GCNN.mat',{'ph_GCNN':ph_GCNN})
savemat('uh_GCNN.mat',{'uh_GCNN':uh_GCNN})

#计算GCN和exact得绝对误差
Dabserror=[]
DabsL2error=[]
for i in range(number_of_unknowns_Darcy):
	Dabs_error = abs(phih_GCNN[i] - phih_D[i])
	Dabserror.append(Dabs_error)
	Dabs_L2_error = (phih_GCNN[i] - phih_D[i]) ** 2
	DabsL2error.append(Dabs_L2_error)
Dabserror = np.array(Dabserror)
Dabs_L2_er = np.sqrt(sum(DabsL2error))


uabserror=[]
uabsL2error=[]
for i in range(number_of_FE_nodes_u):
	uabs_error = abs(uh_GCNN[i]-u_exact[i])
	uabserror.append(uabs_error)
	uabs_L2_error =  (uh_GCNN[i]-u_exact[i])**2
	uabsL2error.append(uabs_L2_error)
uabserror = np.array(uabserror)
uabs_L2_er = np.sqrt (sum(uabsL2error))

pabserror=[]
pabsL2error=[]
for i in range(number_of_FE_nodes_p):
	pabs_error = abs(ph_GCNN[i] - p[i])
	pabserror.append(pabs_error)
	pabs_L2_error = (ph_GCNN[i] - p[i]) ** 2
	pabsL2error.append(pabs_L2_error)
pabserror = np.array(pabserror)
pabs_L2_er = np.sqrt(sum(pabsL2error))
savemat('Dabserror.mat',{'Dabserror':Dabserror})
savemat('pabserror.mat',{'pabserror':pabserror})
savemat('uabserror.mat',{'uabserror':uabserror})
savemat('Erlist_D.mat',{'Erlist_D':Erlist_D})
savemat('Erlist_u.mat',{'Erlist_u':Erlist_u})
savemat('Erlist_u1.mat',{'Erlist_u1':Erlist_u1})
savemat('Erlist_u2.mat',{'Erlist_u2':Erlist_u2})
savemat('Erlist_p.mat',{'Erlist_p':Erlist_p})

# print('Loss=',Loss)
savemat('Loss.mat',{'Loss':Loss})
#无穷范数
D_inf = np.max(abs(phih_GCNN-phih_D))
uh_inf = np.max(abs(uh_GCNN-u_exact))
ph_inf = np.max(abs(ph_GCNN-p))

'''LOSS'''#目前如果ADAM MBGD的训练次数一样时MBGD的LOSS更小
print('Loss=',Loss[-1])
# print('Loss_MBGD=',Loss_MBGD[-1])
# savemat('Loss_MBGD.mat',{'Loss_MBGD':Loss_MBGD})
savemat('Loss.mat',{'Loss':Loss})

'''FEM EXACT ERROR'''
print("D_L2_FEM =",D_L2_FEM)
print("D_Linf_FEM =",D_Linf_FEM)
print("u_L2_FEM =",u_L2_FEM)
print("u_Linf_FEM =",u_Linf_FEM)
print("p_Linf_FEM =",p_Linf_FEM)
print("p_L2_FEM =",p_L2_FEM)

'''GCN EXACT ERROR'''
print("D_L2=",Dabs_L2_er)
print('D_inf',D_inf)
print('erlist_D=',erlist_D)
print("uL2=",uabs_L2_er)
print('uh_inf',uh_inf)
print('erlist_u=',erlist_u)
print('erlist_u1=',erlist_u1)
print('erlist_u2',erlist_u2)
print("pL2=",pabs_L2_er)
print('ph_inf',ph_inf)
print('erlist_p=',erlist_p)