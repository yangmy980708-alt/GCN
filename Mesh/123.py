import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from pycamotk.pyCaMOtk.create_mesh_hcube import mesh_hcube
from pycamotk.pyCaMOtk.mesh import Mesh

'''NS'''
rho=1
nu=1
L1=1
L2 = -0.25
etype='hcube'
N_xNS_4=4
N_yNS_4=4
nelemNS_4=[4,4]
pltit=True
ndim=2
nvar=2
porder=2

"new exam NS"
mshNS_4 = mesh_hcube(etype, np.asarray([[0, L1], [-0.25,0]]), nelemNS_4, porder - 1).getmsh()
e2vcgNS_4 = mshNS_4.e2vcg
xcg2NS_4 = mshNS_4.xcg
#改变结构p
xcg2NS_4=mshNS_4.xcg
xcg2NS_4[1,N_yNS_4+1:2*(N_yNS_4+1)]=xcg2NS_4[1,N_yNS_4+1:2*(N_yNS_4+1)]*0 - (4/32)
xcg2NS_4[1,2*(N_yNS_4+1):3*(N_yNS_4+1)]=xcg2NS_4[1,2*(N_yNS_4+1):3*(N_yNS_4+1)] * 0 - (2/32)
xcg2NS_4[1,3*(N_yNS_4+1):4*(N_yNS_4+1)]=xcg2NS_4[1,3*(N_yNS_4+1):4*(N_yNS_4+1)] * 0 - (1/32)

nnode_p = mshNS_4.xcg.shape[1]
mshNS_4=Mesh(etype,xcg2NS_4,e2vcgNS_4,mshNS_4.e2bnd,2)
print("xcg2NS_4=",xcg2NS_4)
print("e2vcg2NS_4=",e2vcgNS_4)
# xcg2=msh_.xcg
# xcg2[1,N_y+1:2*(N_y+1)]=xcg2[1,N_y+1:2*(N_y+1)]*0 - (7/32)
# xcg2[1,2*(N_y+1):3*(N_y+1)]=xcg2[1,2*(N_y+1):3*(N_y+1)]*0 -(5/32)
# xcg2[1,3*(N_y+1):4*(N_y+1)]=xcg2[1,3*(N_y+1):4*(N_y+1)] * 0 - (3/32)
dataNS_4 = {
    'xcg2NS_4': xcg2NS_4,
    'e2vcgNS_4': e2vcgNS_4
}

# 保存数据到 .mat 文件
savemat('mesh_dataNS_4.mat', dataNS_4)

nelemNS_8=[8,8]
N_xNS_8=8
N_yNS_8=8
mshNS_8 = mesh_hcube(etype, np.asarray([[0, L1], [0,L2]]), nelemNS_8, porder - 1).getmsh()
e2vcgNS_8 = mshNS_8.e2vcg
xcgNS_8=mshNS_8.xcg
xcgNS_8[1,N_yNS_8+1:2*(N_yNS_8+1)]=xcgNS_8[1,N_yNS_8+1:2*(N_yNS_8+1)]*0 - (11/48)
xcgNS_8[1,2*(N_yNS_8+1):3*(N_yNS_8+1)]=xcgNS_8[1,2*(N_yNS_8+1):3*(N_yNS_8+1)]*0 -(10/48)
xcgNS_8[1,3*(N_yNS_8+1):4*(N_yNS_8+1)]=xcgNS_8[1,3*(N_yNS_8+1):4*(N_yNS_8+1)] * 0 - (9/48)
xcgNS_8[1,4*(N_yNS_8+1):5*(N_yNS_8+1)]=xcgNS_8[1,4*(N_yNS_8+1):5*(N_yNS_8+1)] * 0 - (8/48)
xcgNS_8[1,5*(N_yNS_8+1):6*(N_yNS_8+1)]=xcgNS_8[1,5*(N_yNS_8+1):6*(N_yNS_8+1)] * 0 - (6/48)
xcgNS_8[1,6*(N_yNS_8+1):7*(N_yNS_8+1)]=xcgNS_8[1,6*(N_yNS_8+1):7*(N_yNS_8+1)] * 0 - (4/48)
xcgNS_8[1,7*(N_yNS_8+1):8*(N_yNS_8+1)]=xcgNS_8[1,7*(N_yNS_8+1):8*(N_yNS_8+1)] * 0 - (2/48)
mshNS_8=Mesh(etype,xcgNS_8,e2vcgNS_8,mshNS_8.e2bnd,2)

dataNS_8 = {
    'xcg2NS_8': xcgNS_8,
    'e2vcgNS_8': e2vcgNS_8
}

# 保存数据到 .mat 文件
savemat('mesh_dataNS_8.mat', dataNS_8)

'''stokes2'''
L1_stokes=np.pi
L2_stokes=1
N_x_stokes_4=4
N_y_stokes_4=4
nelem_stokes_4=[4,4]
msh_stokes_4 = mesh_hcube(etype, np.asarray([[0, L1_stokes], [0,L2_stokes]]), nelem_stokes_4, porder - 1).getmsh()
xcg_stokes_4=msh_stokes_4.xcg
e2vcg2_stokes_4 = msh_stokes_4.e2vcg
xcg_stokes_4[1,N_y_stokes_4+1:2*(N_y_stokes_4+1)]=xcg_stokes_4[1,N_y_stokes_4+1:2*(N_y_stokes_4+1)]*0 + (2/8)
xcg_stokes_4[1,2*(N_y_stokes_4+1):3*(N_y_stokes_4+1)]=xcg_stokes_4[1,2*(N_y_stokes_4+1):3*(N_y_stokes_4+1)]*0 + (4/8)
xcg_stokes_4[1,3*(N_y_stokes_4+1):4*(N_y_stokes_4+1)]=xcg_stokes_4[1,3*(N_y_stokes_4+1):4*(N_y_stokes_4+1)]*0 + (5/8)
msh_stokes_4=Mesh(etype,xcg_stokes_4,e2vcg2_stokes_4,msh_stokes_4.e2bnd,2)

data_stokes_4 = {
    'xcg2_stokes_4': xcg_stokes_4,
    'e2vcg_stokes_4': e2vcg2_stokes_4
}

# 保存数据到 .mat 文件
savemat('mesh_data_stokes_4.mat', data_stokes_4)

N_x_stokes_8=8
N_y_stokes_8=8
nelem_stokes_8=[8,8]
msh_stokes_8 = mesh_hcube(etype, np.asarray([[0, L1_stokes], [0,L2_stokes]]), nelem_stokes_8, porder - 1).getmsh()
xcg_stokes_8=msh_stokes_8.xcg
e2vcg2_stokes_8 = msh_stokes_8.e2vcg
xcg_stokes_8[1,N_y_stokes_8+1:2*(N_y_stokes_8+1)]=xcg_stokes_8[1,N_y_stokes_8+1:2*(N_y_stokes_8+1)]*0 + (3/16)
xcg_stokes_8[1,2*(N_y_stokes_8+1):3*(N_y_stokes_8+1)]=xcg_stokes_8[1,2*(N_y_stokes_8+1):3*(N_y_stokes_8+1)]*0 + (5/16)
xcg_stokes_8[1,3*(N_y_stokes_8+1):4*(N_y_stokes_8+1)]=xcg_stokes_8[1,3*(N_y_stokes_8+1):4*(N_y_stokes_8+1)]*0 + (7/16)
xcg_stokes_8[1,4*(N_y_stokes_8+1):5*(N_y_stokes_8+1)]=xcg_stokes_8[1,4*(N_y_stokes_8+1):5*(N_y_stokes_8+1)]*0 + (8/16)
xcg_stokes_8[1,5*(N_y_stokes_8+1):6*(N_y_stokes_8+1)]=xcg_stokes_8[1,5*(N_y_stokes_8+1):6*(N_y_stokes_8+1)]*0 + (9/16)
xcg_stokes_8[1,6*(N_y_stokes_8+1):7*(N_y_stokes_8+1)]=xcg_stokes_8[1,6*(N_y_stokes_8+1):7*(N_y_stokes_8+1)] * 0 + (11/16)
xcg_stokes_8[1,7*(N_y_stokes_8+1):8*(N_y_stokes_8+1)]=xcg_stokes_8[1,7*(N_y_stokes_8+1):8*(N_y_stokes_8+1)] * 0 + (14/16)
msh_stokes_8=Mesh(etype,xcg_stokes_8,e2vcg2_stokes_8,msh_stokes_8.e2bnd,2)

data_stokes_8 = {
    'xcg2_stokes_8': xcg_stokes_8,
    'e2vcg_stokes_8': e2vcg2_stokes_8
}

# 保存数据到 .mat 文件
savemat('mesh_data_stokes_8.mat', data_stokes_8)

'''NS dif nu=100'''
rho=1
nu=100
L=1
etype='hcube'
N_x_NS_dif_nu_4=4
N_y_NS_dif_nu_4=4
nelem_NS_dif_nu_4=[4,4]
porder=2
pltit=True
ndim_NS_dif_nu_4=2
nvar_NS_dif_nu_4=2
inletVelocity=1
s=0.4
msh_NS_dif_nu_4 = mesh_hcube(etype, np.asarray([[0, L], [0, L]]), nelem_NS_dif_nu_4, porder - 1).getmsh()
e2vcg2_NS_dif_nu_4 = msh_NS_dif_nu_4.e2vcg
xcg2_NS_dif_nu_4=msh_NS_dif_nu_4.xcg
xcg2_NS_dif_nu_4[1,N_y_NS_dif_nu_4+1:2*(N_y_NS_dif_nu_4+1)]=xcg2_NS_dif_nu_4[1,N_y_NS_dif_nu_4+1:2*(N_y_NS_dif_nu_4+1)]*0 + (3/8)
xcg2_NS_dif_nu_4[1,2*(N_y_NS_dif_nu_4+1):3*(N_y_NS_dif_nu_4+1)]=xcg2_NS_dif_nu_4[1,2*(N_y_NS_dif_nu_4+1):3*(N_y_NS_dif_nu_4+1)]*0 + (5/8)
xcg2_NS_dif_nu_4[1,3*(N_y_NS_dif_nu_4+1):4*(N_y_NS_dif_nu_4+1)]=xcg2_NS_dif_nu_4[1,3*(N_y_NS_dif_nu_4+1):4*(N_y_NS_dif_nu_4+1)]*0 + (7/8)

msh_NS_dif_nu_4=Mesh(etype,xcg2_NS_dif_nu_4,e2vcg2_NS_dif_nu_4,msh_NS_dif_nu_4.e2bnd,2)

data_NS_dif_nu_4 = {
    'xcg2_NS_dif_nu_4': xcg2_NS_dif_nu_4,
    'e2vcg_NS_dif_nu_4': e2vcg2_NS_dif_nu_4
}

# 保存数据到 .mat 文件
savemat('mesh_data_NS_dif_nu_4.mat', data_NS_dif_nu_4)


N_x_NS_dif_nu_8=8
N_y_NS_dif_nu_8=8
nelem_NS_dif_nu_8=[8,8]
porder=2
pltit=True
ndim_NS_dif_nu_8=2
nvar_NS_dif_nu_8=2
inletVelocity=1
s=0.4
msh_NS_dif_nu_8 = mesh_hcube(etype, np.asarray([[0, L], [0, L]]), nelem_NS_dif_nu_8, porder - 1).getmsh()
e2vcg2_NS_dif_nu_8 = msh_NS_dif_nu_8.e2vcg
xcg2_NS_dif_nu_8=msh_NS_dif_nu_8.xcg
xcg2_NS_dif_nu_8[1,N_y_NS_dif_nu_8+1:2*(N_y_NS_dif_nu_8+1)]=xcg2_NS_dif_nu_8[1,N_y_NS_dif_nu_8+1:2*(N_y_NS_dif_nu_8+1)]*0 + (3/16)
xcg2_NS_dif_nu_8[1,2*(N_y_NS_dif_nu_8+1):3*(N_y_NS_dif_nu_8+1)]=xcg2_NS_dif_nu_8[1,2*(N_y_NS_dif_nu_8+1):3*(N_y_NS_dif_nu_8+1)]*0 + (6/16)
xcg2_NS_dif_nu_8[1,3*(N_y_NS_dif_nu_8+1):4*(N_y_NS_dif_nu_8+1)]=xcg2_NS_dif_nu_8[1,3*(N_y_NS_dif_nu_8+1):4*(N_y_NS_dif_nu_8+1)]*0 + (8/16)
xcg2_NS_dif_nu_8[1,4*(N_y_NS_dif_nu_8+1):5*(N_y_NS_dif_nu_8+1)]=xcg2_NS_dif_nu_8[1,4*(N_y_NS_dif_nu_8+1):5*(N_y_NS_dif_nu_8+1)]*0 + (10/16)
xcg2_NS_dif_nu_8[1,5*(N_y_NS_dif_nu_8+1):6*(N_y_NS_dif_nu_8+1)]=xcg2_NS_dif_nu_8[1,5*(N_y_NS_dif_nu_8+1):6*(N_y_NS_dif_nu_8+1)]*0 + (12/16)
xcg2_NS_dif_nu_8[1,6*(N_y_NS_dif_nu_8+1):7*(N_y_NS_dif_nu_8+1)]=xcg2_NS_dif_nu_8[1,6*(N_y_NS_dif_nu_8+1):7*(N_y_NS_dif_nu_8+1)] * 0 + (14/16)
xcg2_NS_dif_nu_8[1,7*(N_y_NS_dif_nu_8+1):8*(N_y_NS_dif_nu_8+1)]=xcg2_NS_dif_nu_8[1,7*(N_y_NS_dif_nu_8+1):8*(N_y_NS_dif_nu_8+1)] * 0 + (15/16)

msh_NS_dif_nu_8=Mesh(etype,xcg2_NS_dif_nu_8,e2vcg2_NS_dif_nu_8,msh_NS_dif_nu_8.e2bnd,2)

data_NS_dif_nu_8= {
    'xcg2_NS_dif_nu_8': xcg2_NS_dif_nu_8,
    'e2vcg_NS_dif_nu_8': e2vcg2_NS_dif_nu_8
}

# 保存数据到 .mat 文件
savemat('mesh_data_NS_dif_nu_8.mat', data_NS_dif_nu_8)