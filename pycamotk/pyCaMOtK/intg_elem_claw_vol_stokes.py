from __future__ import print_function
import numpy as np
import sys

sys.path.insert(0, '../source')
# import TensorFEMCore
from source.FEM_ForwardModel import analyticalstokes_f1, analyticalstokes_f2
from source.TensorFEMCore_stokes import Double

import pdb
################################################################################
def intg_elem_claw_vol_f(Ue,transf_data,elem,elem_data,e):
	[neqn_per_elem,neqn,ndimP1,nq]=elem.Tv_eqn_ref.shape
	[nvar_per_elem,nvar,_,_]=elem.Tv_var_ref.shape
	ndim=ndimP1-1
	wq=elem.wq
	detG=transf_data.detG[:,e]
	Tvar=elem_data.Tv_var_phys[:,:,:,:,e].reshape([nvar_per_elem,nvar*(ndim+1)*nq],
												  order='F')

	Re=np.zeros([neqn_per_elem,1])
	dRe=np.zeros([neqn_per_elem,nvar_per_elem])#尺寸[22,22]

	UQq=np.reshape(Tvar.T.dot(Ue),[nvar,ndim+1,nq],order='F')#UQq=[u u_x u_y]
	w=wq*detG
	for k in range(nq):
		Teqn=elem_data.Tv_eqn_phys[:,:,:,k,e].reshape([neqn_per_elem,neqn*(ndim+1)],
			                                      order='F')
		Tvar=elem_data.Tv_var_phys[:,:,:,k,e].reshape([nvar_per_elem,nvar*(ndim+1)],
			                                      order='F')
		x=transf_data.xq[:,k,e]
		#print("x",x)
		f = function_of_the_right_f(x)
		pars=elem_data.vol_pars[:,k,e]
		SF, dSFdUQ = elem.eqn.srcflux(UQq[:, :, k], f, pars, x)
		SF=SF.flatten(order='F')
		dSFdUQ=np.reshape(dSFdUQ,[neqn*(ndim+1),nvar*(ndim+1)],order='F')
		Re=Re-w[k]*Teqn.dot(SF).reshape(Re.shape,order='F')
		dRe=dRe-w[k]*(Teqn.dot(dSFdUQ.dot(Tvar.T)))
	return Re, dRe

def function_of_the_right_f(xq):
	nu = 1
	#xq=xq.reshape(-1,1)
	f1 = [Double(analyticalstokes_f1(xq.reshape(-1, 1), nu).flatten().reshape(1, -1))]
	f2 = [Double(analyticalstokes_f2(xq.reshape(-1, 1), nu).flatten().reshape(1, -1))]
	result1 = []
	for item in f1:
		result1.append(item[0].item())

	result2 = []
	for item in f2:
		result2.append(item[0].item())

	f_ = np.vstack((result1, result2))
	return f_


