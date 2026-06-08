import torch
from torch import tensor
import numpy as np
import pdb

import sys
sys.path.insert(0, '../pycamotk')
from pycamotk.pyCaMOtk.create_mesh_hcube import mesh_hcube

sys.path.insert(0, '../source')
from source import TensorFEMCore
# from TensorFEMCore import Double
# from FEM_ForwardModel import analyticalNS_f1,analyticalNS_f2
from source.TensorFEMCore import Double, ReshapeFix

"""
####Possion Equation
"""
class setup_linelptc_sclr_base_handcode(object):
	"""docstring for setup_linelptc_sclr_base_handcode"""
	def __init__(self,ndim,K,f,Qb,bnd2nbc):
		self.ndim=ndim
		self.K=K
		self.f=f
		self.Qb=Qb
		self.bnd2nbc=bnd2nbc

		self.I=np.eye(self.ndim)
		if self.K==None:
			self.K=lambda x,el: self.I.reshape(self.ndim**2,1,order='F')
		if self.f==None:
			self.f=lambda x,el: 0
		if self.Qb==None:
			self.Qb=lambda x,n,bnd,el,fc: 0

		self.eqn=LinearEllipticScalarBaseHandcode()
		self.vol_pars_fcn=lambda x,el:np.vstack((self.K(x, el),self.f(x, el),np.nan))
		#bnd_pars_fcn(xqf[:,k],n[:,k],bnd,e,f)
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack((self.K(x,el),
														  self.f(x,el),
														  self.Qb(x,n,bnd,el,fc)))
		#face_pars_fcn(xqf[:,k],n[:,k],e,f,ep,fp)
		"假设后面的几个参数全为1"
		self.dfine = 1
		self.face_pars_fcn = lambda x,n,el1,fc1,el2,fc2:np.vstack((self.K(x,el1),
														  self.f(x,el1),
														  self.dfine,self.dfine,self.dfine,self.dfine))


class LinearEllipticScalarBaseHandcode(object):
	"""docstring for LinearEllipticScalarBaseHandcode"""
	def __init__(self):
		self.neqn=1
		self.nvar=1
		self.ncomp=1

	def srcflux(self,UQ,pars,x):
		"""
		eval_linelptc_base_handcode_srcflux
		"""
		# Extract information from input
		q=UQ[0,1:]#UQ = （u u_x u_y）
		q=ReshapeFix(q,[len(q),1])
		self.ndim=len(q)
		try:#try:尝试执行的代码
			k=np.reshape(pars[0:self.ndim**2],
					 (self.ndim,self.ndim),order='F')
		except:#except:出现错误的处理
			k=torch.reshape(pars[0:self.ndim**2],
					 (self.ndim,self.ndim))
		f=pars[self.ndim**2]

		try:
			temp_flag=(f.requires_grad)
			f=f.reshape([1,1])
		except:
			f=torch.tensor(f).double().to('cuda').reshape([1,1])

		k_ml=torch.tensor(k).double().to('cuda')
		# Define flux and source
		SF=torch.cat((f,-1*torch.mm(k_ml,q)),axis=0)
		#torch.mm(k_ml,q)  是两个矩阵 k_ml与q相乘

		# Define partial derivative
		dSFdU=np.zeros([self.neqn, self.ndim+1, self.ncomp,self.ndim+1])#(1,3,1,3)
		try:
			dSFdU[:,1:,:,1:]=np.reshape(-1*k,[self.neqn, self.ndim,self.ncomp,self.ndim])
			# print("dsfdu",dSFdU)
			# [0 0 0
			#  0 -1 0
			#  0 0 -1]
		except:
			k=k.detach().cpu().numpy()
			dSFdU[:,1:,:,1:]=np.reshape(-1*k,[self.neqn, self.ndim,self.ncomp,self.ndim])
		dSFdU=torch.tensor(dSFdU).double().to('cuda')
		return SF, dSFdU

	def bndstvcflux(self,nbcnbr,UQ,pars,x,n):
		nvar=UQ.shape[0]
		ndim=UQ.shape[1]-1

		Ub=UQ[:,0]#真解
		dUb=np.zeros([nvar,nvar,self.ndim+1])
		dUb[:,:,0]=np.eye(nvar)

		Fn=pars[ndim**2+1]
		dFn=np.zeros([nvar,nvar,self.ndim+1])
		dUb=Double(dUb)
		Fn=Double(Fn)
		dFn=Double(dFn)
		return Ub,dUb,Fn,dFn