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
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack((self.K(x,el),
														  self.f(x,el),
														  self.Qb(x,n,bnd,el,fc)))


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

"""
####Linear Elasticity Equation
"""

class setup_linelast_base_handcode(object):
	"""docstring for setup_linelast_base_handcode"""
	def __init__(self,ndim,lam,mu,f,tb,bnd2nbc):
		self.bnd2nbc=bnd2nbc
		self.eqn=LinearElasticityBaseHandcode(ndim)
		self.vol_pars_fcn=lambda x, el: np.vstack((lam(x,el),
			                                      mu(x,el),
			                                      f(x,el),
			                                      np.zeros([ndim,1])+np.nan))
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack((lam(x, el),
			                                              mu(x, el),
			                                              f(x, el),
			                                              tb(x, n, bnd, el, fc)))

		

class LinearElasticityBaseHandcode(object):
	"""docstring for LinearElasticityBaseHandcode"""
	def __init__(self,ndim):
		self.neqn=ndim
		self.nvar=ndim
		self.bndstvcflux=\
		lambda nbcnbr, UQ, pars, x, n:\
		eval_linelast_base_handcode_bndstvc_intr_bndflux_pars(UQ, pars, x, n)
		self.srcflux=lambda UQ,pars,x:\
		eval_linelast_base_handcode_srcflux(UQ, pars, x)

def eval_linelast_base_handcode_srcflux(UQ, pars, x):
	#print("uq",UQ.shape)
	q=UQ[:,1:]
	ndim=q.shape[0]
	# Define information regarding size of the system
	neqn=ndim
	ncomp=ndim

	# Extract parameters
	lam=pars[0]
	mu=pars[1]
	f=pars[2:2+ndim]
	F=-lam*torch.trace(q)*(Double(np.eye(ndim)))-mu*(q+q.T)
	try:
		S=Double(f.reshape([ndim,1],order='F'))
	except:
		S=f.reshape([ndim,1])
	#pdb.set_trace()
	SF=torch.cat((S,F),axis=1)
	#print('SF=',SF)
	dSFdU=Double(np.zeros([neqn,ndim+1,ncomp,ndim+1]))
	for i in range(ndim):
		for j in range(ndim):
			dSFdU[i,1+i,j,1+j]=dSFdU[i,1+i,j,1+j]-lam
			dSFdU[i,1+j,i,1+j]=dSFdU[i,1+j,i,1+j]-mu
			dSFdU[i,1+j,j,1+i]=dSFdU[i,1+j,j,1+i]-mu
	return SF, dSFdU


def eval_linelast_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n):
	nvar=UQ.shape[0]
	ndim=UQ.shape[1]-1

	Ub=UQ[:,0]
	dUb=np.zeros([nvar,nvar,ndim+1])
	dUb[:,:,0]=np.eye(nvar)
	Fn=-pars[-ndim:]
	dFn=np.zeros([nvar,nvar,ndim+1])
	dUb=Double(dUb)
	Fn=ReshapeFix(Double(Fn),[len(Fn),1],order='F')
	dFn=Double(dFn)
	#print('Fn=',Fn)
	return Ub,dUb,Fn,dFn



"""
#### Inconpressible Navier Stokes Equation
"""
# class setup_ins_base_handcode(object):
# 	"""docstring for setup_ins_base_handcode"""
# 	def __init__(self,ndim,rho,nu,tb,bnd2nbc):
# 		self.eqn=IncompressibleNavierStokes(ndim)
# 		self.bnd2nbc=bnd2nbc
# 		self.vol_pars_fcn=lambda x,el:np.vstack([rho(x, el),
# 			                                     nu(x, el),
# 			                                     np.zeros([ndim+1,1])+np.nan])
# 		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack([rho(x,el),
# 			 										      nu(x,el),
# 			 										      tb(x,n,bnd,el,fc)])
#
#
#
# class IncompressibleNavierStokes(object):
# 	"""docstring for IncompressibleNavierStokes"""
# 	def __init__(self,ndim):
# 		self.ndim=ndim
# 		self.nvar=ndim+1
# 		self.srcflux=lambda UQ,pars,x:\
# 		             eval_ins_base_handcode_srcflux(UQ,pars,x)
# 		self.bndstvcflux=lambda nbcnbr,UQ,pars,x,n:\
# 					     eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n)
#
#
#
# def eval_ins_base_handcode_srcflux(UQ,pars,x):
# 	#print("UQ",UQ)
# 	#print("pars",pars)
# 	u=UQ[:,0]; q=UQ[:,1:]
# 	#print("u",u)
# 	#print("q",q)
# 	#print("UQ",UQ)
# 	#print("pars",pars)
# 	ndim=u.shape[0]-1
# 	neqn=ndim+1
# 	ncomp=ndim+1
# 	rho=pars[0]
# 	nu=pars[1]
# 	v=u[0:ndim]# 提取前ndim个元素
#
# 	v=ReshapeFix(v,[len(v),1],'F')
# 	#v=v.reshape([-1,1],order='F')
#
# 	p=u[-1]
# 	dv=q[0:ndim,:]
# 	#print("dv",dv)
# 	#print("v",v)
#
# 	S=torch.cat([-rho*torch.mm(dv,v),-torch.trace(dv).reshape([1,1])],axis=0)
# 	#S=np.vstack([-rho*dv.dot(v),-np.trace(dv)])
#
# 	#F=np.vstack([-rho*nu*dv+p*np.eye(ndim),
# 	#	         np.zeros([1,ndim])])
# 	F=torch.cat([-rho*nu*dv+p*torch.eye(ndim).double().to('cuda'),
# 	             torch.zeros([1,ndim]).double().to('cuda')],axis=0)
#
#
# 	#SF= np.hstack([S,F])
# 	SF=torch.cat([S,F],axis=1)
#
# 	dSFdUQ=np.zeros([neqn,ndim+1,ncomp,ndim+1])
# 	dSFdUQ[:,0,:,0]=np.vstack([np.hstack([-rho*dv.detach().cpu().numpy(),np.zeros([ndim,1])]), np.zeros([1,ndim+1])])
# 	for i in range(ndim):
# 		dSFdUQ[i,0,i,1:]=-rho*v.detach().cpu().numpy().reshape(dSFdUQ[i,0,i,1:].shape,order='F')
# 	dSFdUQ[-1,0,0:-1,1:]=np.reshape(-np.eye(ndim),[1,ndim,ndim],order='F')
# 	dSFdUQ[0:-1,1:,-1,0]=np.eye(ndim)
# 	for i in range(ndim):
# 		for j in range(ndim):
# 			dSFdUQ[i,1+j,i,1+j]=dSFdUQ[i,1+j,i,1+j]-rho*nu
# 	dSFdUQ=Double(dSFdUQ)
# 	return SF,dSFdUQ
#
# def eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ,pars,x,n):
# 	nvar=UQ.shape[0]
# 	ndim=UQ.shape[1]-1
# 	Ub=UQ[:,0]
# 	dUb=np.zeros([nvar,nvar,ndim+1])
# 	dUb[:,:,0]=np.eye(nvar)
# 	Fn=-pars[-ndim-1:].reshape([-1,1])
# 	dFn=np.zeros([nvar,nvar,ndim+1])
# 	return Ub,Double(dUb),Double(Fn),Double(dFn)
"paper NS with f=0"
class setup_ins_base_handcode(object):
	"""docstring for setup_ins_base_handcode"""

	def __init__(self, ndim, rho, nu, tb, bnd2nbc):
		self.eqn = IncompressibleNavierStokes(ndim)
		self.bnd2nbc = bnd2nbc
		self.vol_pars_fcn = lambda x, el: np.vstack([rho(x, el),
													 nu(x, el),
													 np.zeros([ndim + 1, 1]) + np.nan])
		self.bnd_pars_fcn = lambda x, n, bnd, el, fc: np.vstack([rho(x, el),
																 nu(x, el),
																 tb(x, n, bnd, el, fc)])


class IncompressibleNavierStokes(object):
	"""docstring for IncompressibleNavierStokes"""

	def __init__(self, ndim):
		self.ndim = ndim
		self.nvar = ndim + 1
		self.srcflux = lambda UQ,f, pars, x: \
			eval_ins_base_handcode_srcflux(UQ,f, pars, x)
		self.bndstvcflux = lambda nbcnbr, UQ, pars, x, n: \
			eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ, pars, x, n)


def eval_ins_base_handcode_srcflux(UQ,f, pars, x):
	u = UQ[:, 0];
	q = UQ[:, 1:]
	ndim = u.shape[0] - 1
	neqn = ndim + 1
	ncomp = ndim + 1
	rho = pars[0]
	nu = pars[1]
	v = u[0:ndim]  # 提取前ndim个元素
	v = ReshapeFix(v, [len(v), 1], 'F')
	p = u[-1]
	dv = q[0:ndim, :]
	f_ = torch.tensor(f).reshape(-1, 1).double().to('cuda')

	S = torch.cat([-rho * torch.mm(dv, v)+f_, -torch.trace(dv).reshape([1, 1])], axis=0)
	F = torch.cat([-rho * nu * dv + p * torch.eye(ndim).double().to('cuda'),
				   torch.zeros([1, ndim]).double().to('cuda')], axis=0)

	# SF= np.hstack([S,F])
	SF = torch.cat([S, F], axis=1)

	dSFdUQ = np.zeros([neqn, ndim + 1, ncomp, ndim + 1])
	dSFdUQ[:, 0, :, 0] = np.vstack(
		[np.hstack([-rho * dv.detach().cpu().numpy(), np.zeros([ndim, 1])]), np.zeros([1, ndim + 1])])
	for i in range(ndim):
		dSFdUQ[i, 0, i, 1:] = -rho * v.detach().cpu().numpy().reshape(dSFdUQ[i, 0, i, 1:].shape, order='F')
	dSFdUQ[-1, 0, 0:-1, 1:] = np.reshape(-np.eye(ndim), [1, ndim, ndim], order='F')
	dSFdUQ[0:-1, 1:, -1, 0] = np.eye(ndim)
	for i in range(ndim):
		for j in range(ndim):
			dSFdUQ[i, 1 + j, i, 1 + j] = dSFdUQ[i, 1 + j, i, 1 + j] - rho * nu
	dSFdUQ = Double(dSFdUQ)
	return SF, dSFdUQ


def eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ, pars, x, n):
	nvar = UQ.shape[0]
	ndim = UQ.shape[1] - 1
	Ub = UQ[:, 0]
	dUb = np.zeros([nvar, nvar, ndim + 1])
	dUb[:, :, 0] = np.eye(nvar)
	Fn = -pars[-ndim - 1:].reshape([-1, 1])
	dFn = np.zeros([nvar, nvar, ndim + 1])
	return Ub, Double(dUb), Double(Fn), Double(dFn)


"""
 Navier Stokes Equation HXM PPT
"""


class setup_ins_base_handcode_f(object):
	"""docstring for setup_ins_base_handcode"""

	def __init__(self, ndim, rho, nu, tb, bnd2nbc):
		self.eqn = IncompressibleNavierStokes_f(ndim)
		self.bnd2nbc = bnd2nbc
		self.vol_pars_fcn = lambda x, el: np.vstack([rho(x, el),
													 nu(x, el),
													 np.zeros([ndim + 1, 1]) + np.nan])
		self.bnd_pars_fcn = lambda x, n, bnd, el, fc: np.vstack([rho(x, el),
																 nu(x, el),
																 tb(x, n, bnd, el, fc)])


class IncompressibleNavierStokes_f(object):
	"""docstring for IncompressibleNavierStokes"""

	def __init__(self, ndim):
		self.ndim = ndim
		self.nvar = ndim + 1
		self.srcflux = lambda UQ,f, pars, x: \
			eval_ins_base_handcode_srcflux_NSf(UQ,f, pars, x)
		self.bndstvcflux = lambda nbcnbr, UQ, pars, x, n: \
			eval_ins_base_handcode_bndstvc_intr_bndflux_pars_NSf(UQ, pars, x, n)


def eval_ins_base_handcode_srcflux_NSf(UQ,f, pars, x):
	u = UQ[:, 0]
	q = UQ[:, 1:]
	ndim = u.shape[0] - 1
	neqn = ndim + 1
	ncomp = ndim + 1
	rho = pars[0]
	nu = pars[1]
	v = u[0:ndim]
	v = ReshapeFix(v, [len(v), 1], 'F')
	# v=v.reshape([-1,1],order='F')
	p = u[-1]
	dv = q[0:ndim, :]
	f_=torch.tensor(f).reshape(-1,1).double().to('cuda')
	# print("f_",f_)
	Dv = (dv + dv.T) / 2

	S = torch.cat([-rho*torch.mm(dv,v)+f_,-torch.trace(dv).reshape([1, 1])], axis=0)
	F = torch.cat([-2 * rho * nu * Dv + p * torch.eye(ndim).double().to('cuda'),
				   torch.zeros([1, ndim]).double().to('cuda')], axis=0)
	# SF= np.hstack([S,F])
	SF = torch.cat([S, F], axis=1)

	dSFdUQ = np.zeros([neqn, ndim + 1, ncomp, ndim + 1])
	dSFdUQ[:, 0, :, 0] = np.vstack([np.hstack([-rho * dv.detach().cpu().numpy(), np.zeros([ndim, 1])]), np.zeros([1, ndim + 1])])
	for i in range(ndim):
		dSFdUQ[i, 0, i, 1:] = -rho * v.detach().cpu().numpy().reshape(dSFdUQ[i, 0, i, 1:].shape, order='F')
	dSFdUQ[-1, 0, 0:-1, 1:] = np.reshape(-np.eye(ndim), [1, ndim, ndim], order='F')
	dSFdUQ[0:-1, 1:, -1, 0] = np.eye(ndim)

	dSFdUQ[0,1,0,1] = -2*rho*nu
	dSFdUQ[0,2,0,2] = -rho*nu
	dSFdUQ[0,2,1,1] = -rho*nu
	dSFdUQ[1,1,1,1] = -rho * nu
	dSFdUQ[1,1,0,2] = -rho*nu
	dSFdUQ[1,2,1,2] = -2*rho*nu
	dSFdUQ = Double(dSFdUQ)
	return SF, dSFdUQ

def eval_ins_base_handcode_bndstvc_intr_bndflux_pars_NSf(UQ, pars, x, n):
    nvar = UQ.shape[0]
    ndim = UQ.shape[1] - 1
    Ub = UQ[:, 0]
    dUb = np.zeros([nvar, nvar, ndim + 1])
    dUb[:, :, 0] = np.eye(nvar)
    Fn = -pars[-ndim - 1:].reshape([-1, 1])
    dFn = np.zeros([nvar, nvar, ndim + 1])
    return Ub, Double(dUb), Double(Fn), Double(dFn)






# stokes
class setup_ins_base_handcode_stokes_f(object):

	def __init__(self, ndim, rho, nu, tb, bnd2nbc):
		self.eqn = IncompressibleStokes_f(ndim)
		self.bnd2nbc = bnd2nbc
		self.vol_pars_fcn = lambda x, el: np.vstack([rho(x, el),
													 nu(x, el),
													 np.zeros([ndim + 1, 1]) + np.nan])
		self.bnd_pars_fcn = lambda x, n, bnd, el, fc: np.vstack([rho(x, el),
																 nu(x, el),
																 tb(x, n, bnd, el, fc)])


class IncompressibleStokes_f(object):

	def __init__(self, ndim):
		self.ndim = ndim
		self.nvar = ndim + 1
		self.srcflux = lambda UQ,f, pars, x: \
			eval_ins_base_handcode_srcflux_stokesf(UQ,f, pars, x)
		self.bndstvcflux = lambda nbcnbr, UQ, pars, x, n: \
			eval_ins_base_handcode_bndstvc_intr_bndflux_pars_stokesf(UQ, pars, x, n)


def eval_ins_base_handcode_srcflux_stokesf(UQ,f, pars, x):
	u = UQ[:, 0]
	q = UQ[:, 1:]
	ndim = u.shape[0] - 1
	neqn = ndim + 1
	ncomp = ndim + 1
	rho = pars[0]
	nu = pars[1]
	v = u[0:ndim]
	v = ReshapeFix(v, [len(v), 1], 'F')
	# v=v.reshape([-1,1],order='F')
	p = u[-1]
	dv = q[0:ndim, :]
	f_=torch.tensor(f).reshape(-1,1).double().to('cuda')
	Dv = (dv + dv.T) / 2

	F = torch.cat([-2 * rho * nu * Dv + p * torch.eye(ndim).double().to('cuda'),
				   torch.zeros([1, ndim]).double().to('cuda')], axis=0)
	S = torch.cat([f_.reshape(-1,1), -torch.trace(dv).reshape([1,1])], axis=0)

	# S = torch.cat([f_,-torch.trace(dv).reshape([1, 1])], axis=0)
	# SF= np.hstack([S,F])
	SF = torch.cat([S, F], axis=1)

	dSFdUQ = np.zeros([neqn, ndim + 1, ncomp, ndim + 1])
	for i in range(ndim):
		dSFdUQ[i, 0, :, :] = np.zeros([ndim + 1, ndim + 1])

	dSFdUQ[-1, 0, 0:-1, 1:] = np.reshape(-np.eye(ndim), [1, ndim, ndim], order='F')
	dSFdUQ[0:-1, 1:, -1, 0] = np.eye(ndim)

	dSFdUQ[0,1,0,1] = -2*rho*nu
	dSFdUQ[0,2,0,2] = -rho*nu
	dSFdUQ[0,2,1,1] = -rho*nu
	dSFdUQ[1,1,1,1] = -rho * nu
	dSFdUQ[1,1,0,2] = -rho*nu
	dSFdUQ[1,2,1,2] = -2*rho*nu
	dSFdUQ = Double(dSFdUQ)
	return SF, dSFdUQ

def eval_ins_base_handcode_bndstvc_intr_bndflux_pars_stokesf(UQ, pars, x, n):
    nvar = UQ.shape[0]
    ndim = UQ.shape[1] - 1
    Ub = UQ[:, 0]
    dUb = np.zeros([nvar, nvar, ndim + 1])
    dUb[:, :, 0] = np.eye(nvar)
    Fn = -pars[-ndim - 1:].reshape([-1, 1])
    dFn = np.zeros([nvar, nvar, ndim + 1])
    return Ub, Double(dUb), Double(Fn), Double(dFn)