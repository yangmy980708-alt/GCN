import numpy as np
import pdb

"""  NS_f equation"""
class setup_ins_base_handcode_NSf(object):
	"""docstring for setup_ins_base_handcode"""
	def __init__(self,ndim,rho,nu,tb,bnd2nbc):
		self.eqn=IncompressibleNavierStokes_f(ndim)
		self.bnd2nbc=bnd2nbc
		self.vol_pars_fcn=lambda x,el:np.vstack([rho(x, el),
			                                     nu(x, el),
			                                     np.zeros([ndim+1,1])+np.nan])
		self.bnd_pars_fcn=lambda x,n,bnd,el,fc:np.vstack([rho(x,el),
			 										      nu(x,el),
			 										      tb(x,n,bnd,el,fc)])


class IncompressibleNavierStokes_f(object):
	"""docstring for IncompressibleNavierStokes"""
	def __init__(self,ndim):
		self.ndim=ndim
		self.nvar=ndim+3# neqn=5 u1 u2 p f1 f2
		self.srcflux=lambda UQ,f,pars,x:\
		             eval_ins_base_handcode_srcflux_NSf(UQ,f,pars,x)
		self.bndstvcflux=lambda nbcnbr,UQ,pars,x,n:\
					     eval_ins_base_handcode_bndstvc_intr_bndflux_pars_NSf(UQ,pars,x,n)

def eval_ins_base_handcode_srcflux_NSf(UQ,f,pars,x):
	u=UQ[:,0]; q=UQ[:,1:]
	ndim=u.shape[0]-1
	neqn=ndim+ 1
	ncomp=ndim+1
	rho=pars[0]
	nu=pars[1]
	v=u[0:ndim]
	v=v.reshape([-1,1],order='F')
	p=u[-1]
	dv=q[0:ndim,:]
	# f_ = f.reshape(-1,1)#这一步主要是将两个数值重塑为dim:(2,1)
	Dv = (dv + dv.T)/2

	S=np.vstack([-rho*dv.dot(v)+f,-np.trace(dv)])
	F=np.vstack([-2*rho*nu*Dv+p*np.eye(ndim),
		         np.zeros([1,ndim])])
	SF= np.hstack([S,F])
	dSFdUQ = np.zeros([neqn,ndim+1,ncomp,ndim+1])
	dSFdUQ[:, 0, :, 0] = np.vstack([np.hstack([-rho * (dv.T), np.zeros([ndim, 1])]), np.zeros([1, ndim + 1])])
	for i in range(ndim):
		dSFdUQ[i, 0, i, 1:] = -rho * v.reshape(dSFdUQ[i, 0, i, 1:].shape, order='F')
	dSFdUQ[-1, 0, 0:-1, 1:] = np.reshape(-np.eye(ndim), [1, ndim, ndim], order='F')
	dSFdUQ[0:-1, 1:, -1, 0] = np.eye(ndim)

	dSFdUQ[0,1,0,1] = -2*rho*nu
	dSFdUQ[0,2,0,2] = -rho*nu
	dSFdUQ[0,2,1,1] = -rho*nu
	dSFdUQ[1,1,1,1] = -rho * nu
	dSFdUQ[1,1,0,2] = -rho*nu
	dSFdUQ[1,2,1,2] = -2*rho*nu
	return SF,dSFdUQ


def eval_ins_base_handcode_bndstvc_intr_bndflux_pars_NSf(UQ,pars,x,n):
	nvar=UQ.shape[0]
	ndim=UQ.shape[1]-1
	Ub=UQ[:,0]
	dUb=np.zeros([nvar,nvar,ndim+1])
	dUb[:,:,0]=np.eye(nvar)
	"""
	Ub=UQ[0]=[[u1],[u2],[p]]
	dUb=d[u1]/du1 d[u1]/d(u1_x) d[u1]/d(u1_y)  d[u2]/du1 d[u2]/d(u1_x) d[u2]/d(u1_y)  d[p]/du1 d[p]/d(u1_x) d[p]/d(u1_y)
	    d[u1]/du2 d[u1]/d(u2_x) d[u1]/d(u2_y)    ……
	    d[u1]/dp d[u1]/d(p_x) d[u1]/d(p_y)       ……
	"""
	Fn=-pars[-ndim-1:]
	dFn=np.zeros([nvar,nvar,ndim+1])
	return Ub,dUb,Fn,dFn