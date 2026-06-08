import torch
from torch import tensor
import numpy as np
import pdb

from TensorFEMCore import Double, ReshapeFix

import sys
sys.path.insert(0, '../pycamotk')
from pyCaMOtk.create_mesh_hcube import mesh_hcube

sys.path.insert(0, '../source')
import TensorFEMCore
from TensorFEMCore import Double
from FEM_ForwardModel import analyticalNS_f1,analyticalNS_f2


class setup_ins_base_handcode(object):
	"""docstring for setup_ins_base_handcode"""

	def __init__(self, ndim, rho, nu, bnd2nbc):
		self.eqn = IncompressibleNavierStokes(ndim)
		self.bnd2nbc = bnd2nbc
		self.vol_pars_fcn = lambda x, el: np.vstack([rho(x, el),
													 nu(x, el),
													 np.zeros([ndim + 1, 1]) + np.nan])
		# self.bnd_pars_fcn = lambda x, n, bnd, el, fc: np.vstack([rho(x, el),
		# 														 nu(x, el),
		# 														 tb(x, n, bnd, el, fc)])

class IncompressibleNavierStokes(object):
	"""docstring for IncompressibleNavierStokes"""

	def __init__(self, ndim):
		self.ndim = ndim
		self.nvar = ndim + 1
		self.srcflux = lambda UQ,f, pars, x: \
			eval_ins_base_handcode_srcflux(UQ,f, pars, x)
		# self.bndstvcflux = lambda nbcnbr, UQ, pars, x, n: \
		# 	eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ, pars, x, n)


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


# def eval_ins_base_handcode_bndstvc_intr_bndflux_pars(UQ, pars, x, n):
# 	nvar = UQ.shape[0]
# 	ndim = UQ.shape[1] - 1
# 	Ub = UQ[:, 0]
# 	dUb = np.zeros([nvar, nvar, ndim + 1])
# 	dUb[:, :, 0] = np.eye(nvar)
# 	Fn = -pars[-ndim - 1:].reshape([-1, 1])
# 	dFn = np.zeros([nvar, nvar, ndim + 1])
# 	return Ub, Double(dUb), Double(Fn), Double(dFn)
