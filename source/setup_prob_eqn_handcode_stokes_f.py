import torch
from torch import tensor
import numpy as np
import pdb

from TensorFEMCore_stokes import Double, ReshapeFix

import sys
# sys.path.insert(0, '../pycamotk')
from pycamotk.pyCaMOtk.create_mesh_hcube import mesh_hcube

# sys.path.insert(0, '../source')
import source.TensorFEMCore_stokes
from source.TensorFEMCore_stokes import Double
# from source.FEM_ForwardModel import analyticalNS_f1,analyticalNS_f2

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
		self.bndstvcflux = lambda nbcnbr,UQ, pars, x, n: \
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
	p = u[-1]
	dv = q[0:ndim, :]
	f_=torch.tensor(f).reshape(-1,1).double().to('cuda')
	Dv = (dv + dv.T) / 2

	F = torch.cat([-2 * rho * nu * Dv + p * torch.eye(ndim).double().to('cuda'),
				   torch.zeros([1, ndim]).double().to('cuda')], axis=0)
	S = torch.cat([f_.reshape(-1,1), -torch.trace(dv).reshape([1,1])], axis=0)

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