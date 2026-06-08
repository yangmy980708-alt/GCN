import numpy as np
import pdb
from pyCaMOtk.lfcnsp import LocalFunctionSpace#,LocalFunctionSpace_f
from pyCaMOtk.create_ldof2gdof_cg import create_ldof2gdof_cg,create_ldof2gdof_cg_mixed2
from pyCaMOtk.create_elem_strct import create_elem_strct,create_elem_strct_mixed2
from pyCaMOtk.create_elem_data_no_tb import create_elem_data_no_tb
from pyCaMOtk.create_sparsity_strct import create_sparsity_strct

class create_femsp_cg_mixed2_no_tb(object):
	"""docstring for create_femsp_cg_mixed2"""
	def __init__(self,prob,msh0,neqn1,nvar1,ptest1,ptrial1,
		              e2vcg_test1,e2vcg_trial1,neqn2,nvar2,
		              ptrial2,ptest2,e2vcg_test2,e2vcg_trial2,dbc=None):
		# Extract information from input
		self.prob=prob
		self.etype=msh0.etype
		self.ndim=msh0.xcg.shape[0]
		self.e2e=msh0.e2e
		self.transf_data=msh0.transfdatacontiguous
		self.nquad_per_dim=msh0.lfcnsp.nq0
		if self.etype=='hcube':
			self.polysp='Q'
		elif self.etype=='simplex' or self.etype=='simp':
			self.polysp='P'
		else:
			raise ValueError('Only support simplex or hcube!')

		# Create local function spaces
		self.lfcnsp_eqn1=LocalFunctionSpace(self.polysp,
											self.ndim,
											ptest1,
											'gl',
											self.nquad_per_dim)
		self.lfcnsp_eqn2=LocalFunctionSpace(self.polysp,
											self.ndim,
											ptest2,
											'gl',
											self.nquad_per_dim)
		self.lfcnsp_var1=LocalFunctionSpace(self.polysp,
											self.ndim,
											ptrial1,
											'gl',
											self.nquad_per_dim)
		self.lfcnsp_var2=LocalFunctionSpace(self.polysp,
											self.ndim,
											ptrial2,
											'gl',
											self.nquad_per_dim)
		self.lfcnsp_msh=msh0.lfcnsp
		self.elem=create_elem_strct_mixed2(prob.eqn,neqn1,self.lfcnsp_eqn1,
			                               nvar1,self.lfcnsp_var1,neqn2,
			                               self.lfcnsp_eqn2,nvar2,self.lfcnsp_var2)

		self.elem_data=create_elem_data_no_tb(self.elem.Tv_eqn_ref,
										self.elem.Tvf_eqn_ref, 
                             			self.elem.Tv_var_ref, 
                             			self.elem.Tvf_var_ref, 
                             			self.e2e,self.transf_data,None, 
                             			self.prob.vol_pars_fcn,None,
                             			self.prob.bnd2nbc)
		self.ldof2gdof_eqn=create_ldof2gdof_cg_mixed2(neqn1,e2vcg_test1,neqn2,e2vcg_test2)
		self.ldof2gdof_var=create_ldof2gdof_cg_mixed2(nvar1,e2vcg_trial1,nvar2,e2vcg_trial2)
		self.ldof2gdof_msh=create_ldof2gdof_cg(self.ndim,msh0.e2vcg)
		self.spmat=create_sparsity_strct(self.ldof2gdof_eqn.ldof2gdof,
			                             self.ldof2gdof_var.ldof2gdof)