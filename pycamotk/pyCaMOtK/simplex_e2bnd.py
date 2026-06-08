import pdb
import numpy as np
from pycamotk.pyCaMOtk.geom_mltdim import Simplex
from pycamotk.pyCaMOtk.mesh import Mesh

class setup_e2bnd_msh(object):
    def __init__(self, etype,lims, xcg, e2vcg,porder,ndim,**varargin):
        self.etype = etype
        self.lims = lims
        self.xcg = xcg
        self.e2vcg = e2vcg
        self.ndim = ndim
        self.porder = porder
        self.nf = self.ndim + 1
        self.nelem = e2vcg.shape[1]
        self.varargin = varargin

        "e2bnd : number of edges per element x number of elements"
        self.e2bnd = np.zeros([self.ndim+1, self.nelem]) + np.nan
        self.refhcubeelem = Simplex(self.ndim, self.porder, 'unif')
        self.f2v = self.refhcubeelem.f2n
        for e in range(self.nelem):
            for f in range(self.nf):
                face_nodes = self.xcg[:,self.e2vcg[self.f2v[:, f], e] ]
                for d in range(self.ndim):
                    if float(np.linalg.norm(face_nodes[d, :] - self.lims[d, 0])) == float(0):
                        self.e2bnd[f, e] = d
                    elif float(np.linalg.norm(face_nodes[d, :] - self.lims[d, 1])) == float(0):
                        self.e2bnd[f, e] = self.ndim + d -1
                    else:
                        pass
        if len(self.varargin) == 0:
            self.msh = Mesh(self.etype, self.xcg, self.e2vcg, self.e2bnd)

    def getmsh(self):
     return self.msh




