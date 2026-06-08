import pdb
from pyCaMOtk.geom_mltdim import *
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
from pyCaMOtk.qrule_mltdim import *
from pyCaMOtk.qrule_onedim import *
from pyCaMOtk.poly_mltdim import *
from pycamotk.mesh import *
from pyCaMOtk.lfcnsp_f import LocalFunctionSpace_f
import sys
sys.path.insert(0, '../source')
import TensorFEMCore
from TensorFEMCore import Double
from FEM_ForwardModel import analyticalstokes_f1, analyticalstokes_f2


class function_of_the_right_f(object):
    def __init__(self, lfcnsp_f,mesh):
        self.mesh=mesh
        #self.lfcnsp_f=lfcnsp_f
        self.xq=self.mesh.transfdatacontiguous.xq
        nu = 1
        #self.zq = self.lfcnsp_f.zq
        self.nxq = self.xq.shape[1]
        for k in range(self.nxq):
            f1 = [Double(analyticalstokes_f1(self.xq[:, k].reshape(-1, 1), nu).flatten().reshape(1, -1))]
            f2 = [Double(analyticalstokes_f2(self.xq[:, k].reshape(-1, 1), nu).flatten().reshape(1, -1))]
            result1 = []
            for item in f1:
                result1.append(item[0].item())
            # 将结果转换为NumPy数组
            f1_array = np.array(result1)
            result_f1 = list(f1_array)

            result2 = []
            for item in f2:
                result2.append(item[0].item())
            # 将结果转换为NumPy数组
            f2_array = np.array(result2)
            result_f2 = list(f2_array)

            f_ = np.vstack((result_f1, result_f2))
            return f_