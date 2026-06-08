import pdb
# from pyCaMOtk.geom_mltdim import *
import numpy as np
import matplotlib.pyplot as plt
from pycamotk.pyCaMOtk.qrule_mltdim import *
from pycamotk.pyCaMOtk.qrule_onedim import *
from pycamotk.pyCaMOtk.poly_mltdim import *
from pycamotk.pyCaMOtk.geom_mltdim import *




class LocalFunctionSpace(object):
    def __init__(self, polysp, nsd, porder, qrule0='gl', nq0=None, **kwargs):

        # Create geometry object 创建几何对象
        self.nsd = nsd  # 对应着ndim 维数
        self.porder = porder  # 基函数的阶
        self.desc = polysp  # type
        self.nf = int(self.nsd + 1) * (polysp == 'P') + int(2 * nsd) * (polysp == 'Q')
        if polysp == 'P':
            ndist0 = kwargs['ndist0'] if 'ndist0' in kwargs else 'unif'
            self.geom = Simplex(self.nsd, self.porder, ndist0)
            self.rk = np.linspace(0, 1, self.porder + 1)  # 创建了一个从0到1的等差数列，将其分成self.porder+1个等间距的点
            # raise ValueError('Polynomial space not supported')
        elif polysp == 'Q':
            ndist0 = kwargs['ndist0'] if 'ndist0' in kwargs else 'unif'
            self.geom = Hypercube(self.nsd, self.porder, ndist0)
            self.rk = np.linspace(-1, 1, self.porder + 1)
        else:
            raise ValueError('Geometry not supported')

        # Default input arguments 默认输入参数
        if nq0 is None: nq0 = int(np.max([3 * porder, 1]))  # int((porder+1)/2)*3
        self.nq0 = nq0

        # Create quadrature weights/nodes for reference volume, face, edge
        # 创建参考体、面、边的求积权重/节点
        self.wqd, sq = qrule_onedim(nq0, qrule0)  # edge wqd 表示积分的权重 sq 表示积分点
        """ nq0 积分点个数   qrule0 积分的类型 这里是高斯勒让德求积公式"""
        self.sq = sq
        if nsd > 1:  # nsd 对应着该方程的维数ndim
            # face
            if polysp == 'P':
                self.wqf, rq = qrule_mltdim_simp(nsd - 1, nq0, qrule0)  # 同样的返回积分点及其权重
            elif polysp == 'Q':
                self.wqf, rq = qrule_mltdim_hcube(nsd - 1, nq0, qrule0)
            else:
                raise ValueError('Polynomial space is not supported!')
        else:
            self.wqf = np.ones((1, nq0)) / nq0
            """np.ones((1, nq0))会生成一个形状为(1, nq0)的二维数组，所有元素都是1。然后，通过除以nq0，更新数组的每个元素为1/nq0。"""
            rq = np.zeros((1, nq0))
            """np.zeros((1, nq0))会生成一个形状为(1, nq0)的二维数组，所有元素都是0。"""
        # 这一块儿 是立体的单元所得到高斯权重及积分点
        if polysp == 'P':
            self.wqv, zq = qrule_mltdim_simp(nsd, nq0, qrule0)
        elif polysp == 'Q':
            self.wqv, zq = qrule_mltdim_hcube(nsd, nq0, qrule0)  # volume
        else:
            raise ValueError('Polynomial space is not supported!')
        self.rq = rq  # 立体的 积分点
        self.zq = zq  # 立体的 权重

        # Start zqd
        self.zqd = np.zeros((nsd, len(sq), self.geom.d2n.shape[1]))  # TODO
        """创建一个形状为 (nsd, len(sq), self.geom.d2n.shape[1]) 的全零数组 
        self.zqd，用来存储积分点在每条边上的坐标。"""
        for i in range(self.geom.d2n.shape[1]):
            ndstart = self.geom.zk[:, self.geom.d2n[0, i]]  # Han Gao: ndstart means start node of edge指边的起始节点
            """是由self.geom.zk的所有行和self.geom.d2n的第i列组成。"""
            ndend = self.geom.zk[:, self.geom.d2n[-1, i]]  # Han Gao: ndend means end node of edge 边的末端节点
            dvec = (ndend - ndstart) / 2  # Han Gao: dvec means vector (not a free vector) along the edge
            # 表示沿边缘的向量(不是自由向量)
            ndmid = (ndend + ndstart) / 2  # Han Gao: ndmid means mid point of edge 指边的中点
            for j in range(len(sq)):
                self.zqd[:, j, i] = sq[j] * dvec + ndmid
                """
                self.zqd是一个三维数组，而sq、dvec和ndmid都只是一维数组，所以需要根据索引i和j确定要赋值的位置。
                sq[j]表示对sq数组中索引为j的元素进行访问
                dvec是一个一维数组，它的每个元素与sq[j]进行乘法运算
                ndmid是一个一维数组，它与sq[j] * dvec的结果进行加法运算
                """
        '''
        # Start zqf
        # Han Gao: I belive this method can not be general enough for simplex, 
        # need do some map for simplex based on hcube result
        # Do not delete for check, this is only valid for hcube
        self.zqf_check = np.zeros((nsd, rq.shape[1], self.geom.f2n.shape[1])) # TODO
        for d in range(self.nsd):
            idx_into_idx = [i for i in range(self.nsd) if i != d]
            self.zqf_check[:,:,d] = -1. # Han Gao: use python boradcast!
            self.zqf_check[idx_into_idx,:,d] = rq
            self.zqf_check[:,:,d + self.nsd] = 1. # Han Gao: use python boradcast!
            self.zqf_check[idx_into_idx,:,d + self.nsd] = rq
        '''

        # Han Gao helped by Prof Zahr: Trying to remake it general below下面尝试对其进行概括性改造
        self.zqf = np.zeros((nsd, rq.shape[1], self.geom.f2n.shape[1]))  # TODO
        if nsd > 1:
            if polysp == 'Q':
                coeff = eval_poly_nodal_mltdim_lagr_tensprod(np.linspace(-1, 1, 2), rq, nderiv=1)
            elif polysp == 'P':#一般形单元会使用这个
                coeff =eval_poly_nodal_mltdim_vander(np.linspace(0, 1, 2), rq, polysp='P', nderiv=1, glob_cont=1)
                # coeff = eval_poly_nodal_mltdim_lagr_tensprod(np.linspace(0, 1, 2), rq, nderiv=1)

        else:
            coeff = np.ones((1, 1, rq.shape[1]))
        n2v = invert_idx_array(self.geom.zk.shape[1], self.geom.v2n)#在程序geom_mltdim中
        for d in range(self.nf):  # for loop face
            for i in range(rq.shape[1]):  # for loop quad
                # Han Gao Makes this compact
                '''
                lf2v means local face 2 vortex
                Han Gao made below compact 
                '''
                lf2v = [int(n2v[int(self.geom.f2n[j, d])]) for j in range(self.geom.f2n.shape[0]) if
                        n2v[int(self.geom.f2n[j, d])].shape[0] != 0]
                self.zqf[:, i, d] = np.matmul(self.geom.zk[:, self.geom.v2n[:, lf2v]],
                                              coeff[:, 0, i].reshape(coeff[:, 0, i].shape[0], 1)).transpose()

        # Check if equal
        '''    
        check = np.sum((np.absolute(self.zqf - self.zqf_check)))
        '''

        # Start zq
        self.zqv = zq

        # Han Gao: start reshape
        zqd = self.zqd.reshape(nsd, self.zqd.shape[1] * self.zqd.shape[2],
                               order='F')
        zqf = self.zqf.reshape(nsd, self.zqf.shape[1] * self.zqf.shape[2],
                               order='F')

        # Create basis functions over volume; evaluate at quadrature在卷上创建基函数；在正交表中进行评估
        # on volume, face, edge
        if polysp == 'Q':
            self.Qvd = self.eval_basis_functions(zqd)
            self.Qvf = self.eval_basis_functions(zqf)
            self.Qvv = self.eval_basis_functions(self.zqv)
            self.Qf = self.eval_basis_functions(rq)
        elif polysp == 'P':
            self.Qvd = self.eval_basis_functions(zqd, self.geom.zk)
            self.Qvf = self.eval_basis_functions(zqf, self.geom.zk)
            self.Qvv = self.eval_basis_functions(self.zqv, self.geom.zk)
            if self.nsd == 1 or self.nsd == 2:
                self.Qf = eval_poly_nodal_mltdim_lagr_tensprod(np.linspace(0, 1, self.porder + 1), rq, nderiv=1)
            else:
                self.geom_face = Simplex(self.nsd - 1, self.porder, ndist0)
                self.Qf = self.eval_basis_functions(rq, self.geom_face.zk)

        # Han Gao: start reshape
        # Han Gao: This code is not good, the dimension is
        '''
        numberofbasisxnumberofderivxnumberofquadxnumberof(face or edge)
        We will change the code to make it readable!
        '''
        self.Qvd = self.Qvd.reshape(self.Qvd.shape[0], self.Qvd.shape[1], nq0, int(self.Qvd.shape[2] / nq0), order='F')
        self.Qvf = self.Qvf.reshape(self.Qvf.shape[0], self.Qvf.shape[1], rq.shape[1],
                                    int(self.Qvf.shape[2] / rq.shape[1]), order='F')

        # Han Gao: save rq maybe useful
        self.rq = rq
        # pdb.set_trace()

    def eval_basis_functions(self, z, zk0=None):
        if self.desc == 'Q':
            # NOTE: Specific to hypercube
            zk0 = np.linspace(-1, 1, self.porder + 1)
            return eval_poly_nodal_mltdim_lagr_tensprod(zk0, z, nderiv=1)
        elif self.desc == 'P':
            if zk0 is None:
                zk0 = self.geom.zk
            return eval_poly_nodal_mltdim_vander(zk0, z, 'P', 1)



if __name__ == '__main__':
    #这一行的意思是，当这个脚本作为主程序运行时，以下代码块将会被执行。如果这个文件被导入为模块，以下代码则不会被执行。
    # Test p2d1
    D = 1
    P = 2
    lfcnsp = LocalFunctionSpace('Q', D, P)
    plt.figure()
    plt.plot(lfcnsp.zqv[0,:], [0 for i in range(lfcnsp.zqv.shape[1])], 'o')
    plt.title('quadrature nodes for p2d1 (volume)')
    
    # Test p2d2
    D = 2
    P = 2
    lfcnsp = LocalFunctionSpace('Q', D, P)
    plt.figure()
    plt.plot(lfcnsp.zqv[0,:], lfcnsp.zqv[1,:], 'o')
    plt.title('quadrature nodes for p2d2 (volume)')

    plt.figure()
    plt.plot(lfcnsp.zqf[0,:,0], lfcnsp.zqf[1,:, 0], '*')
    plt.plot(lfcnsp.zqf[0,:,1], lfcnsp.zqf[1,:, 1], '*')
    plt.plot(lfcnsp.zqf[0,:,2], lfcnsp.zqf[1,:, 2], '*')
    plt.plot(lfcnsp.zqf[0,:,3], lfcnsp.zqf[1,:, 3], '*')
    plt.title('quadrature nodes for p2d2 (face)')
    
    plt.figure()
    plt.plot(lfcnsp.zqd[0,:,0], lfcnsp.zqd[1,:, 0], '*')
    plt.plot(lfcnsp.zqd[0,:,1], lfcnsp.zqd[1,:, 1], '*')
    plt.plot(lfcnsp.zqd[0,:,2], lfcnsp.zqd[1,:, 2], '*')
    plt.plot(lfcnsp.zqd[0,:,3], lfcnsp.zqd[1,:, 3], '*')
    plt.title('quadrature nodes for p2d2 (edge)')
    
    # Test p2d3
    D = 3
    P = 2
    lfcnsp = LocalFunctionSpace('Q', D, P)
    plt.figure()
    ax = plt.axes(projection='3d')
    xline = lfcnsp.zqv[0,:]
    yline = lfcnsp.zqv[1,:]
    zline = lfcnsp.zqv[2,:]
    ax.plot3D(xline, yline, zline, 'o')
    plt.title('quadrature nodes for p2d3 (volume)')

    plt.figure()
    ax = plt.axes(projection='3d')
    xline = lfcnsp.zqf[0,:,0]
    yline = lfcnsp.zqf[1,:,0]
    zline = lfcnsp.zqf[2,:,0]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqf[0,:,1]
    yline = lfcnsp.zqf[1,:,1]
    zline = lfcnsp.zqf[2,:,1]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqf[0,:,2]
    yline = lfcnsp.zqf[1,:,2]
    zline = lfcnsp.zqf[2,:,2]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqf[0,:,3]
    yline = lfcnsp.zqf[1,:,3]
    zline = lfcnsp.zqf[2,:,3]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqf[0,:,4]
    yline = lfcnsp.zqf[1,:,4]
    zline = lfcnsp.zqf[2,:,4]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqf[0,:,5]
    yline = lfcnsp.zqf[1,:,5]
    zline = lfcnsp.zqf[2,:,5]
    ax.plot3D(xline, yline, zline, '*')
    plt.title('quadrature nodes for p2d3 (face)')

    plt.figure()
    ax = plt.axes(projection='3d')
    xline = lfcnsp.zqd[0,:,0]
    yline = lfcnsp.zqd[1,:,0]
    zline = lfcnsp.zqd[2,:,0]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,1]
    yline = lfcnsp.zqd[1,:,1]
    zline = lfcnsp.zqd[2,:,1]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,2]
    yline = lfcnsp.zqd[1,:,2]
    zline = lfcnsp.zqd[2,:,2]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,3]
    yline = lfcnsp.zqd[1,:,3]
    zline = lfcnsp.zqd[2,:,3]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,4]
    yline = lfcnsp.zqd[1,:,4]
    zline = lfcnsp.zqd[2,:,4]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,5]
    yline = lfcnsp.zqd[1,:,5]
    zline = lfcnsp.zqd[2,:,5]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,6]
    yline = lfcnsp.zqd[1,:,6]
    zline = lfcnsp.zqd[2,:,6]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,7]
    yline = lfcnsp.zqd[1,:,7]
    zline = lfcnsp.zqd[2,:,7]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,8]
    yline = lfcnsp.zqd[1,:,8]
    zline = lfcnsp.zqd[2,:,8]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,9]
    yline = lfcnsp.zqd[1,:,9]
    zline = lfcnsp.zqd[2,:,9]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,10]
    yline = lfcnsp.zqd[1,:,10]
    zline = lfcnsp.zqd[2,:,10]
    ax.plot3D(xline, yline, zline, '*')
    xline = lfcnsp.zqd[0,:,11]
    yline = lfcnsp.zqd[1,:,11]
    zline = lfcnsp.zqd[2,:,11]
    ax.plot3D(xline, yline, zline, '*')
    plt.title('quadrature nodes for p2d3 (edge)')




    plt.show()    
