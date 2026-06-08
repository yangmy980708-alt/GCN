import numpy as np
import pdb
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../pycamotk')
# from pyCaMOtk.create_mesh_hsphere import mesh_hsphere 
# from pyCaMOtk.setup_linelptc_sclr_base_handcode import setup_linelptc_sclr_base_handcode
# from pyCaMOtk.create_dbc_strct import create_dbc_strct
# from pyCaMOtk.create_femsp_cg import create_femsp_cg
# from pyCaMOtk.solve_fem import solve_fem
# from pyCaMOtk.visualize_fem import visualize_fem



def analyticalPossion(xcg,Tc,Tb=0):
    Ue=Tc*(1-xcg[0,:]**2-xcg[1,:]**2)/4+Tb
    return Ue.flatten()

def analyticalConeInterpolation(xcg,Tc,Tb=0):
    Ue=Tc*(1-np.sqrt(xcg[0,:]**2+xcg[1,:]**2))/4+Tb
    return Ue.flatten()

"文章中的右端项f = 0"
# def analyticalNS_f1(xcg,nu):
#     f1 = xcg[0,:]* 0 + xcg[1,:]*0
#     return f1.flatten()
#
# def analyticalNS_f2(xcg,nu):
#     f2 = xcg[0,:]* 0 + xcg[1,:]*0
#     return f2.flatten()

"NS_f Hexiaoming ppt"
def analyticalNS_u1(xcg,Tc=None,Tb=None):
    Ue_u1 = (xcg[0,:]**2) * (xcg[1,:]**2) + np.exp(-xcg[1,:])
    return Ue_u1.flatten()

def analyticalNS_u2(xcg,Tc=None,Tb=None):
    Ue_u2 = -(2/3) * xcg[0,:] * (xcg[1,:]**3) + 2 - np.pi * np.sin(np.pi * xcg[0,:])
    return Ue_u2.flatten()

def analyticalNS_p(xcg,Tc=None,Tb=None):
    Ue_p = - (2 - np.pi * np.sin(xcg[0,:] * np.pi)) * np.cos(2 * np.pi * xcg[1,:])
    return Ue_p.flatten()

def analyticalNS_f1(xcg,nu):
    f1 = - 2 * nu * xcg[0,:]**2 - 2 * nu * xcg[1,:]**2 - nu * np.exp(-xcg[1,:]) \
         + np.pi**2 * np.cos(np.pi * xcg[0,:]) * np.cos(2 * np.pi * xcg[1,:])
    return f1.flatten()

def analyticalNS_f2(xcg,nu):
    f2 = 4 * nu * xcg[0,:] * xcg[1,:] - nu * np.pi**3 * np.sin(np.pi * xcg[0,:]) \
         + 2 * np.pi * (2 - np.pi * np.sin(np.pi * xcg[0,:])) * np.sin(2 * np.pi * xcg[1,:])
    return f2.flatten()

"NS_f 新算例1 (这个真解算NS不太好 用在stokes的真解)"
# def analyticalNS_u1(xcg,Tc=None,Tb=None):
#     K = 1;
#     c = K / (np.pi ** 2)
#     Ue_u1 =c* (np.pi * np.sin(2 * np.pi * xcg[1,:])) * np.cos(xcg[0,:])
#     return Ue_u1.flatten()
#
# def analyticalNS_u2(xcg,Tc=None,Tb=None):
#     K =1; c = K / (np.pi**2)
#     Ue_u2 = (- 2 * K + c * np.sin(np.pi * xcg[1,:])**2 ) * np.sin(xcg[0,:])
#     return Ue_u2.flatten()
#
# def analyticalNS_p(xcg,Tc=None,Tb=None):
#     Ue_p = xcg[0,:] * 0 + xcg[1,:] * 0
#     return Ue_p.flatten()
#
# def analyticalNS_f1(xcg,nu):
#     K = 1
#     c = K / (np.pi ** 2)
#     f1 = - c**2 * np.pi**2 * np.sin(2* np.pi* xcg[1,:])**2 * np.cos(xcg[0,:]) * np.sin(xcg[0,:])\
#          + 2 * c* np.pi**2 * (-2* K + c * np.sin(np.pi * xcg[1,:])**2) * np.sin(xcg[0,:]) * np.cos(xcg[0,:]) * np.cos(2* np.pi* xcg[1,:])\
#          + 2 * nu * c * np.pi * np.sin(2* np.pi* xcg[1,:]) * np.cos(xcg[0,:])\
#          + 4 * nu * c * np.pi**3 * np.cos(xcg[0,:]) * np.sin(2* np.pi* xcg[1,:])\
#          - 2 * nu * c * np.pi * np.cos(xcg[0,:]) * np.sin( np.pi* xcg[1,:]) * np.cos(np.pi* xcg[1,:])
#     return f1.flatten()
#
# def analyticalNS_f2(xcg,nu):
#     K = 1
#     c = K / (np.pi ** 2)
#     f2 =  c * np.pi * np.sin(2* np.pi* xcg[1,:]) * np.cos(xcg[0,:]) * (-2* K + c* (np.sin(np.pi * xcg[1,:])**2)) * np.cos(xcg[0,:])\
#         + 2 * c * np.pi * (np.sin(xcg[0,:])**2) * np.sin(np.pi* xcg[1,:]) * np.cos(np.pi* xcg[1,:]) * (-2* K + c* (np.sin(np.pi * xcg[1,:])**2))\
#         + nu * (-2* K + c* (np.sin(np.pi * xcg[1,:])**2)) * np.sin(xcg[0,:])\
#         - 4 * nu * c * (np.pi**2) * np.sin(xcg[0,:]) * (np.cos(np.pi * xcg[1,:])**2 - np.sin(np.pi * xcg[1,:])**2)\
#         + 2 * nu * c * (np.pi ** 2) * np.sin(xcg[0,:]) * np.cos(2* np.pi* xcg[1,:])
#     return f2.flatten()

"NS 新算例3 这个不知道f1 f2 算对没 但是出的图还不错"
# def analyticalNS_u1(xcg,Tc=None,Tb=None):
#     Ue_u1 = 10 * (xcg[0,:]**2) * xcg[1,:] * (2*xcg[1,:]-1) * ((xcg[0,:] - 1)**2) * (xcg[1,:])
#     return Ue_u1.flatten()
#
# def analyticalNS_u2(xcg,Tc=None,Tb=None):
#     Ue_u2 = -10* xcg[0,:] * (xcg[1,:]**2) * (2*xcg[0,:]-1)*(xcg[0,:]-1)*((xcg[1,:]-1)**2)
#     return Ue_u2.flatten()
#
# def analyticalNS_p(xcg,Tc=None,Tb=None):
#     Ue_p = (20*xcg[0,:]-10)* (2*xcg[1,:]-1)
#     return Ue_p.flatten()
#
# def analyticalNS_f1(xcg,nu,Tc=None,Tb=None):
#     f1 =  (20 * (2* xcg[1,:]-1) - 40 * xcg[0,:]**2 * (xcg[0,:]-1)**2 * (xcg[1,:]-1)\
#          -20 * xcg[0,:]**2 * (2* xcg[1,:]-1) * (xcg[0,:]-1)**2 - 40*xcg[0,:]**2 * xcg[1,:] * (xcg[0,:]-1)**2\
#          -20 * xcg[0,:]**2 * xcg[1,:] * (2* xcg[1,:]-1) * (xcg[1,:]-1)\
#          - 20 * xcg[1,:] * (2* xcg[1,:]-1) * (xcg[0,:]-1)**2 * (xcg[1,:]-1)\
#          -40 * xcg[0,:] * xcg[1,:] * (2*xcg[0,:]-2) * (2* xcg[1,:]-1) * (xcg[1,:]-1))\
#          -10 * xcg[0,:] * xcg[1,:]**2 * (2*xcg[0,:]-1) * (xcg[0,:]-1) * (xcg[1,:]-1)**2 *(10 * xcg[0,:]**2 * (2*xcg[1,:]-1) * (xcg[0,:]-1)**2 * (xcg[1,:]-1)\
#          +20 * xcg[0,:]**2 * xcg[1,:] * (xcg[0,:]-1)**2 * (xcg[1,:]-1) + 10 * xcg[0,:]**2 * xcg[1,:] * (2*xcg[1,:]-1) * (xcg[0,:]-1)**2)\
#          +10 * xcg[0,:]**2 * xcg[1,:] * (2*xcg[1,:]-1) * (20 * xcg[0,:] * xcg[1,:] * (2*xcg[1,:]-1) * (xcg[0,:]-1)**2 * (xcg[1,:]-1)\
#          +10 * xcg[0,:]**2 * xcg[1,:] * (2* xcg[0,:]-2) * (2*xcg[1,:]-1) * (xcg[1,:]-1)) * (xcg[0,:]-1)**2 *(xcg[1,:]-1)
#     return f1.flatten()
#
# def analyticalNS_f2(xcg,nu,Tc=None,Tb=None):
#     f2 =  2 * (20* xcg[0,:]-10) + 40 * xcg[1,:]**2 * (xcg[0,:]-1) * (xcg[1,:]-1)**2\
#          +20 * xcg[1,:]**2 * (2* xcg[0,:]-1) * (xcg[1,:]-1)**2 + 40 * xcg[0,:] * xcg[1,:]**2 * (xcg[1,:]-1)**2\
#          +20 * xcg[0,:] * xcg[1,:]**2 * (2* xcg[0,:]-1) * (xcg[0,:]-1) + 20 * xcg[0,:] * (2*xcg[0,:]-1) * (xcg[0,:]-1) * (xcg[1,:]-1)**2\
#          +40 * xcg[0,:] * xcg[1,:] * (2* xcg[0,:]-1) * (2* xcg[1,:]-2) * (xcg[0,:]-1)\
#          -10 * xcg[0,:]**2 * xcg[1,:] * (2* xcg[1,:]-1) * (xcg[0,:]-1)**2 * (xcg[1,:]-1) * (10 * xcg[1,:]**2 * (2* xcg[0,:]-1) * (xcg[0,:]-1) * (xcg[1,:]-1)**2\
#          +20 * xcg[0,:] * xcg[1,:]**2 * (xcg[0,:]-1) * (xcg[1,:]-1)**2 + 10 * xcg[0,:] * xcg[1,:]**2 * (2* xcg[0,:]-1) * (xcg[1,:]-1)**2)\
#          +10 * xcg[0,:] * xcg[1,:]**2 * (2* xcg[0,:]-1) * (20 * xcg[0,:] * xcg[1,:] * (2* xcg[0,:]-1) * (xcg[0,:]-1) * (xcg[1,:]-1)**2\
#          +10 * xcg[0,:] * xcg[1,:]**2 * (2* xcg[0,:]-1) * (2* xcg[1,:]-2) * (xcg[0,:]-1)) * (xcg[0,:]-1) * (xcg[1,:]-1)**2
#     return f2.flatten()

"NS 算例  这个算出来不错"
# def analyticalNS_u1(xcg,Tc=None,Tb=None):
#     Ue_u1 = xcg[0,:] + xcg[0,:]**2 - 2 * xcg[0,:] * xcg[1,:] + xcg[0,:]**3 - 3 * xcg[0,:]*xcg[1,:]**2 + xcg[0,:]**2*xcg[1,:]
#     return Ue_u1.flatten()
#
# def analyticalNS_u2(xcg,Tc=None,Tb=None):
#     Ue_u2 = -xcg[1,:] - 2 * xcg[0,:] * xcg[1,:] + xcg[1,:]**2 - 3* xcg[0,:]**2 * xcg[1,:] + xcg[1,:]**3 - xcg[0,:]*xcg[1,:]**2
#     return Ue_u2.flatten()
#
# def analyticalNS_p(xcg,Tc=None,Tb=None):
#     Ue_p = xcg[0,:] * xcg[1,:] + xcg[0,:] + xcg[1,:] + xcg[0,:]**3 * xcg[1,:]**2 - 4/3
#     return Ue_p.flatten()
#
# def analyticalNS_f1(xcg,nu,Tc=None,Tb=None):
#     f1 =  (xcg[0,:] + xcg[0,:]**2 - 2 * xcg[0,:] * xcg[1,:] + xcg[0,:]**3 - 3 * xcg[0,:]*xcg[1,:]**2 + xcg[0,:]**2*xcg[1,:])\
#           *(1 + 2*xcg[0,:] -2*xcg[1,:] + 3* xcg[0,:]**2 - 3* xcg[1,:]**2 + 2 * xcg[0,:]*xcg[1,:])\
#           +(-xcg[1,:] - 2 * xcg[0,:] * xcg[1,:] + xcg[1,:]**2 - 3* xcg[0,:]**2 * xcg[1,:] + xcg[1,:]**3 - xcg[0,:]*xcg[1,:]**2)\
#           *(-2* xcg[0,:] - 6* xcg[0,:]*xcg[1,:] + xcg[0,:]**2)\
#           -2*nu * (2 + 6* xcg[0,:] + 2* xcg[1,:]) + (xcg[1,:] + 3* xcg[1,:]**2 * xcg[0,:]**2 +1) - nu * (-12*xcg[0,:]-2 -2* xcg[1,:])
#     return f1.flatten()
#
# def analyticalNS_f2(xcg,nu,Tc=None,Tb=None):
#     f2 = (xcg[0,:] + xcg[0,:]**2 - 2 * xcg[0,:] * xcg[1,:] + xcg[0,:]**3 - 3 * xcg[0,:]*xcg[1,:]**2 + xcg[0,:]**2*xcg[1,:])\
#          *(-2* xcg[1,:] - 6* xcg[0,:]*xcg[1,:] - xcg[1,:]**2)\
#          +(-xcg[1,:] - 2 * xcg[0,:] * xcg[1,:] + xcg[1,:]**2 - 3* xcg[0,:]**2 * xcg[1,:] + xcg[1,:]**3 - xcg[0,:]*xcg[1,:]**2)\
#          *(-1 - 2* xcg[0,:] + 2* xcg[1,:] - 3* xcg[0,:]**2 +3* xcg[1,:]**2 - 2 * xcg[0,:] * xcg[1,:])\
#          -nu* (-2 - 12* xcg[1,:] + 2* xcg[0,:]) -2* nu *(2+ 6*xcg[1,:] - 2* xcg[0,:]) + (xcg[0,:] + 1 + 2*xcg[0,:]**3 * xcg[1,:])
#     return f2.flatten()



"stokes equation hxm ppt"
# def analyticalstokes_u1(xcg,Tc=None,Tb=None):
#     Ue_u1 = (xcg[0,:]**2) * (xcg[1,:]**2) + np.exp(-xcg[1,:])
#     return Ue_u1.flatten()
#
# def analyticalstokes_u2(xcg,Tc=None,Tb=None):
#     Ue_u2 = -(2/3) * xcg[0,:] * (xcg[1,:]**3) + 2 - np.pi * np.sin(np.pi * xcg[0,:])
#     return Ue_u2.flatten()
#
# def analyticalstokes_p(xcg,Tc=None,Tb=None):
#     Ue_p = - (2 - np.pi * np.sin(xcg[0,:] * np.pi)) * np.cos(2 * np.pi * xcg[1,:])
#     return Ue_p.flatten()
#
# def analyticalstokes_f1(xcg,nu,Tc=None,Tb=None):
#     f1 = -2 * nu * (xcg[0,:]**2) - 2* nu * (xcg[1,:]**2) - nu * np.exp(-xcg[1,:]) \
#          + np.pi**2 * (np.cos(np.pi * xcg[0,:])) * np.cos(2 * np.pi * xcg[1,:])
#     return f1.flatten()
#
# def analyticalstokes_f2(xcg,nu,Tc=None,Tb=None):
#     f2 = 4* nu * xcg[0,:] * xcg[1,:] - nu * np.pi**3 * np.sin(np.pi * xcg[0,:])\
#          + 2* np.pi * (2 - np.pi * np.sin(np.pi * xcg[0,:])) * np.sin(2* np.pi * xcg[1,:])
#     return f2.flatten()

"stokes equation 2"
def analyticalstokes2_u1(xcg,Tc=None,Tb=None):
    K = 1; c = K / (np.pi ** 2)
    Ue_u1 = c * np.pi * np.sin(2 * np.pi * xcg[1,:]) * np.cos(xcg[0,:])
    return Ue_u1.flatten()

def analyticalstokes2_u2(xcg,Tc=None,Tb=None):
    K =1; c = K / (np.pi**2)
    Ue_u2 = (- 2 * K + c * np.sin(np.pi * xcg[1,:])**2 ) * np.sin(xcg[0,:])
    return Ue_u2.flatten()

def analyticalstokes2_p(xcg,Tc=None,Tb=None):
    Ue_p = xcg[0,:] * 0 + xcg[1,:] * 0
    return Ue_p.flatten()

def analyticalstokes2_f1(xcg,nu):
    K = 1
    c = K / (np.pi ** 2)
    f1 =   2 * nu * c * np.pi * np.sin(2* np.pi* xcg[1,:]) * np.cos(xcg[0,:])\
         + 4 * nu * c * np.pi**3 * np.cos(xcg[0,:]) * np.sin(2* np.pi* xcg[1,:])\
         - 2 * nu * c * np.pi * np.cos(xcg[0,:]) * np.sin( np.pi* xcg[1,:]) * np.cos(np.pi* xcg[1,:])
    return f1.flatten()

def analyticalstokes2_f2(xcg,nu):
    K = 1
    c = K / (np.pi ** 2)
    f2 =  2 * nu * c * np.pi ** 2 * np.sin(xcg[0,:]) * np.cos(2* np.pi* xcg[1,:]) \
          + nu * (-2 * K + c * (np.sin(np.pi * xcg[1, :]) ** 2)) * np.sin(xcg[0, :]) \
          - 4 * nu * c * np.pi**2 * np.sin(xcg[0,:]) * (np.cos(np.pi * xcg[1,:])**2 - np.sin(np.pi * xcg[1,:])**2)
    return f2.flatten()

"stokes equation 1"
# def analyticalstokes_u1(xcg,Tc=None,Tb=None):
#     Ue_u1 = 10 * (xcg[0,:]**2) * xcg[1,:] * (2*xcg[1,:]-1) * ((xcg[0,:] - 1)**2) * (xcg[1,:])
#     return Ue_u1.flatten()
#
# def analyticalstokes_u2(xcg,Tc=None,Tb=None):
#     Ue_u2 = -10* xcg[0,:] * (xcg[1,:]**2) * (2*xcg[0,:]-1)*(xcg[0,:]-1)*((xcg[1,:]-1)**2)
#     return Ue_u2.flatten()
#
# def analyticalstokes_p(xcg,Tc=None,Tb=None):
#     Ue_p = (20*xcg[0,:]-10)* (2*xcg[1,:]-1)
#     return Ue_p.flatten()
#
# def analyticalstokes_f1(xcg,nu,Tc=None,Tb=None):
#     f1 =  40*xcg[1,:] - 40*(xcg[0,:]**2) * ((xcg[0,:]-1)**2) * (xcg[1,:]-1) - 20*(xcg[0,:]**2) * (2*xcg[1,:]-1) * ((xcg[0,:]-1)**2)\
#          -40*(xcg[0,:]**2) * xcg[1,:] * ((xcg[0,:]-1)**2) - 20* xcg[1,:]*(2*xcg[1,:]-1)*((xcg[0,:]-1)**2) * (xcg[1,:]-1)\
#          -20*(xcg[0,:]**2) *xcg[1,:]*(2*xcg[1,:]-1)*(xcg[1,:]-1) -40*xcg[0,:]*xcg[1,:] *(2*xcg[0,:]-2)*(2*xcg[1,:]-1)* (xcg[1,:]-1) -20
#     return f1.flatten()
#
# def analyticalstokes_f2(xcg,nu,Tc=None,Tb=None):
#     f2 =  40*xcg[0,:] + 40*(xcg[1,:]**2) *(xcg[0,:]-1)*((xcg[1,:]-1)**2) + 20*(xcg[1,:]**2) * (2*xcg[0,:]-1)*((xcg[1,:]-1)**2)\
#          +40*xcg[0,:]*(xcg[1,:]**2) *((xcg[1,:]-1)**2) + 20*xcg[0,:] *(2*xcg[0,:]-1)*(xcg[0,:]-1)*((xcg[1,:]-1)**2)\
#          +20*xcg[0,:]*(xcg[1,:]**2) *(2*xcg[0,:]-1)*(xcg[0,:]-1) +40*xcg[0,:]*xcg[1,:] *(2*xcg[0,:]-1)*(2*xcg[1,:]-2)*(xcg[0,:]-1) -20
#     return f2.flatten()