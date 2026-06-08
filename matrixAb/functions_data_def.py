import numpy as np

def function_BJ_coefficient(x, y):
    r = 1
    return r

def function_BJSJ_coefficient(x, y):
    r = 1
    return r


def function_f1_Stokes(x, y):
    nu = function_nu(x,y)
    r = -2 * nu * x ** 2 - 2 * nu * y ** 2 - nu * np.exp(-y) + np.pi ** 2 * np.cos(np.pi * x) * np.cos(2 * np.pi * y)
    r += (x ** 2 * y ** 2 + np.exp(-y)) * 2 * x * y ** 2
    r += (-2 * x * y ** 3 / 3 + 2 - np.pi * np.sin(np.pi * x)) * (2 * x ** 2 * y - np.exp(-y))
    return r.flatten()

def function_f2_Stokes(x, y):
    nu = function_nu(x,y)
    r = 4 * nu * x * y - nu * np.pi ** 3 * np.sin(np.pi * x) + 2 * np.pi * (2 - np.pi * np.sin(np.pi * x)) * np.sin(
        2 * np.pi * y)
    r += (x ** 2 * y ** 2 + np.exp(-y)) * (-2 * y ** 3 / 3 - np.pi ** 2 * np.cos(np.pi * x))
    r += (-2 * x * y ** 3 / 3 + 2 - np.pi * np.sin(np.pi * x)) * (-2 * x * y ** 2)
    return r.flatten()


def function_f_Poisson(x, y):
    r = -np.pi ** 3 * np.sin(np.pi * x) * (-y + np.cos(np.pi * (1 - y))) - (2 - np.pi * np.sin(np.pi * x)) * (
                -np.pi ** 2 * np.cos(np.pi * (1 - y)))
    return r.flatten()

def function_fix_p(x, y):
    r = -(2 - np.pi * np.sin(np.pi * x)) * np.cos(2 * np.pi * y)#p_exact_solution
    return r.flatten()

def function_g1_Stokes(x, y):
    r = x**2 * y**2 + np.exp(-y)
    return r.flatten()

def function_g2_Stokes(x, y):
    r = -2 * x * y**3 / 3 + 2 - np.pi * np.sin(np.pi * x)
    return r.flatten()

def function_g_Poisson(x, y):
    r = (2 - np.pi * np.sin(np.pi * x)) * (-y + np.cos(np.pi * (1 - y)))
    return r.flatten()

def function_gravity(x, y):
    r = 1
    return r

def function_relative_depth(x, y):
    r = 0
    return r

def function_gravity_depth(x, y):
    r = function_gravity(x, y) * function_relative_depth(x, y)
    return r.flatten()

def function_k11(x, y):
    r = 1
    return r

def function_k11_BJ_coe(x, y):
    r = function_k11(x, y) * function_BJ_coefficient(x, y)
    return r.flatten()

def function_k12(x, y):
    r = 0
    return r

def function_k12_BJ_coe(x, y):
    r = function_k12(x, y) * function_BJ_coefficient(x, y)
    return r.flatten()

def function_k21(x, y):
    r = 0
    return r

def function_k21_BJ_coe(x, y):
    r = function_k21(x, y) * function_BJ_coefficient(x, y)
    return r.flatten()

def function_k22(x, y):
    r = 1
    return r

def function_k22_BJ_coe(x, y):
    r = function_k22(x, y) * function_BJ_coefficient(x, y)
    return r.flatten()

def function_negativeone(x, y):
    r = -1
    return r

def function_nu(x, y):
    r = 1
    return r

def function_one(x, y):
    r = 1
    return r

def p_exact_solution(x, y):
    r = -(2 - np.pi * np.sin(np.pi * x)) * np.cos(2 * np.pi * y)
    return r.flatten()

def p_exact_solution_x_derivative(x, y):
    r = np.pi**2 * np.cos(np.pi * x) * np.cos(2 * np.pi * y)
    return r.flaten()

def p_exact_solution_y_derivative(x, y):
    r = (2 - np.pi * np.sin(np.pi * x)) * np.sin(2 * np.pi * y) * 2 * np.pi
    return r.flatten()

def phi_exact_solution(x, y):
    r = (2 - np.pi * np.sin(np.pi * x)) * (-y + np.cos(np.pi * (1 - y)))
    return r.flatten()

def phi_exact_solution_x_derivative(x, y):
    r = -np.pi**2 * np.cos(np.pi * x) * (-y + np.cos(np.pi * (1 - y)))
    return r.flatten()

def phi_exact_solution_y_derivative(x, y):
    r = (2 - np.pi * np.sin(np.pi * x)) * (-1 + np.pi * np.sin(np.pi * (1 - y)))
    return r.flatten()

def u1_exact_solution(x, y):
    r = x**2 * y**2 + np.exp(-y)
    return r.flatten()

def u1_exact_solution_x_derivative(x, y):
    r = 2 * x * y**2
    return r.flatten()

def u1_exact_solution_y_derivative(x, y):
    r = 2 * x**2 * y - np.exp(-y)
    return r.flatten()

def u2_exact_solution(x, y):
    r = -2 * x * y**3 / 3 + 2 - np.pi * np.sin(np.pi * x)
    return r.flatten()

def u2_exact_solution_x_derivative(x, y):
    r = -2 * y**3 / 3 - np.pi**2 * np.cos(np.pi * x)
    return r.flatten()

def u2_exact_solution_y_derivative(x, y):
    r = -2 * x * y**2
    return r.flatten()