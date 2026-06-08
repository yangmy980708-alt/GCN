import numpy as np


class Functions:
    def __init__(self):
        pass

    def function_BJ_coefficient(self, x, y):
        r = 1
        return r

    def function_BJSJ_coefficient(self, x, y):
        r = 1
        return r

    def function_f1_Stokes(self, x, y):
        nu = self.function_nu(x, y)
        r = -2 * nu * x ** 2 - 2 * nu * y ** 2 - nu * np.exp(-y) + np.pi ** 2 * np.cos(np.pi * x) * np.cos(
            2 * np.pi * y)
        r += (x ** 2 * y ** 2 + np.exp(-y)) * 2 * x * y ** 2
        r += (-2 * x * y ** 3 / 3 + 2 - np.pi * np.sin(np.pi * x)) * (2 * x ** 2 * y - np.exp(-y))
        return r.flatten()

    def function_f2_Stokes(self,x, y):
        nu = self.function_nu(x, y)
        r = 4 * nu * x * y - nu * np.pi ** 3 * np.sin(np.pi * x) + 2 * np.pi * (2 - np.pi * np.sin(np.pi * x)) * np.sin(
            2 * np.pi * y)
        r += (x ** 2 * y ** 2 + np.exp(-y)) * (-2 * y ** 3 / 3 - np.pi ** 2 * np.cos(np.pi * x))
        r += (-2 * x * y ** 3 / 3 + 2 - np.pi * np.sin(np.pi * x)) * (-2 * x * y ** 2)
        return r.flatten()

    def function_f_Poisson(slef,x, y):
        r = -np.pi ** 3 * np.sin(np.pi * x) * (-y + np.cos(np.pi * (1 - y))) - (2 - np.pi * np.sin(np.pi * x)) * (
                -np.pi ** 2 * np.cos(np.pi * (1 - y)))
        return r.flatten()

    def function_fix_p(self, x, y):
        r = -(2 - np.pi * np.sin(np.pi * x)) * np.cos(2 * np.pi * y)  # p_exact_solution
        return r.flatten()

    def function_g1_Stokes(self, x, y):
        r = x ** 2 * y ** 2 + np.exp(-y)
        return r.flatten()

    def function_g2_Stokes(self, x, y):
        r = -2 * x * y ** 3 / 3 + 2 - np.pi * np.sin(np.pi * x)
        return r.flatten()

    def function_g_Poisson(self, x, y):
        r = (2 - np.pi * np.sin(np.pi * x)) * (-y + np.cos(np.pi * (1 - y)))
        return r.flatten()

    def function_gravity(self, x, y):
        r = 1
        return r

    def function_relative_depth(self, x, y):
        r = 0
        return r

    def function_gravity_depth(self, x, y):
        r = self.function_gravity(x, y) * self.function_relative_depth(x, y)
        return r

    def function_k11(self, x, y):
        r = 1
        return r

    def function_k11_BJ_coe(self, x, y):
        r = self.function_k11(x, y) * self.function_BJ_coefficient(x, y)
        return r

    def function_k12(self, x, y):
        r = 0
        return r

    def function_k12_BJ_coe(self, x, y):
        r = self.function_k12(x, y) * self.function_BJ_coefficient(x, y)
        return r

    def function_k21(self,x, y):
        r = 0
        return r

    def function_k21_BJ_coe(self, x, y):
        r = self.function_k21(x, y) * self.function_BJ_coefficient(x, y)
        return r

    def function_k22(self, x, y):
        r = 1
        return r

    def function_k22_BJ_coe(self, x, y):
        r = self.function_k22(x, y) * self.function_BJ_coefficient(x, y)
        return r

    def function_negativeone(self, x, y):
        r = -1
        return r

    def function_nu(self, x, y):
        r = 1
        return r

    def function_one(self, x, y):
        r = 1
        return r

    def p_exact_solution(self, x, y):
        r = -(2 - np.pi * np.sin(np.pi * x)) * np.cos(2 * np.pi * y)
        return r.flatten()

    def p_exact_solution_x_derivative(self, x, y):
        r = np.pi ** 2 * np.cos(np.pi * x) * np.cos(2 * np.pi * y)
        return r.flaten()

    def p_exact_solution_y_derivative(self, x, y):
        r = (2 - np.pi * np.sin(np.pi * x)) * np.sin(2 * np.pi * y) * 2 * np.pi
        return r.flatten()

    def phi_exact_solution(self, x, y):
        r = (2 - np.pi * np.sin(np.pi * x)) * (-y + np.cos(np.pi * (1 - y)))
        return r.flatten()

    def phi_exact_solution_x_derivative(self, x, y):
        r = -np.pi ** 2 * np.cos(np.pi * x) * (-y + np.cos(np.pi * (1 - y)))
        return r.flatten()

    def phi_exact_solution_y_derivative(self, x, y):
        r = (2 - np.pi * np.sin(np.pi * x)) * (-1 + np.pi * np.sin(np.pi * (1 - y)))
        return r.flatten()

    def u1_exact_solution(self, x, y):
        r = x ** 2 * y ** 2 + np.exp(-y)
        return r.flatten()

    def u1_exact_solution_x_derivative(self, x, y):
        r = 2 * x * y ** 2
        return r.flatten()

    def u1_exact_solution_y_derivative(self, x, y):
        r = 2 * x ** 2 * y - np.exp(-y)
        return r.flatten()

    def u2_exact_solution(self, x, y):
        r = -2 * x * y ** 3 / 3 + 2 - np.pi * np.sin(np.pi * x)
        return r.flatten()

    def u2_exact_solution_x_derivative(self, x, y):
        r = -2 * y ** 3 / 3 - np.pi ** 2 * np.cos(np.pi * x)
        return r.flatten()

    def u2_exact_solution_y_derivative(self, x, y):
        r = -2 * x * y ** 2
        return r.flatten()
