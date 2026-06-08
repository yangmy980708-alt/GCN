import numpy as np


class Functions:
    def __init__(self):
        pass

    import numpy as np

    class NumpyFunctions:
        def __init__(self):
            pass

        def function_BJ_coefficient(self, x, y):
            r = 1
            return r

        def function_BJSJ_coefficient(self, x, y):
            r = 1
            return r

        def function_fix_p(self, x, y):
            r = 0  # p_exact_solution
            return r

        def function_g1_Stokes(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = c * np.pi * np.sin(2 * np.pi * y) * np.cos(x)
            return r.flatten()

        def function_g2_Stokes(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = (-2 * K + c * np.sin(np.pi * y) ** 2) * np.sin(x)
            return r.flatten()

        def function_g_Poisson(self, x, y):
            r = (np.exp(y)-np.exp(-y))*np.sin(x)
            return r.flatten()

        def function_gravity(self, x, y):
            return 1

        def function_relative_depth(self, x, y):
            return 0

        def function_gravity_depth(self, x, y):
            r = self.function_gravity(x, y) * self.function_relative_depth(x, y)
            return r

        def function_k11(self, x, y):
            r = 1
            return r

        def function_k12(self, x, y):
            r = 0
            return r

        def function_k21(self, x, y):
            r = 0
            return r

        def function_k22(self, x, y):
            r = 1
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
            r = 0
            return r

        def p_exact_solution_x_derivative(self, x, y):
            r = 0
            return r

        def p_exact_solution_y_derivative(self, x, y):
            r = 0
            return r

        def phi_exact_solution(self, x, y):
            r = (np.exp(y) - np.exp(-y)) * np.sin(x)
            return r.flatten()

        def phi_exact_solution_x_derivative(self, x, y):
            r = (np.exp(y) - np.exp(-y)) * np.cos(x)
            return r.flatten()

        def phi_exact_solution_y_derivative(self, x, y):
            r = (np.exp(y) + np.exp(-y)) * np.sin(x)
            return r.flatten()

        def u1_exact_solution(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = c * np.pi * np.sin(2 * np.pi * y) * np.cos(x)
            return r.flatten()

        def u1_exact_solution_x_derivative(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = -c * np.pi * np.sin(2 * np.pi * y) * np.sin(x)
            return r.flatten()

        def u1_exact_solution_y_derivative(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = 2 * c * np.pi ** 2 * np.cos(2 * np.pi * y) * np.cos(x)
            return r.flatten()

        def u2_exact_solution(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = (-2 * K + c * np.sin(np.pi * y) ** 2) * np.sin(x)
            return r.flatten()

        def u2_exact_solution_x_derivative(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = (-2 * K + c * np.sin(np.pi * y) ** 2) * np.cos(x)
            return r.flatten()

        def u2_exact_solution_y_derivative(self, x, y):
            K = 1
            c = K / np.pi ** 2
            r = c * np.pi * np.sin(2 * np.pi * y) * np.sin(x)
            return r.flatten()

        def function_f1_Stokes(self, x, y):
            nu = self.function_nu(x, y)
            K = 1
            c = K / np.pi ** 2
            r = (nu * c * np.pi + 4 * nu * c * np.pi ** 3) * np.sin(2 * np.pi * y) * np.cos(x)
            return r.flatten()

        def function_f2_Stokes(self, x, y):
            nu = self.function_nu(x, y)
            K = 1
            c = K / np.pi ** 2
            r = -2 * nu * c * np.pi ** 2 * np.cos(2 * np.pi * y) * np.sin(x) + nu * (
                        -2 * K + c * np.sin(np.pi * y) ** 2) * np.sin(x)
            return r.flatten()

        def function_f_Poisson(self, x, y):
            r = 0
            return r