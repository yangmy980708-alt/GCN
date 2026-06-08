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


        def function_fix_p(self, x, y, t):
            r = self.p_exact_solution(x,y,t)
            return r

        def function_g1_Stokes(self, x, y, t):
            r = x**2 * y**2+ np.exp(-y)
            return r.flatten()

        def function_g2_Stokes(self, x, y, t):
            r = (-2/3) * x * y**3 + 2-np.pi*np.sin(np.pi*x)*np.cos(2*np.pi*t)
            return r.flatten()

        def function_g_Poisson(self, x, y, t):
            r = (2-np.pi*np.sin(np.pi*x))*(-y+np.cos(np.pi*(1-y)))*np.cos(2*np.pi*t)
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

        def function_initial_p(self, x, y):
            r = -(2 - np.pi * np.sin(np.pi * x)) * np.cos(2 * np.pi * y)
            return r

        def p_exact_solution(self, x, y,t):
            r = -(2-np.pi * np.sin(np.pi*x)) * np.cos(2*np.pi*y) * np.cos(2*np.pi * t)
            return r

        def p_exact_solution_x_derivative(self, x, y):
            r = np.pi**2 * np.cos(np.pi*x) * np.cos(2*np.pi*y)
            return r

        def p_exact_solution_y_derivative(self, x, y):
            r = -2*np.pi * np.sin(2*np.pi*y) * (np.pi*np.sin(np.pi*x) - 2)
            return r

        def function_initial_phi(self, x, y):
            r = (2-np.pi*np.sin(np.pi*x))*(-y+np.cos(np.pi*(1-y)))
            return  r

        def phi_exact_solution(self, x, y, t):
            r = (2-np.pi*np.sin(np.pi*x))*(-y+np.cos(np.pi*(1-y)))*np.cos(2*np.pi*t)
            return r.flatten()

        def phi_exact_solution_x_derivative(self, x, y, t):
            r = np.pi**2 * np.cos(2*np.pi*t)* np.cos(np.pi*x)*(y + np.cos(np.pi*y))
            return r.flatten()

        def phi_exact_solution_y_derivative(self, x, y, t):
            r = -np.cos(2*np.pi*t)*(np.pi*np.sin(np.pi*x) - 2)*(np.pi*np.sin(np.pi*y) - 1)
            return r.flatten()

        def function_initial_u1(self,x,y):
            r = x**2 * y**2+ np.exp(-y)
            return r

        def u1_exact_solution(self, x, y):
            r = x**2 * y**2+ np.exp(-y)
            return r.flatten()

        def u1_exact_solution_x_derivative(self, x, y):
            r = 2*x*y**2
            return r.flatten()

        def u1_exact_solution_y_derivative(self, x, y):
            r = 2*y*x**2 - np.exp(-y)
            return r.flatten()

        def function_initial_u2(self, x, y):
            r = (-2 / 3) * x * y ** 3 + 2 - np.pi * np.sin(np.pi * x)
            return r.flatten()

        def u2_exact_solution(self, x, y,t):
            r = (-2/3) * x * y**3 + 2-np.pi*np.sin(np.pi*x)*np.cos(2*np.pi*t)
            return r.flatten()

        def u2_exact_solution_x_derivative(self, x, y):
            r = - np.pi**2 * np.cos(np.pi*x) - (2*y**3)/3
            return r.flatten()

        def u2_exact_solution_y_derivative(self, x, y):
            r = -2*x*y**2
            return r.flatten()

        def function_f1_Stokes(self, x, y,t):
            nu=1
            r = -2 * np.pi * (x**2 * y**2 + np.exp(-y)) * np.sin(2 * np.pi * t)\
            +(-2 * nu * x**2 - 2 * nu * y**2 - nu * np.exp(-y) \
             + np.pi**2 * np.cos(np.pi * x) * np.cos(2 * np.pi * y)) * np.cos(2 * np.pi * t)
            return r.flatten()

        def function_f2_Stokes(self, x, y, t):
            nu=1
            r =-2 * np.pi * (-2/3 * x * y**3 + 2 - np.pi * np.sin(np.pi * x)) * np.sin(2 * np.pi * t)\
               + (4 * nu * x * y - nu * np.pi**3 * np.sin(np.pi * x) \
               + 2 * np.pi * (2 - np.pi * np.sin(np.pi * x)) * np.sin(2 * np.pi * y)) * np.cos(2 * np.pi * t)
            return r.flatten()

        def function_f_Poisson(self, x, y,t):
            r = np.pi**3 * np.cos(2*np.pi*t) * np.sin(np.pi*x) * (y + np.cos(np.pi*y)) \
                - 2 * np.pi * np.sin(2*np.pi*t) * (y + np.cos(np.pi*y)) * (np.pi*np.sin(np.pi*x) - 2) \
                + np.pi**2 * np.cos(2*np.pi*t) * np.cos(np.pi*y) * (np.pi*np.sin(np.pi*x) - 2)
            return r