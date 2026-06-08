import torch

class Functions_torch_t:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)

    def function_BJ_coefficient(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_BJSJ_coefficient(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_fix_p(self, x, y, t):
        r = self.p_exact_solution(x, y, t)
        return r.flatten

    def function_g1_Stokes(self, x, y, t):
        r = x ** 2 * y ** 2 + torch.exp(-y)
        return r.flatten()

    def function_g2_Stokes(self, x, y, t):
        pi = torch.acos(torch.tensor(-1.0))
        r = (-2 / 3) * x * y ** 3 + 2 - pi * torch.sin(pi * x) * torch.cos(2 * pi * t)
        return r.flatten()

    def function_g_Poisson(self, x, y,t):
        pi = torch.acos(torch.tensor(-1.0))
        r = (2-pi*torch.sin(pi*x))*(-y+torch.cos(pi*(1-y)))*torch.cos(2*pi*t)
        return r.flatten()

    def function_gravity(self, x, y):
        return torch.tensor(1.0, device=self.device)

    def function_relative_depth(self, x, y):
        return torch.tensor(0.0, device=self.device)

    def function_gravity_depth(self, x, y):
        r = self.function_gravity(x, y) * self.function_relative_depth(x, y)
        return r

    def function_k11(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_k12(self, x, y):
        r = torch.tensor(0.0, device=self.device)
        return r

    def function_k21(self,x, y):
        r = torch.tensor(0.0, device=self.device)
        return r

    def function_k22(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_negativeone(self, x, y):
        r = -torch.tensor(1.0, device=self.device)
        return r

    def function_nu(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_one(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_initial_p(self, x, y):
        pi = torch.acos(torch.tensor(-1.0))
        r = -(2 - pi * torch.sin(pi * x)) * torch.cos(2 * pi * y)
        return r

    def p_exact_solution(self, x, y, t):
        pi = torch.acos(torch.tensor(-1.0))
        r = -(2-pi * torch.sin(pi*x)) * torch.cos(2*pi*y) * torch.cos(2*pi * t)
        return r.flatten()

    def p_exact_solution_x_derivative(self, x, y):
        pi = torch.acos(torch.tensor(-1.0))
        r = pi**2 * torch.cos(pi*x) * torch.cos(2*pi*y)
        return r.flaten()

    def p_exact_solution_y_derivative(self, x, y):
        pi = torch.acos(torch.tensor(-1.0))
        r = -2* pi * torch.sin(2*pi*y) * (pi*torch.sin(pi*x) - 2)
        return r.flatten()

    def function_initial_phi(self, x, y):
        pi = torch.acos(torch.tensor(-1.0))
        r = (2 - pi * torch.sin(pi * x)) * (-y + torch.cos(pi * (1 - y)))
        return r.flatten()

    def phi_exact_solution(self, x, y, t):
        pi = torch.acos(torch.tensor(-1.0))
        r = (2 - pi * torch.sin(pi * x)) * (-y + torch.cos(pi * (1 - y))) * torch.cos(2 * pi * t)
        return r.flatten()

    def phi_exact_solution_x_derivative(self, x, y, t):
        pi = torch.acos(torch.tensor(-1.0))
        r = pi ** 2 * torch.cos(2 * pi * t) * torch.cos(pi * x) * (y + torch.cos(pi * y))
        return r

    def phi_exact_solution_y_derivative(self, x, y,t):
        pi = torch.acos(torch.tensor(-1.0))
        r = -torch.cos(2*pi*t)*(pi*torch.sin(pi*x) - 2)*(pi*torch.sin(pi*y) - 1)
        return r.flatten()

    def function_initial_u1(self, x, y):
        r = x ** 2 * y ** 2 + torch.exp(-y)
        return r

    def u1_exact_solution(self, x, y):
        r = x**2 * y**2+ torch.exp(-y)
        return r.flatten()

    def u1_exact_solution_x_derivative(self, x, y):
        r = 2*x*y**2
        return r.flatten()

    def u1_exact_solution_y_derivative(self, x, y):
        r = 2*y*x**2 - torch.exp(-y)
        return r.flatten()

    def function_initial_u2(self, x, y):
        pi = torch.acos(torch.tensor(-1.0))
        r = (-2 / 3) * x * y ** 3 + 2 - pi * torch.sin(pi * x)
        return r.flatten()

    def u2_exact_solution(self, x, y,t):
        pi = torch.acos(torch.tensor(-1.0))
        r = (-2 / 3) * x * y ** 3 + 2 - pi * torch.sin(pi * x) * torch.cos(2 * pi * t)
        return r.flatten()

    def u2_exact_solution_x_derivative(self, x, y):
        pi = torch.acos(torch.tensor(-1.0))
        r = - pi**2 * torch.cos(pi*x) - (2*y**3)/3
        return r.flatten()

    def u2_exact_solution_y_derivative(self, x, y):
        r = -2*x*y**2
        return r.flatten()

    def function_f1_Stokes(self, x, y, t):
        nu = self.function_nu(x, y)
        pi = torch.acos(torch.tensor(-1.0))  # 计算 π
        r = -2 * pi * (x ** 2 * y ** 2 + torch.exp(-y)) * torch.sin(2 * pi * t) \
            + (-2 * nu * x ** 2 - 2 * nu * y ** 2 - nu * torch.exp(-y) \
               + pi ** 2 * torch.cos(pi * x) * torch.cos(2 * pi * y)) * torch.cos(2 * pi * t)
        return r.flatten()

    def function_f2_Stokes(self,x, y, t):
        nu = self.function_nu(x, y)
        pi = torch.acos(torch.tensor(-1.0))  # 计算 π
        r = -2 * pi * (-2 / 3 * x * y ** 3 + 2 - pi * torch.sin(pi * x)) * torch.sin(2 * pi * t) \
            + (4 * nu * x * y - nu * pi ** 3 * torch.sin(pi * x) \
               + 2 * pi * (2 - pi * torch.sin(pi * x)) * torch.sin(2 * pi * y)) * torch.cos(2 * pi * t)
        return r.flatten()

    def function_f_Poisson(slef,x, y, t):
        pi = torch.acos(torch.tensor(-1.0))  # 计算 π
        r = pi ** 3 * torch.cos(2 * pi * t) * torch.sin(pi * x) * (y + torch.cos(pi * y)) \
            - 2 * pi * torch.sin(2 * pi * t) * (y + torch.cos(pi * y)) * (pi * torch.sin(pi * x) - 2) \
            + pi ** 2 * torch.cos(2 * pi * t) * torch.cos(pi * y) * (pi * torch.sin(pi * x) - 2)
        return r