import torch

class Functions_torch:
    def __init__(self, device='cuda'):
        self.device = torch.device(device)

    def function_BJ_coefficient(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_BJSJ_coefficient(self, x, y):
        r = torch.tensor(1.0, device=self.device)
        return r

    def function_fix_p(self, x, y):
        r = torch.tensor(0.0, device=self.device)  # p_exact_solution
        return r.flatten()

    def function_g1_Stokes(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))  # 计算 π
        c = K / pi ** 2
        r=c*pi*torch.sin(2*pi*y)*torch.cos(x)
        return r.flatten()

    def function_g2_Stokes(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))  # 计算 π
        c = K / pi ** 2
        r = r=(-2*K+c*torch.sin(pi*y)**2)*torch.sin(x)
        return r.flatten()

    def function_g_Poisson(self, x, y):
        pi = torch.acos(torch.tensor(-1.0))
        # (np.exp(y) - np.exp(-y)) * np.sin(x)
        r = (torch.exp(y) - torch.exp(-y)) * torch.sin(x)
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

    def p_exact_solution(self, x, y):
        r = torch.tensor(0.0, device=self.device)
        return r.flatten()

    def p_exact_solution_x_derivative(self, x, y):
        r = torch.tensor(0.0, device=self.device)
        return r.flaten()

    def p_exact_solution_y_derivative(self, x, y):
        r = torch.tensor(0.0, device=self.device)
        return r.flatten()

    def phi_exact_solution(self, x, y):
        #r=(exp(y)-exp(-y)).*sin(x);
        r = (torch.exp(y) - torch.exp(-y)) * torch.sin(x)
        return r.flatten()

    def phi_exact_solution_x_derivative(self, x, y):
        #r=(exp(y)-exp(-y)).*cos(x);
        r = (torch.exp(y) - torch.exp(-y)) * torch.cos(x)
        return r.flatten()

    def phi_exact_solution_y_derivative(self, x, y):
        #r=(exp(y)+exp(-y)).*sin(x);
        r = (torch.exp(y) + torch.exp(-y)) * torch.sin(x)
        return r.flatten()

    def u1_exact_solution(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))
        c = K / pi ** 2
        r = c * pi * torch.sin(2 * pi * y) * torch.cos(x)
        return r.flatten()

    def u1_exact_solution_x_derivative(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))
        c = K / pi ** 2
        r = -c * pi * torch.sin(2 * pi * y) * torch.sin(x)
        return r.flatten()

    def u1_exact_solution_y_derivative(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))
        c = K / pi ** 2
        r=2 * c * pi**2 * torch.cos(2 * pi * y) * torch.cos(x)
        return r.flatten()

    def u2_exact_solution(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))
        c = K / pi ** 2
        r=(-2*K+c*torch.sin(pi*y)**2)*torch.sin(x)
        return r.flatten()

    def u2_exact_solution_x_derivative(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))
        c = K / pi ** 2
        r=(-2*K+c*torch.sin(pi*y)**2)*torch.cos(x)
        return r.flatten()

    def u2_exact_solution_y_derivative(self, x, y):
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))
        c = K / pi ** 2
        r = c * pi * torch.sin(2 * pi * y) * torch.sin(x)
        return r.flatten()

    def function_f1_Stokes(self, x, y):
        nu = self.function_nu(x, y)
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))  # 计算 π
        c = K / pi ** 2
        r = (nu * c * pi + 4 * nu * c * pi ** 3) * torch.sin(2 * pi * y) * torch.cos(x)
        return r.flatten()

    def function_f2_Stokes(self,x, y):
        nu = self.function_nu(x, y)
        K = torch.tensor(1.0, device=self.device)
        pi = torch.acos(torch.tensor(-1.0))  # 计算 π
        c = K / pi ** 2
        r = -2*nu*c*pi**2* torch.cos(2*pi*y)*torch.sin(x)+nu*(-2*K+c*torch.sin(pi*y)**2)*torch.sin(x)
        return r.flatten()

    def function_f_Poisson(slef,x, y):
        r = torch.tensor(0.0, device=slef.device)
        return r.flatten()