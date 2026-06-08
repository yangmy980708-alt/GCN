import numpy as np

def data_NSf():
    coe_fun_nu = 1
    coe_fun_negativeone = 1

    exactu1 = lambda x, y: (x**2) * (y**2) + np.exp(-y)
    exactu2 = lambda x, y: -2/3 * x * y**3 + 2 - np.pi * np.sin(np.pi * x)
    exactp = lambda x, y: -(2 - np.pi * np.sin(np.pi * x)) * np.cos(2 * np.pi * y)

    f1 = lambda x, y: -2 * x**2 - 2 * y**2 - np.exp(-y) + (np.pi**2) * np.cos(2 * np.pi * y) * np.cos(np.pi * x)
    f2 = lambda x, y: 4 * x * y - (np.pi**3) * np.sin(np.pi * x) + 2 * np.pi * (2 - np.pi * np.sin(np.pi * x)) * np.sin(2 * np.pi * y)

    Du1x = lambda x, y: 2 * x * y**2
    Du1y = lambda x, y: 2 * x**2 * y - np.exp(-y)

    Du2x = lambda x, y: -2/3 * y**3 - np.pi**2 * np.cos(np.pi * x)
    Du2y = lambda x, y: -2 * x * y**2

    Dpx = lambda x, y: np.pi**2 * np.cos(np.pi * x) * np.cos(2 * np.pi * y)
    Dpy = lambda x, y: -(2 - np.pi * np.sin(np.pi * x)) * (-2 * np.pi * np.sin(2 * np.pi * y))

    return {
        'coe_fun_nu': coe_fun_nu,
        'coe_fun_negativeone': coe_fun_negativeone,
        'exactu1': exactu1,
        'exactu2': exactu2,
        'exactp': exactp,
        'f1': f1,
        'f2': f2,
        'Du1x': Du1x,
        'Du1y': Du1y,
        'Du2x': Du2x,
        'Du2y': Du2y,
        'Dpx': Dpx,
        'Dpy': Dpy
    }

# 使用示例
# data = data3()
# x, y = 1.0, 1.0
# print(data['exactu1'](x, y))  # 计算 exactu1 在 (x, y) 处的值
