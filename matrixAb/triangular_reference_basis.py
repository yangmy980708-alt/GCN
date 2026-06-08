import numpy as np
"""basis type =2, dx=0, dy=0"""
def function_b2dx0dy0bi_1(x_hat, y_hat):
    r = 1 - 3 * x_hat - 3 * y_hat + 2 * x_hat ** 2 + 2 * y_hat ** 2 + 4 * x_hat * y_hat
    return r.flatten()

def function_b2dx0dy0bi_2(x_hat,y_hat):
    r = 2 * x_hat**2 - x_hat
    return r.flatten()

def function_b2dx0dy0bi_3(x_hat,y_hat):
    r = 2 * y_hat**2 - y_hat
    return r.flatten()

def function_b2dx0dy0bi_4(x_hat,y_hat):
    r = 4 * x_hat - 4 * x_hat**2 - 4 * x_hat * y_hat
    return r.flatten()

def function_b2dx0dy0bi_5(x_hat,y_hat):
    r = 4 * x_hat * y_hat
    return r.flatten()

def function_b2dx0dy0bi_6(x_hat,y_hat):
    r = 4 * y_hat - 4 * y_hat**2 - 4 * x_hat * y_hat
    return r.flatten()

"""basis type =2, dx=1, dy=0"""
def function_b2dx1dy0bi_1(x_hat, y_hat):
    r = -3 + 4 * x_hat + 4 * y_hat
    return r.flatten()

def function_b2dx1dy0bi_2(x_hat,y_hat):
    r = 4 * x_hat - 1
    return r.flatten()

def function_b2dx1dy0bi_3(x_hat,y_hat):
    r = 0
    return r

def function_b2dx1dy0bi_4(x_hat,y_hat):
    r = 4 - 8 * x_hat - 4 * y_hat
    return r.flatten()

def function_b2dx1dy0bi_5(x_hat,y_hat):
    r = 4 * y_hat
    return r.flatten()

def function_b2dx1dy0bi_6(x_hat,y_hat):
    r = -4 * y_hat
    return r.flatten()

"""basis type =2, dx=0, dy=1"""
def function_b2dx0dy1bi_1(x_hat, y_hat):
    r = -3 + 4 * y_hat + 4 * x_hat
    return r.flatten()

def function_b2dx0dy1bi_2(x_hat,y_hat):
    r = 0
    return r

def function_b2dx0dy1bi_3(x_hat,y_hat):
    r = 4 * y_hat - 1
    return r.flatten()

def function_b2dx0dy1bi_4(x_hat,y_hat):
    r = -4 * x_hat
    return r.flatten()

def function_b2dx0dy1bi_5(x_hat,y_hat):
    r = 4 * x_hat
    return r.flatten()

def function_b2dx0dy1bi_6(x_hat,y_hat):
    r = 4 - 8 * y_hat - 4 * x_hat
    return r.flatten()

"""basis type =2, dx=2, dy=0"""
def function_b2dx2dy0bi_1(x_hat, y_hat):
    r = 4
    return r

def function_b2dx2dy0bi_2(x_hat,y_hat):
    r = 4
    return r

def function_b2dx2dy0bi_3(x_hat,y_hat):
    r = 0
    return r

def function_b2dx2dy0bi_4(x_hat,y_hat):
    r = -8
    return r

def function_b2dx2dy0bi_5(x_hat,y_hat):
    r = 0
    return r

def function_b2dx2dy0bi_6(x_hat,y_hat):
    r = 0
    return r

"""basis type =2, dx=0, dy=2"""
def function_b2dx0dy2bi_1(x_hat, y_hat):
    r = 4
    return r

def function_b2dx0dy2bi_2(x_hat,y_hat):
    r = 0
    return r

def function_b2dx0dy2bi_3(x_hat,y_hat):
    r = 4
    return r

def function_b2dx0dy2bi_4(x_hat,y_hat):
    r = 0
    return r

def function_b2dx0dy2bi_5(x_hat,y_hat):
    r = 0
    return r

def function_b2dx0dy2bi_6(x_hat,y_hat):
    r = -8
    return r

"""basis type =2, dx=1, dy=1"""
def function_b2dx1dy1bi_1(x_hat, y_hat):
    r = 4
    return r

def function_b2dx1dy1bi_2(x_hat,y_hat):
    r = 0
    return r

def function_b2dx1dy1bi_3(x_hat,y_hat):
    r = 0
    return r

def function_b2dx1dy1bi_4(x_hat,y_hat):
    r = -4
    return r

def function_b2dx1dy1bi_5(x_hat,y_hat):
    r = 4
    return r

def function_b2dx1dy1bi_6(x_hat,y_hat):
    r = -4
    return r

"basis type=1 dx=0 dy=0"
def function_b1dx0dy0bi_1(x_hat,y_hat):
    r = 1 - x_hat - y_hat
    return r.flatten()

def function_b1dx0dy0bi_2(x_hat,y_hat):
    r = x_hat
    return r

def function_b1dx0dy0bi_3(x_hat,y_hat):
    r = y_hat
    return r

"basis type=1 dx=1 dy=0"
def function_b1dx1dy0bi_1(x_hat,y_hat):
    r = -1
    return r

def function_b1dx1dy0bi_2(x_hat,y_hat):
    r = 1
    return r

def function_b1dx1dy0bi_3(x_hat,y_hat):
    r = 0
    return r

"basis type=1 dx=0 dy=1"
def function_b1dx0dy1bi_1(x_hat,y_hat):
    r = -1
    return r

def function_b1dx0dy1bi_2(x_hat,y_hat):
    r = 0
    return r

def function_b1dx0dy1bi_3(x_hat,y_hat):
    r = 1
    return r

def triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, derivative_degree_x, derivative_degree_y):

    if basis_type == 2:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 0:
                return function_b2dx0dy0bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b2dx0dy0bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx0dy0bi_3(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx0dy0bi_4(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx0dy0bi_5(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx0dy0bi_6(x_hat,y_hat)

        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 0:
                return function_b2dx1dy0bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b2dx1dy0bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx1dy0bi_3(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx1dy0bi_4(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx1dy0bi_5(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx1dy0bi_6(x_hat,y_hat)

        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 0:
                return function_b2dx0dy1bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b2dx0dy1bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx0dy1bi_3(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx0dy1bi_4(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx0dy1bi_5(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx0dy1bi_6(x_hat,y_hat)

        elif derivative_degree_x == 2 and derivative_degree_y == 0:
            if basis_index == 0:
                return function_b2dx2dy0bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b2dx2dy0bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx2dy0bi_3(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx2dy0bi_4(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx2dy0bi_5(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx2dy0bi_6(x_hat,y_hat)

        elif derivative_degree_x == 0 and derivative_degree_y == 2:
            if basis_index == 0:
                return function_b2dx0dy2bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b2dx0dy2bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx0dy2bi_3(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx0dy2bi_4(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx0dy2bi_5(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx0dy2bi_6(x_hat,y_hat)

        elif derivative_degree_x == 1 and derivative_degree_y == 1:
            if basis_index == 0:
                return function_b2dx1dy1bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b2dx1dy1bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx1dy1bi_3(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx1dy1bi_4(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx1dy1bi_5(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx1dy1bi_6(x_hat,y_hat)

    elif basis_type == 1:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 0:
                return function_b1dx0dy0bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b1dx0dy0bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b1dx0dy0bi_3(x_hat,y_hat)
        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 0:
                return function_b1dx1dy0bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b1dx1dy0bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b1dx1dy0bi_3(x_hat,y_hat)
        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 0:
                return function_b1dx0dy1bi_1(x_hat,y_hat)
            elif basis_index == 1:
                return function_b1dx0dy1bi_2(x_hat,y_hat)
            elif basis_index == 2:
                return function_b1dx0dy1bi_3(x_hat,y_hat)

