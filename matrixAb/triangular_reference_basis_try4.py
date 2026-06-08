import numpy as np
"""basis type =2, dx=0, dy=0"""
def function_b2dx0dy0bi_1(x_hat, y_hat):
    r = 1 - 3 * x_hat - 3 * y_hat + 2 * x_hat ** 2 + 2 * y_hat ** 2 + 4 * x_hat * y_hat
    return r.flattne()

def function_b2dx0dy0bi_2(x_hat,y_hat):
    r = 2 * x_hat**2 - x_hat
    return r.flattne()

def function_b2dx0dy0bi_3(x_hat,y_hat):
    r = 2 * y_hat**2 - y_hat
    return r.flattne()

def function_b2dx0dy0bi_4(x_hat,y_hat):
    r = 4 * x_hat - 4 * x_hat**2 - 4 * x_hat * y_hat
    return r.flattne()

def function_b2dx0dy0bi_5(x_hat,y_hat):
    r = 4 * x_hat * y_hat
    return r.flattne()

def function_b2dx0dy0bi_6(x_hat,y_hat):
    r = 4 * y_hat - 4 * y_hat**2 - 4 * x_hat * y_hat
    return r.flattne()

"""basis type =2, dx=1, dy=0"""
def function_b2dx1dy0bi_1(x_hat, y_hat):
    r = -3 + 4 * x_hat + 4 * y_hat
    return r.flattne()

def function_b2dx1dy0bi_2(x_hat,y_hat):
    r = 4 * x_hat - 1
    return r.flattne()

def function_b2dx1dy0bi_3(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx1dy0bi_4(x_hat,y_hat):
    r = 4 - 8 * x_hat - 4 * y_hat
    return r.flattne()

def function_b2dx1dy0bi_5(x_hat,y_hat):
    r = 4 * y_hat
    return r.flattne()

def function_b2dx1dy0bi_6(x_hat,y_hat):
    r = -4 * y_hat
    return r.flattne()

"""basis type =2, dx=0, dy=1"""
def function_b2dx0dy1bi_1(x_hat, y_hat):
    r = -3 + 4 * y_hat + 4 * x_hat
    return r.flattne()

def function_b2dx0dy1bi_2(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx0dy1bi_3(x_hat,y_hat):
    r = 4 * y_hat - 1
    return r.flattne()

def function_b2dx0dy1bi_4(x_hat,y_hat):
    r = -4 * x_hat
    return r.flattne()

def function_b2dx0dy1bi_5(x_hat,y_hat):
    r = 4 * x_hat
    return r.flattne()

def function_b2dx0dy1bi_6(x_hat,y_hat):
    r = 4 - 8 * y_hat - 4 * x_hat
    return r.flattne()

"""basis type =2, dx=2, dy=0"""
def function_b2dx2dy0bi_1(x_hat, y_hat):
    r = 4
    return r.flattne()

def function_b2dx2dy0bi_2(x_hat,y_hat):
    r = 4
    return r.flattne()

def function_b2dx2dy0bi_3(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx2dy0bi_4(x_hat,y_hat):
    r = -8
    return r.flattne()

def function_b2dx2dy0bi_5(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx2dy0bi_6(x_hat,y_hat):
    r = 0
    return r.flattne()

"""basis type =2, dx=0, dy=2"""
def function_b2dx0dy2bi_1(x_hat, y_hat):
    r = 4
    return r.flattne()

def function_b2dx0dy2bi_2(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx0dy2bi_3(x_hat,y_hat):
    r = 4
    return r.flattne()

def function_b2dx0dy2bi_4(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx0dy2bi_5(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx0dy2bi_6(x_hat,y_hat):
    r = -8
    return r.flattne()

"""basis type =2, dx=1, dy=1"""
def function_b2dx1dy1bi_1(x_hat, y_hat):
    r = 4
    return r.flattne()

def function_b2dx1dy1bi_2(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx1dy1bi_3(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b2dx1dy1bi_4(x_hat,y_hat):
    r = -4
    return r.flattne()

def function_b2dx1dy1bi_5(x_hat,y_hat):
    r = 4
    return r.flattne()

def function_b2dx1dy1bi_6(x_hat,y_hat):
    r = -4
    return r.flattne()

"basis type=1 dx=0 dy=0"
def function_b1dx0dy0bi_1(x_hat,y_hat):
    r = 1 - x_hat - y_hat
    return r.flattne()

def function_b1dx0dy0bi_2(x_hat,y_hat):
    r = x_hat
    return r.flattne()

def function_b1dx0dy0bi_3(x_hat,y_hat):
    r = y_hat
    return r.flattne()

"basis type=1 dx=1 dy=0"
def function_b1dx1dy0bi_1(x_hat,y_hat):
    r = -1
    return r.flattne()

def function_b1dx1dy0bi_2(x_hat,y_hat):
    r = 1
    return r.flattne()

def function_b1dx1dy0bi_3(x_hat,y_hat):
    r = 0
    return r.flattne()

"basis type=1 dx=0 dy=1"
def function_b1dx0dy1bi_1(x_hat,y_hat):
    r = -1
    return r.flattne()

def function_b1dx0dy1bi_2(x_hat,y_hat):
    r = 0
    return r.flattne()

def function_b1dx0dy1bi_3(x_hat,y_hat):
    r = 1
    return r.flattne()

def triangular_reference_basis(x_hat, y_hat, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
    """
    Reference finite element (FE) basis function on triangle ABC (A=(0,0), B=(1,0), C=(0,1)).
    Converted from MATLAB code provided by Xiaoming He, 07/01/2009.
    """

    if basis_type == 2:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 1:
                return function_b2dx0dy0bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx0dy0bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx0dy0bi_3(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx0dy0bi_4(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx0dy0bi_5(x_hat,y_hat)
            elif basis_index == 6:
                return function_b2dx0dy0bi_6(x_hat,y_hat)

        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 1:
                return function_b2dx1dy0bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx1dy0bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx1dy0bi_3(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx1dy0bi_4(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx1dy0bi_5(x_hat,y_hat)
            elif basis_index == 6:
                return function_b2dx1dy0bi_6(x_hat,y_hat)

        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 1:
                return function_b2dx0dy1bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx0dy1bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx0dy1bi_3(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx0dy1bi_4(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx0dy1bi_5(x_hat,y_hat)
            elif basis_index == 6:
                return function_b2dx0dy1bi_6(x_hat,y_hat)

        elif derivative_degree_x == 2 and derivative_degree_y == 0:
            if basis_index == 1:
                return function_b2dx2dy0bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx2dy0bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx2dy0bi_3(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx2dy0bi_4(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx2dy0bi_5(x_hat,y_hat)
            elif basis_index == 6:
                return function_b2dx2dy0bi_6(x_hat,y_hat)

        elif derivative_degree_x == 0 and derivative_degree_y == 2:
            if basis_index == 1:
                return function_b2dx0dy2bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx0dy2bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx0dy2bi_3(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx0dy2bi_4(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx0dy2bi_5(x_hat,y_hat)
            elif basis_index == 6:
                return function_b2dx0dy2bi_6(x_hat,y_hat)

        elif derivative_degree_x == 1 and derivative_degree_y == 1:
            if basis_index == 1:
                return function_b2dx1dy1bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b2dx1dy1bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b2dx1dy1bi_3(x_hat,y_hat)
            elif basis_index == 4:
                return function_b2dx1dy1bi_4(x_hat,y_hat)
            elif basis_index == 5:
                return function_b2dx1dy1bi_5(x_hat,y_hat)
            elif basis_index == 6:
                return function_b2dx1dy1bi_6(x_hat,y_hat)

    elif basis_type == 1:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 1:
                return function_b1dx0dy0bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b1dx0dy0bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b1dx0dy0bi_3(x_hat,y_hat)
        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 1:
                return function_b1dx1dy0bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b1dx1dy0bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b1dx1dy0bi_3(x_hat,y_hat)
        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 1:
                return function_b1dx0dy1bi_1(x_hat,y_hat)
            elif basis_index == 2:
                return function_b1dx0dy1bi_2(x_hat,y_hat)
            elif basis_index == 3:
                return function_b1dx0dy1bi_3(x_hat,y_hat)

