def triangular_reference_basis(x, y, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
    """
    Reference finite element (FE) basis function on triangle ABC (A=(0,0), B=(1,0), C=(0,1)).
    Converted from MATLAB code provided by Xiaoming He, 07/01/2009.
    """
    if basis_type == 2:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 1:
                #lambda x, y ,basis_type, basis_index, derivative_degree_x, derivative_degree_y:
                r =  1 - 3 * x - 3 * y + 2 * x**2 + 2 * y**2 + 4 * x * y
            elif basis_index == 2:
                r = 2 * x**2 - x
            elif basis_index == 3:
                r = 2 * y**2 - y
            elif basis_index == 4:
                r = 4 * x - 4 * x**2 - 4 * x * y
            elif basis_index == 5:
                r = 4 * x * y
            elif basis_index == 6:
                r = 4 * y - 4 * y**2 - 4 * x * y

        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 1:
                r = -3 + 4 * x + 4 * y
            elif basis_index == 2:
                r = 4 * x - 1
            elif basis_index == 3:
                r = 0
            elif basis_index == 4:
                r = 4 - 8 * x - 4 * y
            elif basis_index == 5:
                r = 4 * y
            elif basis_index == 6:
                r = -4 * y

        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 1:
                r = -3 + 4 * y + 4 * x
            elif basis_index == 2:
                r = 0
            elif basis_index == 3:
                r = 4 * y - 1
            elif basis_index == 4:
                r = -4 * x
            elif basis_index == 5:
                r = 4 * x
            elif basis_index == 6:
                r = 4 - 8 * y - 4 * x

        elif derivative_degree_x == 2 and derivative_degree_y == 0:
            if basis_index == 1:
                r = 4
            elif basis_index == 2:
                r = 4
            elif basis_index == 3:
                r = 0
            elif basis_index == 4:
                r = -8
            elif basis_index == 5:
                r = 0
            elif basis_index == 6:
                r = 0

        elif derivative_degree_x == 0 and derivative_degree_y == 2:
            if basis_index == 1:
                r = 4
            elif basis_index == 2:
                r = 0
            elif basis_index == 3:
                r = 4
            elif basis_index == 4:
                r = 0
            elif basis_index == 5:
                r = 0
            elif basis_index == 6:
                r = -8

        elif derivative_degree_x == 1 and derivative_degree_y == 1:
            if basis_index == 1:
                r = 4
            elif basis_index == 2:
                r = 0
            elif basis_index == 3:
                r = 0
            elif basis_index == 4:
                r = -4
            elif basis_index == 5:
                r = 4
            elif basis_index == 6:
                r = -4

    elif basis_type == 1:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 1:
                r = 1 - x - y
            elif basis_index == 2:
                r = x
            elif basis_index == 3:
                r = y

        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 1:
                r = -1
            elif basis_index == 2:
                r = 1
            elif basis_index == 3:
                r = 0

        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 1:
                r = -1
            elif basis_index == 2:
                r = 0
            elif basis_index == 3:
                r = 1

    return r
