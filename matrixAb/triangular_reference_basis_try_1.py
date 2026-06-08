def triangular_reference_basis(x, y, basis_type, basis_index, derivative_degree_x, derivative_degree_y):
    """
    Reference finite element (FE) basis function on triangle ABC (A=(0,0), B=(1,0), C=(0,1)).
    Converted from MATLAB code provided by Xiaoming He, 07/01/2009.
    """
    def create_basis_function(expression):
        return lambda x, y: eval(expression)

    if basis_type == 2:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 1:
                return create_basis_function("1 - 3 * x - 3 * y + 2 * x**2 + 2 * y**2 + 4 * x * y")
            elif basis_index == 2:
                return create_basis_function("2 * x**2 - x")
            elif basis_index == 3:
                return create_basis_function("2 * y**2 - y")
            elif basis_index == 4:
                return create_basis_function("4 * x - 4 * x**2 - 4 * x * y")
            elif basis_index == 5:
                return create_basis_function("4 * x * y")
            elif basis_index == 6:
                return create_basis_function("4 * y - 4 * y**2 - 4 * x * y")
        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 1:
                return create_basis_function("-3 + 4 * x + 4 * y")
            elif basis_index == 2:
                return create_basis_function("4 * x - 1")
            elif basis_index == 3:
                return create_basis_function("0")
            elif basis_index == 4:
                return create_basis_function("4 - 8 * x - 4 * y")
            elif basis_index == 5:
                return create_basis_function("4 * y")
            elif basis_index == 6:
                return create_basis_function("-4 * y")
        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 1:
                return create_basis_function("-3 + 4 * y + 4 * x")
            elif basis_index == 2:
                return create_basis_function("0")
            elif basis_index == 3:
                return create_basis_function("4 * y - 1")
            elif basis_index == 4:
                return create_basis_function("-4 * x")
            elif basis_index == 5:
                return create_basis_function("4 * x")
            elif basis_index == 6:
                return create_basis_function("4 - 8 * y - 4 * x")
        elif derivative_degree_x == 2 and derivative_degree_y == 0:
            if basis_index == 1:
                return create_basis_function("4")
            elif basis_index == 2:
                return create_basis_function("4")
            elif basis_index == 3:
                return create_basis_function("0")
            elif basis_index == 4:
                return create_basis_function("-8")
            elif basis_index == 5:
                return create_basis_function("0")
            elif basis_index == 6:
                return create_basis_function("0")
        elif derivative_degree_x == 0 and derivative_degree_y == 2:
            if basis_index == 1:
                return create_basis_function("4")
            elif basis_index == 2:
                return create_basis_function("0")
            elif basis_index == 3:
                return create_basis_function("4")
            elif basis_index == 4:
                return create_basis_function("0")
            elif basis_index == 5:
                return create_basis_function("0")
            elif basis_index == 6:
                return create_basis_function("-8")
        elif derivative_degree_x == 1 and derivative_degree_y == 1:
            if basis_index == 1:
                return create_basis_function("4")
            elif basis_index == 2:
                return create_basis_function("0")
            elif basis_index == 3:
                return create_basis_function("0")
            elif basis_index == 4:
                return create_basis_function("-4")
            elif basis_index == 5:
                return create_basis_function("4")
            elif basis_index == 6:
                return create_basis_function("-4")

    elif basis_type == 1:
        if derivative_degree_x == 0 and derivative_degree_y == 0:
            if basis_index == 1:
                return create_basis_function("1 - x - y")
            elif basis_index == 2:
                return create_basis_function("x")
            elif basis_index == 3:
                return create_basis_function("y")
        elif derivative_degree_x == 1 and derivative_degree_y == 0:
            if basis_index == 1:
                return create_basis_function("-1")
            elif basis_index == 2:
                return create_basis_function("1")
            elif basis_index == 3:
                return create_basis_function("0")
        elif derivative_degree_x == 0 and derivative_degree_y == 1:
            if basis_index == 1:
                return create_basis_function("-1")
            elif basis_index == 2:
                return create_basis_function("0")
            elif basis_index == 3:
                return create_basis_function("1")

    raise ValueError("Invalid combination of inputs!")