import numpy as np

def generate_Gauss_local_1D(Gauss_point_number, lower_bound, upper_bound):
    """
    Generate the Gauss coefficients and Gauss points on an arbitrary interval [lower_bound, upper_bound]
    using affine transformation.

    Parameters:
        Gauss_coefficient_reference_1D (list or numpy array): Gauss coefficients on the reference interval [-1, 1].
        Gauss_point_reference_1D (list or numpy array): Gauss points on the reference interval [-1, 1].
        lower_bound (float): Lower bound of the target interval.
        upper_bound (float): Upper bound of the target interval.

    Returns:
        tuple:
            Gauss_coefficient_local_1D (numpy array): Gauss coefficients on the interval [lower_bound, upper_bound].
            Gauss_point_local_1D (numpy array): Gauss points on the interval [lower_bound, upper_bound].
    """
    if Gauss_point_number == 4:
        Gauss_coefficient_reference_1D = np.array([0.3478548451, 0.3478548451, 0.6521451549, 0.6521451549])
        Gauss_point_reference_1D = np.array([0.8611363116, -0.8611363116, 0.3399810436, -0.3399810436])
    elif Gauss_point_number == 8:
        Gauss_coefficient_reference_1D = np.array([0.1012285363, 0.1012285363, 0.2223810345, 0.2223810345,
                                                   0.3137066459, 0.3137066459, 0.3626837834, 0.3626837834])
        Gauss_point_reference_1D = np.array([0.9602898565, -0.9602898565, 0.7966664774, -0.7966664774,
                                             0.5255324099, -0.5255324099, 0.1834346425, -0.1834346425])
    elif Gauss_point_number == 2:
        Gauss_coefficient_reference_1D = np.array([1, 1])
        Gauss_point_reference_1D = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])

    Gauss_coefficient_local_1D = (upper_bound - lower_bound) * np.array(Gauss_coefficient_reference_1D) / 2
    Gauss_point_local_1D = (upper_bound - lower_bound) * np.array(Gauss_point_reference_1D) / 2 + (upper_bound + lower_bound) / 2

    return Gauss_coefficient_local_1D, Gauss_point_local_1D
