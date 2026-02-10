import numpy as np

def cross_product_matrix(v):
    """
    Returns the skew-symmetric matrix [v x] such that [v x] * u = v x u
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def inertia_tensor_from_principal_axes(Ixx, Iyy, Izz):
    return np.diag([Ixx, Iyy, Izz])
