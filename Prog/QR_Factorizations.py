import numpy as np

def qr_factorization_GS(A):
    # Obtenemos las dimensiones de A
    m, n = A.shape
    
    # Inicializamos Q y R
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    # Gram-Schmidt para la factorización QR
    for j in range(n):
        # Tomamos la columna j de A
        aj = A[:, j]
        
        # Proyección de a_j en los vectores q previos
        for i in range(j):
            
            qj = Q[:, i]
            
            R[i, j] = np.dot(qj, aj)
            aj = aj - R[i, j] * qj
            
        # Normalizamos a para obtener q_j
        R[j, j] = np.linalg.norm(aj)
        Q[:, j] = aj / R[j, j]
    
    return Q, R

def Givens_rotation_double(a, b):
    # Working with actual trigonometric functions
    # angle = np.arctan2(-a, b)
    # c = np.cos(angle)
    # s = np.sin(angle)

    # Using naive definitions
    # root = np.sqrt(a ** 2 + b ** 2)
    # c = a / root
    # s = -b / root

    # Using Matrix Computations solution
    if b == 0:
        c = 1.0
        s = 0.0
    else:
        if np.abs(b) > np.abs(a):
            tau = - a / b
            s = 1 / (np.sqrt(1 + tau ** 2))
            c = s * tau
        else:
            tau = - b / a
            c = 1 / (np.sqrt(1 + tau ** 2))
            s = c * tau

    return c, s

def Givens_rotation_matrix_double(a, b):
    c, s = Givens_rotation_double(a, b)
    return np.array([[c, -s], [s, c]])

def QR_factorisation_Givens_double(A):

    n, m = A.shape
    R = np.array(A, dtype='float')
    Q = np.eye(n)
    for i in range(m - 1):
        for j in range(n - 1, i, -1):
            G = Givens_rotation_matrix_double(R[j - 1, i], R[j, i])
            R[(j - 1):(j + 1), :] = np.dot(G, R[(j - 1):(j + 1), :])
            Q[(j - 1):(j + 1), :] = np.dot(G, Q[(j - 1):(j + 1), :])
    return Q.T, np.triu(R)


def QR_factorisation_int64(A):
    raise NotImplementedError("Never implemented this...")


