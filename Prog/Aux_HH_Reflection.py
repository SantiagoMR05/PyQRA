import numpy as np

def householder_reflection(a):
    """
    Calcula el vector de Householder para un vector dado.
       input = ak (columna k de la matrix A)
    """
    e = np.zeros_like(a)        #Vector zeros size col(A)
    e[0] = np.linalg.norm(a)    #Inicio componente = ||x||*e1 (10.3)
    v = e - a
    v = v / np.linalg.norm(v)
    return v

def tridiag(matrix):
    """Reduce una matriz simétrica a forma tridiagonal 
        usando transformaciones de Householder.
    """
    n = matrix.shape[0]
    A = matrix.copy()
    for k in range(n - 2):
        """
        Loop para extraer el vector columna "ak"
        """
        ak = A[k+1:, k]
        if np.allclose(ak, 0):
            continue
        # Obtener el vector de Householder
        v = householder_reflection(ak)
        # Crear la matriz de Householder
        Hk = np.eye(n - k - 1) - 2.0 * np.outer(v, v) # vv*
        # Ampliar Hk a una matriz de dimensión completa
        H = np.eye(n)
        H[k+1:, k+1:] = Hk
        # Aplicar transformación de Householder
        A = H @ A @ H
        # print(f"----{k}----")
        # A = np.round(A, 6)
    return A