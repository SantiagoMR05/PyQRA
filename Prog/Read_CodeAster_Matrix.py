import numpy as np
import pandas as pd
import os
import sys
import scipy.linalg as la

# Cargar carpeta Prog
repo_path = os.path.dirname(os.path.abspath(__file__))  
save_path = 'C:/Users/santi/Dropbox/GRUPO-IFIR-FMMH/CURSO-FEM/ALN/TEST_QR_Files'

dfK = pd.read_csv(f'{save_path}/Matrix_K.csv', index_col=0)
dfM = pd.read_csv(f'{save_path}/Matrix_M.csv', index_col=0)

# Cargar matrices K y M desde archivos CSV
K = dfK.to_numpy()
M = dfM.to_numpy()

# Encontrar índices de los autovalores no cero
indices_validos = np.where(np.linalg.eigvals(M) > 1e-10)[0]

# Reducir matrices
M_red = M[np.ix_(indices_validos, indices_validos)]
K_red = K[np.ix_(indices_validos, indices_validos)]

# Bloqueo CB
K_red = K_red[6:, 6:]
M_red = M_red[6:, 6:]

def is_spd(M):
    """Verifica si la matriz M es simétrica definida positiva"""
    return np.allclose(M, M.T) and np.all(np.linalg.eigvals(M) > 0)

def transform_problem(M, K):
    """Transforma el problema generalizado a un problema estándar"""
    if is_spd(M):    
        L = la.cholesky(M, lower=True)  # Factorización de Cholesky M = L L^T
        Linv = la.solve_triangular(L, np.eye(L.shape[0]), lower=True)  # L⁻¹
        A = Linv @ K @ Linv.T  # A = L⁻¹ K L⁻ᵀ
    else:
        raise ValueError("M no es SPD, se necesita otro método (como QZ)")
    return A

# Transformar problema
A = transform_problem(M_red, K_red)

