import numpy as np
import pandas as pd
import os
import sys
import scipy.linalg as la

# Cargar carpeta Prog
prog_path = 'C:/Users/santi/Dropbox/GRUPO-IFIR-FMMH/CURSO-FEM/ALN/PyQRA'
save_path = 'C:/Users/santi/Dropbox/GRUPO-IFIR-FMMH/CURSO-FEM/ALN/TEST_QR_Files'

sys.path.append(prog_path)
sys.path.append(save_path)

# Cargar matrices K y M desde archivos CSV
dfK = pd.read_csv(f'{save_path}/Matrix_K.csv', index_col=0)
dfM = pd.read_csv(f'{save_path}/Matrix_M.csv', index_col=0)

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


#%% Ejecucion de Algoritmos QR

# Configurar algoritmos QR para ver la convergencia

QR_tol = 1e-12
neig = A.shape[0]
neig = 20

#------------------------------------------------------------------------------
print("####--------------- QR ---------------####")

from QRAlgorithm_Pure import qr_algorithm

eigenvalues, Tk, Ts, cm, timeQR = qr_algorithm(A, QR_tol, neig)
# print(np.round(Ts, 4))
 
#------------------------------------------------------------------------------
print("####--------------- QRs ---------------####")

from QRAlgorithm_Shift import qr_alg_shift

eigenvaluess, Tks, Tss, cms, timeQRs  = qr_alg_shift(A, QR_tol, neig)
# print(np.round(Tks, 4)) 

#%% RESULTADOS: Transformar autovalores a frecuencias

freq_QR_pure =  np.sort(np.sqrt(eigenvalues)  / (2*np.pi))
freq_QR_shift = np.sort(np.sqrt(eigenvaluess) / (2*np.pi))

df_list = []

def create_dataframe(array, name):
    
    # Crear el DataFrame con índices como "NUME_ORDRE"
    df = pd.DataFrame({"NUME_ORDRE": np.arange(1, len(array) + 1),
                       name: array})
    
    return df

df1 = create_dataframe(freq_QR_pure, "FREQ_QR_Pure")
df2 = create_dataframe(freq_QR_shift, "FREQ_QR_Shift")

# Comparar resultados contra Code-Aster
df_freq = pd.read_csv('C:/Users/santi/Dropbox/GRUPO-IFIR-FMMH/CURSO-FEM/ALN/TEST_QR_Files/frecuencias.csv', index_col=0)

df_list = df1.copy()
df_list = pd.merge(df_list, df2)
df_list = pd.merge(df_list, df_freq)

