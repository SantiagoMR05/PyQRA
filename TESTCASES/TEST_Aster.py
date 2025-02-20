import numpy as np
from scipy.linalg import hilbert
from random_matrix import gen_matrix_random_SPD
import matplotlib.pyplot as plt

import sys
sys.path.append(r'C:\Users\santi\Dropbox\GRUPO-IFIR-FMMH\CURSO-FEM\ALN\TEST_Aster')

from read_data import cargar_matriz_A

# Llamar a la función que devuelve la matriz A
K, M, A= cargar_matriz_A(5)

def householder_reflection(a):
    """
    Calcula el vector de Householder para un vector dado.
       input = ak (columna k de la matrix A5
    """
    e = np.zeros_like(a)        #Vector zeros size col(A)
    e[0] = np.linalg.norm(a)    #Inicio componente = ||x||*e1 (10.3)
    v = e - a
    v = v / np.linalg.norm(v)
    return v

def tridiag(matrix):
    
    """Reduce una matriz simétrica a forma tridiagonal 
        usando transformaciones de Householder."""
        
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
        """
        
        """
        Hk = np.eye(n - k - 1) - 2.0 * np.outer(v, v) # vv*
        
        # Ampliar Hk a una matriz de dimensión completa
        H = np.eye(n)
        H[k+1:, k+1:] = Hk
        
        # Aplicar transformación de Householder
        A = H @ A @ H
        
        # print(f"----{k}----")
        # A = np.round(A, 6)
    return A

# Iteramos el algoritmo QR para ver la convergencia
QR_tol = 1e-4
neig = A.shape[0]
# neig = 7

print("Matriz original:")
print(A)

T = tridiag(A)
# Tr = np.round(T, 4)

print("\nMatriz tridiagonal:")
# print(Tr)  # Redondear a 4 decimales

#------------------------------------------------------------------------------

print("####--------------- QRs ---------------####")

from QRAlgorithmShift import qr_alg_shift

eigenvaluess, Tks, Tss, cms, timeQRs  = qr_alg_shift(T, QR_tol, neig)
# print(np.round(Tks, 4)) 

freq = np.sqrt(np.maximum(eigenvaluess, 0))

#%%----------------------------------------------------------------------------

# Graficar los valores |t_m,m-1| en función de las iteraciones

plt.figure(figsize=(10, 6))

# plt.loglog(cm[:-1], marker='o', label = "Err. QR Unshift")
# plt.loglog(cms[:-1], marker='o', label = "Err. QR Shift")

# plt.semilogy(cm[:-1], marker='o', label = "Err. QR Unshift")
plt.semilogy(cms[:-1], marker='o', label = "Err. QR Shift", color = "orange")

# plt.title('Sawtooth plot del Algoritmo QR')
plt.xlabel('Número de iteración')
plt.ylabel('Error basado en |t_m,m-1|')
plt.grid(True)
plt.legend()
plt.show()


