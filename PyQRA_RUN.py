import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
import os
import sys

# Cargar carpeta Prog
repo_path = os.path.dirname(os.path.abspath(__file__))  
prog_path = os.path.join(repo_path, "Prog") 
test_path = os.path.join(repo_path, "TESTCASES") 
if os.path.exists(prog_path):
    sys.path.append(prog_path)
    sys.path.append(test_path)
else:
    print(f"Advertencia: La carpeta {prog_path} no existe")
    
# Modulos Propios
from random_matrix import gen_matrix_random_SPD

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

#------------------------------------------------------------------------------
# Entrada de datos de la matriz a analizar.
m = 10  # Tamaño de la matriz

# Opcion 1: Matriz Hilbert.
# A = hilbert(m)  # Matriz de Hilbert (simétrica y positiva definida)

# Opcion 2: Matriz Random.
# A = gen_matrix_random_SPD(15)

# Opcion 3: Matriz Ejercicio 29 guia.
vec = np.arange(m, 0, -1)
A = np.diag(vec) + np.ones(shape=m)

# Opcion 4: TestCase
from Aster01_read_data import cargar_matriz_A
K, M, A= cargar_matriz_A(5)

#------------------------------------------------------------------------------
# Configurar algoritmos QR para ver la convergencia
QR_tol = 1e-12
neig = A.shape[0]
# neig = 15

print("Matriz original:")
print(A)

T = tridiag(A)

print("\nMatriz tridiagonal:")
print(np.round(T, 4))  # Redondear a 4 decimales

#------------------------------------------------------------------------------
"""
Programo QR puro, con while
"""
print("####--------------- QR ---------------####")

from QRAlgorithm_Pure import qr_algorithm

# eigenvalues, Tk, Ts, cm, timeQR = qr_algorithm(T, QR_tol, neig)
# print(np.round(Ts, 4))
 
#------------------------------------------------------------------------------
print("####--------------- QRs ---------------####")

from QRAlgorithm_Shift import qr_alg_shift

eigenvaluess, Tks, Tss, cms, timeQRs  = qr_alg_shift(T, QR_tol, neig)
# print(np.round(Tks, 4)) 

#%%------------------------------------------------------------------------------
# Graficar los valores |t_m,m-1| en función de las iteraciones

plt.figure(figsize=(10, 6))

# plt.semilogy(cm[:-1], marker='o', label = "Err. QR Unshift")
plt.semilogy(cms[:-1], marker='o', label = "Err. QR Shift", color = "orange")

# plt.title('Sawtooth plot del Algoritmo QR')
plt.xlabel('Número de iteración')
plt.ylabel('Error basado en |t_m,m-1|')
plt.grid(True)
plt.legend()
plt.show()


