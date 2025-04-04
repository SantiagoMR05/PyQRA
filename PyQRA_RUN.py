import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert
import os
import sys
import pandas as pd

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
from Aux_HH_Reflection import tridiag


#------------------------------------------------------------------------------
# Entrada de datos de la matriz a analizar.

m = 200  # Tamaño de la matriz

# Opcion 1: Matriz Hilbert.
A = hilbert(m)  # Matriz de Hilbert (simétrica y positiva definida)
T = tridiag(A)

# Opcion 2: Matriz Random.
# A = gen_matrix_random_SPD(15)
# T = tridiag(A)

# Opcion 3: Matriz Ejercicio 29 (Trefethen - Bau, 2022).
vec = np.arange(m, 0, -1)
A = np.diag(vec) + np.ones(shape=m)
T = tridiag(A)

#------------------------------------------------------------------------------
# Configurar algoritmos QR para ver la convergencia

QR_tol = 1e-12
neig = A.shape[0]
neig = 20


#%%

#------------------------------------------------------------------------------
print("####--------------- QR ---------------####")

from QRAlgorithm_Pure import qr_algorithm

eigenvalues, Tk, Ts, cm, timeQR = qr_algorithm(T, QR_tol, neig)
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


