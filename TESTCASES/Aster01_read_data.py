import numpy as np
import pandas as pd
import os
import sys

# Cargar carpeta Prog
repo_path = os.path.dirname(os.path.abspath(__file__))  

def cargar_matriz_A(i):
    
    dfK = pd.read_csv(f'{repo_path}/Aster01_Matrix_K.csv', index_col=0)
    dfM = pd.read_csv(f'{repo_path}/Aster01_Matrix_M.csv', index_col=0)
    
    # Cargar matrices K y M desde archivos CSV
    K = dfK.to_numpy()
    M = dfM.to_numpy()
    
    # K = K[6:, 6:]
    # M = M[6:, 6:]
    
    # Determinar los índices para eliminar (por bloques de 6)
    # i = 0
    start = i * 6 #gdl
    end = (i + 1) * 6  # Este es el rango a eliminar
    
    # Eliminar filas y columnas de la matriz K y M
    K = np.delete(K, slice(start, end), axis=0)  # Eliminar filas
    K = np.delete(K, slice(start, end), axis=1)  # Eliminar columnas
    
    M = np.delete(M, slice(start, end), axis=0)  # Eliminar filas
    M = np.delete(M, slice(start, end), axis=1)  # Eliminar columnas
        
    # Aplicar tolerancia para valores pequeños
    tolerancia = 1e-6
    
    K[np.abs(K) < tolerancia] = 0
    M[np.abs(M) < tolerancia] = 0

    # Verificar y asegurar que M sea SPD
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        
        print("M no es definida positiva. Regularizando...")
        epsilon = 1e-6
        M += epsilon * np.eye(M.shape[0])
        L = np.linalg.cholesky(M)

    # Calcular M^-1 K usando la factorización de Cholesky
    Y = np.linalg.solve(L, K)
    A = np.linalg.solve(L.T, Y)

    A[np.abs(A) < tolerancia] = 0

    return  K, M, A

# Test Function:
# Llamar a la función que devuelve la matriz A
# K, M, A= cargar_matriz_A(5)


