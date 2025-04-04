import numpy as np
import time
import os
import sys

# Cargar carpeta Prog
repo_path = os.path.dirname(os.path.abspath(__file__))  
prog_path = os.path.join(repo_path, "Prog") 
if os.path.exists(prog_path):
    sys.path.append(prog_path)
else:
    print(f"Advertencia: La carpeta {prog_path} no existe")
    
from QR_Factorizations import qr_factorization_GS, QR_factorisation_Givens_double

def qr_algorithm(T, tol, num_eigenvalues):
    start_time = time.time()  # Iniciar el temporizador
    
    m, n = T.shape
    if n != m:
        raise np.linalg.LinAlgError("Array must be square.")
        
    Tk = T.copy()   # Copiar la matriz de entrada para no modificar la original
    Ts = []         # Para almacenar las matrices Tk en cada iteración
    convergence_measure = []  # Para registrar la medida de convergencia
    eigenvalues = np.zeros(num_eigenvalues, dtype='float')  # Almacenar los autovalores encontrados
    
    count = 0  # Contador de autovalores encontrados
    m -= 1 # Indexar matrices Python 
    
    while m > 0:

        # Realizar la factorización QR con Gram-Schmidt
        Q, R = qr_factorization_GS(Tk)
        Tk = R @ Q  # Multiplicación inversa (R por Q) para el próximo paso
        
        # Medir la convergencia usando el valor de la subdiagonal más bajo
        convergence_measure.append(np.abs(Tk[m, m-1]))

        Ts.append(Tk)  # Almacenamos la matriz Tk

        if convergence_measure[-1] < tol:  # Verificamos si la convergencia ha alcanzado el umbral
            eigenvalues[count] = Tk[m, m]  # Guardamos el valor diagonal
            count += 1

            if count >= num_eigenvalues:  # Si ya encontramos los autovalores deseados, salimos del ciclo
                break

            Tk = Tk[:m, :m]  # Reducir la matriz Tk
            m -= 1  # Reducir el tamaño de la matriz para el siguiente ciclo
            
    # Guardar el último autovalor si queda uno solo
    if m == 0:
        eigenvalues[count] = Tk[0, 0]

    # Calcular el tiempo de ejecución
    elapsed_time = time.time() - start_time
    print(f"Convergencia alcanzada en {len(Ts)} pasos")

    return eigenvalues, Tk, Ts, convergence_measure, elapsed_time


