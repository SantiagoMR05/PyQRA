import numpy as np
from QR_Factorizations import qr_factorization_GS, QR_factorisation_Givens_double
import time

def wilkinson_shift(A):
    """
    Calcula el desplazamiento de Wilkinson usando los DOS últimos elementos de la diagonal
    y el elemento justo debajo de la diagonal.
    """ 
    b11 = A[-2, -2]  # Diagonal penúltima
    b22 = A[-1, -1]  # Diagonal última
    b12 = A[-1, -2]  # Elemento fuera de la diagonal
    
    dt = (b11 - b22) / 2
    
    mu = b22 - np.sign(dt) * b12**2 / (abs(dt) + np.linalg.norm([dt, b12]))
    
    return mu

def rayleigh_shift(A):
    """
    Calcula el desplazamiento utilizando el cociente de Rayleigh para la
    submatriz 2x2 de la esquina inferior derecha.
    """
    mu = A[-1, -1]  # Desplazamiento de Rayleigh usando el último elemento diagonal
    return mu

def qr_alg_shift(T, tol, num_eigenvalues):
    start_time = time.time()  # Iniciar el temporizador
    
    m, n = T.shape
    if n != m:
        raise np.linalg.LinAlgError("Array must be square.")
        
    Tk = T.copy()  # Copia de la matriz original
    Ts = []  # Para almacenar las matrices en cada iteración
    convergence_measure = []  # Para registrar la medida de convergencia
    eigenvalues = np.zeros((num_eigenvalues,), dtype='float')  # Almacenar los autovalores encontrados
    count = 0  # Contador de autovalores encontrados
    
    # Inicialización de mu (desplazamiento)
    mu = 0
    
    m -= 1
    
    while m > 0: 
            mu_matrix = np.eye(Tk.shape[0]) * mu  # Crear la matriz identidad multiplicada por mu
    
            Q, R = qr_factorization_GS(Tk - mu_matrix)  # Realizar la factorización QR
            Tk = R @ Q + mu_matrix  # Multiplicación inversa (R por Q) para el próximo paso
            
            # Medir la convergencia usando el valor subdiagonal más bajo
            convergence_measure.append(np.abs(Tk[m, m - 1]))  
            Ts.append(np.round(Tk, 4))  # Almacenamos la matriz redondeada para visualización
            
            # Calcular el desplazamiento de Wilkinson o Rayleigh
            mu = wilkinson_shift(Tk)  # Puedes alternar con rayleigh_shift si lo prefieres
            # print(convergence_measure)
    
            # Condición de convergencia
            if convergence_measure[-1] < tol:  # Verificamos si la convergencia ha alcanzado el umbral
                eigenvalues[count] = Tk[m, m]  # Guardamos el valor diagonal
                Tk = Tk[:m, :m]  # Reducir la matriz Tk
                m -= 1  # Reducir el tamaño de la matriz para el siguiente ciclo
                count += 1
                print(count)
    
            # Salir si alcanzamos el número deseado de autovalores
            if count >= num_eigenvalues:
                eigenvalues[0] = Tk[0, 0]
                break
    
            # Salir si la longitud de convergence_measure excede 500
            if len(convergence_measure) >= 5000:
                print("Se alcanzó el límite de 5000 iteraciones en convergence_measure.")
                
                eigenvalues[count] = Tk[m, m]  # Guardamos el valor diagonal
                Tk = Tk[:m, :m]  # Reducir la matriz Tk
                m -= 1  # Reducir el tamaño de la matriz para el siguiente ciclo
                count += 1
                print(count)
                # break
            
    # Registrar el último eigenvalor encontrado
    eigenvalues[0] = Tk[0, 0]
    
    # Ordenar los autovalores de mayor a menor
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Ordenar en orden descendente
    eigenvalues = eigenvalues[sorted_indices]  # Reorganizar los autovalores
    
    elapsed_time = time.time() - start_time  # Calcular el tiempo transcurrido
    print(f"Convergencia alcanzada en {len(Ts)} pasos")
    
    return eigenvalues, Tk, Ts, convergence_measure, elapsed_time
