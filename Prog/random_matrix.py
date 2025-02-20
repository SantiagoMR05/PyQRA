import numpy as np

def gen_matrix_random_SPD(n):
    # Paso 1: Generar una matriz aleatoria de tamaño n x n
    B = np.random.rand(n, n)
    
    # Paso 2: Generar una matriz simétrica A = B^T B
    A = np.dot(B.T, B)
    
    # Paso 3: Agregar un valor pequeño a la diagonal para asegurar autovalores positivos
    np.fill_diagonal(A, np.diagonal(A) + 1e-5)
    
    return A

# # Ejemplo de uso
# n = 4  # Tamaño de la matriz (n x n)
# matriz = gen_matrix_random_SPD(n)

# print("Matriz generada:")
# print(matriz)

# # Comprobamos los autovalores
# autovalores = np.linalg.eigvals(matriz)
# print("\nAutovalores de la matriz:")
# print(autovalores)
