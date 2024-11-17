import math

# FUNCIÓN QUE DETERMINA EL MEJOR LOCAL SIZE

def factorizar(n: int) -> list[tuple[int, int]]:
    """
    Encuentra todos los factores de un número entero y devuelve pares de factores.

    Inputs:
    - n (int): Número entero a factorizar.

    Outputs:
    - list[tuple[int, int]]: Lista de pares de factores (x, y) tal que x * y = n.
    """
    factores = []
    # Iteramos desde 1 hasta la raíz cuadrada del número
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:  # Si `i` es un factor
            factores.append((i, n // i))  # Agregar el par de factores
    return factores


def optimal_local_size(
    global_size: tuple[int, int], 
    max_compute_units: int, 
    processing_elements: int
) -> list[tuple[int, int]]:
    """
    Determina los tamaños de workgroup (local sizes) compatibles con un tamaño global dado.

    Inputs:
    - global_size (tuple[int, int]): Dimensiones del espacio global de hilos (X, Y).
    - max_compute_units (int): Número máximo de unidades de cómputo en el dispositivo.
    - processing_elements (int): Número total de elementos de procesamiento disponibles.

    Outputs:
    - list[tuple[int, int]]: Lista de pares (local_x, local_y) compatibles.
    """
    tam_x = global_size[0]  # Tamaño global en el eje X
    tam_y = global_size[1]  # Tamaño global en el eje Y
    
    # Factorizamos los elementos de procesamiento
    factores = factorizar(processing_elements)  
    # Ejemplo: para 128 devuelve [(1,128), (2,64), (4,32), (8,16)]

    opciones = []  # Lista para almacenar las combinaciones compatibles
    
    # Recorremos los pares de factores
    for factor in factores:
        local_x, local_y = factor

        # Verificamos si son compatibles con el tamaño global
        if tam_x % local_x == 0 and tam_y % local_y == 0:
            opciones.append((local_x, local_y))  # Agregar como opción válida

        # También consideramos la permutación de los factores
        local_x_perm, local_y_perm = factor[::-1]  # Invertir los factores
        if local_x_perm != local_x and tam_x % local_x_perm == 0 and tam_y % local_y_perm == 0:
            opciones.append((local_x_perm, local_y_perm))  # Agregar la permutación válida
    
    return opciones