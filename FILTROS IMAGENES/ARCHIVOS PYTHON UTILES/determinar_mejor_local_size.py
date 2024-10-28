import math

# FUNCIÓN QUE DETERMINA EL MEJOR LOCAL SIZE


#FACTORIZAR, DEVUELVE UNA LISTA CON TODOS LOS FACTORES
def factorizar(n):
    
    factores = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factores.append((i, n // i))
    return factores

def optimal_local_size(global_size, max_compute_units, processing_elements):
    tam_x = global_size[0]
    tam_y = global_size[1]  
    
    # Factorizamos los elementos de procesamiento
    factores = factorizar(processing_elements)  # Por ejemplo, para 128 devuelve [(1,128), (2,64), (4,32), (8,16)]
    
    # Lista para almacenar las opciones compatibles
    opciones = []
    
    # Recorremos los factores y verificamos compatibilidad
    for factor in factores:
        local_x, local_y = factor
        if tam_x % local_x == 0 and tam_y % local_y == 0:
            opciones.append((local_x, local_y))
        # También consideramos la permutación de los factores
        local_x_perm, local_y_perm = factor[::-1]
        if local_x_perm != local_x and tam_x % local_x_perm == 0 and tam_y % local_y_perm == 0:
            opciones.append((local_x_perm, local_y_perm))
    
    return opciones
