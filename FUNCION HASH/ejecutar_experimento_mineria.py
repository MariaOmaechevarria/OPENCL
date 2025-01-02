'''
ARCHIVO QUE EJECUTA LOS EXPERIMENTOS DE MINERÍA:
    - TARGET SENCILLO MINAR BLOQUE DISTINTOS GLOBAL SIZES
    - TARGET MEDIO MINAR BLOQUE DISTINTOS GLOBAL SIZES
    - DISTINTOS TARGETS GLOBAL SIZE FIJADO
'''



import numpy as np
import funciones_experimento_mining as ex

#RUTA PARA GUARDAR ARCHIVOS (MODIFICAR)

path="C:/Users/Eevee"

# TARGET SENCILLO:
# Este objetivo (target) tiene una dificultad baja, lo que significa que es más probable encontrar un hash válido.
# La dificultad se establece con los primeros bytes menos restrictivos (0x00FFFFFF) y el resto llenos de 0xFFFFFFFF.
target = np.array([0x00FFFFFF] + [0xFFFFFFFF] * 7, dtype=np.uint32)
target_name = 'target_facil'  # Nombre para identificar este experimento
ex.experimento_global_sizes(path, target, target_name)  # Ejecuta el experimento con este target

# TARGET MÁS COMPLICADO:
# Este objetivo (target) tiene una mayor dificultad en comparación con el anterior.
# Los primeros bytes (0x000000FF) hacen que sea mucho menos probable encontrar un hash válido.
target_name = 'target_medio'  # Nombre para identificar este experimento
target = np.array([0x000000FF] + [0xFFFFFFFF] * 7, dtype=np.uint32)
ex.experimento_global_sizes(path, target, target_name)  # Ejecuta el experimento con este target

# COMPARAR MUCHOS TIPOS DISTINTOS DE TARGETS:
# Esta función realiza un análisis comparativo entre diferentes niveles de dificultad.
# Incluye varios targets que van desde dificultad mínima hasta dificultad máxima.
# Genera gráficos y tablas comparativas para evaluar cómo afecta el target al tiempo de ejecución.
ex.comparacion_targets(path)
