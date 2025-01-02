'''
ARCHIVO PARA OBTENER INFORMACIÓN DEL DISPOSITIVO
'''

import pyopencl as cl


# Configurar contexto y seleccionar el dispositivo
platform = cl.get_platforms()[0]  # Selecciona la primera plataforma
device = platform.get_devices()[0]  # Selecciona el primer dispositivo

# Obtener el tipo de dispositivo
device_type = device.get_info(cl.device_info.TYPE)

if device_type == cl.device_type.CPU:
    print("CL_DEVICE_TYPE_CPU")
elif device_type == cl.device_type.GPU:
    print("CL_DEVICE_TYPE_GPU")
elif device_type == cl.device_type.ACCELERATOR:
    print("CL_DEVICE_TYPE_ACCELERATOR")
else:
    print("Other")

# Obtener el Vendor ID del dispositivo
vendor_id = device.get_info(cl.device_info.VENDOR_ID)
print(f"Vendor ID: 0x{vendor_id:04x}")
device_name = device.get_info(cl.device_info.NAME)
print(f"Nombre del dispositivo: {device_name}")


# Obtener el número máximo de Compute Units
max_compute_units = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
print(f"Maximum Compute Units: {max_compute_units}")

# Obtener las dimensiones máximas de trabajo
max_work_item_dimensions = device.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS)
print(f"Maximum Work Item Dimensions: {max_work_item_dimensions}")

# Obtener los tamaños máximos de trabajo
max_work_item_sizes = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)
print(f"Maximum Work Item Sizes: {max_work_item_sizes[0]} x {max_work_item_sizes[1]} x {max_work_item_sizes[2]}")

# Obtener el tamaño máximo del grupo de trabajo
max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
print(f"Maximum Work Group Size: {max_work_group_size}")

# Obtener la frecuencia máxima del reloj
max_clock_frequency = device.get_info(cl.device_info.MAX_CLOCK_FREQUENCY)
print(f"Maximum Clock Frequency: {max_clock_frequency} MHz")



'''
ESTUDIAR LAS CARCATERÍSTICAS DE UN DISPOSITIVO Y DE UN KERNEL DADO
'''


# Crea un contexto y una cola de comando
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

# Un kernel sencillo de ejemplo (suma de dos vectores)
program_source = """__kernel void test_kernel(
    __global uchar* imagen_in, 
    __global uchar* imagen_out, 
    __constant float* filtro, 
    int dim, 
    int ancho, 
    int alto, 
    __local uchar* local_imagen) 
{   
    // Posición del pixel global
    int fila = get_global_id(0);
    int columna = get_global_id(1);
    
    int centro = (dim - 1) / 2;

    // IDs locales
    int local_fila = get_local_id(0);
    int local_columna = get_local_id(1);
    
    // Tamaño del grupo de trabajo
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);
    
    // Dimensiones de la región local con bordes (halo)
    int local_dim_x = local_size_x + 2 * centro;
    int local_dim_y = local_size_y + 2 * centro;
    
    // Índice global para la imagen original
    int global_idx = (fila * ancho + columna) * 3;

    // Cada hebra solo carga los píxeles que le corresponden
    for (int i = local_fila; i < local_dim_x; i += local_size_x) {
        for (int j = local_columna; j < local_dim_y; j += local_size_y) {
            int img_fila = fila - local_fila + i - centro;
            int img_columna = columna - local_columna + j - centro;

            // Índice en la memoria local
            int local_idx = (i * local_dim_y + j) * 3;

            // Manejo de bordes, si el píxel está dentro de los límites de la imagen
            if (img_fila >= 0 && img_fila < alto && img_columna >= 0 && img_columna < ancho) {
                int img_idx = (img_fila * ancho + img_columna) * 3;
                local_imagen[local_idx] = imagen_in[img_idx];
                local_imagen[local_idx + 1] = imagen_in[img_idx + 1];
                local_imagen[local_idx + 2] = imagen_in[img_idx + 2];
            } else {
                // Inicializar píxeles fuera de los límites con 0
                local_imagen[local_idx] = 0;
                local_imagen[local_idx + 1] = 0;
                local_imagen[local_idx + 2] = 0;
            }
        }
    }

    // Sincronizar todas las hebras para asegurar que la carga esté completa
    barrier(CLK_LOCAL_MEM_FENCE);

    // Aplicar el filtro solo si estamos dentro de los límites de la imagen original
    if (fila >= centro && fila < (alto - centro) && columna >= centro && columna < (ancho - centro)) {
        float suma_rojo = 0.0f;
        float suma_verde = 0.0f;
        float suma_azul = 0.0f;

        // Aplicar el filtro convolucional
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                int local_i = local_fila + i;
                int local_j = local_columna + j;
                
                int local_idx = (local_i * local_dim_y + local_j) * 3;
                
                suma_rojo += (float)local_imagen[local_idx] * filtro[i * dim + j];
                suma_verde += (float)local_imagen[local_idx + 1] * filtro[i * dim + j];
                suma_azul += (float)local_imagen[local_idx + 2] * filtro[i * dim + j];
            }
        }

        // Escribir el resultado en la imagen de salida
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out]     = (uchar)clamp(suma_rojo, 0.0f, 255.0f);
        imagen_out[idx_out + 1] = (uchar)clamp(suma_verde, 0.0f, 255.0f);
        imagen_out[idx_out + 2] = (uchar)clamp(suma_azul, 0.0f, 255.0f);
    } else {
        // Manejo de bordes: copiar el píxel sin aplicar filtro
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out]     = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}


"""

# Compila el programa
program = cl.Program(ctx, program_source).build()

# Crea el kernel
kernel = cl.Kernel(program, "test_kernel")

# Obtiene el tamaño máximo de work group soportado para este kernel en el dispositivo
max_work_group_size = kernel.get_work_group_info(cl.kernel_work_group_info.WORK_GROUP_SIZE, device)

print("Tamaño máximo de work group soportado para este kernel:", max_work_group_size)
