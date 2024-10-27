#KERNELS FILTROS IMAGENES

#FILTRO BLANCO Y NEGRO
kernel_filter_black_white="""
__kernel void kernel_filter_black_white(__global uchar* imagen_in,__global uchar* imagen_out,__constant float* filtro,int dim,int ancho,int alto){

  int fila = get_global_id(0);
  int columna = get_global_id(1);

  int centro=(dim-1)/2;
  float suma=0.0f;
  int i,j;

  //Asegurarse de que el píxel esté dentro de los límites

   if (centro <= fila && fila < (alto - centro) && centro <= columna && columna < (ancho - centro)) {

       for(i=-centro;i<=centro;i++){

            for(j=-centro;j<=centro;j++){

               float pixel_value = imagen_in[(fila+i) * ancho + (columna+j)];

                suma += pixel_value * filtro[(i + centro) * dim + (j + centro)];
                }
                }

                imagen_out[fila * ancho + columna]=(uchar)suma;
    }

  else{
    imagen_out[fila * ancho + columna]=imagen_in[fila * ancho + columna];
  }

}

"""



#KERNEL A COLOR:

kernel_filter_color="""
__kernel void kernel_filter_color(__global uchar* imagen_in,__global uchar* imagen_out,__constant float* filtro,int dim,int ancho,int alto){

  int fila = get_global_id(0);
  int columna = get_global_id(1);

  int centro=(dim-1)/2;

  float suma_rojo=0.0f;
  float suma_verde=0.0f;
  float suma_azul=0.0f;

  int i,j;

  //Asegurarse de que el píxel esté dentro de los límites

   if (centro <= fila && fila < (alto - centro) && centro <= columna && columna < (ancho - centro)) {

       for(i=-centro;i<=centro;i++){

            for(j=-centro;j<=centro;j++){

                // Para acceder al valor del pixel

                int idx = ((fila + i) * ancho + (columna + j)) * 3;

                float pixel_rojo = imagen_in[idx];
                float pixel_verde = imagen_in[idx + 1];
                float pixel_azul = imagen_in[idx + 2];

                float valor_filtro = filtro[(i + centro) * dim + (j + centro)];

                suma_rojo += pixel_rojo * valor_filtro;
                suma_verde += pixel_verde * valor_filtro;
                suma_azul += pixel_azul * valor_filtro;


                }
        }
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = (uchar)suma_rojo;
        imagen_out[idx_out + 1] = (uchar)suma_verde;
        imagen_out[idx_out + 2] = (uchar)suma_azul;

    }

    else {
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}
"""
kernel_filter_color_rectangular="""
__kernel void kernel_filter_color_rectangular(__global uchar* imagen_in,
                                   __global uchar* imagen_out,
                                   __constant float* filtro,
                                   int dim_x,       // Ancho del filtro
                                   int dim_y,       // Alto del filtro
                                   int ancho,       // Ancho de la imagen
                                   int alto) {      // Alto de la imagen

    // Obtener las coordenadas globales del píxel actual
    int fila = get_global_id(0);
    int columna = get_global_id(1);

    // Calcular el centro del filtro
    int centro_x = (dim_x - 1) / 2;
    int centro_y = (dim_y - 1) / 2;

    // Inicializar las sumas de los colores
    float suma_rojo = 0.0f;
    float suma_verde = 0.0f;
    float suma_azul = 0.0f;

    // Verificar si el píxel está dentro de los límites de la imagen
    if (centro_y <= fila && fila < (alto - centro_y) && centro_x <= columna && columna < (ancho - centro_x)) {

        // Recorrer el filtro
        for (int i = -centro_y; i <= centro_y; i++) {
            for (int j = -centro_x; j <= centro_x; j++) {

                // Calcular el índice del píxel en la imagen de entrada
                int idx = ((fila + i) * ancho + (columna + j)) * 3;

                // Leer los valores de los píxeles en las tres bandas de color
                float pixel_rojo = imagen_in[idx];
                float pixel_verde = imagen_in[idx + 1];
                float pixel_azul = imagen_in[idx + 2];

                // Obtener el valor del filtro correspondiente
                float valor_filtro = filtro[(i + centro_y) * dim_x + (j + centro_x)];

                // Acumular las sumas para cada canal de color
                suma_rojo += pixel_rojo * valor_filtro;
                suma_verde += pixel_verde * valor_filtro;
                suma_azul += pixel_azul * valor_filtro;
            }
        }

        // Calcular el índice en la imagen de salida
        int idx_out = (fila * ancho + columna) * 3;
        // Asignar los valores calculados a la imagen de salida
        imagen_out[idx_out] = (uchar)clamp(suma_rojo, 0.0f, 255.0f);
        imagen_out[idx_out + 1] = (uchar)clamp(suma_verde, 0.0f, 255.0f);
        imagen_out[idx_out + 2] = (uchar)clamp(suma_azul, 0.0f, 255.0f);
    } else {
        // Si el píxel está fuera de los límites, copiar el valor original
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}


"""

#KERNEL FILTRO SOBEL
kernel_filter_color_sobel="""
__kernel void kernel_filter_color_sobel(__global uchar* imagen_in,__global uchar* imagen_out,__constant float* filtro_X,__constant float* filtro_Y,int dim,int ancho,int alto){

  int fila = get_global_id(0);
  int columna = get_global_id(1);

  int centro=(dim-1)/2;

  float suma_rojo_X=0.0f;
  float suma_verde_X=0.0f;
  float suma_azul_X=0.0f;
  float suma_rojo_Y=0.0;
  float suma_verde_Y=0.0f;
  float suma_azul_Y=0.0f;

  float T_red=0.0f;
  float T_green=0.0f;
  float T_blue=0.0f;

  int i,j;

  //Asegurarse de que el píxel esté dentro de los límites

   if (centro <= fila && fila < (alto - centro) && centro <= columna && columna < (ancho - centro)) {

       for(i=-centro;i<=centro;i++){

            for(j=-centro;j<=centro;j++){

                // Para acceder al valor del pixel

                int idx = ((fila + i) * ancho + (columna + j)) * 3;

                float pixel_rojo = imagen_in[idx];
                float pixel_verde = imagen_in[idx + 1];
                float pixel_azul = imagen_in[idx + 2];

                float valor_filtro_X = filtro_X[(i + centro) * dim + (j + centro)];
                float valor_filtro_Y = filtro_Y[(i + centro) * dim + (j + centro)];

                suma_rojo_X += pixel_rojo * valor_filtro_X;
                suma_verde_X += pixel_verde * valor_filtro_X;
                suma_azul_X += pixel_azul * valor_filtro_X;

                suma_rojo_Y += pixel_rojo * valor_filtro_Y;
                suma_verde_Y += pixel_verde * valor_filtro_Y;
                suma_azul_Y += pixel_azul * valor_filtro_Y;

                }
        }

        T_red = sqrt(suma_rojo_X * suma_rojo_X + suma_rojo_Y * suma_rojo_Y);
        T_green = sqrt(suma_verde_X * suma_verde_X + suma_verde_Y * suma_verde_Y);
        T_blue = sqrt(suma_azul_X * suma_azul_X + suma_azul_Y * suma_azul_Y);




        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = (uchar)T_red;
        imagen_out[idx_out + 1] = (uchar)T_green;
        imagen_out[idx_out + 2] = (uchar)T_blue;

    }

    else {
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}
"""

#KERNEL FILTRO MEDIO:

kernel_median="""
__kernel void kernel_median(__global uchar* imagen_in, __global uchar* imagen_out, int dim, int ancho, int alto) {

    int fila = get_global_id(0);
    int columna = get_global_id(1);

    int centro = (dim - 1) / 2;
    int i, j;

    // Asegurarse de que el píxel esté dentro de los límites
    if (centro <= fila && fila < (alto - centro) && centro <= columna && columna < (ancho - centro)) {

        // Variables privadas (para cada hilo) para almacenar los píxeles vecinos

        uchar ventana_rojo[25]; // Asumiendo una ventana máxima de 5x5,depende de dim
        uchar ventana_verde[25];
        uchar ventana_azul[25];

        //Numero total de elementos en la ventana
        int count = 0;

        // Recorrer la ventana de tamaño `dim x dim`
        for (i = -centro; i <= centro; i++) {
            for (j = -centro; j <= centro; j++) {

                // Calcular el índice del píxel en la imagen de entrada
                int idx = ((fila + i) * ancho + (columna + j)) * 3;

                // Guardar los valores de los píxeles en las ventanas

                ventana_rojo[count] = imagen_in[idx];
                ventana_verde[count] = imagen_in[idx + 1];
                ventana_azul[count] = imagen_in[idx + 2];
                count++;
            }
        }

        // Ordenar los valores de los píxeles en cada canal

        for (int k = 0; k < count - 1; k++) {
            for (int l = k + 1; l < count; l++) {

                if (ventana_rojo[k] > ventana_rojo[l]) {
                    uchar temp = ventana_rojo[k];
                    ventana_rojo[k] = ventana_rojo[l];
                    ventana_rojo[l] = temp;
                }
                if (ventana_verde[k] > ventana_verde[l]) {
                    uchar temp = ventana_verde[k];
                    ventana_verde[k] = ventana_verde[l];
                    ventana_verde[l] = temp;
                }
                if (ventana_azul[k] > ventana_azul[l]) {
                    uchar temp = ventana_azul[k];
                    ventana_azul[k] = ventana_azul[l];
                    ventana_azul[l] = temp;
                }
            }
        }

        // Asignar el valor central al píxel de salida

        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = ventana_rojo[count / 2];
        imagen_out[idx_out + 1] = ventana_verde[count / 2];
        imagen_out[idx_out + 2] = ventana_azul[count / 2];

    } else {
        // Si el píxel está fuera de los límites, mantener el valor original

        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}

"""

#KERNEL MEMORIA LOCAL INEFICIENTE

kernel_filter_color_local="""
__kernel void kernel_filter_color_local(
    __global uchar* imagen_in, 
    __global uchar* imagen_out, 
    __constant float* filtro, 
    int dim, 
    int ancho, 
    int alto, 
    __local uchar* local_imagen) 
{
    int fila = get_global_id(0);
    int columna = get_global_id(1);
    
    int centro = (dim - 1) / 2;

    // IDs locales
    int local_fila = get_local_id(0);
    int local_columna = get_local_id(1);
    
    // Tamaño del grupo de trabajo
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);
    
    // Índice global
    int global_idx = (fila * ancho + columna) * 3;

    // Índice local en memoria local
    int local_idx = ((local_fila + centro) * (local_size_x + 2 * centro) + (local_columna + centro)) * 3;

    // Cargar píxeles en memoria local desde imagen_in
    int count = 0;
    if (fila < alto && columna < ancho) {
        for (int c = 0; c < 3; c++) {
            local_imagen[local_idx + c] = imagen_in[global_idx + c];
            count++;
        }
    }

    // Cargar bordes
    for (int dy = -centro; dy <= centro; dy++) {
        for (int dx = -centro; dx <= centro; dx++) {
            int local_x = local_columna + dx + centro;
            int local_y = local_fila + dy + centro;
            int global_x = columna + dx;
            int global_y = fila + dy;
            
            if (local_x >= 0 && local_x < local_size_x + 2 * centro && 
                local_y >= 0 && local_y < local_size_y + 2 * centro && 
                global_x >= 0 && global_x < ancho && 
                global_y >= 0 && global_y < alto) 
            {
                int local_mem_idx = (local_y * (local_size_x + 2 * centro) + local_x) * 3;
                int global_mem_idx = (global_y * ancho + global_x) * 3;
                local_imagen[local_mem_idx] = imagen_in[global_mem_idx];
                local_imagen[local_mem_idx + 1] = imagen_in[global_mem_idx + 1];
                local_imagen[local_mem_idx + 2] = imagen_in[global_mem_idx + 2];
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Aplicar el filtro
    if (centro <= fila && fila < (alto - centro) && centro <= columna && columna < (ancho - centro)) {
        float suma_rojo = 0.0f;
        float suma_verde = 0.0f;
        float suma_azul = 0.0f;

        for (int i = -centro; i <= centro; i++) {
            for (int j = -centro; j <= centro; j++) {
                int local_idx = ((local_fila + i + centro) * (local_size_x + 2 * centro) + (local_columna + j + centro)) * 3;

                suma_rojo += local_imagen[local_idx] * filtro[(i + centro) * dim + (j + centro)];
                suma_verde += local_imagen[local_idx + 1] * filtro[(i + centro) * dim + (j + centro)];
                suma_azul += local_imagen[local_idx + 2] * filtro[(i + centro) * dim + (j + centro)];
            }
        }

        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = (uchar)clamp(suma_rojo, 0.0f, 255.0f);
        imagen_out[idx_out + 1] = (uchar)clamp(suma_verde, 0.0f, 255.0f);
        imagen_out[idx_out + 2] = (uchar)clamp(suma_azul, 0.0f, 255.0f);
    } else {
        // Manejo de bordes
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}

"""



# KERNEL MEMORIA LOCAL HEBRA MAESTRA

kernel_filter_color_local2 = """
__kernel void kernel_filter_color_local2(
    __global uchar* imagen_in, 
    __global uchar* imagen_out, 
    __constant float* filtro, 
    int dim, 
    int ancho, 
    int alto, 
    __local uchar* local_imagen) 
{   
    // Posición pixel global
    int fila = get_global_id(0);
    int columna = get_global_id(1);
    
    int centro = (dim - 1) / 2;

    // IDs locales
    int local_fila = get_local_id(0);
    int local_columna = get_local_id(1);
    
    // Tamaño del grupo de trabajo
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);
    
    // Índice global
    int global_idx = (fila * ancho + columna) * 3;

    // Variables auxiliares
    int i, j;

    // Acceder a la memoria local solo la hebra 0
    if(local_fila == 0 && local_columna == 0) {
        // Quiero guardar una matriz de tamaño (local_size_x + 2) * (local_size_y + 2) * 3

        for(i = -centro; i < centro+local_size_x; i++) {
            for(j = -centro; j < centro+local_size_y; j++) {
                int fila_actual = fila + i;
                int columna_actual = columna + j;

                if((fila_actual >= 0) && (columna_actual >= 0) && (fila_actual < alto) && (columna_actual < ancho)) {
                    // Índice local en memoria local
                    int local_idx = ((i + centro) * (local_size_y + 2) + (j + centro)) * 3;

                    // Guardar en memoria local
                    for (int c = 0; c < 3; c++) {
                        local_imagen[local_idx + c] = imagen_in[(fila_actual * ancho + columna_actual) * 3 + c];
                    }  
                } else {
                    // Inicializar en caso de estar fuera de los límites
                    int local_idx = (( i + centro) * (local_size_y + 2) + (local_columna + j + centro)) * 3;
                    for (int c = 0; c < 3; c++) {
                        local_imagen[local_idx + c] = 0; // Valor por defecto
                    }
                }
            }
        }
        // Cargar bordes
   
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Aplicar el filtro
    if (centro <= fila && fila < (alto - centro) && centro <= columna && columna < (ancho - centro)) {
        float suma_rojo = 0.0f;
        float suma_verde = 0.0f;
        float suma_azul = 0.0f;

        for (int i = -centro; i <= centro; i++) {
            for (int j = -centro; j <= centro; j++) {
                int local_idx = ((local_fila + i + centro) * (local_size_y + 2) + (local_columna + j + centro)) * 3;

                suma_rojo += local_imagen[local_idx] * filtro[(i + centro) * dim + (j + centro)];
                suma_verde += local_imagen[local_idx + 1] * filtro[(i + centro) * dim + (j + centro)];
                suma_azul += local_imagen[local_idx + 2] * filtro[(i + centro) * dim + (j + centro)];
            }
        }

        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = (uchar)clamp(suma_rojo, 0.0f, 255.0f);
        imagen_out[idx_out + 1] = (uchar)clamp(suma_verde, 0.0f, 255.0f);
        imagen_out[idx_out + 2] = (uchar)clamp(suma_azul, 0.0f, 255.0f);
    } else {
        // Manejo de bordes
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out] = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}
"""
kernel_filter_color_local_rectangular="""
__kernel void kernel_filter_color_local_rectangular(
    __global uchar* imagen_in, 
    __global uchar* imagen_out, 
    __constant float* filtro, 
    int dim_x,      // Ancho del filtro
    int dim_y,      // Alto del filtro
    int ancho, 
    int alto, 
    __local uchar* local_imagen) 
{   
    // Posición del pixel global
    int fila = get_global_id(0);
    int columna = get_global_id(1);
    
    int halo_x = (dim_x - 1) / 2;
    int halo_y = (dim_y - 1) / 2;

    // IDs locales
    int local_fila = get_local_id(0);
    int local_columna = get_local_id(1);
    
    // Tamaño del grupo de trabajo
    int local_size_x = get_local_size(0);
    int local_size_y = get_local_size(1);
    
    // Dimensiones de la región local con bordes (halo)
    int local_dim_x = local_size_x + 2 * halo_y;
    int local_dim_y = local_size_y + 2 * halo_x;

    // Cargar píxeles en memoria local
    for (int i = local_fila; i < local_dim_x; i += local_size_x) {
        for (int j = local_columna; j < local_dim_y; j += local_size_y) {
            int img_fila = fila - halo_y + i;
            int img_columna = columna - halo_x + j;

            // Índice en la memoria local
            int local_idx = (i * local_dim_y + j) * 3;

            // Manejo de bordes
            if (img_fila >= 0 && img_fila < alto && img_columna >= 0 && img_columna < ancho) {
                int img_idx = (img_fila * ancho + img_columna) * 3;
                local_imagen[local_idx] = imagen_in[img_idx];
                local_imagen[local_idx + 1] = imagen_in[img_idx + 1];
                local_imagen[local_idx + 2] = imagen_in[img_idx + 2];
            } else {
                // Inicializar píxeles fuera de los límites con 0
                local_imagen[local_idx] = 0;
                local_imagen[local_idx + 1] = 0;
                local_imagen[local_idx + 1] = 0;
            }
        }
    }

    // Sincronizar todas las hebras
    barrier(CLK_LOCAL_MEM_FENCE);

    // Aplicar el filtro solo si estamos dentro de los límites de la imagen original
    if (fila >= halo_y && fila < (alto - halo_y) && columna >= halo_x && columna < (ancho - halo_x)) {
        float suma_rojo = 0.0f;
        float suma_verde = 0.0f;
        float suma_azul = 0.0f;

        // Aplicar el filtro convolucional
        for (int i = 0; i < dim_y; i++) {  // Recorre alto
            for (int j = 0; j < dim_x; j++) {  // Recorre ancho
                int local_i = local_fila + i;
                int local_j = local_columna + j;

                int local_idx = (local_i * local_dim_y + local_j) * 3;

                suma_rojo += (float)local_imagen[local_idx] * filtro[i * dim_x + j];
                suma_verde += (float)local_imagen[local_idx + 1] * filtro[i * dim_x + j];
                suma_azul += (float)local_imagen[local_idx + 2] * filtro[i * dim_x + j];
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

kernel_filter_color_local3 = """__kernel void kernel_filter_color_local3(
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

kernel_filter_color_local4 = """__kernel void kernel_filter_color_local4(
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
                __global uchar* pixel = &imagen_in[img_idx];
                local_imagen[local_idx]     = pixel[0]; // Rojo
                local_imagen[local_idx + 1] = pixel[1]; // Verde
                local_imagen[local_idx + 2] = pixel[2]; // Azul
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
        uchar pixel_out[3];
        pixel_out[0] = (uchar)clamp(suma_rojo, 0.0f, 255.0f);
        pixel_out[1] = (uchar)clamp(suma_verde, 0.0f, 255.0f);
        pixel_out[2] = (uchar)clamp(suma_azul, 0.0f, 255.0f);

    
        // Copiar los valores RGB a la memoria global en una sola operación
        imagen_out[idx_out]     = pixel_out[0];
        imagen_out[idx_out + 1] = pixel_out[1];
        imagen_out[idx_out + 2] = pixel_out[2];
    } else {
        // Manejo de bordes: copiar el píxel sin aplicar filtro
        int idx_out = (fila * ancho + columna) * 3;
        imagen_out[idx_out]     = imagen_in[idx_out];
        imagen_out[idx_out + 1] = imagen_in[idx_out + 1];
        imagen_out[idx_out + 2] = imagen_in[idx_out + 2];
    }
}
"""


