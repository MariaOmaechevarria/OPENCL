# OPENCL

Repositorio de OpenCL
Este repositorio contiene una colección de scripts y notebooks relacionados con el uso de OpenCL para diversas aplicaciones, incluyendo la multiplicación de matrices, filtros de imágenes y funciones hash. A continuación, se detalla la estructura del repositorio y la descripción de cada archivo y carpeta.

# CARPETAS Y ARCHIVOS

1. **CUDA**
   
    Conjunto de archivos con el objetivo de comparar la ejecución de la multiplicación de matrices en CUDA y en OpenCL. Los archivos son los siguientes:
   - **mult_matrices_basico_cuda.py :** Kernel de multiplicacion basica en cuda y la funcion en python para ejecutar el kernel para unos argumentos dados
   - **mult_matrices_basico_opencl.py :** Kernel de multiplicacion basica en opencl y la funcion en python para ejecutar el kernel para unos argumentos dados
   - **funciones_experimentos_cuda.py:** Funciones para crear tablas, graficos y ejecutar los kernels anteriores para realizar dos experimentos.
     - Comparar cuda vs opencl
     
     - Probar distintos local sizes para obtener los tiempos en cuda
   - **ejecutar_cuda_experimentos.ypnb (ARCHIVO A EJECUTAR) :** Ejecuta los dos experimentos anteriores y almacena los resultados en la carpeta de RESULTADOS, 
         subdividida en dos carpetas, una para cada experimento. 

2. **FILTROS IMAGENES:**

   Conjunto de archivos con el objetivo de estudiar la aplicación de filtro en OpenCL. Los archivos son los siguientes:
        
   -  **filtros.py :** : Archivo con filtros mean,gaussian,sobel  de distintos tamaños
        
   - **kernels_filtros_imagenes.py:** Kernels para aplicar filtros
        
   -  **funciones_ejecutar_kernel_filtros.py:** funciones para aplicar filtros ditintos, con o sin memoria local etc
        
   - **funciones_experimento_filtros.py:** Funciones para realizar experimentos de aplicación de filtros. Obtienen gráficas y tablas.
   - **ejecutar_experimento_filtros.py( A EJECUTAR):** Archivo donde se ejecutan los 4 experimentos. Son los siguientes:

      - Experimento 1 - Local size optimo: Experimento donde para distintos filtros(mean,sobel,gaussian) se calculan los tiempos de ejecucion para distintos tamaños de imagen y distintos local sizes. Resultados obtenidos estan en EXPERIEMNTOS,RESULTADOS ,gaussian, mean y filtro_sobel

      - Experimento 2-  Memoria Local vs not: Comparar todos los kernels de filtros imagenes para filtros de distintos tamaños e imagenes 
               distintas para ver cuál es más óptimo
  
      - Experimento 3- Filtros divididos vs no: Experimento donde se comparan los tiempos de aplicar filtros de manera normal o de manera 
                  dividida para filtros cada vez mas grandes. Los resultados obtenidos se almacenan en RESULTADOS ,COMPARACION DE KERNELS , COMPARACION FILTROS
        
      - Experimento 4- Prueba 1000 veces: Experimento ejecutando filtros de imagnes con distintos local sizes pero calculando el promedio de los tiempos. Ejecutar el kernel 1000 veces y devolver los tiempos medios.Resultados se pueden encontrar en EXPERIMENTOS RESULTADOS 1000veces

   - **EXPERIMENTOS:** Resultados de todos los experimentos anteriores
   - **IMAGENES:** Imagenes sobre las que se van a aplicar los filtros

  3. **FUNCION HASH**
     
     En los siguientes archivos se encuentran funciones para realizar la minería de un bloque del blockchain en PyOpenCl. Los archivos son los siguientes:
     
        - **kernel_mining.py :** Kernel que mina un bloque dado
          
        - **ejecutar_kernel_mineria.py:** Funcion en python que ejecuta el kernel de mineria de un bloque del blockchain en PyOpencl
          
        - **funciones_experimento_mining.py:** Funciones para ejecutar el kernel de mineria y hacer pruebas, Se realizan dos experimentos:
             1. Para un mismo target, probar distintos lcoal sizes y global sizes. 
             2. Fijado un global size, hacer pruebas con distintas combinaciones de local sizes y de targets
                
       - **ejecutar_experimento_mineria.py** Ejecutar experimentos anteriores,resultados guardados en la carpeta RESULTADOS

4. **MULTIPLICACION DE MATRICES:**
   
   En los siguientes archivos se encuentran funciones para realizar la multiplicación de matrices en PyOpenCl. Los archivos son los siguientes:

  
    - **kernels_matrices.py:** Distintos kernels de multiplicacion de matrices
     
    - **funciones_ejecutar_kernel_matrices.py:**  funciones para ejecutar los kernels de multiplicacion de matrices
     
    - **funciones_experimentos_matrices.py:** Funciones que calculan tablas y graficos para distintos experimentos de multiplicar matrices.

    - **ejecutar_experimentos_matrices.py(A EJECUTAR):**  Archivo donde se ejecutan los experimentos del archivo anterior. 
      
       - Experimento 1 - Experimento_local_sizes : Para todos los kernels hacer pruebas de distintos local sizes para determianr el optimo, resutados en RESULTADOS en 
               subcarpetas con el nombre del kernel

       - Experimento 2- Experimento_comparacion_kernels_matrices: Prueba distintos kernels con distintas matrices para obtener los mejores tiempos.Resultados se pueden 
                  encontrar en RESULTADOS /Comparacion kernels

   - **comparar_3_gpus_cpu.py:** Archivo donde se compara la ejecución de todos los dispositivos utilizados durante el proyecto.
          
   - **RESULTADOS:** resultados de los experimentos

5. **INTRUCCIONES EJECUTAR OPENCL:**  Archivo notas que explica como ejecutar opencl en google collab

6. **OBTENER_INFORMACION_DISPOSITIVO:** Archivo python para obtener informacion del dispositivo que se esta utilizando

7. **RESULTADOS GOOGLE COLAB:** Carpeta donde se pueden encontrar los resultados de algunos experimentos de mult matrices en Google Colab GPU y CPU.
   - **mult_mat_basica_google_collab.py:** Archivo  a ejecutar en Google Colab de multiplicación de matrices basica
   - **Mult_Mat_Basica_CPU.csv:** Resultados CPU
   - **Mult_Mat_Basica_GPU.csv:** Resultados GPU
  
8. **RESULTADOS PORTATIL:** Carpeta donde se pueden encontrar los resultados de algunos experimentos de mult matrices y filtros ejecutados en mi portatil personal

9. **OPECNL_SCRIPTS:** Archivos con codigos en OpenCL(NO PYOPENCL)
    - multiplicacion_mat_opencl.cpp : Archivo de prueba en opencl de multiplicación matrices
    - incrementarvector.cpp : Archivo de prueba en opencl de incrementar un vector
      
      
