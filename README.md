# OPENCL

Repositorio de OpenCL
Este repositorio contiene una colección de scripts y notebooks relacionados con el uso de OpenCL para diversas aplicaciones, incluyendo la multiplicación de matrices, filtros de imágenes y funciones hash. A continuación, se detalla la estructura del repositorio y la descripción de cada archivo y carpeta.

# CARPETAS Y ARCHIVOS

1. CUDA 
   Podemos encontrar los siguientes archivos.

       - **mult_matrices_basico_cuda.py :** Kernel de multiplicacion basica en cuda y la funcion en python para ejecutar el kernel para unos argumentos dados
       - **mult_matrices_basico_opencl.py :** Kernel de multiplicacion basica en opencl y la funcion en python para ejecutar el kernel para unos argumentos dados
       - **funciones_experimentos_cuda.py:** Funciones para crear tablas, graficos y ejecutar los kernels anteriores para realizar dos experimentos.
             1.Comparar cuda vs opencl
             2-Probar distintos local sizes para obtener los tiempos en cuda
       - **ejecutar_cuda_experimentos.py (ARCHIVO EJECUTAR) :** Ejecuta los dos experimentos anteriores y almacena los resultados en la carpeta de RESULTADOS, subdividida en dos carpetas, una 
       para cada experimento. Hay una tabla y un grafico para cada una.

2. FILTROS IMAGENES

      Podemos encontrar las siguientes subcarpetas

      - ARCHIVOS PYTHON UTILES:
     
            - determinar_mejor-local_size.py: Determina los mejores local sizes segun la GPU usada
        
            -  filtros.py : Distintos filtros 8mean,gaussian,sobel) y de distintos tamaños
        
            - kernels_filtros_imagenes.py: distintos kernels para aplicar filtros
        
            -  funciones_filtros.py: funciones para aplicar filtros ditintos, con o sin memoria local etc
        
            - experimento_filtros.py: Diversas funciones para realizar distintos experimentos, cada una se ejecuta en su correspondiente jupiter notebook. Obtienen 
                graficos y data frames

            - EXperimento_distintos_filtros_filtro_dividido_vs_no.ypinb: Experimento donde se comparan los tiempos de aplicar filtros de manera normal o de manera 
                  dividida para filtros cada vez mas grandes. Los resultados obtenidos se almacenan en RESULTADOS ,COMPARACION DE KERNELS , COMPARACION FILTROS

            - Pruebas_filtros_local_sizes_GPU.yipnb: Experimento donde para distintos filtros(mean,sobel,gaussian) se calculan los tiempos de ejecucion para 
               distintos tamaños de imagen y distintos local sizes. Resultados obtenidos estan en EXPERIEMNTOS,RESULTADOS ,filtro_gaussian,filtro_mean y 
                       filtro_sobel

            - Pruebas_kernels_filtros_localvsnot.iypnb: Comparar todos los kernels de filtros imagenes para filtros de distintos tamaños e imagenes 
               distintas.Resultados obtenidos en EXPERIMENTOS,RESULTADOS,COMPARACION DE KERNELS, FILTRO3X3,FILTRO5X5...
        
            - experimento_1000veces.ipynb: Experimento ejecutando filtros de imagnes con distintos local sizes pero calculando el promedio de los tiempos. Ejecutar 
                el kernel 1000 veces y devolver los tiempos medios.Resultados se pueden encontrar en EXPERIMENTOS RESULTADOS 1000veces

      - EXPERIMENTOS: Resultados de todos los experimentos anteriores
      - IMAGENES: Imagenes sobre las que se van a aplicar los filtros

  3. FUNCION HASH
     Podemos encontrar los siguientes archivos:
        - SHA-512.py: Implementacion en pyopencl y python de la funcion SHA-512
          
        - SHA-256.cpp : Implementacion en C++ de la funcion SHA-256
          
        - SHA-256+NONCE.py: Implementacion de la funcion SHA-256 en opencl añadiendo un nonce cualquiera
          
        - Mineria_python.py: Funcion en python que realiza el proceso de mineria simulando el blockchain
          
        - Mineria_GPU_def.py: Kernel de  mineria simulando el block chain en OPENCL y funcion que ejecuta el kernel
          
        - Experimento_mining.py: Funciones para ejecutar el kernel de mineria y hacer pruebas, Se realizan dos experimentos:
             1. Para un mismo target, probar distintos lcoal sizes y global sizes
             2. Fijado un global size, hacer pruebas con distintas combinaciones de local sizes y de targets
                
        -Experimento_hash_mining.ipynb: Ejecutar experimentos anteriores,resultados guardados en la carpeta RESULTADOS

4. MULTIPLICACION DE MATRICES:
Se encuentran las siguientes subcarpetas:

   - ARCHIVOS PYTHON UTILES: Se encuentras los siguientes archivos
         - kernels_matrices.py: Distintos kernels de multiplicacion de matrices
     
         - funciones_matrices.py: distintas funciones para ejecutar los kernels de multiplicacion de matrices
     
         - experimentos_matrices.py: Funciones que calculan tablas y graficos para distintos experimentos de multiplicar matrices

         - Experimento_comparacion_kernels_matrices: Prueba distintos kernels con distintas matrices para obtener los mejores tiempos.Resultados se pueden 
                  encontrar en RESULTADOS /Comparacion kernels

        - Experimento_local_sizes.ipynb: Para todos los kernels hacer pruebas de distintos local sizes para determianr el optimo, resutados en RESULTADOS en 
               subcarpetas con el nombre del kernel
          
  - RESULTADOS: resultados de los experimentos

5. INTRUCCIONES EJECUTAR OPENCL: Instrucciones para ejecutar opencl en google collab
6. OBTENER_INFORMACION_DISPOSITIVO: Archivo python para obtener informacion del dispositivo que se esta usando
7. RESULTADOS PORTATIL: Carpeta donde se pueden encontrar los resultados de algunos experimentos de mult matrices y filtros ejecutados en mi portatil personal
8. OPECNL_SCRIPTS: Archivos con codigos en OpenCL(NO PYOPENCL)
    - Archivo de prueba en opencl de multiplicación matrices
      
      
