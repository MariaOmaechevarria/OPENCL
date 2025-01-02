INSTRUCCIONES PARA EJECUTAR OPENCL EN GOOGLE COLLAB 

Paso 1: Instalar los drivers necesarios (10 minutos aproximadamente):
        !sudo apt update
        !sudo apt purge *nvidia* -y
        !sudo apt install nvidia-driver-530 -y


        !pip install pyopencl
        !apt-get install -y pocl-opencl-icd ocl-icd-libopencl1

Paso 2: Ejecutar código en pyopencl
