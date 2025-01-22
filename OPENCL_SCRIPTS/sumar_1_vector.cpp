//Incluir directorio OpenCL
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

// KERNEL PARA INCREMENTAR UN VECTOR
const char* IncrementVector_kernel = R"(
__kernel void IncrementVector_kernel(int dim, __global int* input, __global int* output) {
    //Obtener el índice del work item
    int id = get_global_id(0);

    //Comprobar que estamos dentro del vector
    if (id < dim) {
        //Realizar la suma
        output[id] = input[id] + 1;
    }
}
)";

//Función auxiliar para comprobar si hay algun error
void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        printf("Error en %s: %d\n", operation, err);
        exit(1);
    }
}

//Código en el host
int main() {
    //Obtener el dispositivo donde se va a ejecutar el kernel
    cl_device_type device_type = CL_DEVICE_TYPE_GPU;

    //Inicializar la variable error
    cl_int err;

    //Obtener la plataforma y comprobar si hay algún error
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    checkError(err, "clGetPlatformIDs");

    // Obtener el dispositivo y comprobar si hay algún error
    cl_device_id device;
    err = clGetDeviceIDs(platform, device_type, 1, &device, NULL);
    checkError(err, "clGetDeviceIDs");

    // Obtener el contexto y comprobar si hay algún error
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err, "clCreateContext");

    // Obtener la Command Queue con la propiedad que permite medir el tiempo
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    checkError(err, "clCreateCommandQueueWithProperties");

    // Obtener el Programa para el kernel y comprobar si hay algún error
    cl_program program = clCreateProgramWithSource(context, 1, &IncrementVector_kernel, NULL, &err);
    checkError(err, "clCreateProgramWithSource");

    // Compilar el programa y comprobar el error mediante clGetProgramBuildInfo
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error en la compilación del kernel:\n%s\n", log);
        free(log);
        exit(1);
    }

    // Crear kernel y comprobar si hay algún error
    cl_kernel kernel = clCreateKernel(program, "IncrementVector_kernel", &err);
    checkError(err, "clCreateKernel");

    // Definir tamaño del vector
    cl_int n = 16; 

    // Asignar memoria para el vector input y output
    int* input = (int*)malloc(sizeof(int) * n);
    int* output = (int*)malloc(sizeof(int) * n);

    // Inicializar el vector de entrada
    for (int i = 0; i < n; i++) {
        input[i] = i;
    }

    // Crear buffers de memoria, input solo lectura, output solo escritura
    cl_mem bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, input, &err);
    checkError(err, "clCreateBuffer (bufInput)");
    cl_mem bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * n, NULL, &err);
    checkError(err, "clCreateBuffer (bufOutput)");

    // Escribir en el buffer input el vector inicial y comprobar si hay error
    err = clEnqueueWriteBuffer(command_queue, bufInput, CL_TRUE, 0, sizeof(int) * n, input, 0, NULL, NULL);
    checkError(err, "clEnqueueWriteBuffer");

    // Establecer los argumentos del kernel (dim, vector input, vector output)
    err = clSetKernelArg(kernel, 0, sizeof(cl_int), &n);
    checkError(err, "clSetKernelArg (n)");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufInput);
    checkError(err, "clSetKernelArg (bufInput)");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufOutput);
    checkError(err, "clSetKernelArg (bufOutput)");

    // Definir el tamaño global (nº de work items) y local (nº work items por work group)
    size_t global_size = n;
    size_t local_size = 4;

    // Crear un evento para medir el tiempo de ejecución del kernel
    cl_event event;

    //Variables para medir el tiempo de ejecución
    cl_ulong start_time;
    cl_ulong end_time;

    // Encolar kernel y comprobar si hay algún error
    err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event);
    checkError(err, "clEnqueueNDRangeKernel");

    // Esperar a que se complete el kernel
    clFinish(command_queue);

    // Medir el tiempo de ejecucion del kernel en segundos
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    checkError(err, "clGetEventProfilingInfo (start time)");
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    checkError(err, "clGetEventProfilingInfo (end time)");
    
    //Tiempo de ejecucion del kernel
    double execution_time = (end_time - start_time) / 1e9; 
    printf("Tiempo de ejecucion del kernel: %.3f s\n", execution_time);

    // Leer el buffer de salida, copiar el resultado en output
    err = clEnqueueReadBuffer(command_queue, bufOutput, CL_TRUE, 0, sizeof(int) * n, output, 0, NULL, NULL);
    checkError(err, "clEnqueueReadBuffer");

    // Liberar recursos(recomendado)
    clReleaseMemObject(bufInput);
    clReleaseMemObject(bufOutput);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(input);
    free(output);

    return 0;
}

