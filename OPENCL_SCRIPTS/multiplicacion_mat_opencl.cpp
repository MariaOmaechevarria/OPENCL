#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <iostream>

// KERNEL BÁSICO
const char* MatrixMul_kernel = R"(
__kernel void MatrixMul_kernel(int dim, __global int* A, __global int* B, __global int* C) {
    int fila = get_global_id(0);
    int columna = get_global_id(1);

    int resultado = 0;

    for (int i = 0; i < dim; i++) {
        resultado += A[fila * dim + i] * B[i * dim + columna];
    }

    C[fila * dim + columna] = resultado;
}
)";

int mult_matrices_basica(cl_device_type device_type) {
    cl_int err;

    // Plataforma
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);

    // Dispositivo
    cl_device_id device;
    err = clGetDeviceIDs(platform, device_type, 1, &device, NULL);

    // Contexto
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // Command Queue
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, properties, &err);

    // Programa
    cl_program program = clCreateProgramWithSource(context, 1, &MatrixMul_kernel, NULL, &err);

    // Compilar el programa
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Crear kernel
    cl_kernel kernel = clCreateKernel(program, "MatrixMul_kernel", &err);

    // Definir tamaño de las matrices
    cl_int dim = 1024;

    // Asignar memoria para las matrices
    int* A = (int*)malloc(sizeof(int) * dim * dim);
    int* B = (int*)malloc(sizeof(int) * dim * dim);
    int* C = (int*)malloc(sizeof(int) * dim * dim);

    // Inicializar las matrices
    for (int i = 0; i < dim * dim; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = 0;
    }

    // Crear buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * dim * dim, A, &err);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * dim * dim, B, &err);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * dim * dim, NULL, &err);

    // Escribir buffers
    err = clEnqueueWriteBuffer(command_queue, bufA, CL_TRUE, 0, sizeof(int) * dim * dim, A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(command_queue, bufB, CL_TRUE, 0, sizeof(int) * dim * dim, B, 0, NULL, NULL);

    // Establecer argumentos del kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_int), &dim);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufA);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufB);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufC);

    // Definir el tamaño global y local
    size_t global_size[2] = { dim, dim };
    size_t local_size[2] = { 8, 8 };

    // Crear un evento para medir el tiempo de ejecución del kernel
    cl_event event;

    // Encolar kernel
    cl_ulong start_time;
    cl_ulong end_time;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, &event);

    // Esperar a que se complete el kernel
    clFinish(command_queue);

    // Medir el tiempo de ejecución del kernel
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);

    double execution_time = (end_time - start_time) / 1e9; // Convertir a segundos
    printf("Tiempo de ejecución del kernel: %.3f s\n", execution_time);

    // Leer el buffer de salida y verificar resultados
    err = clEnqueueReadBuffer(command_queue, bufC, CL_TRUE, 0, sizeof(int) * dim * dim, C, 0, NULL, NULL);


    // Liberar recursos
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}

int main() {

    cl_device_type device_type = CL_DEVICE_TYPE_GPU; // O CL_DEVICE_TYPE_CPU para CPU
    mult_matrices_basica(device_type);

    return 0;
}
