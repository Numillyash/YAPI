#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif // !__CUDACC__


__global__ void GAUSS(double* mat, int N, double* det)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    double del;
    if (x < N)
    {
        int check = 1;
        for (int i = 0; i < N - 1; i++) // идем по столбцам матрицы
        {
             // каждый поток ждет, пока остальные потоки блока достигнут этой точки
            if (x > i)
            {
                del = mat[x * N + i] / mat[i * N + i];
                for (int j = i; j < N; j++)
                    mat[x * N + j] -= del * mat[i * N + j];
            }
            else
                break;
        }
        det[x] = mat[(x)*N + x];
        /*for (int i = 0; i < N; i++)
            printf("%f ",det[i]);
        printf("\n%d\n", x);*/
        __syncthreads();
        if (x == N - 1)
        {
            for (int i = 1; i < N; i++)
                det[0] *= det[i];
            det[0] *= check;
        }
    }
}


void print_matrix(int N, double* matrix)
{
    int k = 10;
    printf("┌");
    for (int i = 1; i < N * k; i++)
    {
        if (!(i % k))
            printf("┬");
        else
            printf("─");
    }
    printf("┐\n");

    for (int y = 0; y < N; y++)
    {
        for (int x = 0; x < N; x++)
        {
            printf("│ %7.3f ", matrix[y * N + x]);
        }
        printf("│\n");
        if (y != N - 1)
        {
            printf("├");
            for (int i = 1; i < N * k; i++)
            {
                if (!(i % k))
                    printf("┼");
                else
                    printf("─");
            }
            printf("┤\n");
        }
    }

    printf("└");
    for (int i = 1; i < N * k; i++)
    {
        if (!(i % k))
            printf("┴");
        else
            printf("─");
    }
    printf("┘\n");
}

double determinantCPU(int N, double* matrix)
{

    double det = 1, temp, coefficient;
    for (int i = 0; i < N - 1; i++)
    {
        temp = matrix[i * N + i];       // элемент на строчке, ниже которого будут нули
        for (int j = i + 1; j < N; j++) // идем вниз по строчкам и обнуляем элементы столбца, остальные элементы строки мы вычитаем
        {
            coefficient = matrix[j * N + i] / temp; // коэффициент, на который умножаются элементы вверхней строки для зануления
            for (int s = i; s < N; s++)
                matrix[j * N + s] -= coefficient * matrix[i * N + s]; // идём по строке и из текущего элемента вычитаем (k*элемент сверху)
        }
    }
    for (int i = 0; i < N; i++)
        det *= matrix[i * N + i]; // определитель полученной матрицы равен произведению элементов на главной диагонали
    return det;
}

int main()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    // printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    int N;
    printf("Matrix NxN \nEnter N: ");
    scanf("%d", &N);
    int blockSize;
    printf("Enter blockSize: ");
    scanf("%d", &blockSize);
    double* matrixHost;
    double* matrix;
    matrixHost = (double*)malloc(sizeof(double) * N * N);
    matrix = (double*)malloc(sizeof(double) * N * N);
    // инициализация матрицы
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            matrixHost[i * N + j] = 1 + rand() % 5;
    }
    //print_matrix(N, matrixHost);
    memcpy(matrix, matrixHost, sizeof(double) * N * N);
    // определитель на CPU
    float start = clock();
    double determ = determinantCPU(N, matrixHost);
    float end = clock();

    free(matrixHost);

    printf("CPU determinant = %.2f\n", determ);
    printf("Time CPU: %f\n\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);

    // определитель на GPU
    double determinant = 1;
    double* matrixDevice = 0;
    double* determinant2;
    // Выделяем память для данных, которые будут на GPU
    cudaMalloc((void**)&matrixDevice, (unsigned long long) (N * N * sizeof(double)));
    cudaMalloc((void**)&determinant2, N * sizeof(double));
    // Передаем в matrixDevice значения matrix
    // копирование исходной матрицы в память GPU
    cudaMemcpy(matrixDevice, matrix, N * N * sizeof(double), cudaMemcpyHostToDevice);
    // Запускаем ядро
    start = clock();
    // Запуск ядра из blockSize блока по 1024 потоков
    GAUSS <<<blockSize, 1024 >>> (matrixDevice, N, determinant2);
    cudaThreadSynchronize();
    // передаем значение определителя обратно в CPU
    cudaMemcpy(&determinant, &(determinant2[0]), sizeof(double), cudaMemcpyDeviceToHost);
    end = clock();
    printf("GPU determinant = %.2f\n", determinant);
    printf("Time GPU: %f\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);

    cudaFree(matrixDevice);
    cudaFree(determinant2);
    free(matrix);
}