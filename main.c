#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

void print_matrix(int N, double *matrix)
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

double determ(int N, double *matrix)
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
    srand(time(NULL));
    int N;
    printf("Matrix NxN \nEnter N: ");
    scanf("%d", &N);
    double *matrixHost;
    matrixHost = (double *)malloc(sizeof(double) * N * N);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int r = rand() % 7 - 3;
            double temp = (r == 0 ? 1 : r);
            matrixHost[i * N + j] = (temp);
        }
    }
    if (N <= 8)
        print_matrix(N, matrixHost);
    // определитель на CPU
    float start = clock();
    double det = determ(N, matrixHost);
    float end = clock();
    if (N <= 8)
        print_matrix(N, matrixHost);

    free(matrixHost);

    printf("CPU determinant = %.2f\n", det);
    printf("Time CPU: %.2f msec\n", (end - start) * 1000.0 / CLOCKS_PER_SEC);
}