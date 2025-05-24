#include "matrix.h"
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// ... autres kernels : minus, sum, scalar, sigmoid ...

__global__ void matrixMinusKernel(const double* A, const double* B, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] - B[idx];
}

__global__ void matrixScalarKernel(const double* A, double scalar, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] * scalar;
}

__global__ void matrixSumKernel(const double* A, const double* B, double* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) C[idx] = A[idx] + B[idx];
}



/*matrix_t *alloc_matrix(unsigned rows, unsigned columns)
{
    matrix_t *res = (matrix_t *)malloc(sizeof(matrix_t));
    res->m = (double *)calloc(columns * rows, sizeof(double));
    res->columns = columns;
    res->rows = rows;
    return res;
}

void destroy_matrix(matrix_t *m)
{
    // printf("free %p %p\n", m, m->m);
    free(m->m);
    free(m);
}*/

/***
 *  vv mémoire unifiée vv
 */

matrix_t* alloc_matrix(unsigned rows, unsigned cols) {
    matrix_t* mat;
    size_t total = rows * cols * sizeof(double);
    
    cudaError_t err = cudaMallocManaged(&mat, sizeof(matrix_t));
    if (err != cudaSuccess) {
        printf("❌ alloc_matrix struct failed: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        return NULL;
    }
    err = cudaMallocManaged(&(mat->m), total);
    if (err != cudaSuccess) {
        printf("❌ alloc_matrix m failed: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        cudaFree(mat);
        return NULL;
    }

    mat->rows = rows;
    mat->columns = cols;
    return mat;
}

void destroy_matrix(matrix_t* mat) {

    // libère la matrice
    cudaFree(mat->m);
    cudaFree(mat);
}
/***
 *  ^^ mémoire unifiée ^^
 */

void print_matrix(matrix_t *m, bool is_short)
{
    unsigned lim_rows = 0;
    unsigned lim_col = 0;

    if (is_short)
    {
        lim_rows = MIN(m->rows, 4);
        lim_col = MIN(m->columns, 10);
    }
    else
    {
        lim_rows = m->rows;
        lim_col = m->columns;
    }

    for (int row = 0; row < lim_rows; row++)
    {
        for (int col = 0; col < lim_col; col++)
        {
            printf("%.2lf ", m->m[col + row * m->columns]);
        }
        if (is_short && lim_col != m->columns)
            printf("...");
        printf("\n");
    }
    if (is_short && lim_rows != m->rows)
        printf("...\n");
}

void hadamard_product(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] * m2->m[idx];
    }
}

/*void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] + m2->m[idx];
    }
}*/

void matrix_sum(matrix_t *m1, matrix_t *m2, matrix_t *res){
    int size = m1->rows * m1->columns;
    //double *d_A, *d_B, *d_C;

     // Allocation sur le GPU
    //cudaMalloc(&d_A, size * sizeof(double));
    //cudaMalloc(&d_B, size * sizeof(double));
    //cudaMalloc(&d_C, size * sizeof(double));

    // Copier A et B vers la mémoire GPU
    //cudaMemcpy(d_A, m1->m, size * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, m2->m, size * sizeof(double), cudaMemcpyHostToDevice);

    // Définir combien de threads et de blocs on veut
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Lancer le kernel
    matrixSumKernel<<<blocks, threads>>>(m1->m, m2->m, res->m, size);
    cudaDeviceSynchronize();

    // Copier le résultat de C vers res
    //cudaMemcpy(res->m, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Libération de la mémoire GPU
    //cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

/*void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->columns) &&
           (m1->columns == res->columns) &&
           (m1->rows == m2->rows) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = m1->m[idx] - m2->m[idx];
    }
}*/

void matrix_minus(matrix_t *m1, matrix_t *m2, matrix_t *res) {
    int size = m1->rows * m1->columns;
    //double *d_A, *d_B, *d_C;

     // Allocation sur le GPU
    //cudaMalloc(&d_A, size * sizeof(double));
    //cudaMalloc(&d_B, size * sizeof(double));
    //cudaMalloc(&d_C, size * sizeof(double));

    // Copier A et B vers la mémoire GPU
    //cudaMemcpy(d_A, m1->m, size * sizeof(double), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_B, m2->m, size * sizeof(double), cudaMemcpyHostToDevice);

    // Définir combien de threads et de blocs on veut
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Lancer le kernel
    matrixMinusKernel<<<blocks, threads>>>(m1->m, m2->m, res->m, size);
    cudaDeviceSynchronize();

    // Copier le résultat de C vers res
    //cudaMemcpy(res->m, d_C, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Libération de la mémoire GPU
    //cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

/**
 *
 *  CPU
 *
 */

/*void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert ( (m1->columns == m2->rows)  &&
             (m1->rows == res->rows)    &&
             (m2->columns == res->columns));

    for (int row = 0; row < m1->rows; row ++)
    {
        for (int col = 0; col < m2->columns; col ++)
        {
            int idx = col + row * m2->columns;
            double var = 0.0;

            for (int ii = 0; ii < m1->columns; ii++)
            {
                var += m1->m[ii + row * m1->columns] * m2->m[col + ii * m2->columns];
            }

            res->m[idx] = var;
        }
    }
}*/

__global__ void matrix_dot_kernel(const double *m1, const double *m2, double *res,
                                  int m1_rows, int m1_columns, int m2_columns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m1_rows && col < m2_columns)
    {
        int idx = col + row * m2_columns;
        double var = 0.0;
        for (int ii = 0; ii < m1_columns; ii++)
        {
            var += m1[ii + row * m1_columns] * m2[col + ii * m2_columns];
        }
        res[idx] = var;
    }
}

/*void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    int size_m1 = m1->rows * m1->columns * sizeof(double);
    int size_m2 = m2->rows * m2->columns * sizeof(double);
    int size_res = res->rows * res->columns * sizeof(double);

    double *d_m1, *d_m2, *d_res;

    // Allocation sur le device
    cudaMalloc((void **)&d_m1, size_m1);
    cudaMalloc((void **)&d_m2, size_m2);
    cudaMalloc((void **)&d_res, size_res);

    // Copie des matrices du host vers le device
    cudaMemcpy(d_m1, m1->m, size_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2->m, size_m2, cudaMemcpyHostToDevice);

    // Lancement du kernel CUDA
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((res->columns + 15) / 16, (res->rows + 15) / 16);

    matrix_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_m1, d_m2, d_res, m1->rows, m1->columns, m2->columns);

    // Copie du résultat vers le host
    cudaMemcpy(res->m, d_res, size_res, cudaMemcpyDeviceToHost);

    // Libération mémoire device
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);
}*/

/***
*
*   vv mémoire unifiée vv
*
 */

 void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((res->columns + 15) / 16,
                       (res->rows + 15) / 16);

    matrix_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        m1->m, m2->m, res->m, m1->rows, m1->columns, m2->columns);

    cudaDeviceSynchronize();  // synchronisation 
}

/***
*
*   ^^ mémoire unifiée ^^
*
 */

/*

#define TILE_SIZE 16

__global__ void matrix_dot_kernel_shared(const double* m1, const double* m2, double* res,
                                         int m1_rows, int m1_columns, int m2_columns) {
    __shared__ double tile_m1[TILE_SIZE][TILE_SIZE];
    __shared__ double tile_m2[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double var = 0.0;

    for (int tile = 0; tile < (m1_columns + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Charger les sous-blocs dans la mémoire partagée
        if (row < m1_rows && tile * TILE_SIZE + threadIdx.x < m1_columns)
            tile_m1[threadIdx.y][threadIdx.x] = m1[row * m1_columns + tile * TILE_SIZE + threadIdx.x];
        else
            tile_m1[threadIdx.y][threadIdx.x] = 0.0;

        if (col < m2_columns && tile * TILE_SIZE + threadIdx.y < m1_columns)
            tile_m2[threadIdx.y][threadIdx.x] = m2[(tile * TILE_SIZE + threadIdx.y) * m2_columns + col];
        else
            tile_m2[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();  // Synchroniser les threads du bloc

        // Calculer le produit partiel
        for (int k = 0; k < TILE_SIZE; ++k) {
            var += tile_m1[threadIdx.y][k] * tile_m2[k][threadIdx.x];
        }

        __syncthreads();  // Synchroniser avant de passer à la tuile suivante
    }

    if (row < m1_rows && col < m2_columns)
        res[row * m2_columns + col] = var;
}

void matrix_dot(matrix_t *m1, matrix_t *m2, matrix_t *res)
{
    assert((m1->columns == m2->rows) &&
           (m1->rows == res->rows) &&
           (m2->columns == res->columns));

    int size_m1 = m1->rows * m1->columns * sizeof(double);
    int size_m2 = m2->rows * m2->columns * sizeof(double);
    int size_res = res->rows * res->columns * sizeof(double);

    double *d_m1, *d_m2, *d_res;

    // Allocation sur le device
    cudaMalloc((void **)&d_m1, size_m1);
    cudaMalloc((void **)&d_m2, size_m2);
    cudaMalloc((void **)&d_res, size_res);

    // Copie des matrices du host vers le device
    cudaMemcpy(d_m1, m1->m, size_m1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, m2->m, size_m2, cudaMemcpyHostToDevice);

    // Lancement du kernel CUDA
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
    (m2->columns + TILE_SIZE - 1) / TILE_SIZE,  // nb de blocs en X (colonnes)
    (m1->rows    + TILE_SIZE - 1) / TILE_SIZE   // nb de blocs en Y (lignes)
    );

    matrix_dot_kernel_shared<<<blocksPerGrid, threadsPerBlock>>>(
    d_m1, d_m2, d_res, m1->rows, m1->columns, m2->columns
    );

    // Copie du résultat vers le host
    cudaMemcpy(res->m, d_res, size_res, cudaMemcpyDeviceToHost);

    // Libération mémoire device
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_res);
}*/

void matrix_function(matrix_t *m1, double (*f)(double), matrix_t *res)
{
    assert((m1->columns == res->columns) &&
           (m1->rows == res->rows));

    for (int idx = 0; idx < m1->rows * m1->columns; idx++)
    {
        res->m[idx] = f(m1->m[idx]);
    }
}

void matrix_transpose(matrix_t *m1, matrix_t *res)
{
    assert((m1->columns == res->rows) &&
           (m1->rows == res->columns));

    for (int row = 0; row < m1->rows; row++)
    {
        for (int col = 0; col < m1->columns; col++)
        {
            res->m[row + col * m1->rows] = m1->m[col + row * m1->columns];
        }
    }
}

/*void matrix_scalar(matrix_t *m1, double s, matrix_t *res)
{
    assert((m1->rows == res->rows) &&
           (m1->columns == res->columns));

    for (int idx = 0; idx < m1->columns * m1->rows; idx++)
    {
        res->m[idx] = m1->m[idx] * s;
    }
}*/

void matrix_scalar(matrix_t *m1, double scalar, matrix_t *res)
{
    int size = m1->rows * m1->columns;
    //size_t bytes = size * sizeof(double);

    //double *d_A, *d_C;
    //cudaMalloc(&d_A, bytes);
    //cudaMalloc(&d_C, bytes);

    //cudaMemcpy(d_A, m1->m, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    //matrixScalarKernel<<<blocks, threads>>>(d_A, scalar, d_C, size);
    matrixScalarKernel<<<blocks, threads>>>(m1->m, scalar, res->m, size);
    cudaDeviceSynchronize();

    //cudaMemcpy(res->m, d_C, bytes, cudaMemcpyDeviceToHost);

    //cudaFree(d_A);
    //cudaFree(d_C);
}

void matrix_memcpy(matrix_t *dest, const matrix_t *src)
{
    assert((dest->rows == src->rows) &&
           (dest->columns == src->columns));

    memcpy(dest->m, src->m, src->columns * src->rows * sizeof(double));
}