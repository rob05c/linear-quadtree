//----------------------------------------------------------
// Matrix Multiplication - CUDA Version 2 to run on GPUs
//---------------------------------------------------------
//  By Gita Alaghband, Lan Vu 
//  Use shared memory with higher access speed
//  Updated in 8/8/2011
//-----------------------------------------------------------

#include "test.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>    
#include </usr/include/linux/cuda.h>

#define TILE 16

void InitializeMatrix(float** &x,int n,float value)
{
  x = new float*[n];
  x[0] = new float[n*n];

  for (int i = 1; i < n; i++)	
    x[i] = x[i-1] + n;

  for (int i = 0 ; i < n ; i++)
    for (int j = 0 ; j < n ; j++)
      x[i][j] = value;
}

void DeleteMatrix(float **x,int n)
{
	delete[] x[0];
	delete[] x; 
}

void PrintMatrix(float **x, int n) 
{
  for(int i = 0; i < n; ++i)
  {
    printf("Row %3d: ", i + 1);
    for(int j = 0; j < n; ++j)
      printf("%7.2f", x[i][j]);
    printf("\n");
  }
}

__global__ void MultiplyMatrix_Version2(float* a, float* b, float* c, int n)
{

  __shared__ float A[TILE][TILE];
  __shared__ float B[TILE][TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
	
  int Row = blockIdx.y*TILE + ty;
  int Col = blockIdx.x*TILE + tx;
	
  float value = 0;

  if (Row < n && Col < n) 
  {
    for (int i = 0; i < n ; i += TILE) 
    {
      // Load the matrices from device memory to shared memory
      // Each thread loads one element of each matrix
      A[ty][tx] = a[ n*Row + (i + tx)]; 
      B[ty][tx] = b[ n*(i + ty) + Col]; 
      // Synchronize to make sure the matrices are loaded
      __syncthreads();	

      // Multiply the two matrices
      // Each thread computes one element of the block sub-matrix
      int m = ((n - i) < TILE)? (n - i): TILE;
      for (int j = 0; j < m; j++)  value += A[ty][j] * B[j][tx];

      // Synchronize to make sure that the preceding computation is done before 
      // loading two new sub-matrices of A and B in the next iteration
      __syncthreads();	
			
    }
    c[Row*n + Col] = value;
  }
}

void test_matmul(const size_t matrix_size, const bool print)
{
  float **a, **b,**c; //host pointers
  float *da, *db, *dc; //device pointers
  float runtime;
  int n = matrix_size;
	
  //Initialize the value of matrix a and vetors x, y 
  InitializeMatrix(a,n,1.0);
  InitializeMatrix(b,n,1.0);
  InitializeMatrix(c,n,0.0);

  //Print the input matrices
  if(print)
  {
    printf("Matrix a[n][n]:\n");
    PrintMatrix(a,n); 
    printf("Matrix b[n][n]:\n");
    PrintMatrix(b,n); 
  }
	
  runtime = clock()/(float)CLOCKS_PER_SEC;

  //Declare grid size and block size
  int numblock = n/TILE + ((n%TILE)?1:0);
  dim3 dimGrid(numblock,numblock);	
  dim3 dimBlock(TILE,TILE);	

  //Allocate memory on device
  cudaMalloc((void**)&da, n*n*sizeof(float));
  cudaMalloc((void**)&db, n*n*sizeof(float));
  cudaMalloc((void**)&dc, n*n*sizeof(float));

  //Copy data to the device
  cudaMemcpy(da, a[0], n*n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b[0], n*n*sizeof(float), cudaMemcpyHostToDevice);

  //Do the matrix multiplication on the device (GPU)
  MultiplyMatrix_Version2<<<dimGrid,dimBlock>>>(da,db,dc,n);
	
  cudaThreadSynchronize();

  //Get results from the device
  cudaMemcpy(c[0],dc, n*n*sizeof(float),cudaMemcpyDeviceToHost);

  runtime = clock() - runtime;

  //The matrix is as below:
  if(print)
  {
    printf("Matrix c[n][n]:\n");
    PrintMatrix(c,n); 
  }
  printf("matmul x%d runs in %.2f seconds\n", n, (runtime)/float(CLOCKS_PER_SEC));

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  DeleteMatrix(a,n);	
  DeleteMatrix(b,n);	
  DeleteMatrix(c,n);	
}
