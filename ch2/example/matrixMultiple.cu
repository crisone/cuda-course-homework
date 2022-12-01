#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#define M 512
#define K 512
#define N 512

#define BLOCK_SIZE 16  //block size, each thread to calucate each block

static constexpr int A_SIZE = M * K;
static constexpr int B_SIZE = K * N;
static constexpr int C_SIZE = M * N;

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(rand() % 10 + 1);
	}
}

void printMatrix(float *array, int row, int col)
{
	float *p = array;
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			printf("%10lf", p[x]);
		}
		p = p + col;
		printf("\n");
	}
	return;
}

void  multiplicateMatrixOnHost(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
	for (int i = 0; i < M_p; i++)
	{
		for (int j = 0; j < N_p; j++)
		{
			float sum = 0;
			for (int k = 0; k < K_p; k++)
			{
				sum += array_A[i*K_p + k] * array_B[k*N_p + j];
			}
			array_C[i*N_p + j] = sum;
		}
	}

}

__global__ void multiplicateMatrixOnDevice(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
	int ix = threadIdx.x + blockDim.x*blockIdx.x;//row number
	int iy = threadIdx.y + blockDim.y*blockIdx.y;//col number

	if (ix < N_p && iy < M_p)
	{
		float sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += array_A[iy*K_p + k] * array_B[k*N_p + ix];
		}
		array_C[iy*N_p + ix] = sum;
	}
}

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP

	__shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;


	float Csub = 0.0;

	for (int i = 0; i < (int)(ceil((float)numAColumns / BLOCK_SIZE)); i++)
	{
		//printf("block.x=%d,block.y=%d,threadIdx.x=%d,threadIdx.y=%d,row=%d,col=%d,sharedM[%d][%d]=A[%d],A的值：%f,sharedN[%d][%d]=B[%d],B的值：%f\n",
		//	blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col,
		//	threadIdx.y, threadIdx.x, row*numAColumns + i * BLOCK_SIZE + tx, A[row*numAColumns + i * BLOCK_SIZE + tx],
		//	threadIdx.y, threadIdx.x, (i*BLOCK_SIZE + ty)*numBColumns + col, B[(i*BLOCK_SIZE + ty)*numBColumns + col]);

		if (i*BLOCK_SIZE + tx < numAColumns && row < numARows)
			sharedM[ty][tx] = A[row*numAColumns + i * BLOCK_SIZE + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*BLOCK_SIZE + ty < numBRows && col < numBColumns)
			sharedN[ty][tx] = B[(i*BLOCK_SIZE + ty)*numBColumns + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();


		for (int j = 0; j < BLOCK_SIZE; j++)
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}


	if (row < numCRows && col < numCColumns)
		C[row*numCColumns + col] = Csub;

}

float caltime_cpu(float *h_A, float *h_B, float *hostRef) {
	clock_t start = 0, finish = 0;
	float time;
	start = clock();
	multiplicateMatrixOnHost(h_A, h_B, hostRef, M, K, N);
	finish = clock();
	time = (float)(finish - start) / CLOCKS_PER_SEC;
	return time;
}

float caltime_gpu_normal(float *d_A, float *d_B, float *d_C, float *deviceRef) {

    int dimx = BLOCK_SIZE;
    int dimy = BLOCK_SIZE;

    dim3 block(dimx, dimy);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t gpustart, gpustop;
    float elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);

    multiplicateMatrixOnDevice<<<grid,block>>> (d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);

    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, C_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	return elapsedTime / 1000;
}

float caltime_gpu_shared_mem(float *d_A, float *d_B, float *d_C, float *deviceRef) {
    int dimx = BLOCK_SIZE;
    int dimy = BLOCK_SIZE;

    dim3 block(dimx, dimy);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t gpustart, gpustop;
	float elapsedTime = 0.0;
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);

	matrixMultiplyShared << < grid, block >> > (d_A, d_B, d_C, M, K, K, N, M, N);

	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);

    cudaMemcpy(deviceRef, d_C, C_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
    //         M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);

	return elapsedTime / 1000;
}

float caltime_gpu_math_lib(float *d_A, float *d_B, float *d_C, float *deviceRef) {
    // cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t gpustart, gpustop;
	float elapsedTime = 0.0;

    elapsedTime = 0.0;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);

    float a = 1, b = 0;
    cublasSgemm(
      handle,
      CUBLAS_OP_T,   //矩阵A的属性参数，转置，按行优先
      CUBLAS_OP_T,   //矩阵B的属性参数，转置，按行优先
      M,          //矩阵A、C的行数
      N,          //矩阵B、C的列数
      K,          //A的列数，B的行数，此处也可为B_ROW,一样的
      &a,             //alpha的值
      d_A,            //左矩阵，为A
      K,          //A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
      d_B,            //右矩阵，为B
      N,          //B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
      &b,             //beta的值
      d_C,            //结果矩阵C
      M           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
    );
    // cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(gpustop, 0);
    cudaEventSynchronize(gpustop);

    cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
    cudaEventDestroy(gpustart);
    cudaEventDestroy(gpustop);
    cudaMemcpy(deviceRef, d_C, C_SIZE * sizeof(float), cudaMemcpyDeviceToHost);


	return elapsedTime/1000;
}


bool compare_mat(float* mat_A, float* mat_B, int size) {
	for(int i = 0; i < size; ++i) {
		float diff = mat_A[i] - mat_B[i];
		if(diff > 1e-9 || diff < -1e-9) {
			printf("! incorrect ret: diff [%f] at index %d \n", diff, i);
			return false;
		}
	}
	return true;
}

bool compare_mat_transpose(float* mat_A, float* mat_B, int rows, int cols) {
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			float diff = mat_A[cols * i + j] - mat_B[rows * j + i];
			if(diff > 1e-9 || diff < -1e-9) {
				printf("! incorrect ret: diff [%f] at index %d \n", diff, i);
				return false;
			}
		}
	}
	return true;
}

int main(int argc, char **argv)
{
	float *h_A, *h_B, *hostRef, *deviceRef;
	
	h_A = (float*)malloc(A_SIZE * sizeof(float));
	h_B = (float*)malloc(B_SIZE * sizeof(float));
	hostRef = (float*)malloc(C_SIZE * sizeof(float));
	deviceRef = (float*)malloc(C_SIZE * sizeof(float));

	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, A_SIZE * sizeof(float));
	cudaMalloc((void**)&d_B, B_SIZE * sizeof(float));
	cudaMalloc((void**)&d_C, C_SIZE	* sizeof(float));

	int test_round = 100;
	float t1=0, t2=0, t3=0, t4=0;
	for(int i = 0; i < test_round; ++i) {
	    initial(h_A, A_SIZE);
	    initial(h_B, B_SIZE);

	    // printf("==================== Question Start ====================\n");
	    // printMatrix(h_A, M, K);
	    // printf("x\n");
	    // printMatrix(h_B, K, N);
	    // printf("==================== Question End ====================\n");

	    cudaMemcpy(d_A, h_A, A_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(d_B, h_B, B_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	    // printf("t1 calculating...\n");
	    // t1 += caltime_cpu(h_A, h_B, hostRef);
	    // printMatrix(hostRef, M, N);

	    // printf("t2 calculating...\n");
	    t2 += caltime_gpu_normal(d_A, d_B, d_C, deviceRef);
	    // compare_mat(hostRef, deviceRef, C_SIZE);
	    // printMatrix(deviceRef, M, N);

	    // printf("t3 calculating...\n");
	    t3 += caltime_gpu_shared_mem(d_A, d_B, d_C, deviceRef);
	    // compare_mat(hostRef, deviceRef, C_SIZE);
	    // printMatrix(deviceRef, M, N);

	    // printf("t4 calculating...\n");
	    // t4 += caltime_gpu_math_lib(d_A, d_B, d_C, deviceRef);
	    // compare_mat_transpose(hostRef, deviceRef, M, N);
	    // printMatrix(deviceRef, M, N);
	    // printf("===================== Time summary ==================\n");
	    // printf("t1: %f, t2: %f, t3: %f, t4: %f \n", t1, t2, t3, t4);
	}

	t1 = t1/test_round;
	t2 = t2/test_round;
	t3 = t3/test_round;
	t4 = t4/test_round;

	// printf("===================== Time summary ==================\n");
	printf("%f, %f, %f, %f \n", t1, t2, t3, t4);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(deviceRef);

	cudaDeviceReset();

	return (0);
}

