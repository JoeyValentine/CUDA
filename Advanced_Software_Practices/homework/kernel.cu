
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Windows.h>
#include <assert.h>

const float EPSILON = 0.000001;

#define CUDA_CALL(x) { const cudaError_t a = (x); if(a != cudaSuccess) { printf("\nCuda Error: %s (err_num=%d) at line:%d\n", cudaGetErrorString(a), a, __LINE__); cudaDeviceReset(); assert(0);}}
typedef float TIMER_T;

#define USE_CPU_TIMER 1
#define USE_GPU_TIMER 1

#if USE_CPU_TIMER == 1
__int64 start, freq, end;
#define CHECK_TIME_START() { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }
#else
#define CHECK_TIME_START
#define CHECK_TIME_END(a)
#endif

#if USE_GPU_TIMER == 1
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

void create_device_timer()
{
	CUDA_CALL(cudaEventCreate(&cuda_timer_start));
	CUDA_CALL(cudaEventCreate(&cuda_timer_stop));
}

void destroy_device_timer()
{
	CUDA_CALL( cudaEventDestroy( cuda_timer_start ) );
	CUDA_CALL( cudaEventDestroy( cuda_timer_stop ) );
}

inline void start_device_timer()
{
	cudaEventRecord(cuda_timer_start, CUDA_STREAM_0);
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord(cuda_timer_stop, CUDA_STREAM_0);
	cudaEventSynchronize(cuda_timer_stop);

	cudaEventElapsedTime(&ms, cuda_timer_start, cuda_timer_stop);
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }
#else
#define CHECK_TIME_INIT_GPU()
#define CHECK_TIME_START_GPU()
#define CHECK_TIME_END_GPU(a)
#define CHECK_TIME_DEST_GPU()
#endif

__host__ void cuda_error_check(const char * prefix, const char * postfix)
{
	if (cudaPeekAtLastError() != cudaSuccess)
	{
		printf("%s%s%s", prefix, cudaGetErrorString(cudaGetLastError()), postfix);
		cudaDeviceReset();
		//wait_exit();
		exit(1);
	}
}

__global__ void find_roots_GPU(float *A, float *B, float *C,
	float *X0, float *X1, float *FX0, float *FX1)
{
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	float a, b, c, d, x0, x1, tmp;
	a = A[tid]; b = B[tid]; c = C[tid];
	d = sqrtf(b*b - 4.0f*a*c);
	tmp = 1.0f / (2.0f*a);
	X0[tid] = x0 = (-b - d) * tmp;
	X1[tid] = x1 = (-b + d) * tmp;
	FX0[tid] = (a*x0 + b)*x0 + c;
	FX1[tid] = (a*x1 + b)*x1 + c;
}

void find_roots_CPU(float *A, float *B, float *C,
	float *X0, float *X1, float *FX0, float *FX1, int n) 
{
	int i;
	float a, b, c, d, x0, x1, tmp;
	for (i = 0; i < n; i++) {
		a = A[i]; b = B[i]; c = C[i];
		d = sqrtf(b*b - 4.0f*a*c);
		tmp = 1.0f / (2.0f*a);
		X0[i] = x0 = (-b - d) * tmp;
		X1[i] = x1 = (-b + d) * tmp;
		FX0[i] = (a*x0 + b)*x0 + c;
		FX1[i] = (a*x1 + b)*x1 + c;
	}
}

void main(void) {

	TIMER_T compute_time = 0;
	TIMER_T device_time = 0;

	FILE *fp = fopen( "abc.bin", "rb" );

	if( fp == NULL )
	{
		printf("There does not exists abc.bin.\n");
		return;
	}

	int n_equation;
	fread( &n_equation, sizeof( int ), 1, fp );
	float *A = new float[n_equation];
	float *B = new float[n_equation];
	float *C = new float[n_equation];
	fread(A, sizeof(float), n_equation, fp);
	fread(B, sizeof(float), n_equation, fp);
	fread(C, sizeof(float), n_equation, fp);
	fclose(fp);

	float *X0_cpu = new float[n_equation];
	float *X1_cpu = new float[n_equation];
	float *FX0_cpu = new float[n_equation];
	float *FX1_cpu = new float[n_equation];

	float *X0_gpu = new float[n_equation];
	float *X1_gpu = new float[n_equation];
	float *FX0_gpu = new float[n_equation];
	float *FX1_gpu = new float[n_equation];

	printf( "*** CPU Works...\n" );
	CHECK_TIME_START();
	find_roots_CPU(A, B, C, X0_cpu, X1_cpu, FX0_cpu, FX1_cpu, n_equation);
	CHECK_TIME_END( compute_time );
	printf( " - Finish\n\n" );

	CUDA_CALL(cudaSetDevice(0));

	float *cuda_A, *cuda_B, *cuda_C, *cuda_X0, *cuda_X1, *cuda_FX0, *cuda_FX1;

	CUDA_CALL(cudaMalloc(&cuda_A, sizeof(float) * n_equation));
	CUDA_CALL(cudaMalloc(&cuda_B, sizeof(float) * n_equation));
	CUDA_CALL(cudaMalloc(&cuda_C, sizeof(float) * n_equation));
	CUDA_CALL(cudaMalloc(&cuda_X0, sizeof(float) * n_equation));
	CUDA_CALL(cudaMalloc(&cuda_X1, sizeof(float) * n_equation));
	CUDA_CALL(cudaMalloc(&cuda_FX0, sizeof(float) * n_equation));
	CUDA_CALL(cudaMalloc(&cuda_FX1, sizeof(float) * n_equation));

	printf("*** Copying A, B and C from host to device...\n");
	CUDA_CALL(cudaMemcpy(cuda_A, A, sizeof(float) * n_equation, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cuda_B, B, sizeof(float) * n_equation, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cuda_C, C, sizeof(float) * n_equation, cudaMemcpyHostToDevice));
	printf(" - Finish\n\n");

	CHECK_TIME_INIT_GPU();

	size_t n_threads = (1<<10);
	size_t n_blocks = n_equation / n_threads;

	printf("*** kernel call: find_roots_GPU<<< %d, %d >>>()...\n", n_blocks, n_threads);
	CHECK_TIME_START_GPU();
	find_roots_GPU<<<n_blocks, n_threads >>>(cuda_A, cuda_B, cuda_C, cuda_X0, cuda_X1, cuda_FX0, cuda_FX1);
	cuda_error_check("- ", " FAILED: find_roots_GPU()\n\n");
	//CUDA_CALL(cudaDeviceSynchronize());
	CHECK_TIME_END_GPU(device_time);
	printf( " - Finish\n\n" );

	printf("*** Time taken = %.6fms(CPU), %.6fms(GPU)\n", compute_time, device_time);

	printf("*** Copying X0, X1, FX0, FX1 from device to host...\n");
	CUDA_CALL(cudaMemcpy(X0_gpu, cuda_X0, sizeof(float) * n_equation, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(X1_gpu, cuda_X1, sizeof(float) * n_equation, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(FX0_gpu, cuda_FX0, sizeof(float) * n_equation, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(FX1_gpu, cuda_FX1, sizeof(float) * n_equation, cudaMemcpyDeviceToHost));
	CUDA_CALL( cudaDeviceSynchronize() );
	
	int cnt = 0;
	for( int i = 0; i < n_equation; ++i )
	{
		if( X0_cpu[i] != X0_gpu[i] )
		{
			cnt++;
		}
	}
	printf("\nX0 - %.2f%% numerical errors...\n", cnt / (float)(n_equation) * 100);

	cnt = 0;
	for (int i = 0; i < n_equation; ++i)
	{
		if ( X1_cpu[i] != X1_gpu[i] )
		{
			cnt++;
		}
	}
	printf("X1 - %.2f%% numerical errors...\n", cnt / (float)(n_equation) * 100);

	cnt = 0;
	for (int i = 0; i < n_equation; ++i)
	{
		if ( fabs(FX0_cpu[i] - FX0_gpu[i]) >= EPSILON )
		{
			cnt++;
		}
	}
	printf("FX0 - %.2f%% numerical errors...\n", cnt / (float)(n_equation) * 100);

	cnt = 0;
	for (int i = 0; i < n_equation; ++i)
	{
		if ( fabs(FX1_cpu[i] - FX1_gpu[i]) >= EPSILON )
		{
			cnt++;
		}
	}
	printf("FX1 - %.2f%% numerical errors...\n", cnt / (float)(n_equation) * 100);

	fp = fopen("X0.bin", "wb");
	fwrite(&n_equation, sizeof(int), 1, fp);
	fwrite(X0_gpu, sizeof(float), n_equation, fp);
	fclose(fp);

	fp = fopen("X1.bin", "wb");
	fwrite(&n_equation, sizeof(int), 1, fp);
	fwrite(X1_gpu, sizeof(float), n_equation, fp);
	fclose(fp);

	fp = fopen("FX0.bin", "wb");
	fwrite(&n_equation, sizeof(int), 1, fp);
	fwrite(FX0_gpu, sizeof(float), n_equation, fp);
	fclose(fp);

	fp = fopen("FX1.bin", "wb");
	fwrite(&n_equation, sizeof(int), 1, fp);
	fwrite(FX1_gpu, sizeof(float), n_equation, fp);
	fclose(fp);

	printf(" - Finish\n\n");
	
	cudaFree(cuda_A);
	cudaFree(cuda_B);
	cudaFree(cuda_C);
	cudaFree(cuda_X0);
	cudaFree(cuda_X1);
	cudaFree(cuda_FX0);
	cudaFree(cuda_FX1);

	CHECK_TIME_DEST_GPU();
	
	CUDA_CALL(cudaDeviceReset());

	delete[] A;
	delete[] B;
	delete[] C;
	
	delete[] X0_cpu;
	delete[] X1_cpu;
	delete[] FX0_cpu;
	delete[] FX1_cpu;

	delete[] X0_gpu;
	delete[] X1_gpu;
	delete[] FX0_gpu;
	delete[] FX1_gpu;

}