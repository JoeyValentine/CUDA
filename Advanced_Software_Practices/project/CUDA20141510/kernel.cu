
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Windows.h>
#include <assert.h>

const int ELEM_PER_VECTOR = 32;

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


__global__ void mul_matrix_GPU(float *y, float *mat, float *x)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		result += mat[ eid * ELEM_PER_VECTOR + j ] * x[ vid * ELEM_PER_VECTOR + j ];
	y[ vid * ELEM_PER_VECTOR + eid ] = result;
}

__constant__ float constantMat[ ELEM_PER_VECTOR * ELEM_PER_VECTOR ];
void GenerateConstantMatrix( float *mat )
{
	cudaMemcpyToSymbol( constantMat, mat, sizeof( float ) * ELEM_PER_VECTOR * ELEM_PER_VECTOR );
}


__global__ void mul_matrix_GPU_constant(float *y, float *x)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		result += constantMat[ eid * ELEM_PER_VECTOR + j ] * x[ vid * ELEM_PER_VECTOR + j ];
	y[ vid * ELEM_PER_VECTOR + eid ] = result;
}


__global__ void mul_matrix_GPU_shared(float *y, float* mat, float *x)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	__shared__ float sharedVecX[ 1024 ];
	__shared__ float sharedMatA[ 1024 ];

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )mat )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	sharedVecX[ threadIdx.x ] = x[ tid ];
	unsigned svid = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		result += sharedMatA[ eid * ELEM_PER_VECTOR + j ] * sharedVecX[ svid + j ];

	y[ vid * ELEM_PER_VECTOR + eid ] = result;
}

typedef struct{
	float *elem[ELEM_PER_VECTOR];
}POINTS_SOA;


__global__ void mul_matrix_GPU_constant_SOA(POINTS_SOA y, POINTS_SOA x)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		result += constantMat[ eid * ELEM_PER_VECTOR + j ] * x.elem[j][vid];
	y.elem[eid][vid] = result;
}


__global__ void mul_matrix_GPU_shared_SOA(POINTS_SOA y, float* mat, POINTS_SOA x)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	__shared__ float sharedVecX[ 1024 ];
	__shared__ float sharedMatA[ 1024 ];

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	int accessID = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessID + i * ELEM_PER_VECTOR ] = ( ( float* )mat )[ accessID + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	sharedVecX[ threadIdx.x ] = x.elem[ eid ][ vid ];
	unsigned svid = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		result += sharedMatA[ eid * ELEM_PER_VECTOR + j ] * sharedVecX[ svid + j ];

	y.elem[ eid ][ vid ] = result;
}


__global__ void mul_matrix_GPU_shared_without_bankConflict(float *y, float* mat, float *x)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned vid = tid / ELEM_PER_VECTOR;
	unsigned eid = tid % ELEM_PER_VECTOR;

	__shared__ float sharedVecX[ 1024 ];
	__shared__ float sharedMatA[ 1024 ];

	unsigned ratio = 1024 / blockDim.x; // num of elements in 32x32 Matrix
	int accessIDX = ( threadIdx.x / ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x % ELEM_PER_VECTOR );
	int accessIDX_trans = ( threadIdx.x % ELEM_PER_VECTOR ) * ( ELEM_PER_VECTOR * ratio ) + ( threadIdx.x / ELEM_PER_VECTOR );
	for( unsigned i = 0; i < ratio; ++i )
		sharedMatA[ accessIDX_trans + i * ELEM_PER_VECTOR ] = ( ( float* )mat )[ accessIDX + i * ELEM_PER_VECTOR ];
	__syncthreads( );

	sharedVecX[ threadIdx.x ] = x[ tid ];
	unsigned svid = threadIdx.x / ELEM_PER_VECTOR * ELEM_PER_VECTOR;

	float result = 0.0f;
	for( unsigned j = 0; j < ELEM_PER_VECTOR; ++j )
		result += sharedMatA[ j * ELEM_PER_VECTOR + eid ] * sharedVecX[ svid + j ];

	y[ vid * ELEM_PER_VECTOR + eid ] = result;
}


void mul_matrix_CPU(float *y,float *mat, float *x,int n)
{
	for(int i=0;i<n;++i)
	{
		for(int j=0;j<ELEM_PER_VECTOR;++j)
		{
			y[i*ELEM_PER_VECTOR + j] = 0.0f;
			for(int k=0;k<ELEM_PER_VECTOR;++k)
				y[i*ELEM_PER_VECTOR + j] += mat[j*ELEM_PER_VECTOR + k] * x[i*ELEM_PER_VECTOR + k];
		}
	}
}


void main(void) {

	TIMER_T compute_time = 0;
	TIMER_T device_time = 0;

	FILE *fp = fopen( "gen.bin", "rb" );
	if( fp == NULL )
	{
		printf("No gen.bin\n");
		return;
	}

	int n;
	float *mat = new float[ ELEM_PER_VECTOR * ELEM_PER_VECTOR ];
	fread( &n, sizeof( int ), 1, fp );
	float *x = new float[ ELEM_PER_VECTOR * n ];
	fread( x, sizeof( float ), n * ELEM_PER_VECTOR, fp );
	fread( mat, sizeof( float ), ELEM_PER_VECTOR * ELEM_PER_VECTOR, fp );
	fclose(fp);

	float *y_cpu = new float[ n * ELEM_PER_VECTOR ];
	float *y_gpu = new float[ n * ELEM_PER_VECTOR ];

	POINTS_SOA x_SOA, y_SOA_gpu, cuda_x_SOA, cuda_y_SOA;

	for( int i=0; i<ELEM_PER_VECTOR; ++i )
	{
		x_SOA.elem[i] = (float*)malloc(sizeof(float) * n);
		y_SOA_gpu.elem[i] = (float*)malloc(sizeof(float) * n);
	}

	for( int i=0; i<ELEM_PER_VECTOR; ++i)
		for(int j=0; j<n; ++j)
			x_SOA.elem[i][j] = x[ j * ELEM_PER_VECTOR + i ];

	CHECK_TIME_START();
	mul_matrix_CPU(y_cpu, mat, x, n);
	CHECK_TIME_END( compute_time );
	printf("Elapsed Time by CPU is %f (s).\n", compute_time/1000.0);

	CUDA_CALL(cudaSetDevice(0));

	GenerateConstantMatrix( mat );

	float *cudaY, *cudaX, *cudaMat;

	CUDA_CALL(cudaMalloc(&cudaY, sizeof(float) * n * ELEM_PER_VECTOR));
	CUDA_CALL(cudaMalloc(&cudaX, sizeof(float) * n * ELEM_PER_VECTOR));
	CUDA_CALL(cudaMalloc(&cudaMat, sizeof(float) * ELEM_PER_VECTOR * ELEM_PER_VECTOR));

	for(int i=0;i<ELEM_PER_VECTOR;++i)
	{
		CUDA_CALL(cudaMalloc(&cuda_x_SOA.elem[i], sizeof(float) * n));
		CUDA_CALL(cudaMalloc(&cuda_y_SOA.elem[i], sizeof(float) * n));
	}

	CUDA_CALL(cudaMemcpy(cudaX, x, sizeof(float) * n * ELEM_PER_VECTOR, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cudaMat, mat, sizeof(float) * ELEM_PER_VECTOR * ELEM_PER_VECTOR, cudaMemcpyHostToDevice));

	for(int i=0;i<ELEM_PER_VECTOR;++i)
		CUDA_CALL(cudaMemcpy(cuda_x_SOA.elem[i], x_SOA.elem[i], sizeof(float) * n, cudaMemcpyHostToDevice));

	CHECK_TIME_INIT_GPU();

	size_t _1024Threads = ( 1 << 10 );
	size_t _1024Blocks_perElement = ( n * ELEM_PER_VECTOR ) / _1024Threads;
	size_t _1024Blocks_perVector = n / _1024Threads;

	// global memory
	CHECK_TIME_START_GPU();
	mul_matrix_GPU<<<_1024Blocks_perElement, _1024Threads>>>(cudaY, cudaMat, cudaX);
	cuda_error_check("- ", " FAILED: mul_matrix_GPU()\n\n");
	CHECK_TIME_END_GPU(device_time);
	// error check
	CUDA_CALL(cudaMemcpy(y_gpu, cudaY, sizeof(float)*n * ELEM_PER_VECTOR, cudaMemcpyDeviceToHost));
	CUDA_CALL( cudaDeviceSynchronize() );
	
	int cnt = 0, len = n * ELEM_PER_VECTOR;
	for( int i = 0; i < len; ++i )
		if( y_cpu[ i ] != y_gpu[ i ] )
			cnt++;
	printf("Elapsed Time by GPU1 is %f (s). Error rate is %.2f%%\n", device_time/1000.0, cnt/(float)(n * ELEM_PER_VECTOR)*100);


	// shared memory
	CHECK_TIME_START_GPU();
	mul_matrix_GPU_shared<<<_1024Blocks_perElement, _1024Threads>>>(cudaY, cudaMat, cudaX);
	cuda_error_check("- ", " FAILED: mul_matrix_GPU_shared()\n\n");
	CHECK_TIME_END_GPU(device_time);
	// error check
	CUDA_CALL(cudaMemcpy(y_gpu, cudaY, sizeof(float)*n * ELEM_PER_VECTOR, cudaMemcpyDeviceToHost));
	CUDA_CALL( cudaDeviceSynchronize() );
	
	cnt = 0, len = n * ELEM_PER_VECTOR;
	for( int i = 0; i < len; ++i )
		if( y_cpu[ i ] != y_gpu[ i ] )
			cnt++;
	printf("Elapsed Time by GPU2(shared) is %f (s). Error rate is %.2f%%\n", device_time/1000.0, cnt/(float)(n * ELEM_PER_VECTOR)*100);

	
	// constant memory
	CHECK_TIME_START_GPU();
	mul_matrix_GPU_constant<<<_1024Blocks_perElement, _1024Threads>>>(cudaY, cudaX);
	cuda_error_check("- ", " FAILED: mul_matrix_GPU_constant()\n\n");
	CHECK_TIME_END_GPU(device_time);
	// error check
	CUDA_CALL(cudaMemcpy(y_gpu, cudaY, sizeof(float)*n * ELEM_PER_VECTOR, cudaMemcpyDeviceToHost));
	CUDA_CALL( cudaDeviceSynchronize() );

	cnt = 0, len = n * ELEM_PER_VECTOR;
	for( int i = 0; i < len; ++i )
		if( y_cpu[ i ] != y_gpu[ i ] )
			cnt++;
	printf("Elapsed Time by GPU2(constant) is %f (s). Error rate is %.2f%%\n", device_time/1000.0, cnt/(float)(n * ELEM_PER_VECTOR)*100);
	

	// SOA shared memroy
	CHECK_TIME_START_GPU();
	mul_matrix_GPU_shared_SOA << <_1024Blocks_perElement, _1024Threads>> >(cuda_y_SOA, cudaMat, cuda_x_SOA);
	cuda_error_check("- ", " FAILED: mul_matrix_GPU_shared_SOA()\n\n");
	CHECK_TIME_END_GPU(device_time);
	// error check
	for (int i = 0; i<ELEM_PER_VECTOR; ++i)
		CUDA_CALL(cudaMemcpy(y_SOA_gpu.elem[i], cuda_y_SOA.elem[i], sizeof(float) * n, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaDeviceSynchronize());

	cnt = 0;
	for (int i = 0; i < ELEM_PER_VECTOR; ++i)
		for (int j = 0; j < n; ++j)
			if (y_cpu[j*ELEM_PER_VECTOR + i] != y_SOA_gpu.elem[i][j])
				cnt++;

	printf("Elapsed Time by GPU3(shared) is %f (s). Error rate is %.2f%%\n", device_time/1000.0, cnt / (float)(n * ELEM_PER_VECTOR) * 100);


	// SOA constant memroy
	CHECK_TIME_START_GPU();
	mul_matrix_GPU_constant_SOA<<<_1024Blocks_perElement, _1024Threads>>>(cuda_y_SOA, cuda_x_SOA);
	cuda_error_check("- ", " FAILED: mul_matrix_GPU_constant_SOA()\n\n");
	CHECK_TIME_END_GPU(device_time);
	// error check
	for(int i=0; i<ELEM_PER_VECTOR; ++i)
		CUDA_CALL(cudaMemcpy(y_SOA_gpu.elem[i], cuda_y_SOA.elem[i], sizeof(float) * n, cudaMemcpyDeviceToHost));
	CUDA_CALL( cudaDeviceSynchronize() );

	cnt = 0;
	for( int i = 0; i < ELEM_PER_VECTOR; ++i )
		for( int j = 0; j < n; ++j )
			if( y_cpu[ j*ELEM_PER_VECTOR + i ] != y_SOA_gpu.elem[i][j] )
				cnt++;
	printf("Elapsed Time by GPU3(constant) is %f (s). Error rate is %.2f%%\n", device_time/1000.0, cnt/(float)(n * ELEM_PER_VECTOR)*100);


	// shared memroy without bank_conflict
	CHECK_TIME_START_GPU();
	mul_matrix_GPU_shared_without_bankConflict<< <_1024Blocks_perElement, _1024Threads>> >(cudaY, cudaMat, cudaX);
	cuda_error_check("- ", " FAILED: mul_matrix_GPU_shared_without_bankConflict()\n\n");
	CHECK_TIME_END_GPU(device_time);
	// error check

	CUDA_CALL(cudaMemcpy(y_gpu, cudaY, sizeof(float) * n * ELEM_PER_VECTOR, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaDeviceSynchronize());

	cnt = 0, len = n * ELEM_PER_VECTOR;
	for( int i = 0; i < len; ++i )
		if( y_cpu[ i ] != y_gpu[ i ] )
			cnt++;
	printf("Elapsed Time by GPU4 is %f (s). Error rate is %.2f%%\n", device_time/1000.0, cnt / (float)(n * ELEM_PER_VECTOR) * 100);


	cudaFree(cudaY);
	cudaFree(cudaX);
	cudaFree(cudaMat);
	for(int i=0; i<ELEM_PER_VECTOR; ++i)
	{
		cudaFree(cuda_x_SOA.elem[i]);
		cudaFree(cuda_y_SOA.elem[i]);
	}

	CHECK_TIME_DEST_GPU();
	
	CUDA_CALL(cudaDeviceReset());

	for(int i=0;i<ELEM_PER_VECTOR; ++i)
	{
		free(x_SOA.elem[i]);
		free(y_SOA_gpu.elem[i]);
	}

	delete[] x;
	delete[] y_cpu;
	delete[] y_gpu;
	delete[] mat;
}