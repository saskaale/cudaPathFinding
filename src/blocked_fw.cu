#include <cstdio>
#include <stdio.h>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <climits>

#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "cuda.hpp"

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include "matrix_tools.h"


using namespace std;

void do_BlockedFW(int size, int* mtx);
void do_FW(int size, int* mtx);


//#define COMPARE_WITH_CPU 1



int size;
int * mtx;


void do_FW(int size, int* mtx){
  //prepare matrix of length
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++){
      if(MATRIX_AT(i,j) < 0){   //not an edge
        MATRIX_AT(i,j) = MAXLEN;
      }
    }
  }
  
  //length from i to i is 0
  for(int i = 0; i < size; i++){
    MATRIX_AT(i,i) = 0;
  }


  //Floyd Warshall main loop
  for(int k = 0; k < size; k++){
    for(int i = 0; i < size; i++){
      for(int j = 0; j < size; j++){
        int other = MATRIX_AT(i,k) + MATRIX_AT(k,j);
        if(MATRIX_AT(i,j) > other){
          MATRIX_AT(i,j) = other;
        }
      }
    }
  }
}

#define SOLVE_IJK(i,j,k) \
    int other = MATRIX_AT(i,k) + MATRIX_AT(k,j); \
    if(MATRIX_AT(i,j) > other){ \
	MATRIX_AT(i,j) = other; \
    }


__global__
void kernel_phase1(const int block, int* mtx, const int size){
    __shared__ int d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];

    const int tx = threadIdx.x;	//i
    const int ty = threadIdx.y;	//j

    const int v1 = MATRIX_BLOCKSIZE * block + ty;
    const int v2 = MATRIX_BLOCKSIZE * block + tx;

    int newlen;

    if (v1 < size && v2 < size) 
        d[ty][tx] = MATRIX_AT(v1, v2);
    else
	d[ty][tx] = MAXLEN;


    __syncthreads();

    for(int k = 0; k < MATRIX_BLOCKSIZE; k++){

	newlen = d[ty][k] + d[k][tx];

	__syncthreads();

	if(d[ty][tx] > newlen){
	    d[ty][tx] = newlen;
	}

	__syncthreads();
    }


    if (v1 < size && v2 < size) 
	MATRIX_AT(v1, v2) = d[ty][tx];
}



void phase1(int b, int* mtx, int size){
    for(int k = b*MATRIX_BLOCKSIZE; k < (b+1)*MATRIX_BLOCKSIZE; k++){
        for(int i = b*MATRIX_BLOCKSIZE; i < (b+1)*MATRIX_BLOCKSIZE; i++){
            for(int j = b*MATRIX_BLOCKSIZE; j < (b+1)*MATRIX_BLOCKSIZE; j++){
                SOLVE_IJK(i,j,k);
            }
        }
    }
}


/**Kernel for wake gpu
*
* @param reps dummy variable only to perform some action
*/
__global__ void wake_gpu_kernel(int reps) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reps) return;
}

__global__
void kernel_phase2(const int block, int* mtx, const int size){
    if (blockIdx.x == block) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int v1 = MATRIX_BLOCKSIZE * block + ty;
    int v2 = MATRIX_BLOCKSIZE * block + tx;
    
    __shared__ int primary_d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];
    __shared__ int current_d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];
    
    int newlen;

    if (v1 < size && v2 < size) 
        primary_d[ty][tx] = MATRIX_AT(v1, v2);
    else
	primary_d[ty][tx] = MAXLEN;
    
    // Load i-aligned singly dependent blocks
    if (blockIdx.y == 0)
    {
	v1 = MATRIX_BLOCKSIZE * block + ty;
	v2 = MATRIX_BLOCKSIZE * blockIdx.x + tx;
    }
    // Load j-aligned singly dependent blocks
    else 
    {
	v1 = MATRIX_BLOCKSIZE * blockIdx.x + ty;
	v2 = MATRIX_BLOCKSIZE * block + tx;
    }
    
    if (v1 < size && v2 < size)
	current_d[ty][tx] = MATRIX_AT(v1, v2);
    else
	current_d[ty][tx] = MAXLEN;
    
    __syncthreads();
    
    // i-aligned singly dependent blocks
    if (blockIdx.y == 0)
    {
	for(int i = 0; i < MATRIX_BLOCKSIZE; i++){
	    newlen = primary_d[ty][i] + current_d[i][tx];

	    __syncthreads();
	    if (newlen < current_d[ty][tx])
	    {
		current_d[ty][tx] = newlen;
	    }

	    // Synchronize to make sure that all value are current in block
	    __syncthreads();
	}
    }else{
    // j-aligned singly dependent blocks
	for(int i = 0; i < MATRIX_BLOCKSIZE; i++){
	    newlen = current_d[ty][i] + primary_d[i][tx];

	    __syncthreads();
	    if (newlen < current_d[ty][tx])
	    {
		current_d[ty][tx] = newlen;
	    }

	    // Synchronize to make sure that all value are current in block
	    __syncthreads();
	}
    }

    __syncthreads();

    if (v1 < size && v2 < size) 
	MATRIX_AT(v1, v2) = current_d[ty][tx];
}

__global__
void kernel_phase3(const int block, int* mtx, const int size){
    if (blockIdx.x == block || blockIdx.y == block) return;

    int i;//,j,k;
    int newlen, path;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ int primaryRow_d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];
    __shared__ int primaryCol_d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];


    int v1 = blockDim.y * blockIdx.y + ty;
    int v2 = blockDim.x * blockIdx.x + tx; 


    int v1Row = MATRIX_BLOCKSIZE * block + ty;
    int v2Row = v2;

    int v1Col = v1;
    int v2Col = MATRIX_BLOCKSIZE * block + tx;


    if(v1 < size && v2 < size)
        path = MATRIX_AT(v1, v2);
    else
        path = MAXLEN;

    primaryRow_d[ty][tx] = v1Row < size && v2Row < size ? MATRIX_AT(v1Row, v2Row) : MAXLEN;
    primaryCol_d[ty][tx] = v1Col < size && v2Col < size ? MATRIX_AT(v1Col, v2Col) : MAXLEN;

    __syncthreads();

    for(i = 0; i < MATRIX_BLOCKSIZE; i++){
	newlen = primaryCol_d[ty][i] + primaryRow_d[i][tx];
	if(path > newlen){
	    path = newlen;
	}
    }

    if(v1 < size && v2 < size)
        MATRIX_AT(v1, v2) = path;
}


void phase2(int b, int n, int size, int* mtx){
    //tasks in singly dependent block depend on the independent block
    for(int ib = 0; ib <= n/MATRIX_BLOCKSIZE; ib++){
        for(int k = b*MATRIX_BLOCKSIZE; k < (b+1)*MATRIX_BLOCKSIZE; k++){
            for(int i = b*MATRIX_BLOCKSIZE; i < (b+1)*MATRIX_BLOCKSIZE; i++){
                for(int j = ib*MATRIX_BLOCKSIZE; j < (ib+1)*MATRIX_BLOCKSIZE; j++){
                    if(i >= size || j >= size || k >= size){
                        continue;
                    }
                    SOLVE_IJK(i,j,k);
                }
            }
        }
    }
    
    //j-aligned singly dependent blocks
    for(int jb = 0; jb <= n/MATRIX_BLOCKSIZE; jb++){
        for(int k = b*MATRIX_BLOCKSIZE; k < (b+1)*MATRIX_BLOCKSIZE; k++){
            for(int i = jb*MATRIX_BLOCKSIZE; i < (jb+1)*MATRIX_BLOCKSIZE; i++){
                for(int j = b*MATRIX_BLOCKSIZE; j < (b+1)*MATRIX_BLOCKSIZE; j++){
                    if(i >= size || j >= size || k >= size){
                        continue;
                    }
                    SOLVE_IJK(i,j,k);
                }
            }
        }
    }
}

void phase3(int b, int n, int size, int* mtx){
    for(int ib = 0; ib <= n/MATRIX_BLOCKSIZE; ib++){
        for(int jb = 0; jb <= n/MATRIX_BLOCKSIZE; jb++){
            for(int i = jb*MATRIX_BLOCKSIZE; i < (jb+1)*MATRIX_BLOCKSIZE; i++){
                for(int j = ib*MATRIX_BLOCKSIZE; j < (ib+1)*MATRIX_BLOCKSIZE; j++){
                    for(int k = b*MATRIX_BLOCKSIZE; k < (b+1)*MATRIX_BLOCKSIZE; k++){
                        if(i >= size || j >= size || k >= size){
                            continue;
                        }
                        SOLVE_IJK(i,j,k);
                    }
                }
            }
        }
    }
}

void do_BlockedFW(int size, int* mtx){
  //prepare matrix of length
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++){
      if(MATRIX_AT(i,j) < 0){   //not an edge
        MATRIX_AT(i,j) = MAXLEN;
      }
    }
  }

  //length from i to i is 0
  for(int i = 0; i < size; i++){
    MATRIX_AT(i,i) = 0;
  }

  int* mtx_gpu;
#ifdef COMPARE_WITH_CPU
  int comp;
#endif
  int n = size;	//size of matrix



  HANDLE_ERROR( cudaSetDevice(0) );


  // Initialize the grid and block dimensions here
  dim3 dimGridP1(1, 1, 1);
  dim3 dimGridP2((n - 1) / MATRIX_BLOCKSIZE + 1, 2 , 1);
  dim3 dimGridP3((n - 1) / MATRIX_BLOCKSIZE + 1, (n - 1) / MATRIX_BLOCKSIZE + 1, 1);

  dim3 dimBlockP1(MATRIX_BLOCKSIZE, MATRIX_BLOCKSIZE, 1);
  dim3 dimBlockP2(MATRIX_BLOCKSIZE, MATRIX_BLOCKSIZE, 1);
  dim3 dimBlockP3(MATRIX_BLOCKSIZE, MATRIX_BLOCKSIZE, 1);

//  wake_gpu_kernel<<<1, dimBlockP1>>>(32);




  //copy matrix to cuda
  HANDLE_ERROR( cudaMalloc((void**)&mtx_gpu,MATRIX_SIZE()*sizeof(int)) );

  
  //prepare test matrix
  int* mtx_test = new int[MATRIX_SIZE()];
  memcpy(mtx_test, mtx, MATRIX_SIZE()*sizeof(int));


  HANDLE_ERROR( cudaMemcpy(mtx_gpu, mtx, MATRIX_SIZE()*sizeof(int), cudaMemcpyHostToDevice) );
  

  const int s = MATRIX_BLOCKSIZE;	//size of block

  //Floyd Warshall main loop
  for(int b = 0; b < n/s; b++){
    //independent block first
    kernel_phase1<<<1, dimBlockP1>>>(b,mtx_gpu,size);

#ifdef COMPARE_WITH_CPU
    phase1(b,mtx_test, size);
    HANDLE_ERROR( cudaMemcpy(mtx, mtx_gpu, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost) );
    cout << ".... test1 "<<(comp = memcmp(mtx, mtx_test, MATRIX_SIZE()*sizeof(int)))<<endl;
    if(comp){	cout << "ERROR"<<endl;	exit(EXIT_FAILURE);    }
#endif

    //independent block first
    kernel_phase2<<<dimGridP2, dimBlockP2>>>(b,mtx_gpu,size);


#ifdef COMPARE_WITH_CPU
    phase2(b, n, size, mtx_test);
    HANDLE_ERROR( cudaMemcpy(mtx, mtx_gpu, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost) );
    cout << ".... test2 "<<(comp = memcmp(mtx, mtx_test, MATRIX_SIZE()*sizeof(int)))<<endl;
    if(comp){	cout << "ERROR"<<endl;exit(EXIT_FAILURE);    }
#endif

    //independent block first
    kernel_phase3<<<dimGridP3, dimBlockP3>>>(b,mtx_gpu,size);

#ifdef COMPARE_WITH_CPU
    phase3(b, n, size, mtx_test);
    HANDLE_ERROR( cudaMemcpy(mtx, mtx_gpu, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost) );
    cout << ".... test3 "<<(comp = memcmp(mtx, mtx_test, MATRIX_SIZE()*sizeof(int)))<<endl;
    if(comp){	cout << "ERROR"<<endl; 	exit(EXIT_FAILURE); }
#endif

  }

  HANDLE_ERROR( cudaDeviceSynchronize() );

  HANDLE_ERROR( cudaMemcpy(mtx, mtx_gpu, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost) );

  HANDLE_ERROR( cudaDeviceSynchronize() );

  HANDLE_ERROR( cudaFree(mtx_gpu) );
  
  delete[] mtx_test;

  cout << "after cuda"<<endl;
}


int main(int argc, char* argv[]){
    int s = atoi(argv[1]);
    
    randomMtx(s,size, mtx);

    do_BlockedFW(size, mtx);

//    dump(cout, size, mtx);

//    emptyMem(size, mtx);
}