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

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include "matrix_tools.h"


using namespace std;

void do_BlockedFW(int& size, int*& mtx);
void do_FW(int& size, int*& mtx);



int size;
int * mtx;


void do_FW(int& size, int*& mtx){
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


static void HandleError( cudaError_t err, const char* file, int line) {
    if( err != cudaSuccess){
	printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
	exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError((err), __FILE__, __LINE__))

#define SOLVE_IJK(i,j,k) \
    int other = MATRIX_AT(i,k) + MATRIX_AT(k,j); \
    if(MATRIX_AT(i,j) > other){ \
	MATRIX_AT(i,j) = other; \
    }


__global__
void kernel_phase1(const int block, const int s, int* mtx, const int size){
//    int k = threadidx.x;
    
    __shared__ int d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int v1 = MATRIX_BLOCKSIZE * block + ty;
    const int v2 = MATRIX_BLOCKSIZE * block + tx;

    int newlen;

    d[tx][ty] = MATRIX_AT(v1, v2);

    __syncthreads();

    for(int i = 0; i < MATRIX_BLOCKSIZE; i++){
	newlen = d[ty][i] + d[i][tx];

	__syncthreads();

	if(newlen < d[tx][ty]){
	    d[tx][ty] = newlen;
	}

	__syncthreads();
    }

    __syncthreads();

    MATRIX_AT(v1, v2) = d[tx][ty];
}

__global__
void kernel_phase2(const int block, const int s, int* mtx, const int size){
    if (blockIdx.x == block) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int v1 = MATRIX_BLOCKSIZE * block + ty;
    int v2 = MATRIX_BLOCKSIZE * block + tx;
    
    __shared__ int primary_d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];
    __shared__ int current_d[MATRIX_BLOCKSIZE][MATRIX_BLOCKSIZE];
    
    int newlen;
    
    primary_d[ty][tx] = MATRIX_AT(v1, v2);
    
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
	    newlen = primary_d[ty][i] + current_d[i][tx];

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
    
    MATRIX_AT(v1, v2) = current_d[ty][tx];
}

__global__
void kernel_phase3(const int block, const int s, int* mtx, const int size){
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
    

    path = MATRIX_AT(v1, v2);

    primaryRow_d[ty][tx] = MATRIX_AT(v1Row, v2Row);
    primaryCol_d[ty][tx] = MATRIX_AT(v1Col, v2Col);
    
    __syncthreads();
    
    for(i = 0; i < MATRIX_BLOCKSIZE; i++){
	newlen = primaryCol_d[ty][i] + primaryRow_d[i][tx];
	if(path > newlen){
	    path = newlen;
	}
    }
    
    MATRIX_AT(v1, v2) = path;
}


void do_BlockedFW(int& size, int*& mtx){
  //prepare matrix of length
  for(int i = 0; i < size; i++){
    for(int j = 0; j < size; j++){
      if(MATRIX_AT(i,j) < 0){	//not an edge
	MATRIX_AT(i,j) = MAXLEN;
      }
    }
  }

  //length from i to i is 0
  for(int i = 0; i < size; i++){
    MATRIX_AT(i,i) = 0;
  }
  


  int* mx_gpu;

  cout << "CUDA alloc"<<endl;

  //copy matrix to cuda
  HANDLE_ERROR( cudaMalloc(&mtx_gpu,MATRIX_SIZE()) );


  cout << "COPY TO CUDA"<<endl;

  HANDLE_ERROR( cudaMemcpy(mtx_gpu, mtx, MATRIX_SIZE()*sizeof(int), cudaMemcpyHostToDevice) );


  int n = size;	//size of matrix

  if(n%MATRIX_BLOCKSIZE != 0){
    n += MATRIX_BLOCKSIZE - n%MATRIX_BLOCKSIZE;	//align to size of matrix_blocksize
  }


  // Initialize the grid and block dimensions here
  dim3 dimGridP1(1, 1, 1);
  dim3 dimGridP2((n - 1) / MATRIX_BLOCKSIZE + 1, 2 , 1);
  dim3 dimGridP3((n - 1) / MATRIX_BLOCKSIZE + 1, (n - 1) / MATRIX_BLOCKSIZE + 1, 1);

  dim3 dimBlockP1(MATRIX_BLOCKSIZE, MATRIX_BLOCKSIZE, 1);
  dim3 dimBlockP2(MATRIX_BLOCKSIZE, MATRIX_BLOCKSIZE, 1);
  dim3 dimBlockP3(MATRIX_BLOCKSIZE, MATRIX_BLOCKSIZE, 1);


  const int s = MATRIX_BLOCKSIZE;	//size of block

  //Floyd Warshall main loop
  for(int b = 0; b < n/s; b++){

    cout << "Block: " << b <<endl;

    //independent block first
    kernel_phase1<<<1, dimBlockP1>>>(b,s,md,size);

    kernel_phase2<<<dimGridP2, dimBlockP2>>>(b, s, md, size);

    //most blocks are doubly dependent. Notice that k is now innermost
    kernel_phase3<<<dimGridP3, dimBlockP3>>>(b,s,md,size);
  }


  cudaMemcpy(mtx, md, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(md);
}



void runTests(){
    int* mtx1;
    int mtx1_size;
    int* mtx2;
    int mtx2_size;

    const int TEST_SIZE = 101;

    //prepare mtx1
    randomMtx(TEST_SIZE,mtx1_size, mtx1);

    //copy it to mtx2
    mtx2_size = mtx1_size;
    allocMem(TEST_SIZE,mtx2);

//    cout << "copy "<<(mtx1_size+MATRIX_BLOCKSIZE)*(mtx1_size+MATRIX_BLOCKSIZE)<<endl;
//    cout << "mtx1_size "<<(mtx1_size)<<endl;
    memcpy(mtx2, mtx1, sizeof(int)*(mtx1_size+MATRIX_BLOCKSIZE)*(mtx1_size+MATRIX_BLOCKSIZE));



    cout << "do Blocked" << endl;
    //apply fw cuda on mtx1
    do_BlockedFW(mtx1_size, mtx1);


    cout << "do SIMPLE"<<endl;
    //apply fw on mtx2
    do_FW(mtx2_size, mtx2);


//    dump(cout, mtx1_size, mtx1);
//    dump(cout, mtx2_size, mtx2);


    cout << "compare"<<endl;
    //compare results

    int res = memcmp(mtx1, mtx2, sizeof(int)*(mtx1_size+MATRIX_BLOCKSIZE)*(mtx1_size+MATRIX_BLOCKSIZE));
    cout << "Tests: "<<res<<endl;
    
    emptyMem(mtx2_size,mtx2);
    emptyMem(mtx1_size,mtx1);
//    emptyMem(size, mtx);
}


int main(){
    runTests();
    return 0;
//    load(cin, size, mtx);

//    randomMtx(100,size, mtx);

//    do_BlockedFW(size, mtx);

//    dump(cout, size, mtx);
    
//    emptyMem(size, mtx);
}