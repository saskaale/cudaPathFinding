#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <string.h>
#include <fstream>

#include "matrix_tools.h"
#include "priority_queue.hpp"
#include "cuda.hpp"

using namespace std;

int size;
int * mtx;

#define BURST_SIZE 1024

__global__
void DijkstraCuda(int * mtx, int * dists_gpu, int size, void * heap_mem_pool, int start_idx){
    int source=start_idx+blockIdx.x*BURST_SIZE+threadIdx.x;
    //printf("source %d (%d %d), size %d\n", source, blockIdx.x, threadIdx.x, size);
    if (source>=size) return;
    void * heap_mem=((char*)heap_mem_pool)+HEAP_SIZE(size)*source;
    priority_queue queue(size, heap_mem);
    
    int *dist=dists_gpu+size*source;
    queue.update(source, 0);
    dist[source]=0;
    while(! queue.empty()){
        int c=queue.pop();
        
        while(! queue.empty()){
            int c=queue.pop();
            int dist_to_c=dist[c];
            dist[c]=dist_to_c;
	    #pragma unroll 8
            for (int i=0; i<size; i++) {
                if (MATRIX_AT(c,i)<0) continue; //not a neighbour
                int new_dist_to_i=dist_to_c+MATRIX_AT(c,i);
                if (new_dist_to_i<dist[i]) {
                    queue.update(i, new_dist_to_i);
		    dist[i]=new_dist_to_i;
                }
            }
        }
    }
    if (source==0) printf("kernel ended\n");
}

#define USE_CUDA 1
int main(int argc, char ** argv){
    if (argc<2) load(cin, size, mtx);
    else randomMtx(atoi(argv[1]), size, mtx);
#if USE_CUDA
    int * mtx_gpu;
    int * dists_gpu;
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaMalloc(&mtx_gpu, MATRIX_SIZE()*sizeof(int));
    cudaMemcpy(mtx_gpu, mtx, MATRIX_SIZE()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dists_gpu, MATRIX_SIZE()*sizeof(int));
    void * heap_pool_gpu;
    cudaMalloc(&heap_pool_gpu, HEAP_SIZE(size)*size);
    //cuMemsetD32(heap_pool_gpu, INT_MAX, HEAP_SIZE(size)*size/sizeof(int));
    int block_cnt=(size+BURST_SIZE-1)/BURST_SIZE;
    printf("starting kernel with %d blocks of %d threads\n", block_cnt, BURST_SIZE);
    cudaFuncSetCacheConfig(DijkstraCuda, cudaFuncCachePreferL1);
    DijkstraCuda<<<block_cnt, BURST_SIZE>>>(mtx_gpu, dists_gpu, size, heap_pool_gpu, 0);
    cudaDeviceSynchronize();
    cudaFree(heap_pool_gpu);
    HANDLE_ERROR(cudaGetLastError());
    cudaMemcpy(mtx, dists_gpu, MATRIX_SIZE()*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(mtx_gpu);
    cudaFree(dists_gpu);
#else /*USE_CUDA*/
    do_Dijkstra();
#endif /*USE_CUDA*/
    
    emptyMem(size, mtx);
}
